#!/usr/bin/env python3
"""
vllm_bench.py - Scenario-based vLLM benchmarking tool

- Optional per-scenario model: set `model` or `model_name` in a scenario (override for `vllm serve` / `vllm bench`).
- One or more config files as positional args: `vllm_bench.py a.yaml b.yaml` runs a full study per file.
- Benchmark backend: default `bench.engine` is `vllm_bench` (`vllm bench serve`). Set `bench.engine: guidellm` and
  `bench.guidellm` to run `guidellm benchmark` instead (same server lifecycle). For a dedicated install, set
  `bench.guidellm.env_path` to the venv/conda env root (uses `bin/guidellm` or `bin/python -m guidellm`), or
  `bench.guidellm.executable` to the `guidellm` binary path. See `configs/guidellm_synthetic_profiles.yaml`.
- MLflow: a parent run for the study plus a nested run after each scenario (and final study_dir + metadata on the parent).
  Nested runs also set tags from `logs/vllm_bench_conc*.log` when using vLLM bench (parsed Serving Benchmark Result table).
"""

# CSV columns (every summary row includes all keys; unused fields are empty strings).
SUMMARY_CSV_FIELDNAMES = [
    "scenario",
    "model",
    "port",
    "concurrency",
    "num_prompts",
    "status",
    "runtime_sec",
    "result_file",
    "bench_engine",
    "guidellm_profile",
]

import argparse
import csv
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml


# vLLM `vllm bench serve` may print 40+10 in source, or wider/padded columns in the real log; we parse
# any `name:  <number>` line where the value is the last field (same as the user's sample).
_MLFLOW_BENCH_LOG_END_EQ = re.compile(r"^={50}$")
# Final field is a scalar; label is the vLLM metric name ending at the first ":" on the line
# (sufficient for current bench output, including "Request throughput (req/s):"-style names).
_MLFLOW_BENCH_KV = re.compile(
    r"^(?P<label>.+?:)\s+(?P<val>"
    r"[-+]?(?:\d+\.?\d*|\d*\.\d+)(?:[eE][-+]?\d+)?|"
    r"inf|[-+]?inf|nan|[-+]?nan|Infinity|[-+]?Infinity"
    r")\s*$"
)


def _bench_log_value_looks_scalar(val: str) -> bool:
    t = val.strip()
    if not t:
        return False
    if t.lower() in ("inf", "infinity", "+inf", "-inf", "nan", "+nan", "-nan"):
        return True
    try:
        float(t)
    except ValueError:
        return False
    return True


def _label_to_bench_tag_key(label: str) -> str:
    s = label.strip().rstrip(":")
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s, flags=re.IGNORECASE)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    if not s:
        s = "metric"
    if s[0].isdigit() or s.startswith("_"):
        s = f"m_{s}"
    return s[:200]


def _iter_bench_conc_log_files(logs_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for path in logs_dir.glob("vllm_bench_conc*.log"):
        m = re.match(r"vllm_bench_conc(\d+)\.log$", path.name, re.IGNORECASE)
        if m:
            out.append((int(m.group(1)), path))
    out.sort(key=lambda t: t[0])
    return out


def parse_vllm_bench_serving_log(text: str) -> Dict[str, str]:
    """
    Parse the last 'Serving Benchmark Result' section from a vllm_bench_conc*.log
    and return key/value strings suitable for MLflow tags.
    """
    if "Serving Benchmark Result" not in text:
        return {}
    lines = text.splitlines()
    start: Optional[int] = None
    for i, line in enumerate(lines):
        if "Serving Benchmark Result" in line:
            start = i
    if start is None:
        return {}
    out: Dict[str, str] = {}
    for line in lines[start + 1 :]:
        raw = line.rstrip("\n")
        if _MLFLOW_BENCH_LOG_END_EQ.match(raw):
            break
        if len(raw) < 5:
            continue
        # Skip vLLM section break lines (no trailing scalar); real logs can be >50 columns wide
        m = _MLFLOW_BENCH_KV.match(raw.strip())
        if not m:
            continue
        label, val = m.group("label"), m.group("val")
        if not label or not re.sub(r"[\s-]", "", label):
            continue
        if not _bench_log_value_looks_scalar(val):
            continue
        key = _label_to_bench_tag_key(label)
        out[key] = val
    return out


def _bench_log_tags_for_scenario_dir(scenario_dir: Path) -> Dict[str, str]:
    """
    For each logs/vllm_bench_concN.log, parse the benchmark table and return tags
    prefixed with cN_ (e.g. c32_request_throughput, c32_p50_ttft).
    """
    tags: Dict[str, str] = {}
    logs_dir = scenario_dir / "logs"
    if not logs_dir.is_dir():
        return tags
    parsed = 0
    for conc, log_path in _iter_bench_conc_log_files(logs_dir):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        row = parse_vllm_bench_serving_log(text)
        if row:
            parsed += 1
        prefix = f"c{conc}_"
        for k, v in row.items():
            tags[prefix + k] = v
    tags["bench_serving_log_files_parsed"] = str(parsed)
    return tags


class TeeStream:
    """Mirror writes to terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


class VLLMBenchmark:
    def __init__(self, config_path: str, scenario_filter: Optional[List[str]] = None,
                 nsys_delay: Optional[float] = None, nsys_duration: Optional[float] = None,
                 enable_mlflow: bool = True, mlflow_experiment: Optional[str] = None,
                 mlflow_run_name: Optional[str] = None, mlflow_tracking_uri: Optional[str] = None,
                 mlflow_tags: Optional[Dict[str, str]] = None,
                 cli_command: Optional[List[str]] = None):
        self.config_path = Path(config_path)
        self.scenario_filter = scenario_filter
        self.nsys_delay = nsys_delay
        self.nsys_duration = nsys_duration
        self.enable_mlflow = enable_mlflow
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_run_name = mlflow_run_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_tags = mlflow_tags or {}
        self.cli_command = cli_command or sys.argv[:]
        self.config = self._load_config()
        self.study_dir = self._setup_study_dir()
        self.summary_data = []
        self.server_process = None
        self._stdout_log_handle = None
        self._stderr_log_handle = None
        self._stdout_original = None
        self._stderr_original = None
        self._mlflow_parent_active = False
        self.run_log_path = self.study_dir / "logs" / "benchmark_output.log"
        self.stderr_log_path = self.study_dir / "logs" / "benchmark_stderr.log"
        self.command_file_path = self.study_dir / "metadata" / "command.txt"
        self.nvidia_smi_file_path = self.study_dir / "metadata" / "nvidia-smi.txt"
        self.lscpu_file_path = self.study_dir / "metadata" / "lscpu.txt"
        self.nsys_version_file_path = self.study_dir / "metadata" / "nsys_version.txt"
        self.git_version_file_path = self.study_dir / "metadata" / "git_version.txt"
        self.git_diff_file_path = self.study_dir / "metadata" / "git_diff.patch"
        self.os_release_file_path = self.study_dir / "metadata" / "os-release.txt"

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Return filesystem-safe scenario identifier."""
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)

    def _effective_model_name(self, scenario: Optional[Dict[str, Any]] = None) -> str:
        """Model for a scenario: scenario['model'] or 'model_name' overrides config default."""
        if scenario:
            override = scenario.get("model") or scenario.get("model_name")
            if override is not None and str(override).strip():
                return str(override).strip()
        return str(self.config["model"]["name"])

    def _setup_run_logging(self):
        """Capture stdout to run log; route stderr to dedicated file."""
        self.run_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stdout_original = sys.stdout
        self._stderr_original = sys.stderr
        self._stdout_log_handle = open(self.run_log_path, "a", buffering=1)
        self._stderr_log_handle = open(self.stderr_log_path, "a", buffering=1)
        sys.stdout = TeeStream(self._stdout_original, self._stdout_log_handle)
        sys.stderr = self._stderr_log_handle

    def _teardown_run_logging(self):
        """Restore stdout/stderr and close run log."""
        if self._stdout_original is not None:
            sys.stdout = self._stdout_original
        if self._stderr_original is not None:
            sys.stderr = self._stderr_original
        if hasattr(self, "_stdout_log_handle") and self._stdout_log_handle:
            self._stdout_log_handle.close()
            self._stdout_log_handle = None
        if hasattr(self, "_stderr_log_handle") and self._stderr_log_handle:
            self._stderr_log_handle.close()
            self._stderr_log_handle = None

    def _capture_command_and_system_info(self):
        """Capture command line and machine information into study metadata."""
        metadata_dir = self.study_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        command_contents = [
            f"timestamp={datetime.now().isoformat()}",
            f"cwd={Path.cwd()}",
            f"python={sys.executable}",
            f"argv={shlex.join(self.cli_command)}",
        ]
        self.command_file_path.write_text("\n".join(command_contents) + "\n")

        for cmd, output_file in (
            (["nvidia-smi"], self.nvidia_smi_file_path),
            (["lscpu"], self.lscpu_file_path),
            (["nsys", "--version"], self.nsys_version_file_path),
        ):
            try:
                result = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                output = (
                    f"$ {' '.join(cmd)}\n"
                    f"exit_code={result.returncode}\n\n"
                    f"{result.stdout}"
                )
                if result.stderr:
                    output += f"\n[stderr]\n{result.stderr}"
            except FileNotFoundError:
                output = f"$ {' '.join(cmd)}\ncommand not found on this system\n"
            except Exception as exc:
                output = f"$ {' '.join(cmd)}\nfailed to collect output: {exc}\n"

            output_file.write_text(output)

        # Capture git revision metadata and diffs for reproducibility.
        try:
            git_version = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10
            )
            git_describe = subprocess.run(
                ["git", "describe", "--always", "--dirty", "--tags"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10
            )
            git_version_contents = (
                "$ git rev-parse HEAD\n"
                f"exit_code={git_version.returncode}\n"
                f"{git_version.stdout}\n"
                "$ git describe --always --dirty --tags\n"
                f"exit_code={git_describe.returncode}\n"
                f"{git_describe.stdout}\n"
            )
            if git_version.stderr:
                git_version_contents += f"\n[rev-parse stderr]\n{git_version.stderr}"
            if git_describe.stderr:
                git_version_contents += f"\n[describe stderr]\n{git_describe.stderr}"
            self.git_version_file_path.write_text(git_version_contents)
        except Exception as exc:
            self.git_version_file_path.write_text(f"Failed to capture git version: {exc}\n")

        try:
            git_diff = subprocess.run(
                ["git", "diff", "--no-color"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20
            )
            git_diff_cached = subprocess.run(
                ["git", "diff", "--no-color", "--cached"],
                check=False,
                capture_output=True,
                text=True,
                timeout=20
            )
            git_diff_contents = (
                "$ git diff --no-color\n"
                f"exit_code={git_diff.returncode}\n\n"
                f"{git_diff.stdout}\n"
                "$ git diff --no-color --cached\n"
                f"exit_code={git_diff_cached.returncode}\n\n"
                f"{git_diff_cached.stdout}\n"
            )
            if git_diff.stderr:
                git_diff_contents += f"\n[diff stderr]\n{git_diff.stderr}"
            if git_diff_cached.stderr:
                git_diff_contents += f"\n[cached diff stderr]\n{git_diff_cached.stderr}"
            self.git_diff_file_path.write_text(git_diff_contents)
        except Exception as exc:
            self.git_diff_file_path.write_text(f"Failed to capture git diff: {exc}\n")

        # Store /etc/os-release snapshot for OS provenance.
        try:
            self.os_release_file_path.write_text(Path("/etc/os-release").read_text())
        except Exception as exc:
            self.os_release_file_path.write_text(f"Failed to read /etc/os-release: {exc}\n")

    def _mlflow_set_tracking(self):
        """Set MLflow tracking URI and experiment. Caller must have imported mlflow."""
        import mlflow  # type: ignore

        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        if self.mlflow_experiment:
            mlflow.set_experiment(self.mlflow_experiment)

    def _mlflow_set_tracking_uri_only(self):
        """Re-apply only the tracking URI (for nested run helpers while a parent is active)."""
        import mlflow  # type: ignore

        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

    def _mlflow_begin_parent_run(self) -> bool:
        """
        Start a parent run for the study. Nested child runs (one per scenario) follow.
        """
        if not self.enable_mlflow:
            return False
        try:
            import mlflow  # type: ignore
        except ImportError:
            print("⚠️  MLflow not installed. Skipping MLflow upload.")
            return False

        self._mlflow_set_tracking()

        run_name = self.mlflow_run_name or self.study_dir.name
        print(f"📤 MLflow parent run: {run_name}")

        mlflow.start_run(run_name=run_name)
        if self.mlflow_tags:
            mlflow.set_tags(self.mlflow_tags)
        mlflow.log_param("vllm_bench_parent_run", True)
        mlflow.log_param("config_path", str(self.config_path.resolve()))
        mlflow.log_param("study_dir", str(self.study_dir))
        mlflow.log_param("default_model", self.config["model"]["name"])
        mlflow.log_param("scenario_count", len(self.config["scenarios"]))
        if self.scenario_filter:
            mlflow.log_param("scenario_filter", ",".join(self.scenario_filter))
        if self.nsys_delay is not None:
            mlflow.log_param("nsys_delay_sec", self.nsys_delay)
        if self.nsys_duration is not None:
            mlflow.log_param("nsys_duration_sec", self.nsys_duration)
        return True

    def _mlflow_log_scenario_run(
        self, scenario: Dict[str, Any], model_name: str, scenario_dir: Path
    ):
        """Nested MLflow run: upload one scenario’s directory right after it finishes."""
        if not self.enable_mlflow:
            return
        try:
            import mlflow  # type: ignore
        except ImportError:
            return

        self._mlflow_set_tracking_uri_only()

        sname = scenario.get("name", "unknown")
        run_name = self._sanitize_name(sname)
        print(f"📤 MLflow (scenario run): {run_name}")

        with mlflow.start_run(
            run_name=run_name,
            nested=True,
        ):
            mlflow.set_tag("scenario", sname)
            _bdef = self.config.get("defaults", {}).get("bench", {})
            _bsc = scenario.get("bench", {})
            mlflow.set_tag(
                "bench_engine",
                str({**_bdef, **_bsc}.get("engine", "vllm_bench")).lower(),
            )
            log_tags = _bench_log_tags_for_scenario_dir(scenario_dir)
            if log_tags:
                mlflow.set_tags(log_tags)
            mlflow.log_param("model", model_name)
            mlflow.log_param("config_path", str(self.config_path.resolve()))
            mlflow.log_param("study_dir", str(self.study_dir))
            mlflow.log_param("scenario_name", sname)
            if scenario_dir.is_dir():
                mlflow.log_artifacts(str(scenario_dir), artifact_path="scenario")

    def _mlflow_log_study_artifacts(self):
        """On the active (parent) run, log study-level files (no per-scenario dirs)."""
        if not self.enable_mlflow:
            return
        try:
            import mlflow  # type: ignore
        except ImportError:
            return

        self._mlflow_set_tracking_uri_only()

        for path, ap in (
            (self.nvidia_smi_file_path, "system"),
            (self.lscpu_file_path, "system"),
            (self.command_file_path, "metadata"),
            (self.study_dir / "config.yaml", "metadata"),
            (self.run_log_path, "logs"),
            (self.stderr_log_path, "logs"),
        ):
            p = Path(path)
            if p.is_file():
                mlflow.log_artifact(str(p), artifact_path=ap)
        summ = self.study_dir / "summary.csv"
        if summ.is_file():
            mlflow.log_artifact(str(summ), artifact_path="results")
        # Full study tree (same as legacy single end-upload); per-scenario nested runs
        # already log each `scenario_*` for incremental progress.
        mlflow.log_artifacts(str(self.study_dir), artifact_path="study_dir")

    def _mlflow_end_parent_run(self):
        """Log study-level artifacts and close the parent run."""
        if not self.enable_mlflow:
            return
        try:
            import mlflow  # type: ignore
        except ImportError:
            return

        self._mlflow_set_tracking_uri_only()
        print("📤 MLflow: finalizing parent run (study metadata + summary)")
        self._mlflow_log_study_artifacts()
        # Active run should still be the parent; nested children were closed by their with-blocks.
        mlflow.end_run()
        print("✅ MLflow parent run complete")

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate scenario-based config."""
        print(f"📄 Loading config: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Validate
        if 'model' not in config or 'name' not in config['model']:
            raise ValueError("Invalid config: missing 'model.name'")

        if 'scenarios' not in config or not config['scenarios']:
            raise ValueError("No scenarios found in config")

        # Filter scenarios
        scenarios = config['scenarios']
        if self.scenario_filter:
            scenarios = [s for s in scenarios if s.get('name') in self.scenario_filter]
            if not scenarios:
                raise ValueError(f"No scenarios matched filter: {self.scenario_filter}")
            print(f"🔍 Filtered scenarios: {[s['name'] for s in scenarios]}")

        config['scenarios'] = scenarios
        return config

    def _setup_study_dir(self) -> Path:
        """Create study directory structure."""
        model_name = self.config['model']['name'].replace('/', '_')
        defaults = self.config.get('defaults', {})
        study_name = defaults.get('study_dir', f'Study_{model_name}')

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        study_dir = Path(f"{study_name}_{timestamp}").resolve()
        study_dir.mkdir(parents=True, exist_ok=True)

        print(f"📁 Study directory: {study_dir}")

        # Copy config
        shutil.copy(self.config_path, study_dir / 'config.yaml')

        return study_dir

    def _apply_env_vars(self, env_dict: Dict[str, str]):
        """Apply environment variables."""
        if not env_dict:
            return

        print("🌍 Environment variables:")
        for key, value in env_dict.items():
            os.environ[key] = str(value)
            print(f"  {key}={value}")

    def _wait_for_server(self, port: int, timeout: int = 120) -> bool:
        """Wait for vLLM server to be ready."""
        url = f"http://localhost:{port}/v1/models"
        print(f"⏳ Waiting for server on port {port}...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    print("✅ Server ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)

        print("❌ Server failed to start (timeout)")
        return False

    def _get_nsys_session_id(self) -> Optional[str]:
        """Get the active nsys session ID using awk.

        Expected output format:
        ID         TIME                                      STATE LAUNCH NAME
        <session_id> <timestamp>                            <state> <name>
        """
        try:
            # Use awk to skip header and extract first column (session ID)
            # NR==2 means "line number 2" (first data line after header)
            result = subprocess.run(
                "nsys sessions list | awk 'NR==2 {print $1}'",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5
            )

            session_id = result.stdout.strip()
            if session_id:
                return session_id

        except Exception as e:
            print(f"⚠️  Failed to get nsys session ID: {e}")

        return None

    def _start_server(
        self,
        scenario_name: str,
        port: int,
        params: List[str],
        log_file: Path,
        model_name: Optional[str] = None,
        nsys_profile: bool = False,
        nsys_args: str = "",
        nsys_output_prefix: Optional[Path] = None
    ) -> tuple[Optional[subprocess.Popen], Optional[str]]:
        """Start vLLM server. Returns (process, nsys_session_id)."""
        model_name = (model_name or self.config['model']['name'])

        use_session_mode = False
        cmd = []

        if nsys_profile:
            nsys_launch_args = nsys_args or "--trace=cuda,nvtx,osrt"
            nsys_launch_tokens = shlex.split(nsys_launch_args)
            output_parent = nsys_output_prefix.parent.resolve() if nsys_output_prefix else self.study_dir
            # Check if using --start-later (session mode)
            use_session_mode = any(
                token == "--start-later"
                or token.startswith("--start-later=")
                for token in nsys_launch_tokens
            )

            if use_session_mode:
                print("🔧 Using nsys session mode (start-later)")

            has_output_arg = any(
                token in ("-o", "--output")
                or token.startswith("--output=")
                or (token.startswith("-o") and len(token) > 2)
                for token in nsys_launch_tokens
            )

            if has_output_arg:
                # Keep user-provided output name, but anchor relative paths to scenario profiles dir.
                rewritten_tokens: List[str] = []
                i = 0
                while i < len(nsys_launch_tokens):
                    token = nsys_launch_tokens[i]
                    if token in ("-o", "--output") and i + 1 < len(nsys_launch_tokens):
                        output_val = Path(nsys_launch_tokens[i + 1])
                        if not output_val.is_absolute():
                            output_val = output_parent / output_val
                        rewritten_tokens.extend([token, str(output_val.resolve())])
                        i += 2
                        continue
                    if token.startswith("--output="):
                        output_val = Path(token.split("=", 1)[1])
                        if not output_val.is_absolute():
                            output_val = output_parent / output_val
                        rewritten_tokens.append(f"--output={output_val.resolve()}")
                        i += 1
                        continue
                    if token.startswith("-o") and len(token) > 2:
                        output_val = Path(token[2:])
                        if not output_val.is_absolute():
                            output_val = output_parent / output_val
                        rewritten_tokens.append(f"-o{output_val.resolve()}")
                        i += 1
                        continue
                    rewritten_tokens.append(token)
                    i += 1
                nsys_launch_tokens = rewritten_tokens
            elif nsys_output_prefix:
                nsys_launch_tokens.extend(["-o", str(nsys_output_prefix.resolve())])

            cmd = ["nsys", "profile"] + nsys_launch_tokens

        cmd += ["vllm", "serve", model_name, "--port", str(port)] + params

        print(f"▶️  Starting server: {' '.join(cmd[:5])}...")

        log_file.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid
        )
        relay_thread = threading.Thread(
            target=self._relay_process_output_to_logs,
            args=(process, log_file),
            daemon=True
        )
        relay_thread.start()

        print(f"🔑 PID: {process.pid}")

        # Get session ID if using session mode
        nsys_session_id = None
        if use_session_mode:
            # Wait a moment for nsys to initialize
            time.sleep(2)
            nsys_session_id = self._get_nsys_session_id()
            if nsys_session_id:
                print(f"📊 Nsys session ID: {nsys_session_id}")
            else:
                print("⚠️  Could not retrieve nsys session ID")

        return process, nsys_session_id

    def _stop_server(self, process: subprocess.Popen):
        """Stop vLLM server gracefully."""
        if not process:
            return

        print(f"🛑 Stopping server (PID: {process.pid})")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

        time.sleep(2)

    def _relay_process_output_to_logs(self, process: subprocess.Popen, log_file: Path):
        """Write long-running process output only to scenario log file."""
        if process.stdout is None:
            return

        with open(log_file, "a", buffering=1) as file_stream:
            for line in process.stdout:
                file_stream.write(line)

    def _run_command_logged(
        self,
        cmd: List[str],
        check: bool = False,
        output_file: Optional[Path] = None,
        echo_to_stdout: bool = True,
        echo_to_stderr: bool = True
    ) -> int:
        """Run command and route stdout/stderr independently."""
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        assert process.stdout is not None
        assert process.stderr is not None
        output_handle = None
        try:
            if output_file:
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_handle = open(output_file, "a", buffering=1)

            def stream_stdout():
                for line in process.stdout:
                    if output_handle:
                        output_handle.write(line)
                    if echo_to_stdout:
                        print(line, end="")

            def stream_stderr():
                for line in process.stderr:
                    if output_handle:
                        output_handle.write(line)
                    if echo_to_stderr:
                        print(line, end="", file=sys.stderr)

            stdout_thread = threading.Thread(target=stream_stdout, daemon=True)
            stderr_thread = threading.Thread(target=stream_stderr, daemon=True)
            stdout_thread.start()
            stderr_thread.start()
            stdout_thread.join()
            stderr_thread.join()
        finally:
            if output_handle:
                output_handle.close()

        return_code = process.wait()
        if check and return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        return return_code

    def _normalize_bench_engine(self, bench_config: Dict[str, Any]) -> str:
        engine = str(bench_config.get("engine", "vllm_bench")).strip().lower()
        if engine in ("vllm", "vllm_bench", "bench", "serve"):
            return "vllm_bench"
        if engine in ("guidellm", "guide_llm"):
            return "guidellm"
        raise ValueError(
            f"Unknown bench.engine {engine!r}; use 'vllm_bench' or 'guidellm'."
        )

    def _resolve_guidellm_cmd_prefix(self, g: Dict[str, Any]) -> List[str]:
        """
        Return argv prefix to invoke GuideLLM: either [guidellm], [/env/bin/guidellm],
        or [/env/bin/python, '-m', 'guidellm'] when only the env Python has the package.
        """
        exe = g.get("executable")
        if exe:
            p = Path(str(exe)).expanduser()
            if not p.is_file():
                raise ValueError(f"bench.guidellm.executable is not a file: {p}")
            return [str(p.resolve())]

        env_path = (
            g.get("env_path")
            or g.get("venv")
            or g.get("venv_path")
            or g.get("conda_env")
        )
        if env_path:
            root = Path(str(env_path)).expanduser().resolve()
            if not root.is_dir():
                raise ValueError(f"bench.guidellm.env_path is not a directory: {root}")
            if platform.system() == "Windows":
                bind = root / "Scripts"
                gw = bind / "guidellm.exe"
                pythons = (bind / "python.exe", bind / "python3.exe")
            else:
                bind = root / "bin"
                gw = bind / "guidellm"
                pythons = (bind / "python", bind / "python3")
            if gw.is_file():
                return [str(gw.resolve())]
            for pyexe in pythons:
                if pyexe.is_file():
                    return [str(pyexe.resolve()), "-m", "guidellm"]
            raise ValueError(
                f"GuideLLM not found under env_path {root}: expected {gw} "
                f"or one of {pythons} with the `guidellm` package installed."
            )

        return ["guidellm"]

    def _guidellm_rate_string(self, bench_config: Dict[str, Any]) -> str:
        g = bench_config.get("guidellm") or {}
        r = g.get("rate")
        if r is not None and r != "":
            if isinstance(r, (list, tuple)):
                return ",".join(str(x) for x in r)
            return str(r).strip()
        conc = bench_config.get("concurrencies")
        if conc:
            return ",".join(str(x) for x in conc)
        raise ValueError(
            "GuideLLM requires bench.guidellm.rate or bench.concurrencies to build --rate."
        )

    def _build_vllm_bench_steps(
        self, concurrencies: List[int], cc_mult: int, scenario_dir: Path
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for c in concurrencies:
            out.append(
                {
                    "engine": "vllm_bench",
                    "slug": f"conc{c}",
                    "torch_suffix": f"conc{c}",
                    "concurrency": c,
                    "num_prompts": int(c * cc_mult),
                    "log_file": scenario_dir / "logs" / f"vllm_bench_conc{c}.log",
                }
            )
        return out

    def _build_guidellm_bench_steps(
        self,
        bench_config: Dict[str, Any],
        result_prefix: str,
        scenario_dir: Path,
    ) -> List[Dict[str, Any]]:
        g = bench_config.get("guidellm")
        if not isinstance(g, dict):
            raise ValueError("bench.engine is 'guidellm' but bench.guidellm must be a mapping.")
        out_pfx = g.get("output_prefix", result_prefix)
        results_dir = scenario_dir / "results"
        logs_dir = scenario_dir / "logs"
        steps: List[Dict[str, Any]] = []
        profiles = g.get("profiles")
        if profiles:
            if not isinstance(profiles, list):
                raise ValueError("bench.guidellm.profiles must be a list.")
            for i, p in enumerate(profiles):
                if not isinstance(p, dict):
                    raise ValueError(f"bench.guidellm.profiles[{i}] must be a mapping.")
                label = p.get("label") or p.get("name") or f"profile{i}"
                data = p.get("data")
                if data is None:
                    raise ValueError(f"bench.guidellm profile {label!r} missing 'data'.")
                safe = self._sanitize_name(str(label))
                out_name = f"{out_pfx}-{safe}.json"
                steps.append(
                    {
                        "engine": "guidellm",
                        "slug": f"g_{safe}",
                        "torch_suffix": f"guidellm_{safe}",
                        "label": str(label),
                        "data": data,
                        "log_file": logs_dir / f"guidellm_{safe}.log",
                        "result_json": results_dir / out_name,
                    }
                )
        else:
            data = g.get("data")
            if data is None:
                raise ValueError(
                    "GuideLLM needs bench.guidellm.data or bench.guidellm.profiles (non-empty)."
                )
            lbl = str(g.get("profile_label", "default"))
            safe = self._sanitize_name(lbl)
            raw_out = g.get("outputs")
            if raw_out:
                outputs_name = Path(str(raw_out)).name
            else:
                outputs_name = f"{out_pfx}.json" if lbl == "default" else f"{out_pfx}-{safe}.json"
            steps.append(
                {
                    "engine": "guidellm",
                    "slug": f"g_{safe}",
                    "torch_suffix": f"guidellm_{safe}",
                    "label": lbl,
                    "data": data,
                    "log_file": logs_dir / f"guidellm_{safe}.log",
                    "result_json": results_dir / outputs_name,
                }
            )
        return steps

    def _run_guidellm_benchmark(
        self,
        port: int,
        model_name: str,
        bench_config: Dict[str, Any],
        step: Dict[str, Any],
        bench_log_file: Path,
    ) -> Tuple[str, int]:
        g = bench_config["guidellm"]
        target = str(g.get("target") or f"http://127.0.0.1:{port}")
        processor = str(g.get("processor") or model_name)
        data = step["data"]
        data_str = data if isinstance(data, str) else json.dumps(data)
        rate = self._guidellm_rate_string(bench_config)
        rate_type = str(g.get("rate_type", "concurrent"))
        max_seconds = int(g.get("max_seconds", 450))
        bk = g.get("backend_kwargs", {"timeout": 100000})
        if isinstance(bk, str):
            bk = json.loads(bk)
        if not isinstance(bk, dict):
            raise ValueError("bench.guidellm.backend_kwargs must be a mapping or JSON string.")
        out_dir = step["result_json"].parent
        outputs = step["result_json"].name
        prefix = self._resolve_guidellm_cmd_prefix(g)
        cmd: List[str] = prefix + [
            "benchmark",
            "--target",
            target,
            "--model",
            model_name,
            "--processor",
            processor,
            "--data",
            data_str,
            "--rate-type",
            rate_type,
            "--rate",
            rate,
            "--backend-kwargs",
            json.dumps(bk),
            "--max-seconds",
            str(max_seconds),
            "--output-dir",
            str(out_dir),
            "--outputs",
            outputs,
        ]
        extra = g.get("extra_args")
        if extra:
            if isinstance(extra, str):
                cmd.extend(shlex.split(extra))
            else:
                cmd.extend(str(x) for x in extra)
        label = step.get("label", "default")
        print(f"===== GuideLLM profile: {label} (rate={rate}) =====")
        print(f"🧭 GuideLLM command: {shlex.join(prefix)} …")
        print(f"📝 GuideLLM log: {bench_log_file}")
        print(f"▶ Running: {shlex.join(cmd[:12])}{' …' if len(cmd) > 12 else ''}")
        start_time = time.time()
        try:
            self._run_command_logged(
                cmd,
                check=True,
                output_file=bench_log_file,
                echo_to_stdout=False,
                echo_to_stderr=False,
            )
            status = "success"
        except subprocess.CalledProcessError:
            status = "failed"
        runtime = int(time.time() - start_time)
        print(f"✅ GuideLLM ({label}) complete: {status} ({runtime}s)")
        return status, runtime

    def _run_benchmark(
        self,
        scenario_name: str,
        port: int,
        concurrency: int,
        num_prompts: int,
        input_len: int,
        output_len: int,
        result_file: Path,
        bench_log_file: Path,
        model_name: Optional[str] = None,
        enable_profiling: bool = False
    ) -> tuple[str, int]:
        """Run vLLM benchmark."""
        model_name = (model_name or self.config['model']['name'])

        cmd = [
            "vllm", "bench", "serve",
            "--base-url", f"http://localhost:{port}",
            "--model", model_name,
            "--dataset-name", "random",
            "--random-input-len", str(input_len),
            "--random-output-len", str(output_len),
            "--max-concurrency", str(concurrency),
            "--num-prompts", str(num_prompts),
            "--seed", str(int(time.time())),
            "--save-result",
            "--result-filename", str(result_file),
            "--append-result"
        ]

        if enable_profiling:
            cmd.append("--profile")

        print(f"===== Concurrency: {concurrency} ({num_prompts} prompts) =====")
        print(f"📝 vllm bench log: {bench_log_file}")
        print(f"▶ Running vllm bench at concurrency={concurrency}")

        start_time = time.time()
        try:
            self._run_command_logged(
                cmd,
                check=True,
                output_file=bench_log_file,
                echo_to_stdout=False,
                echo_to_stderr=False
            )
            status = "success"
        except subprocess.CalledProcessError:
            status = "failed"

        runtime = int(time.time() - start_time)
        print(f"✅ Concurrency {concurrency} complete: {status} ({runtime}s)")
        return status, runtime

    def _run_scenario(self, scenario: Dict) -> str:
        """Run a single scenario. Returns the effective model name for this scenario."""
        scenario_name = scenario['name']
        scenario_slug = self._sanitize_name(scenario_name)
        print("\n" + "=" * 50)
        print(f"📋 Scenario: {scenario_name}")
        print("=" * 50)

        # Setup paths
        scenario_dir = self.study_dir / f"scenario_{scenario_name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / 'logs').mkdir(exist_ok=True)
        (scenario_dir / 'results').mkdir(exist_ok=True)
        (scenario_dir / 'profiles').mkdir(exist_ok=True)

        log_file = scenario_dir / 'logs' / f'vllm_server_{scenario_slug}.log'

        # Merge parameters
        base_params = self.config['model'].get('base_params', '')
        scenario_params = scenario.get('params', '')
        full_params = f"{base_params} {scenario_params}".strip().split()

        model_name = self._effective_model_name(scenario)
        print(f"🎯 Model: {model_name}")
        if model_name != str(self.config["model"]["name"]):
            print(f"   (overrides default {self.config['model']['name']!r})")

        # Handle compilation config
        if 'compilation_config' in scenario:
            comp_config = scenario['compilation_config']
            if isinstance(comp_config, str):
                comp_config = yaml.safe_load(comp_config)
            full_params.extend(['--compilation-config', json.dumps(comp_config)])

        port = scenario.get('port', 8000)
        print(f"🌐 Port: {port}")
        print(f"⚙️  Params: {' '.join(full_params[:3])}...")

        # Get benchmark settings
        defaults = self.config.get('defaults', {})
        bench_defaults = defaults.get('bench', {})
        bench_config = {**bench_defaults, **scenario.get('bench', {})}

        bench_engine = self._normalize_bench_engine(bench_config)
        print(f"🔧 Benchmark engine: {bench_engine}")

        input_len = bench_config.get('input_len', 2000)
        output_len = bench_config.get('output_len', 200)
        cc_mult = bench_config.get('cc_mult', 10)
        result_prefix = bench_config.get('result_prefix', f"scenario_{scenario_name}")
        result_file = scenario_dir / 'results' / f"{result_prefix}.json"
        concurrencies = bench_config.get('concurrencies', [1, 32, 128])

        if bench_engine == 'vllm_bench':
            if not concurrencies:
                raise ValueError(f"Scenario '{scenario_name}' has empty concurrencies list")
            bench_steps = self._build_vllm_bench_steps(concurrencies, cc_mult, scenario_dir)
        else:
            bench_steps = self._build_guidellm_bench_steps(
                bench_config, result_prefix, scenario_dir
            )

        first_step_slug = bench_steps[0]['slug']
        guidellm_rate_str = (
            self._guidellm_rate_string(bench_config) if bench_engine == 'guidellm' else ''
        )

        # Profiling settings
        profile_nsys = scenario.get('profile', False)
        nsys_launch_args = scenario.get('profiling', {}).get('nsys_launch_args', '')
        torch_enabled = scenario.get('profiling', {}).get('torch_profiler', {}).get('enabled', False)

        # Setup torch profiler environment BEFORE launching server
        torch_dir = None
        if torch_enabled:
            torch_dir = scenario_dir / 'profiles' / 'torch'
            torch_dir.mkdir(parents=True, exist_ok=True)

            torch_config = scenario.get('profiling', {}).get('torch_profiler', {})
            print(f"🔥 Torch profiler enabled → {torch_dir}")

            os.environ['VLLM_TORCH_PROFILER_DIR'] = str(torch_dir)
            os.environ['VLLM_TORCH_PROFILER_RECORD_SHAPES'] = str(torch_config.get('record_shapes', True)).lower()
            os.environ['VLLM_TORCH_PROFILER_PROFILE_MEMORY'] = str(torch_config.get('profile_memory', True)).lower()
            os.environ['VLLM_TORCH_PROFILER_WITH_STACK'] = str(torch_config.get('with_stack', False)).lower()
            os.environ['VLLM_TORCH_PROFILER_WITH_FLOPS'] = str(torch_config.get('with_flops', False)).lower()

        # Start server
        self.server_process, nsys_session_id = self._start_server(
            scenario_name, port, full_params, log_file, model_name=model_name,
            nsys_profile=profile_nsys, nsys_args=nsys_launch_args,
            nsys_output_prefix=scenario_dir / 'profiles' / f'nsys_{scenario_slug}_{first_step_slug}'
        )

        if not self.server_process or not self._wait_for_server(port):
            print(f"❌ Failed to start server for scenario: {scenario_name}")
            if self.server_process:
                self._stop_server(self.server_process)
            return model_name

        # Run benchmarks (vLLM: one step per concurrency; GuideLLM: one step per profile)
        for step in bench_steps:
            bench_log_file = step['log_file']

            # Update torch profiler prefix for this step
            if torch_enabled:
                os.environ['VLLM_TORCH_PROFILER_PREFIX'] = f"trace_{step['torch_suffix']}"

            # Setup nsys profiling (start in background if delay is specified)
            nsys_file = None
            nsys_stop_timer = None
            nsys_started_event = None
            nsys_stopped_event = None
            if profile_nsys:
                nsys_file = (scenario_dir / 'profiles' / f"nsys_{scenario_slug}_{step['slug']}").resolve()
                nsys_start_args = scenario.get('profiling', {}).get('nsys_start_args', '--force-overwrite=true')
                nsys_stopped_event = threading.Event()

                def start_nsys_profiling():
                    """Start nsys profiling and optional duration timer."""
                    nonlocal nsys_stop_timer
                    if nsys_session_id:
                        # Session-based profiling
                        print(f"🎥 nsys start --session={nsys_session_id} → {nsys_file}.qdrep")
                        cmd = ['nsys', 'start', f'--session={nsys_session_id}']
                        cmd += nsys_start_args.split() + ['--output', str(nsys_file)]
                        self._run_command_logged(cmd)
                    else:
                        # Legacy mode (no session)
                        print(f"🎥 nsys start → {nsys_file}.qdrep")
                        self._run_command_logged(
                            ['nsys', 'start'] + nsys_start_args.split() + ['--output', str(nsys_file)]
                        )

                    # Mark that nsys has started
                    if nsys_started_event:
                        nsys_started_event.set()

                    # If duration is specified, start a timer to auto-stop nsys
                    if self.nsys_duration:
                        print(f"⏱️  Nsys will auto-stop after {self.nsys_duration} seconds from now")

                        def stop_nsys_after_duration():
                            if nsys_session_id:
                                print(f"\n⏰ Duration expired - stopping nsys (session={nsys_session_id})")
                                self._run_command_logged(['nsys', 'stop', f'--session={nsys_session_id}'])
                            else:
                                print("\n⏰ Duration expired - stopping nsys")
                                self._run_command_logged(['nsys', 'stop'])
                            print(f"✅ Saved: {nsys_file}.qdrep")
                            if nsys_stopped_event:
                                nsys_stopped_event.set()

                        nsys_stop_timer = threading.Timer(self.nsys_duration, stop_nsys_after_duration)
                        nsys_stop_timer.start()

                if self.nsys_delay:
                    # Start nsys in background after delay, benchmark continues immediately
                    print(f"⏱️  Nsys will start after {self.nsys_delay} seconds (benchmark starts now)")
                    nsys_started_event = threading.Event()

                    def delayed_nsys_start():
                        time.sleep(self.nsys_delay)
                        start_nsys_profiling()

                    nsys_thread = threading.Thread(target=delayed_nsys_start, daemon=True)
                    nsys_thread.start()
                else:
                    # Start nsys immediately (original behavior)
                    start_nsys_profiling()

            # Run benchmark (starts immediately, concurrent with delayed nsys start if applicable)
            if step['engine'] == 'guidellm':
                status, runtime = self._run_guidellm_benchmark(
                    port, model_name, bench_config, step, bench_log_file
                )
            else:
                status, runtime = self._run_benchmark(
                    scenario_name, port, step['concurrency'], step['num_prompts'],
                    input_len, output_len, result_file, bench_log_file,
                    model_name=model_name,
                    enable_profiling=torch_enabled
                )

            # Stop nsys profiling (only if duration was not specified)
            if profile_nsys:
                if self.nsys_duration:
                    # Duration-based stop is handled by timer
                    # Wait for nsys to start if it hasn't yet
                    if nsys_started_event and not nsys_started_event.is_set():
                        print(f"⏱️  Waiting for delayed nsys to start before duration timer kicks in...")
                        nsys_started_event.wait()
                    print(f"⏱️  Benchmark completed. Waiting for duration-based nsys stop before next step.")
                    if nsys_stop_timer and nsys_stopped_event:
                        wait_timeout = max(self.nsys_duration + 60, 120)
                        if not nsys_stopped_event.wait(timeout=wait_timeout):
                            print("⚠️  Timed out waiting for nsys duration stop; continuing.")
                else:
                    # Manual stop after benchmark completion (original behavior)
                    # Wait for nsys to start if delay was specified
                    if nsys_started_event and not nsys_started_event.is_set():
                        print(f"⏱️  Waiting for delayed nsys to start before stopping...")
                        nsys_started_event.wait()

                    if nsys_session_id:
                        print(f"🛑 nsys stop --session={nsys_session_id}")
                        self._run_command_logged(['nsys', 'stop', f'--session={nsys_session_id}'])
                    else:
                        print("🛑 nsys stop")
                        self._run_command_logged(['nsys', 'stop'])
                    print(f"✅ Saved: {nsys_file}.qdrep")

            # Report torch profiler output
            if torch_dir:
                print(f"✅ Torch traces: {torch_dir}/")
                trace_files = list(torch_dir.glob(f"trace_{step['torch_suffix']}*"))
                for f in trace_files:
                    print(f"    {f.name} ({f.stat().st_size} bytes)")

            # Record summary
            if step['engine'] == 'vllm_bench':
                row_conc = str(step['concurrency'])
                row_prompts = str(step['num_prompts'])
                row_result = str(result_file)
                row_profile = ''
            else:
                row_conc = guidellm_rate_str
                row_prompts = ''
                row_result = str(step['result_json'])
                row_profile = str(step['label'])

            self.summary_data.append({
                'scenario': scenario_name,
                'model': model_name,
                'port': port,
                'concurrency': row_conc,
                'num_prompts': row_prompts,
                'status': status,
                'runtime_sec': runtime,
                'result_file': row_result,
                'bench_engine': step['engine'],
                'guidellm_profile': row_profile,
            })

        # Stop server
        self._stop_server(self.server_process)
        return model_name

    def run(self):
        """Run all scenarios."""
        self._mlflow_parent_active = False
        try:
            self._setup_run_logging()
            self._capture_command_and_system_info()

            print("🚀 Scenario benchmark (vLLM bench or GuideLLM)")
            print(f"🎯 Default model: {self.config['model']['name']}")
            print("   (per-scenario override: set `model` or `model_name` in the scenario)")
            print(f"📊 Scenarios: {len(self.config['scenarios'])}")

            if self._mlflow_begin_parent_run():
                self._mlflow_parent_active = True

            # Apply global environment variables
            env_defaults = self.config.get('defaults', {}).get('env', {})
            self._apply_env_vars(env_defaults)

            # Setup summary file
            summary_file = self.study_dir / 'summary.csv'
            with open(summary_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDNAMES)
                writer.writeheader()

            # Run each scenario; MLflow child run (nested) after each
            for scenario in self.config['scenarios']:
                self._run_scenario(scenario)
                if self._mlflow_parent_active:
                    sdir = self.study_dir / f"scenario_{scenario['name']}"
                    self._mlflow_log_scenario_run(
                        scenario, self._effective_model_name(scenario), sdir
                    )
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            if self.server_process:
                self._stop_server(self.server_process)
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.server_process:
                self._stop_server(self.server_process)
            raise
        finally:
            try:
                summary_file = self.study_dir / 'summary.csv'
                if self.summary_data:
                    with open(summary_file, 'w', newline='') as f:
                        writer = csv.DictWriter(
                            f, fieldnames=SUMMARY_CSV_FIELDNAMES, extrasaction='ignore'
                        )
                        writer.writeheader()
                        writer.writerows(self.summary_data)
                if self._mlflow_parent_active:
                    self._mlflow_end_parent_run()
                    self._mlflow_parent_active = False
            finally:
                self._teardown_run_logging()

        print("\n" + "=" * 50)
        print("🎉 All scenarios completed!")
        print(f"📊 Summary: {self.study_dir / 'summary.csv'}")
        print(f"📁 Study directory: {self.study_dir}")


def main():
    def parse_mlflow_tags(raw_tags: Optional[List[str]]) -> Dict[str, str]:
        """Parse repeated KEY=VALUE arguments into a dict."""
        parsed: Dict[str, str] = {}
        for raw in raw_tags or []:
            if "=" not in raw:
                raise ValueError(f"Invalid MLflow tag '{raw}'. Expected KEY=VALUE format.")
            key, value = raw.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError(f"Invalid MLflow tag '{raw}'. Tag key cannot be empty.")
            parsed[key] = value
        return parsed

    parser = argparse.ArgumentParser(
        description="Scenario-based benchmarking (vLLM bench serve or GuideLLM)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'config',
        nargs='+',
        help='Path(s) to scenario config YAML. Each file is one benchmark run (separate study directory), in order.',
    )
    parser.add_argument(
        '--scenario', '--scenarios',
        dest='scenarios',
        help='Comma-separated list of scenarios to run'
    )
    parser.add_argument(
        '--delay',
        type=float,
        help='Delay (in seconds) before starting nsys profiling session'
    )
    parser.add_argument(
        '--duration',
        type=float,
        help='Duration (in seconds) for nsys profiling. If specified, nsys will stop after this duration instead of at benchmark completion'
    )
    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Disable MLflow artifact upload at the end of the run'
    )
    parser.add_argument(
        '--mlflow-experiment',
        help='MLflow experiment name'
    )
    parser.add_argument(
        '--mlflow-run-name',
        help='MLflow run name (defaults to study directory name)'
    )
    parser.add_argument(
        '--mlflow-tracking-uri',
        help='MLflow tracking URI (optional)'
    )
    parser.add_argument(
        '--mlflow-tag',
        action='append',
        dest='mlflow_tags',
        default=[],
        help='MLflow tag in KEY=VALUE format. Repeat for multiple tags.'
    )

    args = parser.parse_args()

    scenario_filter = None
    if args.scenarios:
        scenario_filter = [s.strip() for s in args.scenarios.split(',')]

    try:
        mlflow_tags = parse_mlflow_tags(args.mlflow_tags)
    except ValueError as e:
        parser.error(str(e))

    exit_code = 0
    n_config = len(args.config)
    for k, config_path in enumerate(args.config):
        if n_config > 1:
            print(f"\n======== Config {k + 1}/{n_config}: {config_path} ========\n")
        try:
            benchmark = VLLMBenchmark(
                config_path,
                scenario_filter,
                nsys_delay=args.delay,
                nsys_duration=args.duration,
                enable_mlflow=not args.no_mlflow,
                mlflow_experiment=args.mlflow_experiment,
                mlflow_run_name=args.mlflow_run_name,
                mlflow_tracking_uri=args.mlflow_tracking_uri,
                mlflow_tags=mlflow_tags,
                cli_command=sys.argv,
            )
            benchmark.run()
        except Exception as e:
            print(f"❌ Fatal error (config: {config_path!r}): {e}", file=sys.stderr)
            exit_code = 1
            break
    if exit_code:
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
