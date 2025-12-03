#!/usr/bin/env python3
"""
vllm_bench.py - Scenario-based vLLM benchmarking tool
"""

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml


class VLLMBenchmark:
    def __init__(self, config_path: str, scenario_filter: Optional[List[str]] = None,
                 nsys_delay: Optional[float] = None, nsys_duration: Optional[float] = None):
        self.config_path = Path(config_path)
        self.scenario_filter = scenario_filter
        self.nsys_delay = nsys_delay
        self.nsys_duration = nsys_duration
        self.config = self._load_config()
        self.study_dir = self._setup_study_dir()
        self.summary_data = []
        self.server_process = None

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
        study_dir = Path(f"{study_name}_{timestamp}")
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
        nsys_profile: bool = False,
        nsys_args: str = ""
    ) -> tuple[Optional[subprocess.Popen], Optional[str]]:
        """Start vLLM server. Returns (process, nsys_session_id)."""
        model_name = self.config['model']['name']

        use_session_mode = False
        cmd = []

        if nsys_profile:
            nsys_launch_args = nsys_args or "--trace=cuda,nvtx,osrt"
            # Check if using --start-later (session mode)
            use_session_mode = "--start-later=true" in nsys_launch_args or "--start-later true" in nsys_launch_args

            if use_session_mode:
                print("🔧 Using nsys session mode (start-later)")

            cmd = ["nsys", "profile"] + nsys_launch_args.split()

        cmd += ["vllm", "serve", model_name, "--port", str(port)] + params

        print(f"▶️  Starting server: {' '.join(cmd[:5])}...")

        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )

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

    def _run_benchmark(
        self,
        scenario_name: str,
        port: int,
        concurrency: int,
        num_prompts: int,
        input_len: int,
        output_len: int,
        result_file: Path,
        enable_profiling: bool = False
    ) -> tuple[str, int]:
        """Run vLLM benchmark."""
        model_name = self.config['model']['name']

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

        start_time = time.time()
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            status = "success"
        except subprocess.CalledProcessError:
            status = "failed"

        runtime = int(time.time() - start_time)
        return status, runtime

    def _run_scenario(self, scenario: Dict):
        """Run a single scenario."""
        scenario_name = scenario['name']
        print("\n" + "=" * 50)
        print(f"📋 Scenario: {scenario_name}")
        print("=" * 50)

        # Setup paths
        scenario_dir = self.study_dir / f"scenario_{scenario_name}"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        (scenario_dir / 'logs').mkdir(exist_ok=True)
        (scenario_dir / 'results').mkdir(exist_ok=True)
        (scenario_dir / 'profiles').mkdir(exist_ok=True)

        log_file = scenario_dir / 'logs' / 'vllm_server.log'

        # Merge parameters
        base_params = self.config['model'].get('base_params', '')
        scenario_params = scenario.get('params', '')
        full_params = f"{base_params} {scenario_params}".strip().split()

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

        concurrencies = bench_config.get('concurrencies', [1, 32, 128])
        input_len = bench_config.get('input_len', 2000)
        output_len = bench_config.get('output_len', 200)
        cc_mult = bench_config.get('cc_mult', 10)
        result_prefix = bench_config.get('result_prefix', f"scenario_{scenario_name}")

        result_file = scenario_dir / 'results' / f"{result_prefix}.json"

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
            scenario_name, port, full_params, log_file,
            nsys_profile=profile_nsys, nsys_args=nsys_launch_args
        )

        if not self.server_process or not self._wait_for_server(port):
            print(f"❌ Failed to start server for scenario: {scenario_name}")
            if self.server_process:
                self._stop_server(self.server_process)
            return

        # Run benchmarks for each concurrency
        for concurrency in concurrencies:
            num_prompts = concurrency * cc_mult

            # Update torch profiler prefix for this concurrency
            if torch_enabled:
                os.environ['VLLM_TORCH_PROFILER_PREFIX'] = f"trace_conc{concurrency}"

            # Setup nsys profiling (start in background if delay is specified)
            nsys_file = None
            nsys_stop_timer = None
            nsys_started_event = None
            if profile_nsys:
                nsys_file = scenario_dir / 'profiles' / f"nsys_conc{concurrency}"
                nsys_start_args = scenario.get('profiling', {}).get('nsys_start_args', '--force-overwrite=true')

                def start_nsys_profiling():
                    """Start nsys profiling and optional duration timer."""
                    if nsys_session_id:
                        # Session-based profiling
                        print(f"🎥 nsys start --session={nsys_session_id} → {nsys_file}.qdrep")
                        cmd = ['nsys', 'start', f'--session={nsys_session_id}']
                        cmd += nsys_start_args.split() + ['--output', str(nsys_file)]
                        subprocess.run(cmd)
                    else:
                        # Legacy mode (no session)
                        print(f"🎥 nsys start → {nsys_file}.qdrep")
                        subprocess.run(['nsys', 'start'] + nsys_start_args.split() + ['--output', str(nsys_file)])

                    # Mark that nsys has started
                    if nsys_started_event:
                        nsys_started_event.set()

                    # If duration is specified, start a timer to auto-stop nsys
                    if self.nsys_duration:
                        print(f"⏱️  Nsys will auto-stop after {self.nsys_duration} seconds from now")

                        def stop_nsys_after_duration():
                            if nsys_session_id:
                                print(f"\n⏰ Duration expired - stopping nsys (session={nsys_session_id})")
                                subprocess.run(['nsys', 'stop', f'--session={nsys_session_id}'])
                            else:
                                print("\n⏰ Duration expired - stopping nsys")
                                subprocess.run(['nsys', 'stop'])
                            print(f"✅ Saved: {nsys_file}.qdrep")

                        timer = threading.Timer(self.nsys_duration, stop_nsys_after_duration)
                        timer.start()

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
            status, runtime = self._run_benchmark(
                scenario_name, port, concurrency, num_prompts,
                input_len, output_len, result_file,
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
                    print(f"⏱️  Benchmark completed. Nsys will be stopped by duration timer.")
                else:
                    # Manual stop after benchmark completion (original behavior)
                    # Wait for nsys to start if delay was specified
                    if nsys_started_event and not nsys_started_event.is_set():
                        print(f"⏱️  Waiting for delayed nsys to start before stopping...")
                        nsys_started_event.wait()

                    if nsys_session_id:
                        print(f"🛑 nsys stop --session={nsys_session_id}")
                        subprocess.run(['nsys', 'stop', f'--session={nsys_session_id}'])
                    else:
                        print("🛑 nsys stop")
                        subprocess.run(['nsys', 'stop'])
                    print(f"✅ Saved: {nsys_file}.qdrep")

            # Report torch profiler output
            if torch_dir:
                print(f"✅ Torch traces: {torch_dir}/")
                trace_files = list(torch_dir.glob(f"trace_conc{concurrency}*"))
                for f in trace_files:
                    print(f"    {f.name} ({f.stat().st_size} bytes)")

            # Record summary
            self.summary_data.append({
                'scenario': scenario_name,
                'port': port,
                'concurrency': concurrency,
                'num_prompts': num_prompts,
                'status': status,
                'runtime_sec': runtime,
                'result_file': str(result_file)
            })

        # Stop server
        self._stop_server(self.server_process)

    def run(self):
        """Run all scenarios."""
        print("🚀 vLLM Scenario Benchmark")
        print(f"🎯 Model: {self.config['model']['name']}")
        print(f"📊 Scenarios: {len(self.config['scenarios'])}")

        # Apply global environment variables
        env_defaults = self.config.get('defaults', {}).get('env', {})
        self._apply_env_vars(env_defaults)

        # Setup summary file
        summary_file = self.study_dir / 'summary.csv'
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'scenario', 'port', 'concurrency', 'num_prompts',
                'status', 'runtime_sec', 'result_file'
            ])
            writer.writeheader()

        # Run each scenario
        try:
            for scenario in self.config['scenarios']:
                self._run_scenario(scenario)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            if self.server_process:
                self._stop_server(self.server_process)
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.server_process:
                self._stop_server(self.server_process)
            raise

        # Write summary
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.summary_data[0].keys())
            writer.writeheader()
            writer.writerows(self.summary_data)

        print("\n" + "=" * 50)
        print("🎉 All scenarios completed!")
        print(f"📊 Summary: {summary_file}")
        print(f"📁 Study directory: {self.study_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM scenario-based benchmark tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('config', help='Path to scenario config YAML file')
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

    args = parser.parse_args()

    scenario_filter = None
    if args.scenarios:
        scenario_filter = [s.strip() for s in args.scenarios.split(',')]

    try:
        benchmark = VLLMBenchmark(
            args.config,
            scenario_filter,
            nsys_delay=args.delay,
            nsys_duration=args.duration
        )
        benchmark.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
