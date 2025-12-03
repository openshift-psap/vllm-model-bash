#!/usr/bin/env python3
"""
vllm_bench_rocm.py - Scenario-based vLLM benchmarking tool for AMD GPUs
Adapted from vllm_bench.py to use rocprof instead of nsys
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


class VLLMBenchmarkROCm:
    def __init__(self, config_path: str, scenario_filter: Optional[List[str]] = None,
                 rocprof_delay: Optional[float] = None, rocprof_duration: Optional[float] = None):
        self.config_path = Path(config_path)
        self.scenario_filter = scenario_filter
        self.rocprof_delay = rocprof_delay
        self.rocprof_duration = rocprof_duration
        self.config = self._load_config()
        self.study_dir = self._setup_study_dir()
        self.summary_data = []
        self.server_process = None
        self.rocprof_process = None

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

    def _start_server(
        self,
        scenario_name: str,
        port: int,
        params: List[str],
        log_file: Path,
        rocprof_profile: bool = False,
        rocprof_args: str = "",
        rocprof_output: Optional[Path] = None,
        attach_mode: bool = False
    ) -> subprocess.Popen:
        """Start vLLM server (with optional rocprof wrapper)."""
        model_name = self.config['model']['name']

        cmd = []

        if rocprof_profile and not attach_mode:
            # Wrap server launch with rocprofv3 (wrapper mode)
            rocprof_launch_args = rocprof_args or "--sys-trace"
            
            # Parse args to list
            args_list = rocprof_launch_args.split()
            
            # Add output directory
            if rocprof_output:
                rocprof_output.mkdir(parents=True, exist_ok=True)
                args_list.extend(["--output-directory", str(rocprof_output)])
            
            # rocprofv3 requires -- separator before the command
            cmd = ["rocprofv3"] + args_list + ["--"]

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
        return process

    def _stop_server(self, process: subprocess.Popen):
        """Stop vLLM server gracefully."""
        if not process:
            return

        print(f"🛑 Stopping server (PID: {process.pid})")
        try:
            # Use SIGINT to allow rocprof to finalize output files
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
        except ProcessLookupError:
            pass

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)

        time.sleep(2)

    def _start_rocprof_attach(
        self, 
        pid: int, 
        output_file: Path, 
        rocprof_args: str = ""
    ) -> subprocess.Popen:
        """Start rocprof in attach mode to profile running process."""
        args_list = rocprof_args.split() if rocprof_args else []
        
        # Build rocprof attach command
        cmd = ["rocprof", "--hip-trace", "--stats", "--timestamp", "on"]
        cmd.extend(args_list)
        cmd.extend(["-d", str(output_file.parent), "-o", output_file.name])
        cmd.extend(["--pid", str(pid)])

        print(f"🎥 Starting rocprof attach (PID: {pid})")
        print(f"   Output: {output_file}")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            return process
        except Exception as e:
            print(f"⚠️  Failed to start rocprof attach: {e}")
            return None

    def _stop_rocprof(self, process: subprocess.Popen):
        """Stop rocprof process."""
        if not process:
            return

        print("🛑 Stopping rocprof")
        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        except Exception as e:
            print(f"⚠️  Error stopping rocprof: {e}")

    def _move_rocprof_files(self, target_dir: Path):
        """Move rocprof generated files to target directory."""
        import glob
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # rocprof with --hip-trace/--hsa-trace creates output in /tmp/rpl_data_*
        # Find the most recent rpl_data directory
        rpl_dirs = sorted(Path('/tmp').glob('rpl_data_*'), key=lambda p: p.stat().st_mtime, reverse=True)
        
        moved_count = 0
        
        if rpl_dirs:
            rpl_dir = rpl_dirs[0]  # Most recent
            print(f"  📁 Found rocprof output: {rpl_dir.name}")
            
            # Move all files from rpl_data directory
            for src in rpl_dir.rglob('*'):
                if src.is_file():
                    # Preserve relative path structure
                    rel_path = src.relative_to(rpl_dir)
                    dst = target_dir / rel_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(src), str(dst))
                    moved_count += 1
            
            # Remove empty rpl_data directory
            shutil.rmtree(rpl_dir, ignore_errors=True)
            
        if moved_count > 0:
            print(f"✅ Moved {moved_count} rocprof file(s) to {target_dir}/")
        else:
            print(f"⚠️  No rocprof output files found")

    def _convert_rocprof_to_json(self, rocprof_dir: Path):
        """Convert rocprof traces to Chrome Tracing JSON format."""
        try:
            # Look for the conversion script
            script_path = Path(__file__).parent / "rocprof_to_json.py"
            
            if not script_path.exists():
                print(f"⚠️  Trace conversion skipped: {script_path.name} not found")
                return
            
            print(f"🔄 Converting traces to Chrome Tracing JSON...")
            
            cmd = [sys.executable, str(script_path), str(rocprof_dir)]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Find the generated JSON file
                json_files = list(rocprof_dir.rglob("chrome_trace.json"))
                if json_files:
                    for json_file in json_files:
                        size_mb = json_file.stat().st_size / (1024 * 1024)
                        print(f"✅ Chrome trace: {json_file.relative_to(self.study_dir)} ({size_mb:.1f} MB)")
                        print(f"   View in Chrome: chrome://tracing")
            else:
                print(f"⚠️  Trace conversion failed: {result.stderr[:200]}")
        
        except subprocess.TimeoutExpired:
            print(f"⚠️  Trace conversion timed out")
        except Exception as e:
            print(f"⚠️  Trace conversion error: {e}")

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
        profile_rocprof = scenario.get('profile', False)
        rocprof_launch_args = scenario.get('profiling', {}).get('rocprof_launch_args', '')
        rocprof_attach_args = scenario.get('profiling', {}).get('rocprof_attach_args', '')
        use_attach_mode = scenario.get('profiling', {}).get('rocprof_attach_mode', False)
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

        # Start server (with wrapper rocprof if not using attach mode)
        rocprof_output_dir = None
        if profile_rocprof and not use_attach_mode:
            rocprof_output_dir = scenario_dir / 'profiles' / 'rocprof_wrapper'
            rocprof_output_dir.mkdir(parents=True, exist_ok=True)

        self.server_process = self._start_server(
            scenario_name, port, full_params, log_file,
            rocprof_profile=(profile_rocprof and not use_attach_mode),
            rocprof_args=rocprof_launch_args,
            rocprof_output=rocprof_output_dir,
            attach_mode=use_attach_mode
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

            # Setup rocprof profiling (attach mode with delay/duration support)
            rocprof_started_event = None
            rocprof_stop_timer = None
            
            if profile_rocprof and use_attach_mode:
                rocprof_output_file = scenario_dir / 'profiles' / f"rocprof_conc{concurrency}"
                rocprof_output_file.parent.mkdir(parents=True, exist_ok=True)

                def start_rocprof_profiling():
                    """Start rocprof attach and optional duration timer."""
                    print(f"🎥 rocprof attach → {rocprof_output_file}")
                    self.rocprof_process = self._start_rocprof_attach(
                        self.server_process.pid,
                        rocprof_output_file,
                        rocprof_attach_args
                    )

                    # Mark that rocprof has started
                    if rocprof_started_event:
                        rocprof_started_event.set()

                    # If duration is specified, start a timer to auto-stop rocprof
                    if self.rocprof_duration and self.rocprof_process:
                        print(f"⏱️  Rocprof will auto-stop after {self.rocprof_duration} seconds from now")

                        def stop_rocprof_after_duration():
                            print(f"\n⏰ Duration expired - stopping rocprof")
                            self._stop_rocprof(self.rocprof_process)
                            print(f"✅ Saved: {rocprof_output_file}")

                        timer = threading.Timer(self.rocprof_duration, stop_rocprof_after_duration)
                        timer.start()

                if self.rocprof_delay:
                    # Start rocprof in background after delay
                    print(f"⏱️  Rocprof will start after {self.rocprof_delay} seconds (benchmark starts now)")
                    rocprof_started_event = threading.Event()

                    def delayed_rocprof_start():
                        time.sleep(self.rocprof_delay)
                        start_rocprof_profiling()

                    rocprof_thread = threading.Thread(target=delayed_rocprof_start, daemon=True)
                    rocprof_thread.start()
                else:
                    # Start rocprof immediately
                    start_rocprof_profiling()

            # Run benchmark
            status, runtime = self._run_benchmark(
                scenario_name, port, concurrency, num_prompts,
                input_len, output_len, result_file,
                enable_profiling=torch_enabled
            )

            # Stop rocprof profiling (only if duration was not specified and using attach mode)
            if profile_rocprof and use_attach_mode:
                if self.rocprof_duration:
                    # Duration-based stop is handled by timer
                    if rocprof_started_event and not rocprof_started_event.is_set():
                        print(f"⏱️  Waiting for delayed rocprof to start before duration timer kicks in...")
                        rocprof_started_event.wait()
                    print(f"⏱️  Benchmark completed. Rocprof will be stopped by duration timer.")
                else:
                    # Manual stop after benchmark completion
                    if rocprof_started_event and not rocprof_started_event.is_set():
                        print(f"⏱️  Waiting for delayed rocprof to start before stopping...")
                        rocprof_started_event.wait()

                    if self.rocprof_process:
                        self._stop_rocprof(self.rocprof_process)
                        print(f"✅ Saved: {rocprof_output_file}")

            # Report torch profiler output
            if torch_dir:
                print(f"✅ Torch traces: {torch_dir}/")
                trace_files = list(torch_dir.glob(f"trace_conc{concurrency}*"))
                for f in trace_files:
                    print(f"    {f.name} ({f.stat().st_size} bytes)")

            # Report rocprof output (wrapper mode)
            if profile_rocprof and not use_attach_mode and rocprof_output_dir:
                print(f"✅ Rocprof wrapper output: {rocprof_output_dir}/")
                output_files = list(rocprof_output_dir.glob("*"))
                for f in output_files:
                    if f.is_file():
                        print(f"    {f.name} ({f.stat().st_size} bytes)")
                
                # Convert traces to Chrome Tracing JSON
                self._convert_rocprof_to_json(rocprof_output_dir)

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
        
        # Move rocprof output files if wrapper mode was used
        if profile_rocprof and not use_attach_mode and rocprof_output_dir:
            # Wait for rocprof to finalize output files
            print("⏳ Waiting for rocprof to finalize output files...")
            time.sleep(5)
            self._move_rocprof_files(rocprof_output_dir)

    def run(self):
        """Run all scenarios."""
        print("🚀 vLLM Scenario Benchmark (ROCm)")
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
            if self.rocprof_process:
                self._stop_rocprof(self.rocprof_process)
            if self.server_process:
                self._stop_server(self.server_process)
        except Exception as e:
            print(f"❌ Error: {e}")
            if self.rocprof_process:
                self._stop_rocprof(self.rocprof_process)
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
        description="vLLM scenario-based benchmark tool for AMD ROCm",
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
        help='Delay (in seconds) before starting rocprof profiling session (attach mode only)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        help='Duration (in seconds) for rocprof profiling. If specified, rocprof will stop after this duration (attach mode only)'
    )

    args = parser.parse_args()

    scenario_filter = None
    if args.scenarios:
        scenario_filter = [s.strip() for s in args.scenarios.split(',')]

    try:
        benchmark = VLLMBenchmarkROCm(
            args.config,
            scenario_filter,
            rocprof_delay=args.delay,
            rocprof_duration=args.duration
        )
        benchmark.run()
    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

