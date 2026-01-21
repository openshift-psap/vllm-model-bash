#!/usr/bin/env python3
"""
vllm_bench.py - Scenario-based vLLM benchmarking tool
"""

import argparse
import copy
import csv
import itertools
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

    def _substitute_variables(self, template: str, context: Dict[str, Any]) -> str:
        """Substitute variables in template string using {variable} syntax."""
        try:
            return template.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing variable in template: {e}")

    def _build_benchmark_command(
        self,
        bench_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[str]:
        """Build benchmark command from config with variable substitution."""
        # Check if new command-based config exists
        if 'command' in bench_config:
            cmd_config = bench_config['command']
            executable = cmd_config.get('executable', 'vllm')
            
            # Substitute variables in executable path
            executable = self._substitute_variables(executable, context)
            
            # Build args list with variable substitution
            args = []
            for arg in cmd_config.get('args', []):
                # Substitute variables in each argument
                substituted_arg = self._substitute_variables(str(arg), context)
                args.append(substituted_arg)
            
            return [executable] + args
        
        # Legacy mode: build vllm bench command
        model_name = context.get('model_name', self.config['model']['name'])
        port = context.get('port', 8000)
        concurrency = context.get('concurrency', 1)
        num_prompts = context.get('num_prompts', 10)
        input_len = context.get('input_len', 2000)
        output_len = context.get('output_len', 200)
        result_file = context.get('result_file', '')
        enable_profiling = context.get('enable_profiling', False)

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

        return cmd

    def _run_benchmark(
        self,
        scenario_name: str,
        port: int,
        concurrency: Optional[int],
        num_prompts: Optional[int],
        input_len: int,
        output_len: int,
        result_file: Path,
        bench_config: Dict[str, Any],
        scenario_dir: Path,
        log_file: Optional[Path] = None,
        enable_profiling: bool = False
    ) -> tuple[str, int]:
        """Run benchmark with configurable client."""
        model_name = self.config['model']['name']
        
        # Build context for variable substitution (without timeout first)
        context = {
            'port': port,
            'model_name': model_name,
            'concurrency': concurrency,
            'num_prompts': num_prompts,
            'input_len': input_len,
            'output_len': output_len,
            'seed': int(time.time()),
            'result_file': str(result_file),
            'result_dir': str(scenario_dir / 'results'),
            'scenario_dir': str(scenario_dir),
            'scenario_name': scenario_name,
            'timestamp': int(time.time()),
            'study_dir': str(self.study_dir),
            'enable_profiling': enable_profiling,
        }
        
        # Add any additional variables from bench_config
        if 'variables' in bench_config:
            for key, value in bench_config['variables'].items():
                # Substitute nested variables
                if isinstance(value, str) and '{' in value:
                    value = self._substitute_variables(value, context)
                context[key] = value
        
        # Process timeout value (may need variable substitution)
        timeout_raw = bench_config.get('timeout', 3600)
        if isinstance(timeout_raw, str) and '{' in timeout_raw:
            # Timeout contains variables, substitute them
            timeout_str = self._substitute_variables(timeout_raw, context)
            timeout_value = float(timeout_str)
        else:
            timeout_value = float(timeout_raw)
        
        # Add timeout to context for use in command arguments
        context['timeout'] = timeout_value

        # Build command
        cmd = self._build_benchmark_command(bench_config, context)
        
        # Set working directory if specified
        work_dir = None
        if 'work_dir' in bench_config:
            work_dir_str = self._substitute_variables(bench_config['work_dir'], context)
            work_dir = Path(work_dir_str)
            work_dir.mkdir(parents=True, exist_ok=True)

        # Set environment variables if specified
        env = os.environ.copy()
        if 'env' in bench_config:
            for key, value in bench_config['env'].items():
                env_value = self._substitute_variables(str(value), context)
                env[key] = env_value

        # Create log file if not provided
        if log_file is None:
            logs_dir = scenario_dir / 'logs'
            logs_dir.mkdir(parents=True, exist_ok=True)
            if concurrency is not None:
                log_file = logs_dir / f'benchmark_conc{concurrency}.log'
            else:
                log_file = logs_dir / f'benchmark_{int(time.time())}.log'
        else:
            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)

        # Print status
        if concurrency is not None and num_prompts is not None:
            print(f"===== Concurrency: {concurrency} ({num_prompts} prompts) =====")
        else:
            print(f"===== Running benchmark =====")
        print(f"🔧 Command: {' '.join(cmd[:8])}...")
        print(f"📝 Logging to: {log_file}")

        start_time = time.time()
        try:
            # Capture stdout and stderr to log file
            with open(log_file, 'w') as log_f:
                # Write command header
                log_f.write(f"Command: {' '.join(cmd)}\n")
                log_f.write(f"Working directory: {work_dir or os.getcwd()}\n")
                log_f.write(f"Start time: {datetime.now().isoformat()}\n")
                log_f.write("=" * 80 + "\n\n")
                log_f.flush()
                
                # Run command and capture output
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    cwd=work_dir,
                    env=env,
                    timeout=timeout_value
                )
                
                # Write footer
                log_f.write("\n" + "=" * 80 + "\n")
                log_f.write(f"End time: {datetime.now().isoformat()}\n")
                log_f.write(f"Status: success\n")
                log_f.write(f"Return code: {result.returncode}\n")
            
            status = "success"
        except subprocess.CalledProcessError as e:
            # Write error to log file
            with open(log_file, 'a') as log_f:
                log_f.write(f"\n{'=' * 80}\n")
                log_f.write(f"End time: {datetime.now().isoformat()}\n")
                log_f.write(f"Status: failed\n")
                log_f.write(f"Return code: {e.returncode}\n")
            status = "failed"
        except subprocess.TimeoutExpired:
            # Write timeout to log file
            with open(log_file, 'a') as log_f:
                log_f.write(f"\n{'=' * 80}\n")
                log_f.write(f"End time: {datetime.now().isoformat()}\n")
                log_f.write(f"Status: timeout\n")
            status = "timeout"

        runtime = int(time.time() - start_time)
        print(f"✅ Benchmark completed. Log saved to: {log_file}")
        return status, runtime

    def _generate_param_combinations(self, param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameter values from ranges.
        
        Example:
            param_ranges = {'max_num_seq': [256, 512], 'gpu_memory_util': [0.9, 0.95]}
            Returns: [
                {'max_num_seq': 256, 'gpu_memory_util': 0.9},
                {'max_num_seq': 256, 'gpu_memory_util': 0.95},
                {'max_num_seq': 512, 'gpu_memory_util': 0.9},
                {'max_num_seq': 512, 'gpu_memory_util': 0.95}
            ]
        """
        if not param_ranges:
            return [{}]
        
        import itertools
        
        keys = list(param_ranges.keys())
        values = [param_ranges[key] for key in keys]
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations

    def _apply_param_to_args(self, base_params: List[str], param_name: str, param_value: Any) -> List[str]:
        """Apply a parameter value to the params list, replacing or adding the argument.
        
        Converts param_name like 'max_num_seq' to '--max-num-seq'.
        """
        arg_name = f"--{param_name.replace('_', '-')}"
        
        # Remove existing argument if present
        new_params = []
        i = 0
        while i < len(base_params):
            if base_params[i] == arg_name:
                # Skip the argument and its value
                i += 2
                continue
            new_params.append(base_params[i])
            i += 1
        
        # Add new argument
        new_params.extend([arg_name, str(param_value)])
        
        return new_params

    def _run_scenario(self, scenario: Dict):
        """Run a single scenario, optionally with parameter ranges."""
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

        # Get benchmark settings
        defaults = self.config.get('defaults', {})
        bench_defaults = defaults.get('bench', {})
        bench_config = {**bench_defaults, **scenario.get('bench', {})}

        # Get iteration settings - support datasets/scenarios for MLPerf or concurrencies for vllm bench
        datasets = bench_config.get('datasets', None)
        scenarios_list = bench_config.get('scenarios', None)  # Renamed to avoid conflict with scenario dict
        concurrencies = bench_config.get('concurrencies', None)
        
        input_len = bench_config.get('input_len', 2000)
        output_len = bench_config.get('output_len', 200)
        cc_mult = bench_config.get('cc_mult', 10)
        result_prefix = bench_config.get('result_prefix', f"scenario_{scenario_name}")

        # Generate iteration combinations
        # Priority: datasets/scenarios (MLPerf) > concurrencies (vllm bench)
        if datasets is not None or scenarios_list is not None:
            # MLPerf-style iteration over datasets and/or scenarios
            # If only one is specified, use it alone; if both, use cartesian product
            if datasets is not None and scenarios_list is not None:
                # Both specified: cartesian product
                iteration_items = []
                for dataset, scenario_type in itertools.product(datasets, scenarios_list):
                    iteration_items.append({'dataset': dataset, 'scenario': scenario_type})
                print(f"📊 MLPerf iteration: {len(datasets)} dataset(s) × {len(scenarios_list)} scenario(s) = {len(iteration_items)} run(s)")
            elif datasets is not None:
                # Only datasets specified
                iteration_items = [{'dataset': d} for d in datasets]
                print(f"📊 MLPerf iteration: {len(datasets)} dataset(s)")
            else:
                # Only scenarios specified
                iteration_items = [{'scenario': s} for s in scenarios_list]
                print(f"📊 MLPerf iteration: {len(scenarios_list)} scenario(s)")
        elif concurrencies is not None:
            # vllm bench-style iteration over concurrencies
            iteration_items = [{'concurrency': c} for c in concurrencies]
            print(f"📊 Concurrency iteration: {len(concurrencies)} value(s)")
        else:
            # Default: single concurrency of 1
            iteration_items = [{'concurrency': 1}]
            print(f"📊 Default: single run with concurrency=1")

        # Check for parameter ranges
        param_ranges = scenario.get('param_ranges', {})
        param_combinations = self._generate_param_combinations(param_ranges)
        
        if param_ranges:
            print(f"🔄 Parameter ranges detected: {list(param_ranges.keys())}")
            print(f"   Will run {len(param_combinations)} parameter combination(s)")

        # Run scenario for each parameter combination
        for param_idx, param_combo in enumerate(param_combinations):
            if param_ranges:
                combo_suffix = "_".join([f"{k}_{v}" for k, v in sorted(param_combo.items())])
                combo_scenario_name = f"{scenario_name}_{combo_suffix}"
                combo_result_prefix = f"{result_prefix}_{combo_suffix}"
                print(f"\n{'='*50}")
                print(f"📋 Parameter combination {param_idx + 1}/{len(param_combinations)}: {param_combo}")
                print(f"{'='*50}")
            else:
                combo_suffix = ""  # Empty when no param ranges
                combo_scenario_name = scenario_name
                combo_result_prefix = result_prefix

            log_file = scenario_dir / 'logs' / f'vllm_server_{combo_scenario_name}.log'

            # Merge parameters
            base_params = self.config['model'].get('base_params', '')
            scenario_params = scenario.get('params', '')
            full_params = f"{base_params} {scenario_params}".strip().split()

            # Apply parameter combination to params
            for param_name, param_value in param_combo.items():
                full_params = self._apply_param_to_args(full_params, param_name, param_value)

            # Handle compilation config
            if 'compilation_config' in scenario:
                comp_config = scenario['compilation_config']
                if isinstance(comp_config, str):
                    comp_config = yaml.safe_load(comp_config)
                full_params.extend(['--compilation-config', json.dumps(comp_config)])

            port = scenario.get('port', 8000)
            print(f"🌐 Port: {port}")
            print(f"⚙️  Params: {' '.join(full_params[:5])}...")

            # Note: result_file is set here but iter_result_file is used in the loop
            # This is kept for backward compatibility but iter_result_file includes param combo info
            result_file = scenario_dir / 'results' / f"{combo_result_prefix}.json"

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
                combo_scenario_name, port, full_params, log_file,
                nsys_profile=profile_nsys, nsys_args=nsys_launch_args
            )

            if not self.server_process or not self._wait_for_server(port):
                print(f"❌ Failed to start server for scenario: {combo_scenario_name}")
                if self.server_process:
                    self._stop_server(self.server_process)
                continue  # Skip to next parameter combination

            # Run benchmarks for each iteration item (dataset/scenario or concurrency)
            for iter_idx, iter_item in enumerate(iteration_items):
                # Extract iteration values
                concurrency = iter_item.get('concurrency', None)
                dataset = iter_item.get('dataset', None)
                scenario_type = iter_item.get('scenario', None)
                
                # Calculate num_prompts if concurrency is available
                num_prompts = concurrency * cc_mult if concurrency is not None else None
                
                # Build iteration suffix for file naming
                # For logs, exclude dataset name (only use scenario or concurrency)
                iter_parts = []
                if dataset is not None:
                    iter_parts.append(f"dataset_{dataset}")
                if scenario_type is not None:
                    iter_parts.append(f"scenario_{scenario_type}")
                if concurrency is not None:
                    iter_parts.append(f"conc{concurrency}")
                iter_suffix = "_".join(iter_parts) if iter_parts else f"iter{iter_idx}"
                
                # Build log suffix (without dataset name)
                # Include param range info if param_ranges are used
                log_parts = []
                if param_ranges and combo_suffix:
                    # Add param combo suffix to log file name
                    log_parts.append(combo_suffix)
                if scenario_type is not None:
                    log_parts.append(f"scenario_{scenario_type}")
                if concurrency is not None:
                    log_parts.append(f"conc{concurrency}")
                log_suffix = "_".join(log_parts) if log_parts else f"iter{iter_idx}"
                
                # Create unique result file name that includes:
                # 1. Parameter combination (if param_ranges used) - already in combo_result_prefix
                # 2. Iteration info (dataset/scenario/concurrency) - in iter_suffix
                # This ensures each run gets its own file and nothing gets overwritten
                # When using param_ranges, each combination gets a unique file
                # When using iterations (datasets/scenarios/concurrencies), each gets a unique file
                if iter_suffix:
                    iter_result_file = scenario_dir / 'results' / f"{combo_result_prefix}_{iter_suffix}.json"
                else:
                    # No iterations, just use the result prefix (which includes param combo if used)
                    iter_result_file = scenario_dir / 'results' / f"{combo_result_prefix}.json"
                
                print(f"📄 Result file: {iter_result_file.name}")

                # Update torch profiler prefix
                if torch_enabled:
                    prefix_parts = []
                    if concurrency is not None:
                        prefix_parts.append(f"conc{concurrency}")
                    if dataset is not None:
                        prefix_parts.append(f"ds_{dataset}")
                    if scenario_type is not None:
                        prefix_parts.append(f"sc_{scenario_type}")
                    prefix = "_".join(prefix_parts) if prefix_parts else f"iter{iter_idx}"
                    os.environ['VLLM_TORCH_PROFILER_PREFIX'] = f"trace_{prefix}"

                # Setup nsys profiling (start in background if delay is specified)
                nsys_file = None
                nsys_stop_timer = None
                nsys_started_event = None
                if profile_nsys:
                    nsys_file = scenario_dir / 'profiles' / f"nsys_{combo_scenario_name}_{iter_suffix}"
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

                # Set output directory to scenario's results directory (for MLPerf output-dir)
                # When param_ranges are used, create a subdirectory for each parameter combination
                if param_ranges:
                    # Create subdirectory for this parameter combination
                    param_subdir = scenario_dir / 'results' / combo_suffix
                    param_subdir.mkdir(parents=True, exist_ok=True)
                    iter_output_dir = param_subdir
                    print(f"📁 MLPerf output directory: {iter_output_dir}")
                else:
                    # No param ranges, use scenario's results directory directly
                    iter_output_dir = scenario_dir / 'results'
                    iter_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Update bench_config variables with iteration values
                iter_bench_config = copy.deepcopy(bench_config)
                if 'variables' not in iter_bench_config:
                    iter_bench_config['variables'] = {}
                # Merge iteration values into variables (don't overwrite existing)
                if dataset is not None:
                    iter_bench_config['variables']['dataset'] = dataset
                    iter_bench_config['variables']['dataset_path'] = dataset  # Common alias
                if scenario_type is not None:
                    iter_bench_config['variables']['scenario'] = scenario_type
                    iter_bench_config['variables']['scenario_type'] = scenario_type  # Common alias
                if concurrency is not None:
                    iter_bench_config['variables']['concurrency'] = concurrency
                
                # Set result_dir and output_dir to the scenario's results directory
                # This ensures MLPerf output-dir goes to the scenario's results directory
                iter_bench_config['variables']['result_dir'] = str(iter_output_dir)
                iter_bench_config['variables']['output_dir'] = str(iter_output_dir)  # Common alias for MLPerf

                # Create log file for this benchmark iteration (without dataset name)
                logs_dir = scenario_dir / 'logs'
                logs_dir.mkdir(parents=True, exist_ok=True)
                bench_log_file = logs_dir / f'benchmark_{log_suffix}.log'
                
                # Run benchmark (starts immediately, concurrent with delayed nsys start if applicable)
                status, runtime = self._run_benchmark(
                    combo_scenario_name, port, concurrency, num_prompts,
                    input_len, output_len, iter_result_file,
                    iter_bench_config, scenario_dir,
                    log_file=bench_log_file,
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
                    if concurrency is not None:
                        trace_files = list(torch_dir.glob(f"trace_conc{concurrency}*"))
                    else:
                        trace_files = list(torch_dir.glob(f"trace_{iter_suffix}*"))
                    for f in trace_files:
                        print(f"    {f.name} ({f.stat().st_size} bytes)")

                # Record summary
                param_info = f"_{combo_suffix}" if param_ranges else ""
                summary_row = {
                    'scenario': f"{scenario_name}{param_info}",
                    'port': port,
                    'status': status,
                    'runtime_sec': runtime,
                    'result_file': str(iter_result_file),
                    'params': str(param_combo) if param_combo else ""
                }
                # Add iteration-specific fields
                if concurrency is not None:
                    summary_row['concurrency'] = concurrency
                    summary_row['num_prompts'] = num_prompts
                if dataset is not None:
                    summary_row['dataset'] = dataset
                if scenario_type is not None:
                    summary_row['mlperf_scenario'] = scenario_type
                self.summary_data.append(summary_row)

            # Stop server after all concurrencies for this parameter combination
            self._stop_server(self.server_process)
            time.sleep(2)  # Brief pause between parameter combinations

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
        summary_fieldnames = [
            'scenario', 'port', 'concurrency', 'num_prompts',
            'dataset', 'mlperf_scenario',
            'status', 'runtime_sec', 'result_file', 'params'
        ]
        with open(summary_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
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
        if self.summary_data:
            with open(summary_file, 'w', newline='') as f:
                # Use all keys from data, but ensure standard order
                all_keys = set()
                for row in self.summary_data:
                    all_keys.update(row.keys())
                fieldnames = [k for k in summary_fieldnames if k in all_keys]
                fieldnames.extend([k for k in sorted(all_keys) if k not in fieldnames])
                writer = csv.DictWriter(f, fieldnames=fieldnames)
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
