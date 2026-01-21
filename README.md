# vLLM Benchmark Tool

A flexible, scenario-based benchmarking tool for vLLM servers with support for multiple benchmark clients (vLLM bench, MLPerf, custom clients).

## Features

- **Configurable Benchmark Clients**: Support for vLLM bench, MLPerf, and custom benchmark clients via YAML configuration
- **Parameter Ranges**: Automatically test multiple server parameter combinations
- **MLPerf Support**: Native support for MLPerf harness and run scripts with dataset/scenario iteration
- **Profiling Integration**: Built-in support for Nsight Systems (nsys) and PyTorch profiler
- **Organized Results**: Automatic directory structure for logs, results, and profiles
- **Variable Substitution**: Dynamic variable substitution in commands and arguments

## Installation

```bash
# Ensure you have the required dependencies
pip install pyyaml requests

# Make the script executable
chmod +x vllm_bench.py
```

## Quick Start

### Basic Usage

```bash
./vllm_bench.py configs/models/gpt-oss-20b.yaml
```

### Run Specific Scenarios

```bash
./vllm_bench.py configs/models/gpt-oss-20b.yaml --scenarios baseline,async_scheduling
```

### With Profiling

```bash
# Delay nsys start by 30 seconds
./vllm_bench.py configs/models/gpt-oss-20b.yaml --delay 30

# Auto-stop nsys after 60 seconds
./vllm_bench.py configs/models/gpt-oss-20b.yaml --duration 60
```

## Configuration

### Basic YAML Structure

```yaml
model:
  name: openai/gpt-oss-20b
  base_params: "--max-model-len 8192 --gpu-memory-utilization 0.95"

defaults:
  bench:
    concurrencies: [1, 32, 128]
    input_len: 2000
    output_len: 200
    cc_mult: 10
  env:
    CUDA_VISIBLE_DEVICES: "0"

scenarios:
  - name: baseline
    port: 8000
    params: "--max-num-seq 512"
```

### Using Custom Benchmark Clients

```yaml
defaults:
  bench:
    command:
      executable: "python"
      args:
        - "my_benchmark.py"
        - "--server-url"
        - "http://localhost:{port}"
        - "--concurrency"
        - "{concurrency}"
    variables:
      custom_var: "value"
    timeout: 3600
```

### MLPerf Configuration

```yaml
defaults:
  bench:
    command:
      executable: "python3"
      args:
        - "harness_main.py"
        - "--dataset-path"
        - "{dataset}"
        - "--scenario"
        - "{scenario}"
        - "--output-dir"
        - "{output_dir}"
    datasets:
      - "v4/perf/perf_eval_ref.parquet"
      - "v4/acc/acc_eval_ref.parquet"
    scenarios:
      - "Server"
      - "Offline"
```

### Parameter Ranges

```yaml
scenarios:
  - name: test_params
    port: 8000
    params: "--async-scheduling"
    param_ranges:
      max_num_seq: [256, 512, 1024]
      gpu_memory_utilization: [0.85, 0.90, 0.95]
```

## Variable Substitution

Available variables for use in command arguments:

- `{port}` - Server port
- `{model_name}` - Model name from config
- `{concurrency}` - Current concurrency level
- `{num_prompts}` - Calculated: concurrency * cc_mult
- `{input_len}` - Input length
- `{output_len}` - Output length
- `{seed}` - Random seed (timestamp)
- `{result_file}` - Full path to result file
- `{result_dir}` - Directory for results
- `{output_dir}` - Output directory (same as result_dir)
- `{scenario_dir}` - Scenario directory
- `{scenario_name}` - Scenario name
- `{timestamp}` - Current timestamp
- `{study_dir}` - Study directory
- `{timeout}` - Timeout value from config
- Custom variables from `variables` section

## Output Structure

```
Study_ModelName_2026-01-20_20-50-29/
├── config.yaml                    # Copy of input config
├── summary.csv                    # Summary of all runs
└── scenario_<name>/
    ├── logs/
    │   ├── vllm_server_<name>.log
    │   └── benchmark_<suffix>.log
    ├── results/
    │   ├── <result_prefix>.json
    │   └── <param_combo>/         # If param_ranges used
    │       └── (MLPerf output files)
    └── profiles/
        ├── nsys_*.qdrep
        └── torch/
            └── trace_*.json
```

## Examples

See the `configs/` directory for example configurations:

- `configs/models/gpt-oss-20b.yaml` - Basic vLLM bench example
- `configs/mlperf_harness_example.yaml` - MLPerf harness example
- `configs/mlperf_run_example.yaml` - MLPerf run example
- `configs/param_ranges_example.yaml` - Parameter ranges example

## Command Line Options

```
usage: vllm_bench.py [-h] [--scenario SCENARIOS] [--delay DELAY] [--duration DURATION] config

positional arguments:
  config                Path to scenario config YAML file

optional arguments:
  -h, --help            show this help message and exit
  --scenario SCENARIOS, --scenarios SCENARIOS
                        Comma-separated list of scenarios to run
  --delay DELAY         Delay (in seconds) before starting nsys profiling session
  --duration DURATION   Duration (in seconds) for nsys profiling. If specified, nsys will stop after this duration instead of at benchmark completion
```

## Tips

1. **Parameter Ranges**: Use `param_ranges` to test multiple server configurations automatically
2. **MLPerf Iteration**: Use `datasets` and `scenarios` lists for MLPerf-style iteration
3. **Log Files**: All benchmark client output is captured in `logs/benchmark_*.log`
4. **Result Organization**: When using param_ranges, each combination gets its own subdirectory
5. **Profiling**: Enable nsys with `profile: true` and configure in `profiling.nsys_launch_args`

## Troubleshooting

### Server fails to start
- Check server logs in `scenario_*/logs/vllm_server_*.log`
- Verify port is not already in use
- Check GPU memory availability

### Benchmark client errors
- Check benchmark logs in `scenario_*/logs/benchmark_*.log`
- Verify command path and arguments in YAML
- Check variable substitution (use print statements or check logs)

### Missing variables
- Ensure all variables used in command args are defined
- Check that iteration variables (dataset, scenario, concurrency) are available
- Verify timeout is set if using `{timeout}` variable

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - Code architecture and design
- [CHANGELOG_BENCHMARK_CLIENT.md](CHANGELOG_BENCHMARK_CLIENT.md) - Feature changelog
- [BENCHMARK_CLIENT_DESIGN.md](BENCHMARK_CLIENT_DESIGN.md) - Design documentation
