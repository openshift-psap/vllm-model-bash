# Benchmark Client Refactoring - Changelog

## Overview
Refactored `vllm_bench.py` to support configurable benchmark clients (vllm bench, MLPerf, custom clients) via YAML configuration, implementing Design Option 1 (Template-Based Command).

## New Features

### 1. MLPerf-Style Iteration (Datasets & Scenarios)
For MLPerf benchmarks, you can iterate over datasets and scenarios instead of concurrencies:

```yaml
defaults:
  bench:
    datasets:
      - "v4/perf/perf_eval_ref.parquet"
      - "v4/acc/acc_eval_ref.parquet"
    scenarios:
      - "Server"
      - "Offline"
    command:
      executable: "python3"
      args:
        - "harness_main.py"
        - "--dataset-path"
        - "{dataset}"  # Substituted from datasets list
        - "--scenario"
        - "{scenario}"  # Substituted from scenarios list
```

**Features:**
- Specify only `datasets` to iterate over datasets
- Specify only `scenarios` to iterate over scenarios  
- Specify both for cartesian product (all combinations)
- Variables `{dataset}` and `{scenario}` are automatically available
- Also available as `{dataset_path}` and `{scenario_type}` aliases
- **Automatic output directory**: MLPerf results are saved directly to the scenario's results directory
  - Format: `{scenario_dir}/results/`
  - Available as `{output_dir}` and `{result_dir}` variables
  - Perfect for MLPerf `--output-dir` argument - all results for a scenario go to the same directory

### 2. Configurable Benchmark Clients
You can now specify any benchmark client command in YAML instead of being limited to `vllm bench serve`.

#### Example: MLPerf Harness
```yaml
defaults:
  bench:
    command:
      executable: "python3"
      args:
        - "harness_main.py"
        - "--server-url"
        - "http://localhost:{port}"
        - "--server-target-qps"
        - "{target_qps}"
    variables:
      target_qps: "{concurrency}"  # Maps concurrency to QPS
    concurrencies: [2, 4, 8]
```

#### Example: Custom Client
```yaml
defaults:
  bench:
    command:
      executable: "/path/to/custom_benchmark"
      args:
        - "--endpoint"
        - "http://localhost:{port}/v1/completions"
        - "--threads"
        - "{concurrency}"
        - "--output"
        - "{result_file}"
```

### 3. Variable Substitution
Support for dynamic variable substitution in command arguments using `{variable_name}` syntax.

#### Available Variables:
- `{port}` - Server port
- `{model_name}` - Model name from config
- `{concurrency}` - Current concurrency level
- `{num_prompts}` - Calculated: concurrency * cc_mult
- `{input_len}` - Input length
- `{output_len}` - Output length
- `{seed}` - Random seed (timestamp)
- `{result_file}` - Full path to result file
- `{result_dir}` - Directory for results
- `{scenario_dir}` - Scenario directory
- `{scenario_name}` - Scenario name
- `{timestamp}` - Current timestamp
- `{study_dir}` - Study directory
- Custom variables from `variables` section

#### Example:
```yaml
bench:
  command:
    executable: "python"
    args:
      - "benchmark.py"
      - "--server"
      - "http://localhost:{port}"
      - "--qps"
      - "{target_qps}"
  variables:
    target_qps: "{concurrency}"  # Nested substitution
```

### 4. Parameter Ranges
Run the server with multiple parameter values automatically. The benchmark will run for each parameter combination.

#### Example: Single Parameter Range
```yaml
scenarios:
  - name: test_max_num_seq
    port: 8000
    params: "--async-scheduling"
    param_ranges:
      max_num_seq: [256, 512, 1024]
    bench:
      result_prefix: "max_num_seq_test"
```

This will:
1. Start server with `--max-num-seq 256`, run benchmarks
2. Stop server, start with `--max-num-seq 512`, run benchmarks
3. Stop server, start with `--max-num-seq 1024`, run benchmarks

#### Example: Multiple Parameter Ranges (Cartesian Product)
```yaml
scenarios:
  - name: test_multiple_params
    port: 8000
    params: "--async-scheduling"
    param_ranges:
      max_num_seq: [256, 512]
      gpu_memory_utilization: [0.85, 0.90, 0.95]
```

This will run 6 combinations (2 × 3):
- max_num_seq=256, gpu_memory_utilization=0.85
- max_num_seq=256, gpu_memory_utilization=0.90
- max_num_seq=256, gpu_memory_utilization=0.95
- max_num_seq=512, gpu_memory_utilization=0.85
- max_num_seq=512, gpu_memory_utilization=0.90
- max_num_seq=512, gpu_memory_utilization=0.95

### 5. Additional Configuration Options

#### Working Directory
```yaml
bench:
  work_dir: "{scenario_dir}/bench"  # Change working directory for benchmark
```

#### Environment Variables
```yaml
bench:
  env:
    CUDA_VISIBLE_DEVICES: "0"
    MLPERF_CONFIG: "/path/to/config.json"
```

#### Timeout
```yaml
bench:
  timeout: 7200  # Timeout in seconds (default: 3600)
```

## Backward Compatibility

The legacy format is still supported. If no `command` section is specified, it defaults to the original `vllm bench serve` behavior.

### Legacy Format (Still Works)
```yaml
defaults:
  bench:
    concurrencies: [1, 32, 128]
    input_len: 2000
    output_len: 200
    cc_mult: 10
```

### New Format (Recommended)
```yaml
defaults:
  bench:
    command:
      executable: "vllm"
      args:
        - "bench"
        - "serve"
        - "--base-url"
        - "http://localhost:{port}"
        - "--model"
        - "{model_name}"
        # ... other args
    variables:
      input_len: 2000
      output_len: 200
    concurrencies: [1, 32, 128]
    cc_mult: 10
```

## Example Configurations

See the following example files:
- `configs/mlperf_harness_example.yaml` - MLPerf harness_main.py example
- `configs/mlperf_run_example.yaml` - MLPerf run_mlperf.py example
- `configs/param_ranges_example.yaml` - Parameter ranges examples

## Implementation Details

### Key Methods Added:
1. `_substitute_variables()` - Handles variable substitution in templates
2. `_build_benchmark_command()` - Builds command from config with variable substitution
3. `_generate_param_combinations()` - Generates parameter combinations from ranges
4. `_apply_param_to_args()` - Applies parameter values to server args

### Changes to Existing Methods:
1. `_run_benchmark()` - Now accepts `bench_config` and `scenario_dir`, uses configurable command
2. `_run_scenario()` - Now handles parameter ranges and iterates over combinations

## Migration Guide

### For Existing Configs:
No changes required! Existing YAML configs will continue to work as before.

### For New MLPerf Configs:
1. Add `command` section with your MLPerf command
2. Use `{port}`, `{concurrency}`, etc. for dynamic values
3. Set `variables` for any custom variables
4. Optionally add `work_dir`, `env`, `timeout`

### For Parameter Ranges:
1. Add `param_ranges` section to scenario
2. Specify parameter names (without `--`) and list of values
3. Parameter names are converted from `snake_case` to `--kebab-case` automatically

## Notes

- Parameter ranges create separate server runs for each combination
- Results are saved with parameter combination info in the filename
- Summary CSV includes a `params` column showing parameter combinations
- Server is stopped and restarted between parameter combinations
