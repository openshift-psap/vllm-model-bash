# Benchmark Client Design Proposal

## Overview
Refactor `vllm_bench.py` to support configurable benchmark clients (vllm bench, mlperf, custom clients) via YAML configuration.

## Design Option 1: Template-Based Command (Recommended)

### YAML Configuration Structure

```yaml
defaults:
  bench:
    # Legacy mode (backward compatible)
    client: "vllm"  # or "mlperf", "custom"
    
    # New flexible mode
    command:
      executable: "vllm"  # or full path
      args:
        - "bench"
        - "serve"
        - "--base-url"
        - "http://localhost:{port}"
        - "--model"
        - "{model_name}"
        - "--dataset-name"
        - "random"
        - "--random-input-len"
        - "{input_len}"
        - "--random-output-len"
        - "{output_len}"
        - "--max-concurrency"
        - "{concurrency}"
        - "--num-prompts"
        - "{num_prompts}"
        - "--seed"
        - "{seed}"
        - "--save-result"
        - "--result-filename"
        - "{result_file}"
        - "--append-result"
    
    # Variables available for substitution
    variables:
      port: 8000
      model_name: "{model.name}"  # Reference to model config
      input_len: 2000
      output_len: 200
      concurrency: 1  # Will be overridden per iteration
      num_prompts: 10  # Will be calculated: concurrency * cc_mult
      seed: "{timestamp}"  # Dynamic: current timestamp
      result_file: "{scenario_dir}/results/{result_prefix}.json"
    
    # Environment variables for benchmark client
    env:
      CUDA_VISIBLE_DEVICES: "0"
    
    # Working directory for benchmark client
    work_dir: "{scenario_dir}/bench"
    
    # Timeout for benchmark execution
    timeout: 3600  # seconds
    
    # Result parsing (optional, for custom clients)
    result_parser:
      type: "json"  # or "csv", "custom"
      path: "{result_file}"
      metrics:
        - "throughput"
        - "latency_p50"
        - "latency_p99"
```

### Example: MLPerf Configuration

```yaml
defaults:
  bench:
    client: "mlperf"
    command:
      executable: "python"
      args:
        - "/path/to/mlperf/benchmark.py"
        - "--server"
        - "http://localhost:{port}"
        - "--scenario"
        - "Server"
        - "--qps"
        - "{qps}"
        - "--duration"
        - "60"
        - "--output-dir"
        - "{result_dir}"
    variables:
      port: 8000
      qps: 100  # Will be varied per concurrency
      result_dir: "{scenario_dir}/results"
    env:
      MLPERF_CONFIG: "/path/to/config.json"
```

### Example: Custom Client

```yaml
scenarios:
  - name: "custom_bench"
    bench:
      command:
        executable: "/usr/local/bin/my_benchmark"
        args:
          - "--endpoint"
          - "http://localhost:{port}/v1/completions"
          - "--threads"
          - "{concurrency}"
          - "--requests"
          - "{num_prompts}"
          - "--output"
          - "{result_file}"
      variables:
        port: 8000
        concurrency: 32
        num_prompts: 320
        result_file: "{scenario_dir}/results/custom.json"
```

## Design Option 2: Structured Client Configuration

### YAML Structure

```yaml
defaults:
  bench:
    client: "vllm"  # Built-in client type
    
    # Client-specific configuration
    vllm:
      dataset: "random"
      input_len: 2000
      output_len: 200
      seed: "auto"  # or specific number
    
    mlperf:
      scenario: "Server"
      duration: 60
      target_qps: 100
      config_file: "/path/to/config.json"
    
    custom:
      executable: "/path/to/benchmark"
      args_template: |
        --endpoint http://localhost:{port}
        --concurrency {concurrency}
        --output {result_file}
      env:
        BENCHMARK_MODE: "load"
```

## Design Option 3: Plugin System

### Structure

```yaml
defaults:
  bench:
    client_plugin: "vllm_bench_client"  # Python module path
    config:
      # Plugin-specific config
      dataset: "random"
      input_len: 2000
      # ...
```

### Plugin Interface

```python
class BenchmarkClient:
    def build_command(self, context: Dict) -> List[str]:
        """Build command with arguments."""
        pass
    
    def parse_results(self, result_file: Path) -> Dict:
        """Parse benchmark results."""
        pass
    
    def get_metrics(self, results: Dict) -> Dict:
        """Extract metrics from results."""
        pass
```

## Recommended Implementation: Option 1 (Template-Based)

### Key Features:
1. **Backward Compatibility**: Support legacy `bench` config format
2. **Variable Substitution**: Use `{variable_name}` syntax
3. **Dynamic Variables**: Support computed values (timestamp, calculated paths)
4. **Environment Variables**: Allow per-client env vars
5. **Result Parsing**: Optional result parsing for summary metrics

### Variable Substitution Context:
- `{port}` - Server port
- `{model_name}` - Model name from config
- `{concurrency}` - Current concurrency level
- `{num_prompts}` - Calculated: concurrency * cc_mult
- `{input_len}` - Input length
- `{output_len}` - Output length
- `{seed}` - Random seed (timestamp or config value)
- `{result_file}` - Full path to result file
- `{result_dir}` - Directory for results
- `{scenario_dir}` - Scenario directory
- `{scenario_name}` - Scenario name
- `{timestamp}` - Current timestamp
- `{study_dir}` - Study directory

### Implementation Changes:

1. **New Method**: `_build_benchmark_command()` - Builds command from template
2. **New Method**: `_substitute_variables()` - Handles variable substitution
3. **Refactor**: `_run_benchmark()` - Use configurable command instead of hardcoded
4. **New Method**: `_parse_benchmark_results()` - Optional result parsing

### Migration Path:

1. **Phase 1**: Add new `command` config, keep legacy support
2. **Phase 2**: Auto-detect legacy format and convert internally
3. **Phase 3**: (Optional) Deprecate legacy format with warning

## Example YAML Configurations

### Legacy (Backward Compatible)
```yaml
defaults:
  bench:
    concurrencies: [1, 32, 128]
    input_len: 2000
    output_len: 200
    cc_mult: 10
```

### New Format (vLLM Bench)
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
        - "--dataset-name"
        - "random"
        - "--random-input-len"
        - "{input_len}"
        - "--random-output-len"
        - "{output_len}"
        - "--max-concurrency"
        - "{concurrency}"
        - "--num-prompts"
        - "{num_prompts}"
        - "--save-result"
        - "--result-filename"
        - "{result_file}"
    variables:
      input_len: 2000
      output_len: 200
    concurrencies: [1, 32, 128]
    cc_mult: 10
```

### MLPerf Format
```yaml
defaults:
  bench:
    command:
      executable: "python"
      args:
        - "/opt/mlperf/inference/language/llm/run_benchmark.py"
        - "--server"
        - "http://localhost:{port}"
        - "--scenario"
        - "Server"
        - "--qps"
        - "{qps}"
        - "--duration"
        - "60"
        - "--output-dir"
        - "{result_dir}"
    variables:
      qps: 100
    concurrencies: [100, 200, 400]  # Maps to QPS values
    cc_mult: 1  # Not used for MLPerf
```

## Benefits

1. **Flexibility**: Support any benchmark client
2. **Backward Compatible**: Existing configs continue to work
3. **Extensible**: Easy to add new clients
4. **Maintainable**: Clear separation of concerns
5. **Testable**: Command building is isolated and testable
