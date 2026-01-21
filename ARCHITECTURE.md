# vLLM Benchmark Tool - Architecture & Code Flow

## Overview

The vLLM Benchmark Tool is a scenario-based benchmarking framework that orchestrates vLLM server instances and benchmark clients. It supports multiple benchmark clients (vLLM bench, MLPerf, custom) through a flexible YAML configuration system.

## Architecture

### Core Components

```
VLLMBenchmark (Main Class)
├── Config Loading & Validation
├── Study Directory Management
├── Scenario Execution Engine
│   ├── Parameter Range Generator
│   ├── Server Lifecycle Manager
│   ├── Benchmark Client Executor
│   └── Profiling Integration
└── Results Aggregation
```

### Class Structure

#### `VLLMBenchmark`

Main orchestrator class that manages the entire benchmarking workflow.

**Key Responsibilities:**
- Load and validate YAML configuration
- Create study directory structure
- Execute scenarios sequentially
- Aggregate results into summary CSV

**Key Methods:**
- `_load_config()` - Load and validate YAML config
- `_setup_study_dir()` - Create study directory with timestamp
- `_run_scenario()` - Execute a single scenario
- `run()` - Main entry point, runs all scenarios

#### Scenario Execution Flow

```
1. Load Config
   └──> Validate model, scenarios
   └──> Filter scenarios if --scenario flag used

2. Setup Study Directory
   └──> Create timestamped directory
   └──> Copy config.yaml

3. For Each Scenario:
   ├──> Generate Parameter Combinations (if param_ranges)
   │    └──> Cartesian product of all parameter values
   │
   ├──> For Each Parameter Combination:
   │    ├──> Start vLLM Server
   │    │    ├──> Apply parameter values to server args
   │    │    ├──> Optionally wrap with nsys profile
   │    │    └──> Wait for server ready
   │    │
   │    ├──> Generate Iteration Items
   │    │    ├──> MLPerf: datasets × scenarios
   │    │    └──> vLLM bench: concurrencies
   │    │
   │    ├──> For Each Iteration:
   │    │    ├──> Setup Profiling (nsys/torch)
   │    │    ├──> Build Benchmark Command
   │    │    │    ├──> Substitute variables
   │    │    │    └──> Use configurable command or legacy vllm bench
   │    │    ├──> Execute Benchmark
   │    │    │    ├──> Capture stdout/stderr to log
   │    │    │    └──> Apply timeout
   │    │    ├──> Stop Profiling
   │    │    └──> Record Summary
   │    │
   │    └──> Stop Server
   │
   └──> Write Summary CSV
```

## Code Flow Details

### 1. Configuration Loading

**File:** `_load_config()`

```python
1. Load YAML file
2. Validate required fields (model.name, scenarios)
3. Filter scenarios if scenario_filter provided
4. Return config dict
```

**Key Data Structures:**
- `config['model']` - Model configuration
- `config['defaults']` - Global defaults
- `config['scenarios']` - List of scenario configs

### 2. Parameter Range Processing

**File:** `_generate_param_combinations()`

```python
Input: {'max_num_seq': [256, 512], 'gpu_memory_utilization': [0.85, 0.95]}
Output: [
    {'max_num_seq': 256, 'gpu_memory_utilization': 0.85},
    {'max_num_seq': 256, 'gpu_memory_utilization': 0.95},
    {'max_num_seq': 512, 'gpu_memory_utilization': 0.85},
    {'max_num_seq': 512, 'gpu_memory_utilization': 0.95}
]
```

**File:** `_apply_param_to_args()`

Converts parameter names to command-line arguments:
- `max_num_seq` → `--max-num-seq`
- `gpu_memory_utilization` → `--gpu-memory-utilization`

### 3. Iteration Generation

**File:** `_run_scenario()` - Iteration logic

**MLPerf Mode:**
```python
if datasets or scenarios_list:
    # Cartesian product: datasets × scenarios
    iteration_items = [
        {'dataset': d, 'scenario': s}
        for d, s in product(datasets, scenarios_list)
    ]
```

**vLLM Bench Mode:**
```python
elif concurrencies:
    iteration_items = [
        {'concurrency': c} for c in concurrencies
    ]
```

### 4. Command Building

**File:** `_build_benchmark_command()`

**New Command Mode (Configurable):**
```python
1. Get executable from config['command']['executable']
2. Substitute variables in executable path
3. For each arg in config['command']['args']:
   - Substitute variables using context
   - Add to args list
4. Return [executable] + args
```

**Legacy Mode (vLLM Bench):**
```python
1. Build hardcoded vllm bench serve command
2. Use values from context
3. Return command list
```

**Variable Substitution:**
- Uses Python's `str.format()` with `{variable}` syntax
- Context contains all available variables
- Supports nested substitution

### 5. Benchmark Execution

**File:** `_run_benchmark()`

```python
1. Build context for variable substitution
   - Add standard variables (port, model_name, etc.)
   - Add variables from bench_config['variables']
   - Process timeout value

2. Build command using _build_benchmark_command()

3. Setup working directory (if work_dir specified)

4. Setup environment variables (if env specified)

5. Create log file path

6. Execute command:
   - Redirect stdout/stderr to log file
   - Apply timeout
   - Capture return code

7. Write footer to log (status, return code)

8. Return (status, runtime)
```

### 6. Server Management

**File:** `_start_server()`

```python
1. Build server command
   - Add nsys profile wrapper if enabled
   - Add vllm serve command
   - Add model name, port, params

2. Start process with subprocess.Popen
   - Redirect output to log file
   - Use process group (setsid) for cleanup

3. If nsys session mode:
   - Wait for nsys initialization
   - Get session ID

4. Return (process, nsys_session_id)
```

**File:** `_wait_for_server()`

```python
1. Poll /v1/models endpoint
2. Check every 2 seconds
3. Timeout after 120 seconds (default)
4. Return True if ready, False if timeout
```

**File:** `_stop_server()`

```python
1. Send SIGTERM to process group
2. Wait up to 5 seconds
3. If still running, send SIGKILL
4. Wait 2 seconds for cleanup
```

### 7. Profiling Integration

**Nsight Systems (nsys):**

**Launch Mode:**
- Server started with `nsys profile` wrapper
- Profiling starts immediately with server

**Session Mode (start-later):**
- Server started with `nsys profile --start-later=true`
- Profiling controlled separately with `nsys start/stop`
- Supports delayed start and duration-based stop

**PyTorch Profiler:**
- Configured via environment variables
- `VLLM_TORCH_PROFILER_DIR` - Output directory
- `VLLM_TORCH_PROFILER_PREFIX` - File prefix (updated per iteration)
- Traces saved automatically by vLLM

### 8. Output Organization

**Directory Structure:**
```
Study_<name>_<timestamp>/
├── config.yaml
├── summary.csv
└── scenario_<name>/
    ├── logs/
    │   ├── vllm_server_<name>.log
    │   └── benchmark_<suffix>.log
    ├── results/
    │   ├── <prefix>_<suffix>.json
    │   └── <param_combo>/  # If param_ranges
    └── profiles/
        ├── nsys_*.qdrep
        └── torch/
```

**File Naming:**
- **Result files**: `{prefix}_{param_combo}_{iteration_suffix}.json`
- **Log files**: `benchmark_{param_combo}_{scenario}_{concurrency}.log`
- **MLPerf output**: `{scenario_dir}/results/{param_combo}/` (if param_ranges)

## Key Design Decisions

### 1. Template-Based Command System

**Why:** Maximum flexibility for different benchmark clients without code changes.

**How:** YAML defines command template with variable placeholders, variables substituted at runtime.

### 2. Parameter Ranges as Separate Server Runs

**Why:** Each parameter combination needs a fresh server instance to ensure clean state.

**How:** Server stopped and restarted for each parameter combination.

### 3. MLPerf Iteration Over Datasets/Scenarios

**Why:** MLPerf doesn't use concurrency in the same way as vLLM bench.

**How:** Separate iteration mechanism for datasets × scenarios cartesian product.

### 4. Automatic Log Capture

**Why:** Benchmark clients may not log to files, or logs may be empty.

**How:** All stdout/stderr captured and saved to log files automatically.

### 5. Deep Copy for Bench Config

**Why:** Prevent modification of original config when adding iteration variables.

**How:** Use `copy.deepcopy()` before modifying variables per iteration.

## Extension Points

### Adding New Benchmark Clients

1. Add command configuration to YAML:
```yaml
bench:
  command:
    executable: "your_client"
    args: ["--arg", "{variable}"]
```

2. Define iteration mechanism:
   - Use `concurrencies` for concurrency-based iteration
   - Use `datasets`/`scenarios` for MLPerf-style iteration
   - Use custom variables in `variables` section

3. Variables automatically available in command args

### Adding New Profiling Tools

1. Add profiling detection in `_run_scenario()`
2. Add setup logic before server start (if needed)
3. Add start/stop logic in iteration loop
4. Update summary CSV with profiling info

## Error Handling

- **Config validation**: Raises ValueError with clear messages
- **Server startup failure**: Logs error, continues to next scenario
- **Benchmark failure**: Captures in log, records "failed" status
- **Timeout**: Handles subprocess.TimeoutExpired, records "timeout" status
- **Variable substitution**: Raises ValueError if variable missing

## Performance Considerations

- **Sequential execution**: Scenarios run one at a time (by design)
- **Server reuse**: Server reused across iterations within a parameter combination
- **Log buffering**: Logs written immediately (no buffering) for real-time monitoring
- **Memory**: Deep copy of config per iteration (minimal overhead)

## Future Enhancements

- Parallel scenario execution
- Result parsing and metric extraction
- Automated report generation
- Integration with experiment tracking systems
- Support for distributed benchmarks
