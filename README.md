# vllm-model-bash

Scenario-based benchmarking and profiling helpers for vLLM serving workloads.

## What This Does

For each scenario in a YAML config, `vllm_bench.py`:

1. Launches `vllm serve` with scenario-specific parameters.
2. Runs the configured **benchmark backend** across one or more steps (see below).
3. Saves benchmark JSON output and a cross-scenario summary CSV.
4. Optionally collects:
   - Nsight Systems (`nsys`) traces
   - PyTorch Profiler traces

**Benchmark backends** (`defaults.bench.engine` / per-scenario `bench.engine`):

- **`vllm_bench`** (default): runs **`vllm bench serve`** once per entry in `bench.concurrencies`, with the same CLI shape as before GuideLLM support.
- **`guidellm`**: runs **`guidellm benchmark`** using `bench.guidellm` (optional `env_path` / `executable` for a dedicated environment). See `configs/guidellm_synthetic_profiles.yaml`.

This is useful for repeatable performance studies, regression tracking, and profiling runs.

## Requirements

### Python and Package Dependencies

```bash
pip install -r requirements.txt
```

### Optional System Tools

Install these if you use the legacy shell scripts or profiling workflows:

```bash
sudo apt-get install -y jq curl
pip install yq
```

Profiling tools (optional but recommended for GPU analysis):
- Nsight Systems (`nsys`)
- Nsight Compute (`ncu`)
- PyTorch Profiler (enabled through scenario config)

## Recommended Workflow (`vllm_bench.py`)

### CLI Syntax

```bash
python vllm_bench.py <config.yaml> [--scenario name1,name2] [--delay SEC] [--duration SEC] [--mlflow-experiment NAME] [--mlflow-run-name NAME] [--mlflow-tracking-uri URI] [--mlflow-tag KEY=VALUE]
```

### Quick checks (`vllm bench serve` unchanged by default)

The default engine is **`vllm_bench`**. Omitting `bench.engine` matches historical behavior. Small configs under `configs/examples/` are meant for regression smoke tests:

```bash
# One concurrency, short sequences (~fastest GPU check)
python vllm_bench.py configs/examples/vllm_bench_smoke.yaml --scenario smoke

# Two concurrencies, two scenarios (exercises the per-concurrency loop + logs)
python vllm_bench.py configs/examples/vllm_bench_multiconc.yaml
```

See `configs/examples/README.md` for a short index.

### Common Commands

```bash
# Run all scenarios
python vllm_bench.py configs/models/gpt-oss-20b.yaml

# Run a single scenario
python vllm_bench.py configs/models/gpt-oss-20b.yaml --scenario baseline

# Run multiple scenarios
python vllm_bench.py configs/models/gpt-oss-20b.yaml --scenarios baseline,async_scheduling

# Start nsys after benchmark has already started
python vllm_bench.py configs/models/gpt-oss-20b.yaml --scenario baseline --delay 15

# Collect nsys for fixed duration (seconds)
python vllm_bench.py configs/models/gpt-oss-20b.yaml --scenario baseline --duration 30

# Upload artifacts to a specific MLflow experiment
python vllm_bench.py configs/models/gpt-oss-20b.yaml --mlflow-experiment vllm-bench --mlflow-run-name gptoss20b-baseline

# Add MLflow tags (repeat --mlflow-tag for multiple)
python vllm_bench.py configs/models/gpt-oss-20b.yaml --mlflow-tag team=perf --mlflow-tag model_family=gptoss

# Disable MLflow uploads
python vllm_bench.py configs/models/gpt-oss-20b.yaml --no-mlflow
```

### Argument Details

- `config`: YAML file with model/defaults/scenarios.
- `--scenario` / `--scenarios`: comma-separated scenario names to run.
- `--delay`: delay before `nsys start` (useful with warmup-heavy startup).
- `--duration`: stop nsys after fixed time instead of end-of-benchmark.
- `--mlflow-experiment`: MLflow experiment name.
- `--mlflow-run-name`: MLflow run name override.
- `--mlflow-tracking-uri`: custom MLflow tracking URI.
- `--mlflow-tag`: MLflow tag as `KEY=VALUE` (repeatable).
- `--no-mlflow`: skip MLflow upload.

## Config Shape

At minimum, your config should contain:

```yaml
model:
  name: meta-llama/Llama-3.1-8B-Instruct
  base_params: "--gpu-memory-utilization 0.9"

defaults:
  study_dir: Study_llama
  env:
    VLLM_USE_V1: "1"
  bench:
    # engine: vllm_bench   # optional; default is vllm_bench (vllm bench serve)
    concurrencies: [1, 8, 32]
    input_len: 1024
    output_len: 128
    cc_mult: 10

scenarios:
  - name: baseline
    port: 8000
    params: "--max-model-len 8192"
    bench:
      concurrencies: [1, 16, 64]
    profile: true
    profiling:
      nsys_launch_args: "--trace=cuda,nvtx,osrt --start-later=true"
      nsys_start_args: "--force-overwrite=true --gpu-metrics-devices=cuda-visible"
      torch_profiler:
        enabled: true
```

For **GuideLLM**, set `bench.engine: guidellm` and a `bench.guidellm` block (`env_path`, `data` or `profiles`, etc.); see `configs/guidellm_synthetic_profiles.yaml`.

## Output Layout

Each run creates a timestamped study directory:

```text
<study_dir>_<timestamp>/
  config.yaml
  summary.csv
  scenario_<name>/
    logs/
      vllm_server_<scenario_slug>.log
      vllm_bench_conc<k>.log           # vllm_bench engine (one per concurrency)
      guidellm_<profile_slug>.log      # guidellm engine (one per profile step)
    results/
      <result_prefix>.json             # vllm bench (append across concurrencies)
      <prefix>-<profile>.json          # guidellm (one file per profile)
    profiles/
      nsys_server.qdrep|.nsys-rep      # when using direct `nsys profile` mode
      nsys_conc<k>.qdrep|.nsys-rep     # when using start/stop session mode
      torch/
        trace_conc<k>*
```

Notes:
- `summary.csv` aggregates every benchmark step. Columns include `bench_engine` and `guidellm_profile` (empty when using `vllm_bench`).
- Nsight output now writes under each scenario `profiles/` directory.
- Exact Nsight extension varies by nsys version (`.qdrep` and/or `.nsys-rep`).
- Run output is captured in `logs/benchmark_output.log`.
- MLflow nested runs tag `bench_engine` and, for `vllm_bench`, parse metrics from `vllm_bench_conc*.log` into run tags when present.
- MLflow upload includes the full study directory, `nvidia-smi`, `lscpu`, command metadata, config, and run log.

## Legacy Script

`vllm_bench.sh` remains available for older workflows/configs:

```bash
bash vllm_bench.sh config.yaml
bash vllm_bench.sh configs/models/gpt-oss-20b.yaml --scenario baseline
```

## Developer Notes

For code flow and debugging notes, see `CODE_README.md`.

