# vllm-model-bash

Scenario-based benchmarking and profiling helpers for vLLM serving workloads.

## What This Does

For each scenario in a YAML config, `vllm_bench.py`:

1. Launches `vllm serve` with scenario-specific parameters.
2. Runs `vllm bench serve` across one or more concurrency points.
3. Saves benchmark JSON output and a cross-scenario summary CSV.
4. Optionally collects:
   - Nsight Systems (`nsys`) traces
   - PyTorch Profiler traces

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

## Dataset Generator

Before running benchmarks you need a JSONL dataset. Use the included
`dataset_generator/` tools to create guidellm-compatible datasets with
configurable ISL/OSL distributions and prefix-caching behaviour — no GPU or
running model required (only the tokenizer is downloaded).

### Quick start

```bash
# Single shared prefix, uniform output tokens
python dataset_generator/generate_dataset.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --buckets "1-500:132,500-1000:1092,1000-2000:54,2000-4000:454,4000-8000:1955,8000-15000:7243,15000-30000:17" \
  --prefix-ratio 0.8 \
  --output dataset.jsonl

# Production traffic profile — scale 411k real requests down to 50k prompts,
# bucketed OSL distribution, 61% prefix ratio
python dataset_generator/generate_dataset.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --buckets "200-500:75,500-1000:285,1000-2000:591,2000-5000:110875,5000-10000:226250,10000-20000:46390,20000-50000:26804" \
  --output-buckets "5-20:10308,20-50:185814,50-100:138520,100-200:49762,200-500:17092,500-1000:7383,1000-2000:2626,2000-5000:350" \
  --prefix-ratio 0.61 \
  --total 50000 \
  --output dataset_production.jsonl

# Split into warmup + run chunks
python dataset_generator/split_dataset.py \
  --input dataset.jsonl \
  --chunk-size 1200 --num-chunks 9 --output-dir /mnt/results
```

Then pass the file to `vllm bench serve` or guidellm:

```bash
guidellm benchmark run --data /mnt/results/dataset.jsonl ...
```

See [`dataset_generator/README.md`](dataset_generator/README.md) for the full
argument reference, flag-compatibility matrix, prefix-caching modes, and
end-to-end workflow examples (including multi-prefix / Airbnb-style production
profiles).

### End-to-end example: generate → benchmark

```bash
# 1. Generate
python dataset_generator/generate_dataset.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --buckets "1-500:132,500-1000:1092,1000-2000:54,2000-4000:454,4000-8000:1955,8000-15000:7243,15000-30000:17" \
  --prefix-ratio 0.8 \
  --output /mnt/results/dataset.jsonl

# 2. Split (8 rate chunks + 1 warmup)
python dataset_generator/split_dataset.py \
  --input /mnt/results/dataset.jsonl \
  --chunk-size 1200 --num-chunks 9 --output-dir /mnt/results

# 3. Benchmark
python vllm_bench.py configs/models/gpt-oss-20b.yaml --scenario baseline
```

## Datasets (MLPerf)

To download the standard MLPerf inference datasets:

```bash
bash download_mlperf_datasets.sh
```

## Recommended Workflow (`vllm_bench.py`)

### CLI Syntax

```bash
python vllm_bench.py <config.yaml> [--scenario name1,name2] [--delay SEC] [--duration SEC] [--mlflow-experiment NAME] [--mlflow-run-name NAME] [--mlflow-tracking-uri URI] [--mlflow-tag KEY=VALUE]
```

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

## Output Layout

Each run creates a timestamped study directory:

```text
<study_dir>_<timestamp>/
  config.yaml
  summary.csv
  scenario_<name>/
    logs/
      vllm_server.log
    results/
      <result_prefix>.json
    profiles/
      nsys_server.qdrep|.nsys-rep      # when using direct `nsys profile` mode
      nsys_conc<k>.qdrep|.nsys-rep     # when using start/stop session mode
      torch/
        trace_conc<k>*
```

Notes:
- `summary.csv` aggregates every scenario/concurrency run.
- Nsight output now writes under each scenario `profiles/` directory.
- Exact Nsight extension varies by nsys version (`.qdrep` and/or `.nsys-rep`).
- Run output is captured in `logs/benchmark_output.log`.
- MLflow upload includes the full study directory, `nvidia-smi`, `lscpu`, command metadata, config, and run log.

## Legacy Script

`vllm_bench.sh` remains available for older workflows/configs:

```bash
bash vllm_bench.sh config.yaml
bash vllm_bench.sh configs/models/gpt-oss-20b.yaml --scenario baseline
```

## Developer Notes

For code flow and debugging notes, see `CODE_README.md`.
