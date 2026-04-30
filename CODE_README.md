# vllm_bench.py Code Guide

This document explains the main code paths in `vllm_bench.py`, with emphasis on benchmark engines, the per-step loop, profiling, and log routing.

## High-Level Execution Flow

1. `main()` parses CLI arguments and builds `VLLMBenchmark`.
2. `VLLMBenchmark.__init__()` loads config and creates `study_dir`.
3. `run()` sets up logging + metadata capture, then iterates over scenarios.
4. `_run_scenario()`:
   - creates scenario directories
   - merges `defaults.bench` with `scenario.bench` and resolves **`bench.engine`** via `_normalize_bench_engine()` (default **`vllm_bench`**)
   - builds a list of **bench steps**:
     - **`vllm_bench`**: one step per `concurrencies` entry (`_build_vllm_bench_steps`)
     - **`guidellm`**: one step per synthetic profile or a single `data` profile (`_build_guidellm_bench_steps`)
   - starts `vllm serve`
   - for each step:
     - optional nsys / torch env for that step’s slug
     - runs **`vllm bench serve`** or **`guidellm benchmark`** (see `_run_benchmark` / `_run_guidellm_benchmark`)
   - after all steps, stops the server
5. `run()` writes `summary.csv` and uploads artifacts to MLflow (if enabled).

## `vllm bench serve` behavior (regression note)

When **`bench.engine`** is omitted or set to **`vllm_bench`** (aliases: `vllm`, `serve`, `bench`), the driver still calls **`_run_benchmark()`**, which builds the same **`vllm bench serve`** argument list as before multi-engine support: base URL, model, random dataset lengths, `--max-concurrency`, `--num-prompts`, seed, `--save-result` / `--append-result`, and optional `--profile` for torch.

What **did** change around vLLM bench (surface area only):

- Each scenario prints **`Benchmark engine: vllm_bench`**.
- **`summary.csv`** adds columns **`bench_engine`** and **`guidellm_profile`** (empty for vLLM bench rows).
- MLflow nested runs set tag **`bench_engine`** and may add tags parsed from **`vllm_bench_conc*.log`**.
- Nsys initial capture prefix uses the **first step’s slug** (e.g. `conc1`) instead of always `conc{concurrencies[0]}`—identical when the first list element is still the minimum concurrency.

Use **`configs/examples/vllm_bench_smoke.yaml`** and **`configs/examples/vllm_bench_multiconc.yaml`** for quick end-to-end checks.

## GuideLLM path (short)

- **`_resolve_guidellm_cmd_prefix`**: `executable`, or `env_path` / `venv` / `conda_env` → `bin/guidellm` or `bin/python` `-m` `guidellm`, else `guidellm` on `PATH`.
- **`_run_guidellm_benchmark`**: `prefix + ["benchmark", ...]` with `--data` JSON, `--rate`, `--output-dir`, `--outputs`, etc.

## Concurrency / profiling behavior

The **step** loop lives inside `_run_scenario()`:

- A single `vllm serve` process is started before the loop.
- For each step, `num_prompts = concurrency * cc_mult` applies only to **`vllm_bench`** steps.
- Profiling start/stop is tied to each step (nsys output basename includes the step **`slug`**, e.g. `conc32` or `g_isl1000_osl1000`).

### nsys start/stop per step

Inside each loop iteration:

- Start:
  - If session mode (`--start-later`) is active, run `nsys start --session=<id>`.
  - Otherwise run `nsys start ... --output <scenario/profiles/nsys_<slug>_<step_slug>>`.
- Stop:
  - If `--duration` is set, a timer stops nsys and code waits for completion before continuing.
  - Otherwise code stops nsys explicitly after benchmark completion.

This ensures start/stop completes per step before moving to the next one.

## Logging/Output routing (current)

The script intentionally separates outputs into different files/streams:

1. **`vllm serve`** output:
   - Written to `scenario_<name>/logs/vllm_server_<scenario_slug>.log`
   - Not echoed into global stderr stream

2. **`vllm bench serve`** output (`vllm_bench` engine):
   - Written to `scenario_<name>/logs/vllm_bench_conc<k>.log`
   - Not echoed to console streams

3. **`guidellm benchmark`** output (`guidellm` engine):
   - Written to `scenario_<name>/logs/guidellm_<profile_slug>.log`

4. Other command output (`nsys` control commands and script prints):
   - Normal informational messages go to stdout
   - stderr is routed to `logs/benchmark_stderr.log`
   - stdout is mirrored to `logs/benchmark_output.log`

## Files to check while debugging

Per scenario:

- `scenario_<name>/logs/vllm_server_<scenario_slug>.log`
- `scenario_<name>/logs/vllm_bench_conc<k>.log` (vLLM bench)
- `scenario_<name>/logs/guidellm_*.log` (GuideLLM)
- `scenario_<name>/profiles/` (nsys and torch traces)
- `scenario_<name>/results/*.json`

Run-level:

- `logs/benchmark_output.log` (stdout mirror + progress)
- `logs/benchmark_stderr.log` (stderr stream)
- `summary.csv`

## Debug checklist for missing vLLM bench steps

1. Confirm configured concurrencies in merged config (defaults + scenario override).
2. Confirm **`bench.engine`** is `vllm_bench` (or unset).
3. Check each `vllm_bench_conc<k>.log` file exists.
4. Look in `benchmark_output.log` for:
   - `Running vllm bench at concurrency=<k>`
   - `Concurrency <k> complete`
5. Check nsys wait/stop messages for stalls in timed profiling mode.
6. Verify server stayed healthy in `vllm_server_*.log` across all concurrencies.
