# vllm_bench.py Code Guide

This document explains the main code paths in `vllm_bench.py`, with emphasis on concurrency loops, profiling, and log routing.

## High-Level Execution Flow

1. `main()` parses CLI arguments and builds `VLLMBenchmark`.
2. `VLLMBenchmark.__init__()` loads config and creates `study_dir`.
3. `run()` sets up logging + metadata capture, then iterates over scenarios.
4. `_run_scenario()`:
   - creates scenario directories
   - starts `vllm serve`
   - loops over all configured concurrencies
   - for each concurrency:
     - starts nsys profiling (if enabled)
     - runs `vllm bench serve`
     - stops nsys profiling
   - after all concurrencies, stops the server
5. `run()` writes summary + uploads artifacts to MLflow (if enabled).

## Concurrency + Profiling Behavior

The concurrency loop is inside `_run_scenario()`:

- `concurrencies` are read from merged defaults + scenario override.
- A single `vllm serve` process is started before the loop.
- For each concurrency:
  - `num_prompts = concurrency * cc_mult`
  - benchmark runs once at that concurrency
  - profiling start/stop occurs for that concurrency
- After the loop completes, server is terminated via `_stop_server()`.

### nsys Start/Stop Per Concurrency

Inside each loop iteration:

- Start:
  - If session mode (`--start-later`) is active, run `nsys start --session=<id>`.
  - Otherwise run `nsys start ... --output <scenario/profiles/nsys_concX>`.
- Stop:
  - If `--duration` is set, a timer stops nsys and code waits for completion before continuing.
  - Otherwise code stops nsys explicitly after benchmark completion.

This ensures start/stop is done per concurrency before moving to the next one.

## Logging/Output Routing (Current)

The script intentionally separates outputs into different files/streams:

1. `vllm serve` output:
   - Written only to `scenario_<name>/logs/vllm_server.log`
   - Not echoed into global stderr stream

2. `vllm bench serve` output:
   - Written only to `scenario_<name>/logs/vllm_bench_conc<k>.log`
   - Not echoed to console streams

3. Other command output (`nsys` control commands and script prints):
   - Normal informational messages go to stdout
   - stderr is routed to `logs/benchmark_stderr.log`
   - stdout is mirrored to `logs/benchmark_output.log`

## Files to Check While Debugging

Per scenario:

- `scenario_<name>/logs/vllm_server.log`
- `scenario_<name>/logs/vllm_bench_conc<k>.log`
- `scenario_<name>/profiles/` (nsys and torch traces)
- `scenario_<name>/results/*.json`

Run-level:

- `logs/benchmark_output.log` (stdout mirror + progress)
- `logs/benchmark_stderr.log` (stderr stream)
- `summary.csv`

## Debug Checklist for Missing Concurrency Runs

1. Confirm configured concurrencies in merged config (defaults + scenario override).
2. Check each `vllm_bench_conc<k>.log` file exists.
3. Look in `benchmark_output.log` for:
   - `Running vllm bench at concurrency=<k>`
   - `Concurrency <k> complete`
4. Check nsys wait/stop messages for stalls in timed profiling mode.
5. Verify server stayed healthy in `vllm_server.log` across all concurrencies.

