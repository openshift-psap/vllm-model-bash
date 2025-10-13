# vllm-model-bash
Scripts for vllm-model-bash efforts 

## Usage
```bash
bash vllm_bench.sh config.yaml
```
### Example configs/test.yaml

# 🚀 vLLM Benchmark Harness (`vllm_bench.sh`)

This harness automates:
- Launching and monitoring `vllm serve` for multiple models
- Running `vllm bench serve` benchmarks with per-model overrides
- Collecting Nsight Systems (`nsys`) profiling traces
- Generating structured results and per-model summaries

---

## 🧠 Overview

Each benchmark run:
1. Launches a vLLM server based on the YAML config
2. Runs concurrency sweeps and collects latency/throughput metrics
3. Optionally profiles GPU activity via Nsight Systems (nsys), and/or PyTorch Profiler
4. Produces organized outputs under a specified directory

Ideal for performance characterization, MLPerf inference testing, and multi-level GPU profiling at scale.

---

## ⚙️ Requirements

### Dependencies

Install these packages:

```bash
sudo apt-get install jq curl -y
pip install yq
```

### Profiling Tools (Optional)

For GPU profiling capabilities:
- **Nsight Systems**: System-wide performance analysis, CUDA graph tracing
- **Nsight Compute**: Detailed kernel-level analysis
- **PyTorch Profiler**: Python/PyTorch-level CPU and GPU profiling with memory tracking

---

## 🔥 Profiling Options

### 1. Nsight Systems (nsys)

Captures system-wide GPU activity, CUDA graphs, and NVTX ranges.

```yaml
profiling:
  nsys_launch_args: "--trace=cuda,nvtx,osrt --cuda-graph-trace=node"
  nsys_start_args: "--force-overwrite=true --gpu-metrics-devices=cuda-visible"
```

**Outputs**: `.qdrep` files viewable in Nsight Systems GUI

### 2. PyTorch Profiler

Captures Python-level CPU/GPU activity, memory allocations, and operator traces.

```yaml
profiling:
  torch_profiler:
    enabled: true
    record_shapes: true        # Record tensor shapes
    profile_memory: true       # Track memory allocations
    with_stack: false          # Include Python stack traces
    with_flops: false          # Include FLOP estimates
```

**Outputs**:
- Chrome trace files (`.json`) - viewable in `chrome://tracing`
- PyTorch `.pt` trace files - loadable with `torch.profiler.load()`

### 3. Combined Profiling

You can enable both nsys and torch profiler simultaneously:

```yaml
profile: true  # Enables nsys
profiling:
  nsys_launch_args: "--trace=cuda,nvtx,osrt --cuda-graph-trace=node"
  nsys_start_args: "--force-overwrite=true --gpu-metrics-devices=cuda-visible"

  torch_profiler:
    enabled: true
    record_shapes: true
    profile_memory: true
```
