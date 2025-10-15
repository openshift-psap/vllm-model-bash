# vllm-model-bash
Scripts for vllm-model-bash efforts 

## Overview

Each benchmark run:
1. Launches a vLLM server based on the YAML config
2. Runs concurrency sweeps and collects latency/throughput metrics
3. Optionally profiles GPU activity via Nsight Systems (nsys), and/or PyTorch Profiler
4. Produces organized outputs under a specified directory

Ideal for performance characterization, MLPerf inference testing, and multi-level GPU profiling at scale.

---

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

**Python script** (recommended for scenario-based configs):
```bash
# Run all scenarios
python vllm_bench.py configs/models/gpt-oss-20b.yaml

# Run specific scenario(s)
python vllm_bench.py configs/models/gpt-oss-20b.yaml --scenario baseline
python vllm_bench.py configs/models/gpt-oss-20b.yaml --scenarios baseline,async_scheduling
```

**Bash script** (legacy support):
```bash
# Works with both old and new config formats
bash vllm_bench.sh config.yaml
bash vllm_bench.sh configs/models/gpt-oss-20b.yaml --scenario baseline
```

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
