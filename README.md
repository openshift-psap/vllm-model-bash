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
1. Launches one or more vLLM servers based on the YAML config  
2. Runs concurrency sweeps and collects latency/throughput metrics  
3. Optionally profiles GPU activity via Nsight Systems  
4. Produces organized outputs under a `Study_*` directory  

Ideal for performance characterization, MLPerf inference testing, and GPU profiling at scale.

---

## ⚙️ Requirements

### Dependencies

Install these packages:

```bash
sudo apt-get install jq curl -y
pip install yq
```
