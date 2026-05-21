# Dataset Generator

Generate guidellm-compatible JSONL datasets with configurable ISL/OSL distributions and prefix caching behavior. Supports single shared prefix, multiple independent prefix groups with popularity skew, and variable per-group prefix lengths for realistic production traffic modeling.

## What it does

1. Loads the target model's tokenizer
2. Generates one or more independent "source prompts" as random token sequences (one per prefix group)
3. For each prompt in the dataset:
   - Samples a total token length from the ISL bucket distribution
   - Assigns the prompt to a prefix group (single group by default, or weighted across N groups)
   - Computes the prefix length: either `floor(length * prefix_ratio)` or a fixed per-group length from `--prefix-length`
   - Takes that many tokens from the assigned group's source prompt as the prefix
   - Fills the remaining tokens with unique random tokens
   - Samples an output token count (uniform range or weighted bucket distribution)
   - Decodes the full token sequence to text
4. Writes each prompt as a JSONL line with `prompt` and `output_tokens_count` fields

## Quick Start

```bash
# Single prefix, uniform output tokens
python generate_dataset.py \
  --model "openai/gpt-oss-20b" \
  --buckets "1-500:132,500-1000:1092,1000-2000:54,2000-4000:454,4000-8000:1955,8000-15000:7243,15000-30000:17" \
  --prefix-ratio 0.8 \
  --output dataset.jsonl

# Multi-prefix production profile with variable system prompt lengths
python generate_dataset.py \
  --model "<your-model-name>" \
  --buckets "100-890:50,890-3700:22,3700-8000:12,8000-15800:11,15800-19200:4,19200-25000:1" \
  --output-buckets "1-9:50,9-50:21,50-282:24,282-457:4,457-1000:1" \
  --prefix-length "200-500:30,500-1000:50,1000-2000:20" \
  --num-prefixes 100 \
  --total 50000 \
  --output dataset_production.jsonl
```

Then pass the file to guidellm:

```bash
guidellm benchmark run --data /mnt/results/dataset.jsonl ...
```

## Arguments

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model` | yes | - | Model name/path for tokenizer (e.g. `openai/gpt-oss-20b`) |
| `--buckets` | yes | - | ISL distribution: comma-separated `min-max:count` pairs for prompt token lengths |
| `--prefix-ratio` | one of | - | Prefix = fraction of each prompt's token length (0.0-1.0). Scales with prompt length. Mutually exclusive with `--prefix-length` |
| `--prefix-length` | one of | - | Fixed prefix length in tokens. Accepts a single int (`800`) or bucket specs (`200-500:30,500-1000:50`) for variable per-group lengths. Mutually exclusive with `--prefix-ratio` |
| `--num-prefixes` | no | `1` | Number of independent prefix groups. When >1, each prompt is assigned to a group; only same-group prompts share a prefix |
| `--prefix-popularity` | no | `zipf` | Distribution of prompts across prefix groups: `uniform`, `zipf`, or `zipf:<alpha>` |
| `--output-tokens-range` | no | `1-100` | OSL as uniform range (e.g. `1-100`). Cannot be used with `--output-buckets` |
| `--output-buckets` | no | - | OSL distribution: comma-separated `min-max:weight` pairs. Cannot be used with `--output-tokens-range` |
| `--total` | no | - | Scale all bucket counts proportionally to this total. Cannot be used with `--multiply` |
| `--multiply` | no | `1` | Multiply every bucket count by this factor. Cannot be used with `--total` |
| `--output` | no | `dataset.jsonl` | Output file path |
| `--seed` | no | `42` | Random seed for reproducibility |
| `--no-shuffle` | no | `false` | Disable shuffling (output is shuffled by default) |

## Flag Compatibility

| | `--output-tokens-range` | `--output-buckets` | `--multiply` | `--total` | `--prefix-ratio` | `--prefix-length` |
|---|---|---|---|---|---|---|
| `--output-tokens-range` | - | ERROR | OK | OK | OK | OK |
| `--output-buckets` | ERROR | - | OK | OK | OK | OK |
| `--multiply` | OK | OK | - | ERROR | OK | OK |
| `--total` | OK | OK | ERROR | - | OK | OK |
| `--prefix-ratio` | OK | OK | OK | OK | - | ERROR |
| `--prefix-length` | OK | OK | OK | OK | ERROR | - |

## ISL Distribution (`--buckets`)

Each bucket is `min-max:count` where `min` and `max` are token counts and `count` is the number of prompts to generate in that range. Prompt lengths are sampled uniformly within the range.

Example: `1-500:132,500-1000:1092` generates 132 prompts with 1-500 tokens and 1092 prompts with 500-1000 tokens.

When used with `--total`, the counts are treated as relative proportions and scaled to sum to the target.

### Scaling to a Target Size (`--total`)

When you have production bucket counts (e.g. from a monitoring dashboard showing 411k total requests) but want a smaller dataset for benchmarking, use `--total` to scale proportionally:

```bash
# Production counts sum to ~411,000. Scale down to 50,000 prompts.
python generate_dataset.py \
  --model "openai/gpt-oss-20b" \
  --buckets "200-500:75,500-1000:285,1000-2000:591,2000-5000:110875,5000-10000:226250,10000-20000:46390,20000-50000:26804" \
  --prefix-ratio 0.61 \
  --total 50000 \
  --output dataset_50k.jsonl
```

The script computes a scale factor (`50000 / 411270 = 0.1216`) and applies it to each bucket:

| Bucket | Raw Count | Scaled to 50k |
|---|---|---|
| 200-500 | 75 | 9 |
| 500-1,000 | 285 | 35 |
| 1,000-2,000 | 591 | 72 |
| 2,000-5,000 | 110,875 | 13,479 |
| 5,000-10,000 | 226,250 | 27,502 |
| 10,000-20,000 | 46,390 | 5,640 |
| 20,000-50,000 | 26,804 | 3,263 |

Rounding errors are corrected on the largest bucket to ensure the total is exact.

## OSL Distribution (`--output-buckets`)

### Uniform (default)

When `--output-buckets` is NOT specified, each prompt's output token count is sampled uniformly from the range:

```bash
--output-tokens-range "1-100"   # uniform random between 1 and 100
```

If neither `--output-tokens-range` nor `--output-buckets` is specified, defaults to `"1-100"`.

### Bucketed (production traffic matching)

When `--output-buckets` is specified, each prompt's output token count is sampled from a weighted distribution:

```bash
--output-buckets "5-20:10308,20-50:185814,50-100:138520,100-200:49762,200-500:17092,500-1000:7383,1000-2000:2626,2000-5000:350"
```

For each prompt:
1. A bucket is chosen via weighted random selection (weights = the count values)
2. An output token count is sampled uniformly within that bucket's [min, max] range

The counts do NOT need to sum to the total number of prompts -- they are only used as relative weights.

## Prefix Caching

### Single prefix (`--prefix-ratio`, default)

All prompts share one source prompt. Each prompt's prefix is `source[:floor(L * ratio)]`, so shorter prompts' prefixes are strict subsets of longer prompts'. Every request hits the same prefix in the KV cache.

```bash
--prefix-ratio 0.8   # 80% of each prompt is shared prefix
```

### Multiple independent prefixes (`--num-prefixes`)

For realistic production traffic where many independent system prompts (agents, tools, personas) compete for KV cache space. Generates N independent source prompts; each dataset prompt is assigned to one group via weighted random selection.

```bash
--num-prefixes 100                # 100 independent prefix groups
--prefix-popularity zipf          # Zipf(alpha=1.0) popularity distribution
```

Prompts in the same group share a common prefix. Prompts in different groups share nothing.

#### Prefix popularity (`--prefix-popularity`)

Controls how prompts are distributed across prefix groups:

| Value | Distribution | Use case |
|---|---|---|
| `zipf` (default) | Zipf(alpha=1.0) -- rank-1 gets ~2x rank-2, ~3x rank-3, etc. | Realistic: a few agents handle most traffic |
| `zipf:<alpha>` | Zipf with custom exponent | Higher alpha = more skewed toward top prefixes |
| `uniform` | Equal probability for all groups | Worst-case cache pressure (no hot prefixes) |

The script prints the actual prefix assignment distribution at the end so you can verify the skew matches your target.

### Prefix length modes

| Mode | Flag | Behavior | Best for |
|---|---|---|---|
| Proportional | `--prefix-ratio 0.6` | Prefix = 60% of each prompt's token length. Longer prompts get longer prefixes. | Single-prefix benchmarks where prefix/suffix ratio is the variable |
| Fixed uniform | `--prefix-length 800` | Every group's prefix is 800 tokens (clamped if prompt is shorter). | Multi-prefix where all system prompts are similar length |
| Fixed variable | `--prefix-length "200-500:30,500-1000:50,1000-2000:20"` | Each group draws its own fixed prefix length from a weighted distribution. | Production modeling where different agents have different system prompt sizes |

With variable prefix lengths, each group's length is sampled once at generation time and stays fixed for all prompts in that group -- matching how real system prompts work (each agent has a fixed-size prompt, but different agents have different sizes).

### Relationship to cache hit rate

The actual cache hit rate in a KV cache engine depends on prefix size, prefix count, popularity distribution, cache capacity, and eviction policy:

- **Few prefixes + large prefix** = high cache hit rate (most requests share a cached prefix)
- **Many prefixes + small prefix** = low cache hit rate (cache thrashing, small prefixes)
- **Many prefixes + Zipf popularity** = moderate hit rate (hot prefixes stay cached, cold ones don't)

Tune these parameters empirically against your engine to hit a target cache hit rate.

## Splitting into Chunks

guidellm reads file-based datasets sequentially and `--random-seed` does not change the sampling order. To get different prompts per benchmark run (e.g. one rate per run), split the dataset into non-overlapping chunks using `split_dataset.py`:

```bash
python split_dataset.py \
  --input /mnt/results/dataset.jsonl \
  --chunk-size 1200 \
  --num-chunks 9 \
  --output-dir /mnt/results
```

This produces `dataset_chunk_0.jsonl` through `dataset_chunk_8.jsonl`, each with 1200 unique prompts. Then pass a different chunk to each guidellm run:

```bash
guidellm benchmark run --data /mnt/results/dataset_chunk_0.jsonl --data-samples 1200 ...
guidellm benchmark run --data /mnt/results/dataset_chunk_1.jsonl --data-samples 1200 ...
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | yes | - | Path to the input JSONL dataset |
| `--chunk-size` | yes | - | Number of records per chunk |
| `--num-chunks` | yes | - | Number of chunks to produce |
| `--output-dir` | no | `.` | Directory for output files |
| `--output-prefix` | no | input filename | Prefix for output filenames |

## Backward Compatibility

All commands from previous versions work identically. Use `--prefix-ratio` for the original behavior:

```bash
# Original 80% prefix caching benchmark dataset -- UNCHANGED
python generate_dataset.py \
  --model "openai/gpt-oss-20b" \
  --buckets "1-500:132,500-1000:1092,1000-2000:54,2000-4000:454,4000-8000:1955,8000-15000:7243,15000-30000:17" \
  --prefix-ratio 0.8 \
  --output-tokens-range "1-100" \
  --output dataset.jsonl

# 3x multiplied dataset for sustained load testing -- UNCHANGED
python generate_dataset.py \
  --model "openai/gpt-oss-20b" \
  --buckets "1-500:132,500-1000:1092,1000-2000:54,2000-4000:454,4000-8000:1955,8000-15000:7243,15000-30000:17" \
  --prefix-ratio 0.8 \
  --multiply 3 \
  --output dataset_3x.jsonl
```

With `--prefix-ratio` and `--num-prefixes 1` (the defaults), the RNG sequence is identical to the original code -- same seed produces byte-for-byte identical output.

## Prerequisites

No running model or GPU required. The `--model` argument is only used to download the tokenizer (a few small config files from Hugging Face), not the model weights. You can run this script on any machine with `transformers` installed.

## Dependencies

- `transformers` (for tokenizer)
- Python standard library (`json`, `argparse`, `random`, `math`)

All available in the guidellm container.

## End-to-end Workflows

### Workflow 1: Original benchmark (80% prefix, uniform output tokens)

```bash
# 1. Generate dataset
python generate_dataset.py \
  --model "openai/gpt-oss-20b" \
  --buckets "1-500:132,500-1000:1092,1000-2000:54,2000-4000:454,4000-8000:1955,8000-15000:7243,15000-30000:17" \
  --prefix-ratio 0.8 \
  --output /mnt/results/dataset.jsonl

# 2. Split into 9 equal chunks (8 rates + 1 warmup)
python split_dataset.py \
  --input /mnt/results/dataset.jsonl \
  --chunk-size 1200 --num-chunks 9 --output-dir /mnt/results

# 3. Run benchmarks
bash run_16x1_prefix_caching_dataset_per_qps.sh --warmup --show-distribution
```

### Workflow 2: Production traffic profile (61% prefix, bucketed output tokens)

```bash
# 1. Generate 50,000 prompts matching production distribution
python generate_dataset.py \
  --model "openai/gpt-oss-20b" \
  --buckets "200-500:75,500-1000:285,1000-2000:591,2000-5000:110875,5000-10000:226250,10000-20000:46390,20000-50000:26804" \
  --output-buckets "5-20:10308,20-50:185814,50-100:138520,100-200:49762,200-500:17092,500-1000:7383,1000-2000:2626,2000-5000:350" \
  --prefix-ratio 0.61 \
  --total 50000 \
  --output /mnt/results/dataset_production.jsonl

# 2. Split into warmup (6,000) + run (44,000)
head -n 6000 /mnt/results/dataset_production.jsonl > /mnt/results/dataset_chunk_0.jsonl
tail -n +6001 /mnt/results/dataset_production.jsonl > /mnt/results/dataset_chunk_1.jsonl

# 3. Run benchmark
RATES="200" DATA_SAMPLES=44000 WARMUP_DATA_SAMPLES=6000 WARMUP_RATE=100 \
DATASET_DIR="/mnt/results" DATASET_PREFIX="dataset_chunk" \
bash run_16x1_prefix_caching_dataset_per_qps.sh --warmup --show-distribution
```

### Workflow 3: 24K sustained load (production profile, single rate)

```bash
# 1. Generate 30,000 prompts (24K data + 6K warmup) with production distribution
python generate_dataset.py \
  --model "openai/gpt-oss-20b" \
  --buckets "200-500:75,500-1000:285,1000-2000:591,2000-5000:110875,5000-10000:226250,10000-20000:46390,20000-50000:26804" \
  --output-buckets "5-20:10308,20-50:185814,50-100:138520,100-200:49762,200-500:17092,500-1000:7383,1000-2000:2626,2000-5000:350" \
  --prefix-ratio 0.61 \
  --total 30000 \
  --output /mnt/results/dataset_production_24k.jsonl

# 2. Split: first 6,000 = warmup (chunk 0), remaining 24,000 = data (chunk 1)
head -6000 /mnt/results/dataset_production_24k.jsonl > /mnt/results/dataset_production_24k_0.jsonl
tail -24000 /mnt/results/dataset_production_24k.jsonl > /mnt/results/dataset_production_24k_1.jsonl

# 3. Run sustained-load benchmark at QPS 200
RATES="200" \
DATA_SAMPLES=24000 \
WARMUP_DATA_SAMPLES=6000 \
DATASET_DIR="/mnt/results" \
DATASET_PREFIX="dataset_production_24k" \
OUTPUT_PREFIX="gptoss20b-llmd-16x1-prefix-caching-production-optimized-24k" \
bash run_16x1_prefix_caching_dataset_per_qps.sh --warmup --show-distribution 2>&1 | tee /mnt/results/run_production_optimized_24k.log
```

### Workflow 4: Sustained load with large scaled dataset

```bash
# 1. Generate 100,000 prompts (for sustained high-QPS testing)
python generate_dataset.py \
  --model "openai/gpt-oss-20b" \
  --buckets "200-500:75,500-1000:285,1000-2000:591,2000-5000:110875,5000-10000:226250,10000-20000:46390,20000-50000:26804" \
  --output-buckets "5-20:10308,20-50:185814,50-100:138520,100-200:49762,200-500:17092,500-1000:7383,1000-2000:2626,2000-5000:350" \
  --prefix-ratio 0.61 \
  --total 100000 \
  --output /mnt/results/dataset_100k.jsonl

# 2. Split into warmup (10,000) + run (90,000)
head -n 10000 /mnt/results/dataset_100k.jsonl > /mnt/results/dataset_100k_chunk_0.jsonl
tail -n +10001 /mnt/results/dataset_100k.jsonl > /mnt/results/dataset_100k_chunk_1.jsonl

# 3. Run single sustained-load benchmark
RATES="300" DATA_SAMPLES=90000 WARMUP_DATA_SAMPLES=10000 WARMUP_RATE=100 \
DATASET_DIR="/mnt/results" DATASET_PREFIX="dataset_100k_chunk" \
OUTPUT_PREFIX="gptoss20b-production-profile-sustained" \
bash run_16x1_prefix_caching_dataset_per_qps.sh --warmup --show-distribution
```

### Workflow 5: Multi-prefix production profile (e.g. Airbnb 235B)

Replicates a production profile with hundreds of independent system prompts, ~40% prefix cache hit rate, and ISL/OSL distributions derived from real traffic percentiles:

| | p50 | p95 | p99 | avg |
|---|---|---|---|---|
| ISL | 890 | 15,800 | 19,200 | 3,700 |
| OSL | 9 | 282 | 457 | 71 |

The ISL/OSL bucket specs are constructed from these percentiles (percentile values used as bucket boundaries, weights computed to hit all four targets simultaneously).

Each of the 100 prefix groups represents a distinct agent/system prompt. Different agents have different system prompt lengths, modeled with `--prefix-length` bucket specs. Each group draws a fixed prefix length once from this distribution, then every prompt in that group uses that length:

```bash
# 1. Generate 50,000 prompts matching the production ISL/OSL distribution
#    with 100 independent prefix groups (Zipf popularity),
#    each with a variable-length system prompt prefix (200-2000 tokens)
python generate_dataset.py \
  --model "<your-model-name>" \
  --buckets "100-890:50,890-3700:22,3700-8000:12,8000-15800:11,15800-19200:4,19200-25000:1" \
  --output-buckets "1-9:50,9-50:21,50-282:24,282-457:4,457-1000:1" \
  --prefix-length "200-500:30,500-1000:50,1000-2000:20" \
  --num-prefixes 100 \
  --prefix-popularity zipf \
  --total 50000 \
  --output /mnt/results/dataset_airbnb_profile.jsonl

# 2. Split into warmup + run
head -n 6000 /mnt/results/dataset_airbnb_profile.jsonl > /mnt/results/dataset_airbnb_chunk_0.jsonl
tail -n +6001 /mnt/results/dataset_airbnb_profile.jsonl > /mnt/results/dataset_airbnb_chunk_1.jsonl

```

Key tuning knobs for hitting ~40% prefix cache hit rate:
- `--prefix-length` — system prompt size distribution; larger prefixes = more tokens cacheable per request
- `--num-prefixes` — more prefixes = more cache pressure = lower hit rate
- `--prefix-popularity zipf:<alpha>` — higher alpha = more skewed = hot prefixes stay cached
- Adjust the prefix length buckets and monitor `vllm:prefix_cache_hits_total / vllm:prefix_cache_queries_total`
