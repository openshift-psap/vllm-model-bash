#!/usr/bin/env python3
"""
Generate a guidellm-compatible JSONL dataset with shared-prefix prompts.

Each prompt in the dataset shares a common prefix (taken from a single randomly
generated source prompt) whose length is a configurable fraction of the
prompt's total token length.  The remaining tokens are unique random tokens.

Usage (legacy -- still works):
    python generate_dataset.py \
        --model "openai/gpt-oss-20b" \
        --buckets "1-500:132,500-1000:1092,1000-2000:54,2000-4000:454,4000-8000:1955,8000-15000:7243,15000-30000:17" \
        --prefix-ratio 0.8 \
        --output-tokens-range "1-100" \
        --output dataset.jsonl \
        --seed 42

Usage (production profile with scaling):
    python generate_dataset.py \
        --model "openai/gpt-oss-20b" \
        --buckets "200-500:75,500-1000:285,1000-2000:591,2000-5000:110875,5000-10000:226250,10000-20000:46390,20000-50000:26804" \
        --output-buckets "5-20:10308,20-50:185814,50-100:138520,100-200:49762,200-500:17092,500-1000:7383,1000-2000:2626,2000-5000:350" \
        --prefix-ratio 0.61 \
        --total 50000 \
        --output dataset_production_profile.jsonl
"""

import argparse
import json
import math
import random
import sys

from transformers import AutoTokenizer


def parse_buckets(spec: str) -> list[tuple[int, int, int]]:
    """Parse 'lo-hi:count,...' into [(lo, hi, count), ...]."""
    buckets = []
    for part in spec.split(","):
        part = part.strip()
        range_str, count_str = part.split(":")
        lo_str, hi_str = range_str.split("-")
        buckets.append((int(lo_str), int(hi_str), int(count_str)))
    return buckets


def parse_range(spec: str) -> tuple[int, int]:
    """Parse 'lo-hi' into (lo, hi)."""
    lo, hi = spec.split("-")
    return int(lo), int(hi)


def scale_buckets_to_total(buckets: list[tuple[int, int, int]], total: int) -> list[tuple[int, int, int]]:
    """Scale bucket counts proportionally to sum to exactly `total`.

    Preserves the relative distribution of counts across buckets.
    Rounding errors are corrected on the largest bucket.
    """
    raw_total = sum(count for _, _, count in buckets)
    if raw_total == 0:
        print("ERROR: bucket counts sum to 0, cannot scale", file=sys.stderr)
        sys.exit(1)

    scale = total / raw_total
    scaled = [(lo, hi, max(1, round(count * scale))) for lo, hi, count in buckets]

    diff = total - sum(c for _, _, c in scaled)
    if diff != 0:
        largest_idx = max(range(len(scaled)), key=lambda i: scaled[i][2])
        lo, hi, c = scaled[largest_idx]
        scaled[largest_idx] = (lo, hi, c + diff)

    return scaled


def sample_output_tokens(output_buckets: list[tuple[int, int, int]], rng: random.Random) -> int:
    """Sample an output token count from a weighted bucket distribution.

    Each bucket's count is used as a relative weight. A bucket is chosen
    via weighted random selection, then a value is sampled uniformly within
    that bucket's [lo, hi] range.
    """
    weights = [count for _, _, count in output_buckets]
    bucket = rng.choices(output_buckets, weights=weights, k=1)[0]
    lo, hi, _ = bucket
    return rng.randint(lo, hi)


def generate_prefix_weights(num_prefixes: int, popularity: str | None) -> list[float]:
    """Generate probability weights for assigning prompts to prefix groups.

    Supports 'uniform' (equal weight) and 'zipf[:alpha]' (power-law, default
    alpha=1.0).  Returns a list of num_prefixes floats usable with
    random.choices(weights=...).
    """
    if num_prefixes == 1:
        return [1.0]

    if popularity is None or popularity == "zipf":
        return [1.0 / (i + 1) for i in range(num_prefixes)]

    if popularity.startswith("zipf:"):
        alpha = float(popularity.split(":", 1)[1])
        return [1.0 / (i + 1) ** alpha for i in range(num_prefixes)]

    if popularity == "uniform":
        return [1.0] * num_prefixes

    raise ValueError(
        f"Unknown --prefix-popularity '{popularity}'. "
        "Use 'uniform', 'zipf', or 'zipf:<alpha>'."
    )


def resolve_prefix_lengths(spec: str, num_prefixes: int, rng: random.Random) -> list[int]:
    """Parse --prefix-length into a per-group length array.

    If spec is a plain integer, every group gets the same length.
    If spec is bucket specs (min-max:weight,...), each group independently
    samples a fixed length from the weighted distribution.
    """
    if ":" not in spec:
        length = int(spec)
        return [length] * num_prefixes

    buckets = parse_buckets(spec)
    weights = [count for _, _, count in buckets]
    lengths = []
    for _ in range(num_prefixes):
        bucket = rng.choices(buckets, weights=weights, k=1)[0]
        lo, hi, _ = bucket
        lengths.append(rng.randint(lo, hi))
    return lengths


def get_valid_token_ids(tokenizer) -> list[int]:
    """Return token IDs that are safe to sample (no special tokens)."""
    special = set(tokenizer.all_special_ids)
    vocab_size = tokenizer.vocab_size
    return [tid for tid in range(vocab_size) if tid not in special]


def build_source_text(tokenizer, rng: random.Random, valid_ids: list[int], min_tokens: int):
    """Generate source text that stably tokenizes to at least `min_tokens` tokens.

    We over-generate raw token IDs, decode to text, then re-encode to get a
    stable token sequence that round-trips perfectly through encode/decode.
    """
    raw_ids = [rng.choice(valid_ids) for _ in range(int(min_tokens * 1.2))]
    text = tokenizer.decode(raw_ids, skip_special_tokens=True)
    stable_ids = tokenizer.encode(text)
    return text, stable_ids


def make_prompt(tokenizer, source_ids: list[int], prefix_tok_len: int,
                target_len: int, rng: random.Random, valid_ids: list[int]) -> str:
    """Build a prompt of exactly `target_len` tokens with a shared prefix.

    Decodes source_ids[:prefix_tok_len] as the prefix, appends random suffix
    tokens, then re-encodes and truncates/pads to hit the exact target length.
    """
    suffix_budget = int((target_len - prefix_tok_len) * 1.3) + 10
    suffix_raw = [rng.choice(valid_ids) for _ in range(suffix_budget)]

    prefix_text = tokenizer.decode(source_ids[:prefix_tok_len], skip_special_tokens=True)
    suffix_text = tokenizer.decode(suffix_raw, skip_special_tokens=True)
    combined = prefix_text + suffix_text

    ids = tokenizer.encode(combined)
    if len(ids) >= target_len:
        ids = ids[:target_len]
    else:
        while len(ids) < target_len:
            extra = [rng.choice(valid_ids) for _ in range(target_len - len(ids) + 5)]
            suffix_text += tokenizer.decode(extra, skip_special_tokens=True)
            combined = prefix_text + suffix_text
            ids = tokenizer.encode(combined)
        ids = ids[:target_len]

    return tokenizer.decode(ids, skip_special_tokens=True)


def print_config_summary(buckets, raw_buckets, output_buckets, out_tok_range,
                         prefix_ratio, group_prefix_lengths, total_prompts, seed,
                         scaled, num_prefixes=1, prefix_popularity=None,
                         prefix_weights=None):
    """Print a human-readable summary of the generation configuration."""
    print("")
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)

    if scaled:
        raw_total = sum(c for _, _, c in raw_buckets)
        print(f"  Prompt buckets (raw total: {raw_total:,}, scaled to {total_prompts:,}):")
        for (lo, hi, count), (_, _, raw_count) in zip(buckets, raw_buckets):
            pct = (count / total_prompts * 100) if total_prompts > 0 else 0
            print(f"    [{lo:,}, {hi:,}]: {raw_count:,} -> {count:,} prompts ({pct:.1f}%)")
    else:
        print(f"  Prompt buckets ({total_prompts:,} total):")
        for lo, hi, count in buckets:
            pct = (count / total_prompts * 100) if total_prompts > 0 else 0
            print(f"    [{lo:,}, {hi:,}]: {count:,} prompts ({pct:.1f}%)")

    if output_buckets:
        total_weight = sum(c for _, _, c in output_buckets)
        print(f"  Output token distribution (weighted sampling):")
        for lo, hi, count in output_buckets:
            pct = (count / total_weight * 100) if total_weight > 0 else 0
            print(f"    [{lo:,}, {hi:,}]: weight {count:,} ({pct:.1f}%)")
    else:
        lo, hi = out_tok_range
        print(f"  Output tokens: uniform random in [{lo}, {hi}]")

    if prefix_ratio is not None:
        print(f"  Prefix mode: ratio {prefix_ratio} (scales with prompt length)")
    elif group_prefix_lengths:
        unique_lengths = set(group_prefix_lengths)
        if len(unique_lengths) == 1:
            print(f"  Prefix mode: fixed {group_prefix_lengths[0]:,} tokens per group")
        else:
            mn, mx = min(group_prefix_lengths), max(group_prefix_lengths)
            avg = sum(group_prefix_lengths) / len(group_prefix_lengths)
            print(f"  Prefix mode: variable per group (min={mn:,}, avg={avg:,.0f}, max={mx:,})")
    if num_prefixes > 1:
        pop_label = prefix_popularity or "zipf"
        print(f"  Prefix groups: {num_prefixes} independent prefixes ({pop_label} popularity)")
        if prefix_weights and num_prefixes <= 20:
            total_w = sum(prefix_weights)
            top = sorted(enumerate(prefix_weights), key=lambda x: -x[1])[:5]
            parts = [f"#{i}={w/total_w*100:.1f}%" for i, w in top]
            print(f"    Top prefix weights: {', '.join(parts)}")
    print(f"  Seed: {seed}")
    print("=" * 60)
    print("")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a guidellm-compatible JSONL dataset with shared-prefix prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Legacy: uniform output tokens, manual bucket counts
  python generate_dataset.py \\
    --model "openai/gpt-oss-20b" \\
    --buckets "1-500:132,500-1000:1092,8000-15000:7243" \\
    --prefix-ratio 0.8 \\
    --output-tokens-range "1-100" \\
    --output dataset.jsonl

  # Scale up with --multiply (existing behavior)
  python generate_dataset.py \\
    --model "openai/gpt-oss-20b" \\
    --buckets "1-500:132,500-1000:1092,8000-15000:7243" \\
    --prefix-ratio 0.8 --multiply 3 --output dataset_3x.jsonl

  # Production profile: bucketed output tokens + scale to target size
  python generate_dataset.py \\
    --model "openai/gpt-oss-20b" \\
    --buckets "200-500:75,500-1000:285,1000-2000:591,2000-5000:110875,5000-10000:226250,10000-20000:46390,20000-50000:26804" \\
    --output-buckets "5-20:10308,20-50:185814,50-100:138520,100-200:49762,200-500:17092,500-1000:7383,1000-2000:2626,2000-5000:350" \\
    --prefix-ratio 0.61 --total 50000 \\
    --output dataset_production.jsonl
""",
    )
    parser.add_argument(
        "--model", required=True,
        help="Model name or path used to load the tokenizer (e.g. openai/gpt-oss-20b).",
    )
    parser.add_argument(
        "--buckets", required=True,
        help=(
            "Comma-separated bucket specs for prompt token lengths: min-max:count. "
            "E.g. '1-500:132,500-1000:1092,1000-2000:54'. "
            "Counts can be raw production numbers when used with --total."
        ),
    )
    parser.add_argument(
        "--prefix-ratio", type=float, default=None,
        help=(
            "Fraction of each prompt's tokens that come from the shared prefix (0.0-1.0). "
            "Prefix length scales with prompt length. "
            "Exactly one of --prefix-ratio or --prefix-length is required."
        ),
    )
    parser.add_argument(
        "--prefix-length", default=None,
        help=(
            "Fixed prefix length in tokens, clamped to prompt length. "
            "Accepts a single integer (e.g. '800') for uniform length across all groups, "
            "or bucket specs (e.g. '200-500:30,500-1000:50,1000-2000:20') to sample "
            "a different fixed length per prefix group from a weighted distribution. "
            "Exactly one of --prefix-ratio or --prefix-length is required."
        ),
    )
    parser.add_argument(
        "--output-tokens-range", default=None,
        help=(
            "Min-max range for output_tokens_count per prompt (e.g. '1-100'). "
            "Output tokens are sampled uniformly within this range. "
            "Cannot be used together with --output-buckets. "
            "Default: '1-100' (when --output-buckets is not specified)."
        ),
    )
    parser.add_argument(
        "--output-buckets", default=None,
        help=(
            "Comma-separated bucket specs for output token distribution: min-max:count. "
            "Counts are used as relative weights for weighted random sampling. "
            "E.g. '5-20:10308,20-50:185814,50-100:138520,100-200:49762'. "
            "Cannot be used together with --output-tokens-range."
        ),
    )
    parser.add_argument(
        "--output", default="dataset.jsonl",
        help="Output JSONL file path (default: dataset.jsonl).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--no-shuffle", action="store_true", default=False,
        help="Disable shuffling of output records (default: shuffle is enabled).",
    )
    parser.add_argument(
        "--multiply", type=int, default=1,
        help="Multiply every bucket count by this factor (default: 1). Cannot be used with --total.",
    )
    parser.add_argument(
        "--total", type=int, default=None,
        help=(
            "Target total number of prompts. Proportionally scales all bucket counts "
            "to sum to this number while maintaining their relative distribution. "
            "Cannot be used with --multiply (other than the default of 1)."
        ),
    )
    parser.add_argument(
        "--num-prefixes", type=int, default=1,
        help=(
            "Number of independent prefix groups (default: 1). "
            "When >1, generates N independent source prompts and assigns each "
            "dataset prompt to a prefix group. Prompts sharing the same group "
            "have a common prefix; prompts in different groups share nothing."
        ),
    )
    parser.add_argument(
        "--prefix-popularity", default=None,
        help=(
            "How to distribute prompts across prefix groups. "
            "'uniform' = equal probability, 'zipf' = Zipf(alpha=1.0), "
            "'zipf:<alpha>' = Zipf with custom exponent. "
            "Default: 'zipf' when --num-prefixes > 1, ignored when 1."
        ),
    )
    args = parser.parse_args()

    # --- Validate flag combinations ---
    if args.output_buckets and args.output_tokens_range:
        print("ERROR: --output-buckets and --output-tokens-range are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if args.total is not None and args.multiply != 1:
        print("ERROR: --total and --multiply cannot be used together.", file=sys.stderr)
        sys.exit(1)

    if args.multiply < 1:
        print(f"ERROR: --multiply must be >= 1, got {args.multiply}", file=sys.stderr)
        sys.exit(1)

    if args.total is not None and args.total < 1:
        print(f"ERROR: --total must be >= 1, got {args.total}", file=sys.stderr)
        sys.exit(1)

    if args.num_prefixes < 1:
        print(f"ERROR: --num-prefixes must be >= 1, got {args.num_prefixes}", file=sys.stderr)
        sys.exit(1)

    if args.prefix_ratio is not None and args.prefix_length is not None:
        print("ERROR: --prefix-ratio and --prefix-length are mutually exclusive.", file=sys.stderr)
        sys.exit(1)

    if args.prefix_ratio is None and args.prefix_length is None:
        print("ERROR: exactly one of --prefix-ratio or --prefix-length is required.", file=sys.stderr)
        sys.exit(1)

    if args.prefix_length is not None and ":" not in args.prefix_length:
        try:
            val = int(args.prefix_length)
        except ValueError:
            print(f"ERROR: --prefix-length must be an integer or bucket spec, got '{args.prefix_length}'", file=sys.stderr)
            sys.exit(1)
        if val < 1:
            print(f"ERROR: --prefix-length must be >= 1, got {val}", file=sys.stderr)
            sys.exit(1)

    # --- Parse buckets ---
    raw_buckets = parse_buckets(args.buckets)
    buckets = raw_buckets

    # --- Apply scaling ---
    scaled = False
    if args.total is not None:
        buckets = scale_buckets_to_total(raw_buckets, args.total)
        scaled = True
    elif args.multiply > 1:
        buckets = [(lo, hi, count * args.multiply) for lo, hi, count in raw_buckets]

    # --- Parse output token config ---
    output_buckets = None
    out_tok_range = None
    if args.output_buckets:
        output_buckets = parse_buckets(args.output_buckets)
    else:
        range_spec = args.output_tokens_range if args.output_tokens_range else "1-100"
        out_tok_range = parse_range(range_spec)

    prefix_ratio = args.prefix_ratio
    rng = random.Random(args.seed)

    if prefix_ratio is not None and not 0.0 <= prefix_ratio <= 1.0:
        print(f"ERROR: --prefix-ratio must be between 0.0 and 1.0, got {prefix_ratio}", file=sys.stderr)
        sys.exit(1)

    num_prefixes = args.num_prefixes
    prefix_popularity = args.prefix_popularity
    prefix_weights = generate_prefix_weights(num_prefixes, prefix_popularity)

    # Resolve per-group prefix lengths (consumes rng before source prompt generation)
    if args.prefix_length is not None:
        group_prefix_lengths = resolve_prefix_lengths(args.prefix_length, num_prefixes, rng)
    else:
        group_prefix_lengths = None

    max_bucket_hi = max(hi for _, hi, _ in buckets)
    if prefix_ratio is not None:
        source_min_tokens = math.ceil(max_bucket_hi * prefix_ratio)
    else:
        source_min_tokens = max(group_prefix_lengths)
    total_prompts = sum(count for _, _, count in buckets)

    # --- Print configuration summary ---
    print_config_summary(buckets, raw_buckets, output_buckets, out_tok_range,
                         prefix_ratio, group_prefix_lengths, total_prompts, args.seed,
                         scaled, num_prefixes, prefix_popularity, prefix_weights)

    # --- Load tokenizer ---
    print(f"Loading tokenizer for '{args.model}' ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    valid_ids = get_valid_token_ids(tokenizer)
    print(f"  Vocab size: {tokenizer.vocab_size}, usable tokens: {len(valid_ids)}")

    print(f"Generating {num_prefixes} source prompt(s) (>= {source_min_tokens} stable tokens max) ...")
    source_prompts = []
    for i in range(num_prefixes):
        min_tok = group_prefix_lengths[i] if group_prefix_lengths else source_min_tokens
        _, s_ids = build_source_text(tokenizer, rng, valid_ids, min_tok)
        source_prompts.append(s_ids)
        if num_prefixes <= 10 or (i + 1) % 50 == 0:
            pfx_info = f", prefix={group_prefix_lengths[i]}" if group_prefix_lengths else ""
            print(f"  Source prompt {i}: {len(s_ids)} stable tokens{pfx_info}")
    if num_prefixes > 10 and num_prefixes % 50 != 0:
        pfx_info = f", prefix={group_prefix_lengths[-1]}" if group_prefix_lengths else ""
        print(f"  Source prompt {num_prefixes - 1}: {len(source_prompts[-1])} stable tokens{pfx_info}")

    print(f"Generating {total_prompts:,} prompts across {len(buckets)} buckets ...")

    # --- Generate prompts ---
    records = []
    prefix_assign_counts = [0] * num_prefixes
    for lo, hi, count in buckets:
        print(f"  Bucket [{lo:,}, {hi:,}]: {count:,} prompts")
        for _ in range(count):
            L = rng.randint(lo, hi)

            if num_prefixes == 1:
                prefix_idx = 0
            else:
                prefix_idx = rng.choices(range(num_prefixes), weights=prefix_weights, k=1)[0]

            if prefix_ratio is not None:
                prefix_tok_len = int(math.floor(L * prefix_ratio))
            else:
                prefix_tok_len = min(group_prefix_lengths[prefix_idx], L)

            prefix_assign_counts[prefix_idx] += 1

            text = make_prompt(tokenizer, source_prompts[prefix_idx], prefix_tok_len,
                               L, rng, valid_ids)

            if output_buckets:
                out_tok = sample_output_tokens(output_buckets, rng)
            else:
                out_tok = rng.randint(out_tok_range[0], out_tok_range[1])

            records.append({"prompt": text, "output_tokens_count": out_tok})

            if len(records) % 500 == 0:
                print(f"    ... {len(records):,}/{total_prompts:,} generated")

    # --- Shuffle ---
    if not args.no_shuffle:
        print(f"Shuffling {len(records):,} records ...")
        rng.shuffle(records)

    # --- Write output ---
    with open(args.output, "w") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    label = "shuffled " if not args.no_shuffle else ""
    print(f"\nDone. Wrote {len(records):,} {label}prompts to {args.output}")

    # --- Print output token distribution summary ---
    if output_buckets:
        print("\nOutput token distribution (actual):")
        out_counts = {}
        for record in records:
            ot = record["output_tokens_count"]
            for lo, hi, _ in output_buckets:
                if lo <= ot <= hi:
                    key = f"[{lo}, {hi}]"
                    out_counts[key] = out_counts.get(key, 0) + 1
                    break
        for lo, hi, _ in output_buckets:
            key = f"[{lo}, {hi}]"
            c = out_counts.get(key, 0)
            pct = (c / len(records) * 100) if records else 0
            print(f"  {key}: {c:,} ({pct:.1f}%)")

    if num_prefixes > 1:
        print(f"\nPrefix group assignment (actual):")
        active = sum(1 for c in prefix_assign_counts if c > 0)
        print(f"  Active groups: {active}/{num_prefixes}")
        top_n = min(10, num_prefixes)
        ranked = sorted(enumerate(prefix_assign_counts), key=lambda x: -x[1])
        for rank, (idx, cnt) in enumerate(ranked[:top_n]):
            pct = (cnt / len(records) * 100) if records else 0
            print(f"  #{idx}: {cnt:,} prompts ({pct:.1f}%)")
        if num_prefixes > top_n:
            rest = sum(c for _, c in ranked[top_n:])
            pct = (rest / len(records) * 100) if records else 0
            print(f"  ... remaining {num_prefixes - top_n} groups: {rest:,} prompts ({pct:.1f}%)")


if __name__ == "__main__":
    main()
