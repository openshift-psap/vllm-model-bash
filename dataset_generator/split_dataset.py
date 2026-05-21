#!/usr/bin/env python3
"""
Split a shuffled JSONL dataset into N non-overlapping chunks.

Usage:
    python split_dataset.py --input dataset.jsonl --chunk-size 1200 --num-chunks 9

Produces: dataset_chunk_0.jsonl, dataset_chunk_1.jsonl, ..., dataset_chunk_8.jsonl
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Split JSONL into non-overlapping chunks")
    parser.add_argument("--input", required=True, help="Path to the input JSONL file")
    parser.add_argument("--chunk-size", type=int, required=True, help="Number of records per chunk")
    parser.add_argument("--num-chunks", type=int, required=True, help="Number of chunks to produce")
    parser.add_argument("--output-dir", default=".", help="Directory for output files (default: .)")
    parser.add_argument("--output-prefix", default=None,
                        help="Output filename prefix (default: input filename without extension)")
    args = parser.parse_args()

    with open(args.input) as f:
        lines = f.readlines()

    total = len(lines)
    needed = args.chunk_size * args.num_chunks
    if needed > total:
        print(f"Error: need {needed} records ({args.chunk_size} × {args.num_chunks}) "
              f"but dataset only has {total}", file=sys.stderr)
        sys.exit(1)

    prefix = args.output_prefix
    if prefix is None:
        import os
        prefix = os.path.splitext(os.path.basename(args.input))[0]

    for i in range(args.num_chunks):
        start = i * args.chunk_size
        end = start + args.chunk_size
        chunk = lines[start:end]

        out_path = f"{args.output_dir}/{prefix}_chunk_{i}.jsonl"
        with open(out_path, "w") as f:
            f.writelines(chunk)

        print(f"Chunk {i}: rows [{start}:{end}] -> {out_path} ({len(chunk)} records)")

    print(f"\nDone. {args.num_chunks} chunks of {args.chunk_size} from {total} total records.")
    unused = total - needed
    if unused > 0:
        print(f"({unused} records unused)")


if __name__ == "__main__":
    main()
