#!/usr/bin/env python3
"""Extract the `id` field of each line of a JSONL sparse-retrieval file into a .npy.

The output `<out>.npy` is a string array where entry `i` is the `id` field
from JSONL line `i`. Downstream evaluation uses it to translate between
external IDs (what BMP/SP emits in TREC, taken verbatim from `id`) and
positional indices (what `groundtruth.tsv` uses).

JSONL schema (from the seismic README):
    {"id": <int|str>, "content": "<str>" (optional), "vector": {"<token>": <float>, ...}}

Memory: O(N) for the array (N=#lines), O(1) for streaming.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", required=True, type=Path, help="Input JSONL path")
    p.add_argument("--out", required=True, type=Path, help="Output .npy path (string array)")
    p.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="If set, fail unless the JSONL has exactly this many lines",
    )
    args = p.parse_args()

    if not args.jsonl.is_file():
        print(f"ERROR: input not found: {args.jsonl}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)

    ids: list[str] = []
    with args.jsonl.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: malformed JSON at line {lineno}: {e}", file=sys.stderr)
                return 2
            if "id" not in obj:
                print(f"ERROR: missing `id` at line {lineno}", file=sys.stderr)
                return 3
            ids.append(str(obj["id"]))
            if lineno % 500_000 == 0:
                print(f"  read {lineno:,} lines", file=sys.stderr)

    n = len(ids)
    if args.expected_count is not None and n != args.expected_count:
        print(
            f"ERROR: line count mismatch: got {n}, expected {args.expected_count}",
            file=sys.stderr,
        )
        return 4

    arr = np.array(ids, dtype=object)
    # numpy "object" array of Python strings is fine for save; consumers do str(x).
    np.save(args.out, arr, allow_pickle=True)
    print(f"wrote {n:,} ids -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
