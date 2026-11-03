#!/usr/bin/env python3
"""kANNolo-Rerank with RaBitQ-ext 8-bit rescoring, on the builds selected for kANNolo-Rerank.

Reads a collection's selection (analysis/select_configs.py), takes every kANNolo-Rerank build that is
the fastest at some accuracy target in 90..99, and runs the same build with `rerank_bits = 8` over
the same search grid (definitions/kannolo_rerank_ext8).

    python dense/run_ext8.py --vibe <vibe> --root <data> --dataset dpr-jina-768-normalized --budget-s 5400 \
        --selection selections/dense_dpr.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_dense as rd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--runs", type=int, default=3)
    rd.add_path_args(ap)
    args = ap.parse_args()
    rd.set_paths(args)

    with open(args.selection) as fh:
        winners = {r["result_file"] for r in json.load(fh)["per_accuracy"]["kannolo-rerank"]}
    results = rd.vibe_import("vibe.results")
    f16 = rd.load_grid(args.dataset, "kannolo-rerank", args.k)
    ext8 = {tuple(map(rd._hashable, d.arguments[:-1])): d
            for d in rd.load_grid(args.dataset, "kannolo-rerank-ext8", args.k)}
    log = os.path.join(rd.LOGS, f"dense_{args.dataset}.log")
    for d in f16:
        path, _ = results.build_result_filepath(args.dataset, args.k, d, None, False)
        if not any(path.endswith(w.split("results/", 1)[-1]) for w in winners):
            continue
        target = ext8[tuple(map(rd._hashable, d.arguments))]
        secs = rd.run_one(args.dataset, args.k, int(4 * args.budget_s), "kannolo-rerank-ext8",
                          target, log, args.runs)
        print(f"kannolo-rerank-ext8 {target.arguments[1:]}  build "
              f"{'n/a' if secs is None else f'{secs:.1f} s'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
