#!/usr/bin/env python3
"""Read VIBE's result hdf5 tree and emit one JSON row per (config, query-args) point.

Accuracy@10 is computed against the ground truth shipped with the collection. VIBE stores the
retrieved ids per query, so accuracy is a set intersection against `neighbors[:, :10]`.

    python dense/collect_results.py --vibe-root <vibe> --data-root <data> --dataset agnews-mxbai-1024-euclidean --out results/dense/agnews.jsonl
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

DEFAULT_ROOT = os.environ.get("KANNOLO_BENCH_ROOT")


def accuracy_at_k(retrieved, shipped, k):
    """|retrieved ∩ exact| / k, in percent, averaged over queries."""
    nq = shipped.shape[0]
    hits = np.empty(nq, dtype=np.float64)
    for i in range(nq):
        got = {int(x) for x in retrieved[i][:k] if x >= 0}
        hits[i] = len(got & {int(x) for x in shipped[i][:k]})
    return 100.0 * hits.mean() / k


def rows(vibe_root, dataset, k, data_root):
    gt_path = os.path.join(data_root, "datasets", dataset, "neighbors_top10.npy")
    shipped = np.load(gt_path)[:, :k]

    result_dir = os.path.join(vibe_root, "results", dataset, str(k))
    if not os.path.isdir(result_dir):
        print(f"no results under {result_dir}", file=sys.stderr)
        return

    for root, _, files in os.walk(result_dir):
        for filename in sorted(files):
            if not filename.endswith(".hdf5"):
                continue
            path = os.path.join(root, filename)
            with h5py.File(path, "r") as f:
                for group in f.keys():
                    g = f[group]
                    attrs = {kk: vv for kk, vv in g.attrs.items()}
                    retrieved = np.asarray(g["neighbors"])
                    row = {
                        "dataset": dataset,
                        "k": k,
                        "algo": str(attrs.get("algo", "")),
                        "name": str(attrs.get("name", "")),
                        "accuracy_at_k": accuracy_at_k(retrieved, shipped, k),
                        "qps": float(attrs.get("best_qps", float("nan"))),
                        "mean_us": 1e6 * float(attrs.get("best_search_time", float("nan"))),
                        "build_time_s": float(attrs.get("build_time", float("nan"))),
                        "uss_delta_kb": float(attrs.get("index_size", float("nan"))),
                        "run_count": int(attrs.get("run_count", 0)),
                        "result_file": os.path.relpath(path, vibe_root),
                        "query_args": group,
                    }
                    # Per-pass QPS, recorded by the patched vibe/runner.py.
                    if "pass_qps" in attrs:
                        passes = [float(x) for x in np.asarray(attrs["pass_qps"]).ravel()]
                        row["pass_qps"] = passes
                        # With --runs 3 the first pass is the warm-up; the spread is between the
                        # two timed passes.
                        timed = passes[1:] if len(passes) >= 3 else passes
                        if len(timed) > 1 and min(timed) > 0:
                            row["pass_spread_pct"] = 100.0 * (max(timed) - min(timed)) / min(timed)
                            row["n_timed_passes"] = len(timed)
                    if "space_usage_bytes" in attrs:
                        row["space_usage_bytes"] = int(attrs["space_usage_bytes"])
                    if "build_placement" in attrs:
                        row["build_placement"] = json.loads(str(attrs["build_placement"]))
                    yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--vibe-root", default=os.environ.get("KANNOLO_BENCH_VIBE"), required=not os.environ.get("KANNOLO_BENCH_VIBE"))
    ap.add_argument("--data-root", default=DEFAULT_ROOT, required=DEFAULT_ROOT is None,
                    help="the directory holding datasets/ (or $KANNOLO_BENCH_ROOT)")
    ap.add_argument("--out", help="append JSON lines here as well as stdout")
    args = ap.parse_args()

    out = open(args.out, "a") if args.out else None
    for row in rows(args.vibe_root, args.dataset, args.k, args.data_root):
        line = json.dumps(row)
        print(line)
        if out:
            out.write(line + "\n")
    if out:
        out.close()


if __name__ == "__main__":
    main()
