#!/usr/bin/env python3
"""A CLI for command-line timing of kANNolo's sparse two-stage index (`SparseRerankHNSW` Python binding).

The Rust binary `hnsw_rerank_search` is the multivector rerank path
(`RerankIndex::new(sparse_index, multivec_dataset)`, `DenseMultiVectorView`), not the sparse one.
The sparse two-stage index is an L1-pruned sparse first stage plus an exact sparse residual
rescoring, exposed through the Python binding; this CLI times it from the command line.

A Python CLI does not bring back the per-query overhead of in-process timing: the binding's
`batch_search` takes the whole query set in one call, so Python is entered once per pass.
`num_threads=1` is the binding's documented setting for "a serial loop with no rayon
involvement ... to reproduce single-thread benchmarks that pin the process with
`numactl --physcpubind`", which is the CPU pinning used for command-line timing.

The index is loaded with `residual_rerank=True`, which makes the two stages sum to the full dot
product.

Emits `pass_done: N` on stdout after each pass (the boundary `driver.measure_cli` timestamps
on its own clock) and writes `query_id\tdoc_id\trank\tscore`, the shape every other CLI here
writes and `groundtruth.tsv` uses.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import kannolo  # noqa: E402
from sparse_data import read_seismic  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index-file", required=True)
    p.add_argument("--query-file", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--k-candidates", type=int, required=True)
    p.add_argument("--ef-search", type=int, required=True)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--beta", type=int, default=None)
    p.add_argument("--lambda", dest="lambda_", type=float, default=None)
    p.add_argument("--first-stage-value-type", default="f16")
    p.add_argument("--graph-type", default="standard")
    p.add_argument("--num-runs", type=int, default=3)
    a = p.parse_args()

    print("Loading the index", file=sys.stderr, flush=True)
    index = kannolo.SparseRerankHNSW.load(
        a.index_file, a.first_stage_value_type, a.graph_type, residual_rerank=True
    )

    q = read_seismic(a.query_file)
    comps = q.components.astype(np.int32, copy=False)
    vals = q.values.astype(np.float32, copy=False)
    offs = q.offsets.astype(np.int64, copy=False)
    n_queries = len(offs) - 1
    print(f"Loaded {n_queries} queries", file=sys.stderr, flush=True)

    ids = None
    for i_run in range(a.num_runs):
        _, ids = index.batch_search(
            comps, vals, offs,
            a.k_candidates, a.k, a.ef_search,
            a.alpha, a.beta, a.lambda_,
            1,      # num_threads: serial, no rayon; the pinned single-core setting
        )
        # Pass boundary: the driver timestamps this line on its own clock.
        print(f"pass_done: {i_run + 1}", flush=True)

    ids = np.asarray(ids).reshape(n_queries, a.k)
    with open(a.output_path, "w") as fh:
        for qi in range(n_queries):
            for rank, doc in enumerate(ids[qi], start=1):
                fh.write(f"{qi}\t{int(doc)}\t{rank}\t0\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
