#!/usr/bin/env python3
"""A Lane-B CLI for PyANN, which ships none (it is a Python library).

**One `search_batch` call per query (`nq = 1`), because a larger batch is a different search.**
`SparseGraphSearcher::SearchBatch` computes its pruning threshold from a max over the whole batch:

    float max = 0.0f;
    for (int i = 0; i < indptr[nq]; ++i) max = std::max(max, data[i]);
    ...
        if (data[j] > max * budget) { ...keep this term... }

At `nq = 1` that max is the query's own largest term; over the whole query file it is the
largest term of any query, so more terms are pruned and accuracy falls (and no longer matches
the Lane-A accuracy of the same configuration). So PyANN's Lane-B row still pays a per-call
binding overhead that the other systems' rows do not, which makes it conservative.

`set_num_threads(1)` is process-wide and mandatory: PyANN's omp team is sized globally, so without
it a search pinned to one core would still spin up a full thread team on that core.

PyANN's ids are not in similarity order: it returns its heap unsorted. Accuracy@10 is the set
intersection |R n G| / k, so this does not affect it; the ranks written below are positional
fillers, and this file must not be used for any rank-sensitive metric.

The "index file" PyANN's module writes is a two-line pointer to the CSR and the graph, since the
searcher holds no other state; this reads it the same way.
"""
import argparse
import os
import sys

import numpy as np
import pyanns


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--index-file", required=True)
    p.add_argument("--query-file", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("-k", type=int, default=10)
    p.add_argument("--ef", type=int, required=True)
    p.add_argument("--budget", type=float, default=0.1)
    p.add_argument("--num-runs", type=int, default=3)
    a = p.parse_args()

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from sparse_data import read_seismic

    with open(a.index_file) as fh:
        csr_path, graph_path = [line.strip() for line in fh][:2]
    print("Loading the index", file=sys.stderr, flush=True)
    pyanns.set_num_threads(1)   # process-wide; see the module docstring
    searcher = pyanns.SparseGrapSearcher(csr_path, graph_path)
    searcher.set_ef(a.ef)

    q = read_seismic(a.query_file)
    n_queries = len(q.offsets) - 1
    indptr = q.offsets.astype(np.int32, copy=False)
    comps = q.components.astype(np.int32, copy=False)
    vals = q.values.astype(np.float32, copy=False)
    print(f"Loaded {n_queries} queries", file=sys.stderr, flush=True)

    # Per-query slices are prepared before the timed passes: converting them inside would time
    # our own marshalling. The per-call binding overhead itself cannot be removed.
    per_q = [(np.array([0, int(indptr[i + 1] - indptr[i])], dtype=np.int32),
              comps[indptr[i]:indptr[i + 1]],
              vals[indptr[i]:indptr[i + 1]]) for i in range(n_queries)]

    out = np.empty((n_queries, a.k), dtype=np.int64)
    for i_run in range(a.num_runs):
        for qi, (ip, c, v) in enumerate(per_q):
            r = searcher.search_batch(1, ip, c, v, a.k, a.budget)
            out[qi] = np.asarray(r)[:a.k]
        # Lane-B pass boundary: the driver timestamps this line on its own clock.
        print(f"pass_done: {i_run + 1}", flush=True)

    ids = out
    with open(a.output_path, "w") as fh:
        for qi in range(n_queries):
            for rank, doc in enumerate(ids[qi], start=1):
                fh.write(f"{qi}\t{int(doc)}\t{rank}\t0\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
