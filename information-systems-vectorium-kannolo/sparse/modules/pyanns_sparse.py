"""PyANN under this harness (Lane A): a sparse graph index, compared against kANNolo's HNSW.

NeurIPS'23 BigANN sparse-track winner, `hhy3/pyanns` (commit in sparse/env.example.sh). API facts
below are read from `python/bindings.cc` and `pyanns/searcher/sparse_*.hpp`.

**1. It reads its own CSR file, so `fit` converts the collection first.**
`pyanns::IndexSparse::add` mmaps a blob laid out as

    int64 n, int64 m, int64 nnz          (the third field is skipped by the reader)
    int64 indptr[n + 1]
    int32 indices[nnz]
    float32 data[nnz]

which is the same layout SINDI reads, so one converter serves both. The conversion depends only
on the collection, so it is cached on disk and paid by the first build of a collection only.
Its time (`convert_s`) and the graph build time (`graph_build_s`) are recorded separately.

**2. Its sparse search API is batch-only.** Both `IndexSparse::search_batch` and
`SparseGraphSearcher::batch_search` take `nq` and run an `#pragma omp parallel for` over the
queries. It is called with `nq = 1`, which is one query at a time on one core; each call still
pays a fixed binding overhead (wrapping three numpy arrays, allocating a result buffer) that
per-query APIs do not. A larger batch would change the search itself (see
sparse/tools/pyanns_search_cli.py), so the Lane-B CLI also calls it once per query.

**3. Search must be forced to one thread.** `pyanns.set_num_threads` sets the OpenMP team size
process-wide; left at the default an `nq = 1` search would still spin up a full thread team on
one pinned core. It is set to 64 for the build and 1 for the search, explicitly, and recorded.
"""

import os
import time

import numpy as np

import pyanns

from base import SparseANN, SPARSE_ROOT

# Where converted CSR files live. One per collection, keyed by the documents path, because the
# conversion is a property of the collection.
CSR_CACHE = os.environ.get("PYANNS_CSR_CACHE", os.path.join(SPARSE_ROOT, "csr"))


def _csr_path(documents_path):
    stem = os.path.basename(os.path.dirname(os.path.dirname(documents_path))) or "collection"
    parent = os.path.basename(os.path.dirname(documents_path))
    return os.path.join(CSR_CACHE, f"{stem}__{parent}__documents.csr")


def convert_seismic_to_csr(documents_path, out_path, queries_path=None):
    """Write the Seismic-format collection as the CSR blob PyANN and SINDI both read.

    `m` (the vocabulary size) must cover every component in *both* the documents and the queries,
    because a searcher that sizes its inverted index from `m` will index out of range on a query
    term it has never seen. The scan below takes the maximum over both files rather than
    assuming the documents' maximum covers the queries.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def scan(path):
        raw = np.fromfile(path, dtype=np.uint8)
        n = int(raw[:4].view(np.uint32)[0])
        pos, lens, comp_max = 4, np.empty(n, dtype=np.int64), 0
        starts = np.empty(n, dtype=np.int64)
        for i in range(n):
            nnz = int(raw[pos:pos + 4].view(np.uint32)[0])
            pos += 4
            starts[i] = pos
            lens[i] = nnz
            pos += 8 * nnz
        return raw, n, lens, starts

    raw, n, lens, starts = scan(documents_path)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float32)
    for i in range(n):
        a, b, p, ln = int(indptr[i]), int(indptr[i + 1]), int(starts[i]), int(lens[i])
        indices[a:b] = raw[p:p + 4 * ln].view(np.uint32).astype(np.int32)
        data[a:b] = raw[p + 4 * ln:p + 8 * ln].view(np.float32)

    m = int(indices.max()) + 1 if nnz else 0
    if queries_path:
        qraw, qn, qlens, qstarts = scan(queries_path)
        for i in range(qn):
            p, ln = int(qstarts[i]), int(qlens[i])
            if ln:
                m = max(m, int(qraw[p:p + 4 * ln].view(np.uint32).max()) + 1)

    tmp = out_path + ".partial"
    with open(tmp, "wb") as fh:
        np.array([n, m, nnz], dtype=np.int64).tofile(fh)
        indptr.tofile(fh)
        indices.tofile(fh)
        data.tofile(fh)
    os.replace(tmp, out_path)  # never leave a half-written CSR that a later run would trust
    return n, m, nnz


class PyannsSparseHNSW(SparseANN):
    """Sparse HNSW: `r` is our `M`, `l` our `ef_construction`, `ef` our `ef_search`."""

    def __init__(self, r, l, drop_ratio=0.0):
        self.r = r
        self.l = l
        self.drop_ratio = drop_ratio
        self.ef = None
        self.budget = None
        self.searcher = None
        self.csr_path = None
        self.graph_path = None
        self.convert_s = None
        self.graph_build_s = None
        self._probe_us = None

    def fit(self, documents_path):
        pyanns.set_num_threads(64)  # set the build thread count explicitly, never infer it

        self.csr_path = _csr_path(documents_path)
        t0 = time.perf_counter()
        if not os.path.exists(self.csr_path):
            # The queries file is needed only to size the vocabulary, and it is NOT a constructor
            # argument: every constructor argument lands in `build_args` and therefore in the
            # `result_file` stem, where a path's dots and slashes are exactly what
            # `driver.result_file_key` refuses. It sits beside the documents in every collection,
            # so it is derived rather than passed.
            queries_path = os.path.join(os.path.dirname(documents_path), "queries.bin")
            convert_seismic_to_csr(documents_path, self.csr_path,
                                   queries_path if os.path.exists(queries_path) else None)
        self.convert_s = time.perf_counter() - t0

        self.graph_path = f"{self.csr_path}.r{self.r}_l{self.l}.graph"
        t0 = time.perf_counter()
        graph = pyanns.SparseHNSWIndex(self.r, self.l).build(self.csr_path)
        self.graph_build_s = time.perf_counter() - t0
        graph.save(self.graph_path)

        # The class name carries upstream's typo ("Grap"); it is not a mistake here.
        self.searcher = pyanns.SparseGrapSearcher(self.csr_path, self.graph_path)
        pyanns.set_num_threads(1)  # search: sequential, one query at a time, single core

    def set_query_arguments(self, ef, budget):
        self.ef = ef
        self.budget = budget
        self.searcher.set_ef(ef)

    def query(self, components, values, k):
        # nq = 1: one query at a time through a batch-only API. The omp loop inside has a single
        # iteration and the process is pinned to one core, so this is a sequential single-core
        # search, paying a per-call binding overhead the per-query APIs do not.
        #
        # These ids are NOT in similarity order, unlike every other module here and unlike what
        # `base.SparseANN.query` asks for: PyANN returns its heap without a final sort. This does
        # not affect `driver.accuracy_at_k`, which is a set intersection |R n G| / k. Any
        # rank-sensitive metric (MRR, nDCG) would need a sort here first.
        indptr = np.array([0, len(components)], dtype=np.int32)
        ids = self.searcher.search_batch(
            1, indptr,
            components.astype(np.int32, copy=False),
            values.astype(np.float32, copy=False),
            k, self.budget,
        )
        return [int(x) for x in ids[:k]]

    def get_additional(self):
        return {
            "convert_s": self.convert_s,
            "graph_build_s": self.graph_build_s,
            "csr_path": self.csr_path,
            "graph_path": self.graph_path,
            "build_threads": 64,
            "search_threads": 1,
        }

    def index_bytes(self):
        total = 0
        for p in (self.csr_path, self.graph_path):
            if p and os.path.exists(p):
                total += os.path.getsize(p)
        return total or None

    def save(self, path):
        # The graph and the CSR are already on disk and the searcher holds no other state, so
        # "saving" is recording where they are. Written as a small text file so the driver's
        # one-path-per-`result_file` convention and `--reuse-indexes` both work unchanged.
        with open(path, "w") as fh:
            fh.write(f"{self.csr_path}\n{self.graph_path}\n")

    def load(self, path):
        with open(path) as fh:
            self.csr_path, self.graph_path = [line.strip() for line in fh][:2]
        pyanns.set_num_threads(1)
        self.searcher = pyanns.SparseGrapSearcher(self.csr_path, self.graph_path)

    def __str__(self):
        return (f"PyANN-SparseHNSW(R={self.r}, L={self.l}, dropRatio={self.drop_ratio}, "
                f"ef={self.ef}, budget={self.budget})")
