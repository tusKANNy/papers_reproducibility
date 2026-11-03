"""Readers for the sparse collections' own file formats.

Two formats, both fixed by the collections:

* **Seismic binary** (`documents.bin`, `queries.bin`), little-endian:
  `u32 n_vectors`, then per vector `u32 nnz`, `nnz x u32` component ids, `nnz x f32` values.
  Components are stored in ascending order. `vectorium::readers::read_seismic_format` reads
  this format, so nothing is converted for any system except LSP.

* **`groundtruth.tsv`**: four tab-separated columns: query id, document id, rank (1-based),
  score. The collections ship it at k = 10.

Only the *queries* are read here. A collection's documents are handed to each system as a file
path and read by that system's own loader, so the driver never holds 12--27 GB of documents in
Python beside the index.
"""

import numpy as np


class SparseQueries:
    """Queries in CSR form: `components`, `values`, and `offsets` of length `n + 1`."""

    __slots__ = ("components", "values", "offsets", "n", "max_component")

    def __init__(self, components, values, offsets):
        self.components = components
        self.values = values
        self.offsets = offsets
        self.n = len(offsets) - 1
        self.max_component = int(components.max()) if components.size else -1

    def __len__(self):
        return self.n

    def get(self, i):
        """The i-th query as `(components, values)` views into the flat arrays."""
        a, b = self.offsets[i], self.offsets[i + 1]
        return self.components[a:b], self.values[a:b]

    def nnz_mean(self):
        return float(np.diff(self.offsets).mean())


def read_seismic(path, limit=None):
    """Read a Seismic-format file into `SparseQueries`.

    Reads the whole file into memory once and slices it, rather than issuing three reads per
    vector: the query files are a few MB, and the document files are never read here.
    """
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"{path}: too short to hold a header")

    n_vectors = int(raw[:4].view(np.uint32)[0])
    if limit is not None:
        n_vectors = min(n_vectors, limit)

    # Walk the length prefixes once to build the offsets, then gather components and values.
    offsets = np.zeros(n_vectors + 1, dtype=np.int64)
    starts = np.zeros(n_vectors, dtype=np.int64)  # byte offset of each vector's component block
    pos = 4
    for i in range(n_vectors):
        nnz = int(raw[pos : pos + 4].view(np.uint32)[0])
        pos += 4
        starts[i] = pos
        offsets[i + 1] = offsets[i] + nnz
        pos += 8 * nnz  # u32 components followed by f32 values

    total = int(offsets[-1])
    components = np.empty(total, dtype=np.uint32)
    values = np.empty(total, dtype=np.float32)
    for i in range(n_vectors):
        a, b = int(offsets[i]), int(offsets[i + 1])
        nnz = b - a
        p = int(starts[i])
        components[a:b] = raw[p : p + 4 * nnz].view(np.uint32)
        values[a:b] = raw[p + 4 * nnz : p + 8 * nnz].view(np.float32)

    return SparseQueries(components, values, offsets)


def seismic_header(path):
    """`(n_vectors, nnz_total, max_component_seen)` without materializing the vectors.

    `max_component_seen` is exact: it scans every component. Used to check a collection against
    the `u16` component path before a build depends on it.
    """
    with open(path, "rb") as fh:
        n_vectors = int(np.frombuffer(fh.read(4), dtype=np.uint32)[0])
        nnz_total = 0
        max_component = -1
        for _ in range(n_vectors):
            nnz = int(np.frombuffer(fh.read(4), dtype=np.uint32)[0])
            comps = np.frombuffer(fh.read(4 * nnz), dtype=np.uint32)
            fh.seek(4 * nnz, 1)
            nnz_total += nnz
            if nnz:
                max_component = max(max_component, int(comps.max()))
    return n_vectors, nnz_total, max_component


def read_groundtruth(path, k=10, n_queries=None):
    """Read `groundtruth.tsv` into an `(n_queries, k)` int64 array of document ids.

    Rows are ordered by rank within a query. A query with fewer than `k` rows is padded with
    `-1`, which never matches a returned id. A short ground-truth row therefore lowers the
    accuracy it can reach and never counts as a hit.
    """
    qids, dids, ranks = [], [], []
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                raise ValueError(f"{path}:{lineno}: expected 4 fields, got {len(parts)}")
            qids.append(int(parts[0]))
            dids.append(int(parts[1]))
            ranks.append(int(parts[2]))

    qids = np.asarray(qids, dtype=np.int64)
    dids = np.asarray(dids, dtype=np.int64)
    ranks = np.asarray(ranks, dtype=np.int64)

    n = (int(qids.max()) + 1) if n_queries is None else n_queries
    gt = np.full((n, k), -1, dtype=np.int64)
    keep = (ranks >= 1) & (ranks <= k) & (qids < n)
    gt[qids[keep], ranks[keep] - 1] = dids[keep]
    return gt
