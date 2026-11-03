"""VIBE module for kANNolo's dense HNSW over f16.

The module is minimal: `query()` contains only the binding call and the padding filter. There
is no normalization (every dataset here is `normalized`, `euclidean` or `ip`, never `cosine`),
no dtype coercion and no reshaping. The only array manipulation,
`ascontiguousarray(...).ravel()`, happens once in `fit()`, because the builder takes a flat
array plus an explicit `dim`.

`kannolo.DensePlainHNSW` stores the collection as f16 internally. It covers all three graph
backends and the distance-adaptive early exit, so this module needs no new Rust code.
"""

import numpy as np

import kannolo

from ..base.module import BaseANN

# VIBE's `distance` string -> the metric kANNolo's bindings parse. `normalized` is cosine
# over pre-normalized vectors, which ranks them as dot product does.
METRIC = {
    "euclidean": "euclidean",
    "normalized": "ip",
    "ip": "ip",
}


class Kannolo(BaseANN):
    def __init__(self, metric, m, ef_construction, graph_type):
        if metric not in METRIC:
            raise NotImplementedError(f"kANNolo does not handle metric {metric!r}")
        self.metric = METRIC[metric]
        self.m = m
        self.ef_construction = ef_construction
        self.graph_type = graph_type
        self.ef_search = None
        self.early_exit_threshold = None

    def fit(self, X):
        dim = X.shape[1]
        flat = np.ascontiguousarray(X, dtype=np.float32).ravel()
        self.index = kannolo.DensePlainHNSW.build_from_array(
            flat,
            dim,
            self.m,
            self.ef_construction,
            self.metric,
            self.graph_type,
        )

    def set_query_arguments(self, ef_search, early_exit_threshold=None):
        self.ef_search = ef_search
        self.early_exit_threshold = early_exit_threshold

    def query(self, v, n):
        ids = self.index.search(v, n, self.ef_search, self.early_exit_threshold)[1]
        # `search` pads a short result list to k with id = -1 (push_results, pylib/common.rs).
        # VIBE evaluates X_train[candidates], where -1 would silently index the last row.
        return ids[ids >= 0]

    def get_additional(self):
        return {"space_usage_bytes": self.index.space_usage_bytes()}

    def __str__(self):
        lam = "none" if self.early_exit_threshold is None else f"{self.early_exit_threshold:g}"
        return (
            f"kANNolo(graph={self.graph_type}, M={self.m}, "
            f"efC={self.ef_construction}, efS={self.ef_search}, lambda={lam})"
        )
