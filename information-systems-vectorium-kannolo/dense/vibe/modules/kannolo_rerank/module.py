"""VIBE module for kANNolo's dense two-stage index (`DenseRerankHNSW`, kANNolo-Rerank).

An HNSW first stage over RaBitQ / RaBitQ-ext codes retrieves `k_candidates` candidates, which are
rescored against a second representation of the collection: plain `f16` by default, or RaBitQ-ext
codes with `rerank_bits` bits per component (the ext-8 configuration, `rerank_bits = 8`). The graph
is built over `f16`; the codes never take part in construction.
"""

import numpy as np

import kannolo

from ..base.module import BaseANN

METRIC = {
    "euclidean": "euclidean",
    "normalized": "ip",
    "ip": "ip",
}


class KannoloRerank(BaseANN):
    def __init__(self, metric, m, ef_construction, graph_type, encoder_spec, rerank_bits=0,
                 residual_rerank=False):
        """`encoder_spec` is `[encoder, pq_subspaces, rabitq_total_bits, rabitq_query_bits]`.

        The four values form one argument because the encoder parameters depend on `encoder`.
        The config declares it in `arg_groups`, which passes the list through as a single value.

        `rerank_bits` is the second stage's encoding: `0` is `f16`, `2`/`4`/`8` RaBitQ-ext codes.
        `residual_rerank` stores `x - x̂` in the second stage and scores `first stage + rerank`,
        an exact decomposition under inner product; it needs a `rabitq-ext` first stage.
        """
        if metric not in METRIC:
            raise NotImplementedError(f"kANNolo does not handle metric {metric!r}")
        encoder, pq_subspaces, rabitq_total_bits, rabitq_query_bits = encoder_spec
        self.metric = METRIC[metric]
        self.m = m
        self.ef_construction = ef_construction
        self.graph_type = graph_type
        self.encoder = encoder
        self.pq_subspaces = pq_subspaces
        self.rabitq_total_bits = rabitq_total_bits
        self.rabitq_query_bits = rabitq_query_bits
        self.rerank_bits = rerank_bits
        self.residual_rerank = residual_rerank
        self.k_candidates = None
        self.ef_search = None
        self.alpha = None
        self.early_exit_threshold = None

    def fit(self, X):
        dim = X.shape[1]
        flat = np.ascontiguousarray(X, dtype=np.float32).ravel()
        self.index = kannolo.DenseRerankHNSW.build_from_array(
            flat,
            dim,
            self.m,
            self.ef_construction,
            self.metric,
            self.graph_type,
            self.encoder,
            self.pq_subspaces,
            self.rabitq_total_bits,
            self.rabitq_query_bits,
            self.rerank_bits,
            self.residual_rerank,
        )

    # VIBE passes query arguments by position, in the order of the `query_args:` keys in
    # config.yml: k_candidates, ef_multiplier, alpha, early_exit_threshold[, query_bits].
    # A different order would raise no error, but every value would go to the wrong parameter.
    def set_query_arguments(self, k_candidates, ef_multiplier, alpha=None,
                            early_exit_threshold=None, query_bits=None):
        self.k_candidates = k_candidates
        # The first-stage beam is a multiple of k_candidates, so it is not a separate grid axis.
        self.ef_search = max(int(round(ef_multiplier * k_candidates)), k_candidates)
        self.ef_multiplier = ef_multiplier
        self.alpha = alpha
        self.early_exit_threshold = early_exit_threshold
        # The query width only changes how the query is encoded, so one index serves every width.
        # Only the 1-bit `rabitq` encoder quantizes its query.
        if query_bits is not None:
            if self.encoder != "rabitq":
                raise ValueError(
                    f"query_bits is only meaningful for the 1-bit rabitq encoder, "
                    f"not {self.encoder!r}"
                )
            self.index.set_query_bits(query_bits)
            self.rabitq_query_bits = query_bits

    def query(self, v, n):
        # (query, k, k_candidates, ef_search, alpha, beta, early_exit_threshold); beta, the
        # rescoring early exit, is not used.
        ids = self.index.search(
            v, n, self.k_candidates, self.ef_search, self.alpha, None,
            self.early_exit_threshold,
        )[1]
        return ids[ids >= 0]

    def get_additional(self):
        return {"space_usage_bytes": self.index.space_usage_bytes()}

    def __str__(self):
        if self.encoder == "pq":
            enc = f"pq{self.pq_subspaces}"
        elif self.encoder == "rabitq":
            enc = f"rabitq1-q{self.rabitq_query_bits}"
        else:
            enc = f"rabitq{self.rabitq_total_bits}"
        alpha = "off" if self.alpha is None else f"{self.alpha:g}"
        lam = "none" if self.early_exit_threshold is None else f"{self.early_exit_threshold:g}"
        rr = "f16" if not self.rerank_bits else f"ext{self.rerank_bits}"
        if self.residual_rerank:
            rr += "-resid"
        return (
            f"kANNolo-Rerank({enc}+{rr}, graph={self.graph_type}, M={self.m}, "
            f"efC={self.ef_construction}, kc={self.k_candidates}, "
            f"efS={self.ef_search}, alpha={alpha}, lambda={lam})"
        )
