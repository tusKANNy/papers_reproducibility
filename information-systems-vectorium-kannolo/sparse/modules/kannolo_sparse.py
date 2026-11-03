"""kANNolo's two sparse indexes (in-process timing), plus an exhaustive index for checking accuracy.

The module is named `kannolo_sparse` because a module called `kannolo.py` on the path would
shadow the compiled `kannolo` extension it imports: `import kannolo` inside it would resolve to
itself.

Three classes:

* `KannoloSparseHNSW`   -- single-stage HNSW. With kannolo >= 0.15.0 the index is
                           blocked-sparse (f16 values, graph built over f16); with earlier
                           releases it is plain f16.
* `KannoloSparseRerank` -- two-stage HNSW. The collection is L1-pruned at factor `a` into a
                           first stage and its exact complement. The residual rescoring sums the
                           two scores, so at f16/f16 the final score equals the score of the
                           single-stage index. Accuracy can be lost only through candidate
                           recall.
* `KannoloSparseFlat`   -- exhaustive scan. Not reported: it gives the 100% point and is used to
                           check the driver's accuracy function.

Both indexes take their construction parameters from sparse/grids.py.
"""

import os

import numpy as np

import kannolo

from base import SparseANN


def _as_query(components, values):
    """kANNolo's bindings take int32 components and float32 values."""
    return components.astype(np.int32, copy=False), values.astype(np.float32, copy=False)


class KannoloSparseHNSW(SparseANN):
    """Single-stage sparse HNSW, dot product: plain f16 before kannolo 0.15.0, blocked-sparse
    from 0.15.0. Indexes saved by one do not load in the other."""

    def __init__(self, m, ef_construction, graph_type="standard"):
        self.m = m
        self.ef_construction = ef_construction
        self.graph_type = graph_type
        self.ef_search = None
        self.lambda_ = None
        self.index = None

    def fit(self, documents_path):
        # By keyword: kannolo 0.15.0 removed `metric` (dot product only), so the positional call
        # of earlier releases would bind "dotproduct" to `graph_type`.
        self.index = kannolo.SparsePlainHNSW.build_from_file(
            documents_path,
            m=self.m,
            ef_construction=self.ef_construction,
            graph_type=self.graph_type,
        )

    def set_query_arguments(self, ef_search, early_exit_threshold=None):
        self.ef_search = ef_search
        self.lambda_ = early_exit_threshold

    def query(self, components, values, k):
        c, v = _as_query(components, values)
        _, ids = self.index.search(c, v, k, self.ef_search, self.lambda_)
        return ids

    def index_bytes(self):
        # The library's own SpaceUsage, recorded beside the USS delta so the two can be compared.
        return self.index.space_usage_bytes()

    def save(self, path):
        self.index.save(path)

    def load(self, path):
        # `graph_type` must match the build call: index files are not self-describing, and
        # "standard"/"permuted" share one on-disk representation.
        self.index = kannolo.SparsePlainHNSW.load(path, graph_type=self.graph_type)

    def __str__(self):
        return (
            f"kANNolo-HNSW(M={self.m}, efC={self.ef_construction}, graph={self.graph_type}, "
            f"efSearch={self.ef_search}, lambda={self.lambda_})"
        )


#: kannolo < 0.12.0 splits at build with `build_l1_pruned` and sums the stages with
#: `residuals=True` at search; 0.12.0 and later fix both at build. Same index either way.
_PRE_012 = hasattr(kannolo.SparseRerankHNSW, "build_l1_pruned")


class KannoloSparseRerank(SparseANN):
    """Two-stage sparse HNSW: L1-pruned first stage, exact residual rescoring.

    Built with `residual_rerank=True` (kannolo >= 0.12.0): the rerank stage holds the pruned-away
    components and the two scores are summed into the full dot product. Older releases build the
    same index through `build_l1_pruned` plus `residuals=True` at search time.
    """

    def __init__(self, m, ef_construction, prune_percentage,
                 first_stage_value_type="f16", graph_type="standard"):
        self.m = m
        self.ef_construction = ef_construction
        self.prune_percentage = prune_percentage
        self.first_stage_value_type = first_stage_value_type
        self.graph_type = graph_type
        self.k_candidates = None
        self.ef_search = None
        self.alpha = None
        self.beta = None
        self.lambda_ = None
        self.index = None

    def fit(self, documents_path):
        # The split happens inside the build call, so the pruning cost is charged to the build,
        # like every system's data preparation.
        if _PRE_012:
            self.index = kannolo.SparseRerankHNSW.build_l1_pruned(
                documents_path, self.prune_percentage, m=self.m,
                ef_construction=self.ef_construction,
                first_stage_value_type=self.first_stage_value_type, graph_type=self.graph_type)
            return
        self.index = kannolo.SparseRerankHNSW.build_from_file(
            documents_path,
            m=self.m,
            ef_construction=self.ef_construction,
            first_stage_value_type=self.first_stage_value_type,
            graph_type=self.graph_type,
            prune_percentage=self.prune_percentage,
            residual_rerank=True,
        )

    def set_query_arguments(self, k_candidates, ef_search=None, alpha=None, beta=None,
                            early_exit_threshold=None):
        self.k_candidates = k_candidates
        # At high accuracy the two-stage index is limited by k_candidates, so ef_search is not
        # swept independently: it is set from k_candidates (default: equal to it).
        self.ef_search = ef_search if ef_search is not None else k_candidates
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = early_exit_threshold

    def query(self, components, values, k):
        c, v = _as_query(components, values)
        _, ids = self.index.search(
            c, v,
            self.k_candidates,
            k,
            self.ef_search,
            self.alpha,
            self.beta,
            self.lambda_,
            **({"residuals": True} if _PRE_012 else {}),
        )
        return ids

    def index_bytes(self):
        # Both stages together. For an L1-pruned pair this shows that the decomposition does not
        # double the index: the two stages split the original non-zeroes between them.
        return self.index.space_usage_bytes()

    def save(self, path):
        self.index.save(path)

    def load(self, path):
        if _PRE_012:
            self.index = kannolo.SparseRerankHNSW.load(path, self.first_stage_value_type, self.graph_type)
            return
        self.index = kannolo.SparseRerankHNSW.load(
            path, self.first_stage_value_type, self.graph_type, residual_rerank=True
        )

    def __str__(self):
        return (
            f"kANNolo-HNSW-Rerank(M={self.m}, efC={self.ef_construction}, "
            f"a={self.prune_percentage}, first={self.first_stage_value_type}, "
            f"graph={self.graph_type}, kCand={self.k_candidates}, efSearch={self.ef_search}, "
            f"alpha={self.alpha}, beta={self.beta}, lambda={self.lambda_})"
        )


class KannoloSparseFlat(SparseANN):
    """Exhaustive sparse scan: the 100% point, and a check of the driver's accuracy function."""

    def __init__(self):
        self.index = None

    def fit(self, documents_path):
        self.index = kannolo.SparseFlatIndex.build_from_file(documents_path)

    def set_query_arguments(self):
        return

    def query(self, components, values, k):
        c, v = _as_query(components, values)
        _, ids = self.index.search(c, v, k)
        return ids

    def __str__(self):
        return "kANNolo-SparseFlat(exhaustive)"
