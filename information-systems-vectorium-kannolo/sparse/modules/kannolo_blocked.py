"""kANNolo's two sparse systems on the `blocked-sparse` value type, Lane B.

Built and searched by kANNolo's own Rust CLIs: kANNolo 0.15.1 (vectorium 0.11.1) plus a
`pass_done: N` marker per pass in `hnsw_search` / `hnsw_rerank_search_sparse`
(sparse/patches/kannolo-v0.15.0-pass-markers.patch), which the driver times between.

Both systems store blocked-sparse vectors with `f16` values, and both build their graph over the
`f16` collection before re-encoding it. The two-stage index's blocked first stage
(`SparseFirstStage::Blocked`) re-encodes both stages, so the graph is that of an `f16` build and
the scores are the same `f16` products. Accuracy is measured on the blocked index, not assumed
from the `f16` one. Both builds are charged in full.
"""
import os

from base import SPARSE_ROOT

from lane_b_retime import KannoloHnsw, _RetimeCLI, NUM_RUNS

BIN = os.environ.get("KANNOLO_BS_BIN", os.path.join(SPARSE_ROOT, "repos", "kannolo", "target", "release"))
VALUE_TYPE = "blocked-sparse"
PROVENANCE = {"value_type": VALUE_TYPE, "kannolo": "0.15.1 + pass markers", "vectorium": "0.11.1"}


class KannoloHnswBlocked(KannoloHnsw):
    """Single-stage sparse HNSW, blocked-sparse, built by `hnsw_build`."""

    def build_stages(self, documents_path, queries_path, index_path):
        b = self.build_args
        return [("hnsw_build", [os.path.join(BIN, "hnsw_build"),
                                "--data-file", documents_path, "--output-file", index_path,
                                "--dataset-type", "sparse", "--value-type", VALUE_TYPE,
                                "--component-type", "u16", "--encoder", "plain",
                                "--graph-type", b["graph_type"], "--distance", "dotproduct",
                                "--m", str(int(b["m"])),
                                "--ef-construction", str(int(b["ef_construction"]))], True)]

    def search_argv(self, index_path, k):
        argv = super().search_argv(index_path, k)
        argv[0] = os.path.join(BIN, "hnsw_search")
        argv[argv.index("--value-type") + 1] = VALUE_TYPE
        return argv

    def get_additional(self):
        return dict(PROVENANCE)


class KannoloRerankBlocked(_RetimeCLI):
    """Two-stage sparse HNSW, both stages blocked-sparse, built by `hnsw_rerank_build_sparse`."""

    def __init__(self, m, ef_construction, prune_percentage, first_stage_value_type=VALUE_TYPE,
                 graph_type="standard"):
        if first_stage_value_type != VALUE_TYPE:
            raise ValueError(f"KannoloRerankBlocked builds {VALUE_TYPE}, got {first_stage_value_type}")
        self.build_args = dict(m=m, ef_construction=ef_construction, graph_type=graph_type,
                               prune_percentage=prune_percentage,
                               first_stage_value_type=first_stage_value_type)
        self.name = f"kANNolo-Rerank-Blocked({self.build_args})"
        self.k_candidates = self.ef_search = self.alpha = self.beta = self.lambda_ = None

    def build_stages(self, documents_path, queries_path, index_path):
        b = self.build_args
        return [("hnsw_rerank_build_sparse",
                 [os.path.join(BIN, "hnsw_rerank_build_sparse"),
                  "--data-file", documents_path, "--output-file", index_path,
                  "--prune-percentage", str(float(b["prune_percentage"])), "--residual-rerank",
                  "--first-stage-value-type", VALUE_TYPE, "--graph-type", b["graph_type"],
                  "--m", str(int(b["m"])), "--ef-construction", str(int(b["ef_construction"]))],
                 True)]

    def set_query_arguments(self, k_candidates, ef_search=None, alpha=None, beta=None,
                            early_exit_threshold=None):
        self.k_candidates = k_candidates
        self.ef_search = ef_search if ef_search is not None else k_candidates
        self.alpha, self.beta, self.lambda_ = alpha, beta, early_exit_threshold

    def search_argv(self, index_path, k):
        self._out = self._results_path(index_path)
        argv = [os.path.join(BIN, "hnsw_rerank_search_sparse"),
                "--index-file", index_path,
                "--query-file", os.environ["RETIME_QUERIES"],
                "--output-path", self._out,
                "--residual-rerank", "--first-stage-value-type", VALUE_TYPE,
                "--graph-type", self.build_args["graph_type"],
                "-k", str(int(k)), "--k-candidates", str(int(self.k_candidates)),
                "--ef-search", str(int(self.ef_search)), "--num-runs", str(NUM_RUNS)]
        if self.alpha is not None:
            argv += ["--alpha", str(float(self.alpha))]
        if self.beta is not None:
            argv += ["--beta", str(int(self.beta))]
        if self.lambda_ is not None:
            argv += ["--early-termination", "distance-adaptive", "--lambda", str(float(self.lambda_))]
        else:
            argv += ["--early-termination", "none"]
        return argv

    def get_additional(self):
        return dict(PROVENANCE)
