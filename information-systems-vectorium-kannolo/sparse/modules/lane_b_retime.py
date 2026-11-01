"""Lane-B command-line drivers for the systems tuned in Lane A (kANNolo, PyANN) and for Seismic.

The systems tuned in-process (Lane A) are timed for the reported numbers through their own
command-line tools (Lane B), on the selected configurations only and on the indexes Lane A
saved. No index is rebuilt: these classes refuse to build, and sparse/run_sparse.py joins each
re-timed row's `build_time_s` from the Lane-A row of the same build.

Each search CLI must print a `pass_done: N` marker at the end of every pass over the query file,
which is what the driver times between. The patches in sparse/patches add these markers and,
where needed, clear the result buffer between passes so that appending results does not grow
across passes inside the timed region. Every pass goes over the whole query file, as Seismic,
LSP and SINDI do: repeating one query `num_runs` times back to back would be more cache-friendly
than the competitors' passes.

Accuracy does not depend on the lane: the same index, parameters and queries return the same
ids, so a re-timed row must reproduce its Lane-A accuracy, and one that does not is driving a
different index or search.

Ids are read from each CLI's own results file (`query_id\\tdoc_id\\trank\\tscore`, the
`groundtruth.tsv` shape), written after the timed passes, so reading it costs the measurement
nothing.

`--num-runs 3` everywhere: one warm-up pass plus the two timed passes the driver reports.
"""

import os
import re
import subprocess
import sys

import numpy as np

from base import SparseCLI, SPARSE_ROOT

SEISMIC_REPO = os.environ.get("SEISMIC_REPO", os.path.join(SPARSE_ROOT, "repos", "seismic"))
KANNOLO_REPO = os.environ.get("KANNOLO_REPO", os.path.join(SPARSE_ROOT, "repos", "kannolo"))
#: Our CLIs are in sparse/tools/; the env vars override them.
_TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
PYANNS_CLI = os.environ.get("PYANNS_CLI", os.path.join(_TOOLS, "pyanns_search_cli.py"))

#: PyANN lives in its own venv (it pins older numpy and pybind11 versions), so its CLI cannot be
#: run with the driver's interpreter. kANNolo's rerank CLI can, and does.
PYANNS_PYTHON = os.environ.get(
    "PYANNS_PYTHON", os.path.join(SPARSE_ROOT, "venv-pyanns", "bin", "python"))

#: `KannoloRerank` times kANNolo's sparse two-stage index through our Python CLI on the
#: `SparseRerankHNSW` binding (`hnsw_rerank_search` is the multivector rerank path). `batch_search`
#: takes the whole query set in one call, so Python is entered once per pass rather than once per
#: query, and the per-query interpreter overhead of Lane A does not apply.
KANNOLO_RERANK_CLI = os.environ.get(
    "KANNOLO_RERANK_CLI", os.path.join(_TOOLS, "kannolo_rerank_cli.py"))

#: 1 warm-up + 2 timed. See the module docstring.
NUM_RUNS = int(os.environ.get("RETIME_NUM_RUNS", "3"))

_PASS = re.compile(r"^pass_done:\s*(\d+)\s*$")


class _RetimeCLI(SparseCLI):
    """Shared behaviour: never build, read ids from the CLI's results file, count `pass_done`."""

    def build_stages(self, documents_path, queries_path, index_path):
        raise SystemExit(
            f"{type(self).__name__} must never build. This module re-times configurations whose "
            "indexes are already on disk and whose build times are the Lane-A measurements. "
            "Reaching here means --reuse-indexes did not find "
            f"{index_path}, which usually means --algo does not match the name the index was "
            "written under (the reuse path keys on it).")

    def shared_preparation(self, index_path):
        return []

    def _results_path(self, index_path):
        return f"{index_path}.retime.tsv"

    def pass_boundary(self, line):
        m = _PASS.match(line)
        return float(m.group(1)) if m else None

    def parse_ids(self, stdout_text, n_queries, k):
        """Read the CLI's own results file: `query_id\\tdoc_id\\trank\\tscore`."""
        path = getattr(self, "_out", None)
        if not path or not os.path.exists(path):
            raise SystemExit(
                f"{type(self).__name__}: results file {path!r} not written. Without returned ids "
                "accuracy cannot be computed, and a system's self-reported recall is never used.")
        ids = np.full((n_queries, k), -1, dtype=np.int64)
        seen = set()
        with open(path) as fh:
            for line in fh:
                f = line.split("\t")
                if len(f) < 3:
                    continue
                q, doc, rank = int(f[0]), int(f[1]), int(f[2])
                if q < n_queries and 1 <= rank <= k:
                    ids[q, rank - 1] = doc
                    seen.add(q)
        if not seen:
            return None
        # Unanswered queries are scored as misses and counted by driver.answer_counts; more than
        # 1 % missing is refused as a truncated run.
        if n_queries - len(seen) > n_queries // 100:
            raise SystemExit(
                f"{type(self).__name__} returned results for {len(seen)} of {n_queries} queries.")
        return ids

    def index_bytes(self, index_path):
        return os.path.getsize(index_path) if os.path.exists(index_path) else None

    def get_additional(self):
        return {"retimed_from_lane_a": True, "retime_num_runs": NUM_RUNS,
                "retime_note": ("search timed in Lane B on the index built and measured in "
                                "Lane A; build_time_s is that Lane-A measurement")}


class Seismic(_RetimeCLI):
    """Seismic through `perf_inverted_index`. Build args are baked into the index path."""

    def __init__(self, n_postings, centroid_fraction, min_cluster_size, summary_energy,
                 max_fraction, doc_cut, nknn):
        self.build_args = dict(n_postings=n_postings, centroid_fraction=centroid_fraction,
                               min_cluster_size=min_cluster_size, summary_energy=summary_energy,
                               max_fraction=max_fraction, doc_cut=doc_cut, nknn=nknn)
        self.name = f"Seismic-LaneB({self.build_args})"
        self.query_cut = self.heap_factor = self.n_knn = None
        self.first_sorted = False

    def set_query_arguments(self, query_cut, heap_factor, n_knn=0, first_sorted=False):
        self.query_cut, self.heap_factor = query_cut, heap_factor
        self.n_knn, self.first_sorted = n_knn, first_sorted

    def search_argv(self, index_path, k):
        self._out = self._results_path(index_path)
        argv = [os.path.join(SEISMIC_REPO, "target", "release", "perf_inverted_index"),
                "--index-file", index_path,
                "--query-file", os.environ["RETIME_QUERIES"],
                "--output-path", self._out,
                "--n-queries", os.environ.get("RETIME_N_QUERIES", "3452"),
                "-k", str(int(k)),
                "--n-runs", str(NUM_RUNS),
                "--query-cut", str(int(self.query_cut)),
                "--heap-factor", str(float(self.heap_factor)),
                "--n-knn", str(int(self.n_knn or 0))]
        if self.first_sorted:
            argv.append("--first-sorted")
        return argv


class SeismicWave(Seismic):
    """Seismic-Wave is the same binary and the same index format; `n_knn > 0` is what uses the
    kNN graph the build paid for, so a zero here would be searching a different system."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.name = f"Seismic-Wave-LaneB({self.build_args})"

    def set_query_arguments(self, query_cut, heap_factor, n_knn, first_sorted=False):
        if not n_knn:
            raise ValueError("SeismicWave with n_knn=0 never consults the graph it built.")
        super().set_query_arguments(query_cut, heap_factor, n_knn, first_sorted)


class KannoloHnsw(_RetimeCLI):
    """Single-stage sparse HNSW through `hnsw_search`, with f16 values."""

    def __init__(self, m, ef_construction, graph_type="standard"):
        self.build_args = dict(m=m, ef_construction=ef_construction, graph_type=graph_type)
        self.graph_type = graph_type
        self.name = f"kANNolo-HNSW-LaneB({self.build_args})"
        self.ef_search = self.lambda_ = None

    def set_query_arguments(self, ef_search, early_exit_threshold=None):
        self.ef_search, self.lambda_ = ef_search, early_exit_threshold

    def search_argv(self, index_path, k):
        self._out = self._results_path(index_path)
        argv = [os.path.join(KANNOLO_REPO, "target", "release", "hnsw_search"),
                "--index-file", index_path,
                "--query-file", os.environ["RETIME_QUERIES"],
                "--output-path", self._out,
                "--dataset-type", "sparse", "--value-type", "f16",
                "--component-type", "u16", "--encoder", "plain",
                "--graph-type", self.graph_type, "--distance", "dotproduct",
                "-k", str(int(k)), "--ef-search", str(int(self.ef_search)),
                "--num-runs", str(NUM_RUNS)]
        # lambda relaxes the stopping rule; it is only meaningful with the adaptive strategy, and
        # the Lane-A module passes None to mean "off". Mirrored exactly rather than defaulted.
        if self.lambda_ is not None:
            argv += ["--early-termination", "distance-adaptive", "--lambda", str(float(self.lambda_))]
        else:
            argv += ["--early-termination", "none"]
        return argv


class KannoloRerank(_RetimeCLI):
    """Two-stage index through sparse/tools/kannolo_rerank_cli.py, which loads it with
    `residual_rerank=True`: the two stages sum to the full dot product."""

    def __init__(self, m, ef_construction, prune_percentage, first_stage_value_type="f16",
                 graph_type="standard"):
        self.build_args = dict(m=m, ef_construction=ef_construction,
                               prune_percentage=prune_percentage,
                               first_stage_value_type=first_stage_value_type,
                               graph_type=graph_type)
        self.graph_type = graph_type
        self.name = f"kANNolo-HNSW-Rerank-LaneB({self.build_args})"
        self.k_candidates = self.ef_search = None
        self.alpha = self.beta = self.lambda_ = None

    def set_query_arguments(self, k_candidates, ef_search=None, alpha=None, beta=None,
                            early_exit_threshold=None):
        self.k_candidates = k_candidates
        self.ef_search = ef_search if ef_search is not None else k_candidates
        self.alpha, self.beta, self.lambda_ = alpha, beta, early_exit_threshold

    def search_argv(self, index_path, k):
        self._out = self._results_path(index_path)
        argv = [sys.executable, KANNOLO_RERANK_CLI,
                "--index-file", index_path,
                "--query-file", os.environ["RETIME_QUERIES"],
                "--output-path", self._out,
                "--first-stage-value-type", self.build_args["first_stage_value_type"],
                "--graph-type", self.graph_type,
                "-k", str(int(k)), "--k-candidates", str(int(self.k_candidates)),
                "--ef-search", str(int(self.ef_search)),
                "--num-runs", str(NUM_RUNS)]
        if self.alpha is not None:
            argv += ["--alpha", str(float(self.alpha))]
        if self.beta is not None:
            argv += ["--beta", str(int(self.beta))]
        if self.lambda_ is not None:
            argv += ["--lambda", str(float(self.lambda_))]
        return argv


class Pyanns(_RetimeCLI):
    """PyANN through our CLI (sparse/tools/pyanns_search_cli.py); PyANN ships none.

    Its sparse API is batch-only (`search_batch` takes `nq` and an `#pragma omp` team). The CLI
    calls it once per query (`nq = 1`), because a larger batch prunes query terms against the
    batch-wide maximum and so changes the search (see the CLI's docstring). The CLI pins
    `OMP_NUM_THREADS=1` for search, as the Lane-A module does, or a single pinned core would still
    spin up a thread team of the machine's size.
    """

    def __init__(self, r, l, drop_ratio=0.0):
        self.build_args = dict(r=r, l=l, drop_ratio=drop_ratio)
        self.name = f"PyANN-SparseHNSW-LaneB({self.build_args})"
        self.ef = self.budget = None

    def set_query_arguments(self, ef, budget=0.1):
        self.ef, self.budget = ef, budget

    def search_argv(self, index_path, k):
        self._out = self._results_path(index_path)
        return [PYANNS_PYTHON, PYANNS_CLI,
                "--index-file", index_path,
                "--query-file", os.environ["RETIME_QUERIES"],
                "--output-path", self._out,
                "-k", str(int(k)), "--ef", str(int(self.ef)),
                "--budget", str(float(self.budget)),
                "--num-runs", str(NUM_RUNS)]
