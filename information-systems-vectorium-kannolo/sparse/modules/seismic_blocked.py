"""Seismic and Seismic-Wave on the `blocked-sparse` value type, with command-line timing.

`BlockedSparseEncoder` (Vectorium) stores u16 components and f16 values packed into 64-byte
blocks with a SIMD dot product. This module runs Seismic's Rust CLIs from `$SEISMIC_BS_REPO`,
built with `--value-type blocked-sparse` support (Seismic 0.6.0 with the `pass_done` marker
patch, sparse/patches/seismic-cc065cf-pass-markers.patch; see sparse/env.example.sh).

The build argv follows the authors' `scripts/run_experiments.py`: it never passes `--block-size`
and omits `kmeans-pruning-factor`. The driver measures the build time, so it is charged like
every other build with command-line timing. `--knn 50` makes the index Seismic-Wave and includes
the kNN graph in the charged build.

Blocked-sparse and `f16` builds with the same parameters compute the same scores but can visit
different candidates, so accuracy is measured on the blocked index, never assumed.
"""
import json
import os

from base import SPARSE_ROOT

from cli_retime import _RetimeCLI, NUM_RUNS

REPO = os.environ.get("SEISMIC_BS_REPO", os.path.join(SPARSE_ROOT, "repos", "seismic"))
VALUE_TYPE = "blocked-sparse"


class SeismicBlocked(_RetimeCLI):
    """Seismic (fork CLIs), `blocked-sparse`. Unlike the re-timing classes, it builds its own index."""


    def __init__(self, n_postings, centroid_fraction, min_cluster_size, summary_energy,
                 max_fraction, doc_cut, nknn):
        self.build_args = dict(n_postings=n_postings, centroid_fraction=centroid_fraction,
                               min_cluster_size=min_cluster_size, summary_energy=summary_energy,
                               max_fraction=max_fraction, doc_cut=doc_cut, nknn=nknn)
        self.name = f"{type(self).__name__}({self.build_args})"
        self.query_cut = self.heap_factor = self.n_knn = None
        self.first_sorted = False

    def build_stages(self, documents_path, queries_path, index_path):
        b = self.build_args
        build = [os.path.join(REPO, "target", "release", "build_inverted_index"),
                 "--input-file", documents_path, "--output-file", index_path,
                 "--n-postings", str(int(b["n_postings"])),
                 "--centroid-fraction", str(float(b["centroid_fraction"])),
                 "--knn", str(int(b["nknn"])),
                 "--clustering-algorithm", "random-kmeans-inverted-index-approx",
                 "--summary-energy", str(float(b["summary_energy"])),
                 "--kmeans-doc-cut", str(int(b["doc_cut"])),
                 "--min-cluster-size", str(int(b["min_cluster_size"])),
                 "--max-fraction", str(float(b["max_fraction"])),
                 "--component-type", "u16", "--value-type", VALUE_TYPE,
                 "--pruning-strategy", "global-threshold"]
        # The CLI appends `.index.seismic`; the rename puts the index where the driver's reuse and
        # size checks look. It is a file move, so it is not charged.
        return [("build_inverted_index", build, True),
                ("rename", ["mv", f"{index_path}.index.seismic", index_path], False)]

    def set_query_arguments(self, query_cut, heap_factor, n_knn=0, first_sorted=False):
        self.query_cut, self.heap_factor = query_cut, heap_factor
        self.n_knn, self.first_sorted = n_knn, first_sorted

    def search_argv(self, index_path, k):
        self._out = self._results_path(index_path)
        argv = [os.path.join(REPO, "target", "release", "perf_inverted_index"),
                "--index-file", index_path,
                "--query-file", os.environ["RETIME_QUERIES"],
                "--output-path", self._out,
                "--n-queries", os.environ.get("RETIME_N_QUERIES", "6980"),
                "-k", str(int(k)), "--n-runs", str(NUM_RUNS),
                "--query-cut", str(int(self.query_cut)),
                "--heap-factor", str(float(self.heap_factor)),
                "--n-knn", str(int(self.n_knn or 0)),
                "--component-type", "u16", "--value-type", VALUE_TYPE]
        if self.first_sorted:
            argv.append("--first-sorted")
        return argv

    def get_additional(self):
        # SEISMIC_BS_PROVENANCE (JSON) overrides the Seismic build recorded on the row.
        env = os.environ.get("SEISMIC_BS_PROVENANCE")
        if env:
            return dict(json.loads(env), value_type=VALUE_TYPE)
        return {"value_type": VALUE_TYPE, "seismic": "0.6.0 (cc065cf) + pass markers", "vectorium": "0.11.0"}


class SeismicWaveBlocked(SeismicBlocked):
    """Seismic-Wave on the fork, `blocked-sparse`: the same CLIs with the build's `--knn 50`."""


    def set_query_arguments(self, query_cut, heap_factor, n_knn, first_sorted=False):
        if not n_knn:
            raise ValueError("SeismicWave with n_knn=0 never consults the graph it built.")
        super().set_query_arguments(query_cut, heap_factor, n_knn, first_sorted)
