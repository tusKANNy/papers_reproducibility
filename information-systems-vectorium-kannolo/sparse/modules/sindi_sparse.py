"""SINDI under this harness: Lane B, because the PoC ships binaries and no per-query API.

The PoC (replicate-sindi-poc, commit in sparse/env.example.sh) ships only
`sparse/scripts/{sindi_index_build,sindi_index_search,generate_gt}`, with no Python bindings and
no per-query entry point. `sindi_index_search` takes the whole query file, runs it three times,
and prints one throughput number.

**This module drives a patched binary, and the patch is required**
(`patches/0003-sindi-search-pass-markers-and-ids-dump.patch`, added to the PoC by
sparse/patches/replicate-sindi-poc-4d22e55.patch). Upstream `sindi_index_search` cannot be
measured here for two reasons:

* **It emits no pass boundary.** The three `KnnSearch` calls (`sindi_index_search.cpp:166, 167,
  172`) run back to back with nothing printed between them, so neither where the index load
  ends nor where one pass stops and the next starts is visible.
* **It emits no ids.** They exist at `:187` (`result->GetIds()`) and are consumed only by its own
  `cal_recall`. A system's self-reported recall is never used, so without the ids there is no
  accuracy to compute.

The patch adds a `pass_done: <n>` line after each of the three searches and an `ids:` dump after
the last, and touches nothing else: the markers are single stdout writes between passes, the dump
happens after `end_time`, and upstream's own `qps` is computed from the same interval as before.

**The self-report is kept as a cross-check.** SINDI's `high_resolution_clock` brackets the whole
`KnnSearch` call of pass 3, which is our second timed pass, so `sindi_self_qps` and our
measurement of that pass should agree closely; a large gap means our pass boundaries are not
where we think they are.

**`alpha` is the paper's alpha directly: mass retained, lower alpha = more pruning.** It is the
opposite of mainline antgroup/vsag's `doc_prune_ratio`. Do not invert it (see the PoC's
`utils.cpp:64`, `total_mass * alpha`).

**`"dim": 30000` in the PoC's build JSON is inert.** `Sindi::build` derives `data_dim_` from the
data itself (`src/index/sindi.cpp:110-136`), so a larger vocabulary is not truncated by it. The
real bound is `query_dense(data_dim_)` at `sindi.cpp:248`, sized from the documents only: a query
term id above the document maximum would write out of bounds. `check_vocabulary` checks this
per collection.
"""

import os
import re
import subprocess
import sys

import numpy as np

from base import SparseCLI, SPARSE_ROOT

#: The PoC build tree (built with its build_poc_vsag.sh; see sparse/env.example.sh).
POC_BUILD = os.environ.get(
    "SINDI_POC_BUILD", os.path.join(SPARSE_ROOT, "build", "vsag-build-release"))
POC_REPO = os.environ.get(
    "SINDI_POC_REPO", os.path.join(SPARSE_ROOT, "repos", "replicate-sindi-poc"))

_PASS = re.compile(r"^pass_done:\s*(\d+)\s*$")
_IDS = re.compile(r"^ids:\s*(\d+)((?:\s+-?\d+)*)\s*$")  # zero ids allowed: an unanswered query
_SELF_QPS = re.compile(r"^qps:\s*([0-9eE.+\-]+)\s*$", re.MULTILINE)
_SELF_RECALL = re.compile(r"^recall:\s*([0-9eE.+\-]+)\s*$", re.MULTILINE)


class Sindi(SparseCLI):
    """SINDI's inverted index over the collection, via the PoC CLIs.

    Build arguments in the order sparse/grids.py declares them, which is the order
    the `result_file` stem carries and therefore the order `dominance.py`'s monotone indices
    refer to.
    """

    def __init__(self, lambda_window, alpha, use_reorder):
        self.lambda_window = lambda_window
        self.alpha = alpha
        self.use_reorder = use_reorder
        self.name = f"Sindi(lambda={lambda_window}, alpha={alpha}, use_reorder={use_reorder})"
        self.beta = None
        self.gamma = None
        self._topk = None
        self._self_qps = None
        self._self_recall = None

    # -- build ---------------------------------------------------------------------------------

    def build_stages(self, documents_path, queries_path, index_path):
        """CSR conversion, then `sindi_index_build`. Both inside the bracket.

        **The CSR conversion is timed but not charged**: a conversion cost depends on the format
        the collection ships in, not on the method. It runs on every build, and
        `build_conversion_s` records it beside the charged number.

        **`lambda` must lie in [10000, 100000]** or `sindi_index_build` aborts with SIGABRT
        (`failed to create index(invalid argument)`). The range is enforced by the library and
        not documented in the PoC.

        **Ground truth is not a build stage.** `sindi_index_search` requires a `gt_top<k>.bin` to
        compute its own recall; it is a prerequisite produced once per collection, outside the
        build, and `_gt_path` fails if it is missing.
        """
        seismic_dir = os.path.dirname(os.path.abspath(documents_path))
        if os.path.dirname(os.path.abspath(queries_path)) != seismic_dir:
            raise SystemExit(
                "convert_seismic_to_csr.py takes a --seismic-dir holding both documents.bin and "
                f"queries.bin, but --documents is in {seismic_dir} and --queries is not. Point "
                "both at the collection's own directory."
            )
        csr_dir = self._csr_dir_for(index_path)
        return [
            # Not charged: seismic .bin -> CSR is a pure format conversion, and the format the
            # collection ships in is not SINDI's choice.
            ("convert", [sys.executable,
                         os.path.join(POC_REPO, "scripts", "convert_seismic_to_csr.py"),
                         "--seismic-dir", seismic_dir,
                         "--out-dir", csr_dir], False),
            # Charged: the index build proper, including `use_reorder`'s document reordering,
            # which is data preparation.
            ("index", [os.path.join(POC_BUILD, "sparse", "scripts", "sindi_index_build"),
                       os.path.join(csr_dir, "base.csr"),
                       str(int(self.lambda_window)),
                       str(float(self.alpha)),
                       "1" if self.use_reorder else "0",
                       index_path], True),
        ]

    # -- search --------------------------------------------------------------------------------

    @staticmethod
    def _csr_dir_for(index_path):
        """Where base.csr / queries.csr / gt_top<k>.bin live: beside the index.

        Derived from `index_path` on every call rather than remembered from `build_stages`,
        because `build_stages` does not always run: with `--reuse-indexes` the driver skips the
        build and goes straight to searching, so state set during a build would be `None`.
        """
        return os.path.dirname(index_path) or "."

    def _gt_path(self, index_path, k):
        """The PoC's own `gt_top<k>.bin`, which the search binary requires as argv[3].

        SINDI's search tool computes its own recall against this file, so it must exist even
        though our accuracy is computed from the dumped ids. sparse/run_sparse.py writes it from
        the shipped `groundtruth.tsv` with sparse/tools/gt_tsv_to_bin.py. Fails rather than
        generating it, so a missing or stale file is never silently replaced.
        """
        csr_dir = self._csr_dir_for(index_path)
        path = os.path.join(csr_dir, f"gt_top{k}.bin")
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {path}. Generate it and verify it before any search:\n"
                f"  {POC_BUILD}/sparse/scripts/generate_gt {csr_dir}/base.csr "
                f"{csr_dir}/queries.csr {path} {k}\n"
                f"  python {POC_REPO}/scripts/compare_gt.py --bin {path} --tsv <groundtruth.tsv>\n"
                "STOP if compare_gt.py reports anything but a 100% exact set match."
            )
        return path

    def search_argv(self, index_path, k):
        self._topk = k
        return [
            os.path.join(POC_BUILD, "sparse", "scripts", "sindi_index_search"),
            index_path,
            os.path.join(self._csr_dir_for(index_path), "queries.csr"),
            self._gt_path(index_path, k),
            str(float(self.beta)),
            str(int(self.gamma)),
            str(int(k)),
            "1",  # num_threads: sequential, one query at a time, single core.
        ]

    def set_query_arguments(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma

    def pass_boundary(self, line):
        """`pass_done: <n>`, from patch 0003. `True` because SINDI reports no per-pass timing --
        its single self-reported `qps` covers pass 3 only and is picked up in `parse_ids`."""
        return True if _PASS.match(line) else None

    def parse_ids(self, stdout_text, n_queries, k):
        """The `ids:` dump from patch 0003, plus SINDI's own qps/recall stashed for the row.

        The self-reports are collected here rather than in `get_additional` because this is the
        only method handed the CLI's output. The driver calls `get_additional()` after
        `measure_lane_b` has returned, so what is stashed here is on the row.
        """
        m = _SELF_QPS.search(stdout_text)
        self._self_qps = float(m.group(1)) if m else None
        m = _SELF_RECALL.search(stdout_text)
        self._self_recall = float(m.group(1)) if m else None

        ids = np.full((n_queries, k), -1, dtype=np.int64)
        seen = 0
        for line in stdout_text.splitlines():
            m = _IDS.match(line)
            if not m:
                continue
            q = int(m.group(1))
            row = [int(t) for t in m.group(2).split()]
            if q < n_queries:
                if row:
                    ids[q, :min(k, len(row))] = row[:k]
                seen += 1
        if seen == 0:
            # Patch 0003 is not in the binary being run. Say so precisely: the alternative
            # reading -- "SINDI found nothing" -- would be a silent 0 % accuracy row.
            return None
        # Unanswered queries are scored as misses and counted by driver.answer_counts; more than
        # 1 % missing from the dump is refused as a truncated run.
        if n_queries - seen > n_queries // 100:
            raise SystemExit(
                f"sindi_index_search dumped ids for {seen} of {n_queries} queries. A partial dump "
                "would compute accuracy over a query set that is not the collection's."
            )
        return ids

    def index_bytes(self, index_path):
        return os.path.getsize(index_path) if os.path.exists(index_path) else None

    def get_additional(self):
        """SINDI's own numbers, recorded beside ours and never in place of them.

        `sindi_self_qps` times pass 3 with `high_resolution_clock` around the whole `KnnSearch`
        call. Pass 3 is our second timed pass, so the two should agree; a large gap means our
        pass boundaries are not where we think they are. `sindi_self_recall` is recorded for the
        same cross-check and is never the reported accuracy: the driver computes accuracy from
        the dumped ids against the shipped ground truth.
        """
        return {
            "sindi_self_qps": self._self_qps,
            "sindi_self_recall": self._self_recall,
            "sindi_poc_build": POC_BUILD,
            "patched_for_observability": "patches/0003-sindi-search-pass-markers-and-ids-dump",
        }


def check_vocabulary(documents_path, queries_path):
    """Guard `sindi.cpp:248`'s `query_dense(data_dim_)`: query terms must not exceed the base's.

    Run per collection before its first build: a violation is an out-of-bounds write.
    """
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import sparse_data

    _, _, max_doc = sparse_data.seismic_header(documents_path)
    _, _, max_qry = sparse_data.seismic_header(queries_path)
    ok = max_qry <= max_doc
    print(f"max document term {max_doc}, max query term {max_qry}: "
          f"{'OK' if ok else 'OUT OF BOUNDS -- do not build'}")
    return ok
