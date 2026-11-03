"""SINDI under this harness, with command-line timing because SINDI ships binaries and no per-query API.

SINDI's code (the `sparse` branch of Roxanne0321/vsag, commit in sparse/install.sh) ships only
`sparse/scripts/{sindi_index_build,sindi_index_search,generate_gt}`, with no Python bindings and
no per-query entry point. `sindi_index_search` takes the whole query file, runs it three times,
and prints one throughput number.

**This module drives a patched binary, and the patch is required**
(sparse/patches/vsag-93e034f.patch, applied by sparse/install.sh). Upstream `sindi_index_search` cannot be
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
`KnnSearch` call of pass 3, which is our second timed pass. `sindi_self_qps` and our
measurement of that pass should therefore agree closely. A large gap means that our pass
boundaries are misplaced.

**`alpha` is the paper's alpha: the mass retained, so a lower alpha prunes more.** It is the
opposite of mainline antgroup/vsag's `doc_prune_ratio`. Do not invert it (see vsag's
`utils.cpp:64`, `total_mass * alpha`).

**`"dim": 30000` in SINDI's build JSON is inert.** `Sindi::build` derives `data_dim_` from the
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

#: The vsag build tree (built by sparse/install.sh).
SINDI_BUILD = os.environ.get(
    "SINDI_BUILD", os.path.join(SPARSE_ROOT, "build", "vsag-build-release"))
TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")

_PASS = re.compile(r"^pass_done:\s*(\d+)\s*$")
_IDS = re.compile(r"^ids:\s*(\d+)((?:\s+-?\d+)*)\s*$")  # zero ids allowed: an unanswered query
_SELF_QPS = re.compile(r"^qps:\s*([0-9eE.+\-]+)\s*$", re.MULTILINE)
_SELF_RECALL = re.compile(r"^recall:\s*([0-9eE.+\-]+)\s*$", re.MULTILINE)


class Sindi(SparseCLI):
    """SINDI's inverted index over the collection, via its command-line tools.

    Build arguments follow the order of sparse/grids.py. The `result_file` stem uses the same
    order, and so do the monotone indices of `dominance.py`.
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
        """CSR conversion, then `sindi_index_build`. Both are inside the timed build.

        **The CSR conversion is timed but not charged**: its cost depends on the format the
        collection ships in, and not on the method. It runs on every build, and
        `build_conversion_s` records it beside the charged number.

        **`lambda` must lie in [10000, 100000]** or `sindi_index_build` aborts with SIGABRT
        (`failed to create index(invalid argument)`). The range is enforced by the library and
        not documented in SINDI's code.

        **Ground truth is not a build stage.** `sindi_index_search` requires a `gt_top<k>.bin` to
        compute its own recall. It is a prerequisite produced once per collection, outside the
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
                         os.path.join(TOOLS, "convert_seismic_to_csr.py"),
                         "--seismic-dir", seismic_dir,
                         "--out-dir", csr_dir], False),
            # Charged: the index build proper, including `use_reorder`'s document reordering,
            # which is data preparation.
            ("index", [os.path.join(SINDI_BUILD, "sparse", "scripts", "sindi_index_build"),
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

        Derived from `index_path` on every call, and not stored by `build_stages`, because
        `build_stages` does not always run: with `--reuse-indexes` the driver skips the
        build and goes straight to searching, so state set during a build would be `None`.
        """
        return os.path.dirname(index_path) or "."

    def _gt_path(self, index_path, k):
        """SINDI's own `gt_top<k>.bin`, which the search binary requires as argv[3].

        SINDI's search tool computes its own recall against this file, so it must exist even
        though our accuracy is computed from the dumped ids. sparse/run_sparse.py writes it from
        the shipped `groundtruth.tsv` with sparse/tools/gt_tsv_to_bin.py. This method fails if the
        file is missing and never generates it, so a missing or stale file is never silently
        replaced.
        """
        csr_dir = self._csr_dir_for(index_path)
        path = os.path.join(csr_dir, f"gt_top{k}.bin")
        if not os.path.exists(path):
            raise SystemExit(
                f"missing {path}. Generate it and verify it before any search:\n"
                f"  {SINDI_BUILD}/sparse/scripts/generate_gt {csr_dir}/base.csr "
                f"{csr_dir}/queries.csr {path} {k}\n"
                f"  python {TOOLS}/compare_gt.py --bin {path} --tsv <groundtruth.tsv>\n"
                "STOP if compare_gt.py reports anything but a 100% exact set match."
            )
        return path

    def search_argv(self, index_path, k):
        self._topk = k
        return [
            os.path.join(SINDI_BUILD, "sparse", "scripts", "sindi_index_search"),
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
        """`pass_done: <n>`, added by the vsag patch. Returns `True` because SINDI reports no
        per-pass timing: its single self-reported `qps` covers pass 3 only and is read in
        `parse_ids`."""
        return True if _PASS.match(line) else None

    def parse_ids(self, stdout_text, n_queries, k):
        """The `ids:` dump added by the vsag patch, plus SINDI's own qps/recall stored for the row.

        The self-reports are collected here, and not in `get_additional`, because this is the
        only method that receives the CLI's output. The driver calls `get_additional()` after
        `measure_cli` has returned, so the values stored here reach the row.
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
            # The binary being run does not include the vsag patch. Returning None reports this;
            # reading it as "SINDI found nothing" would give a silent 0 % accuracy row.
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
        """SINDI's own numbers, recorded next to ours. They never replace ours.

        `sindi_self_qps` times pass 3 with `high_resolution_clock` around the whole `KnnSearch`
        call. Pass 3 is our second timed pass, so the two should agree. A large gap means that our
        pass boundaries are misplaced. `sindi_self_recall` is recorded for the
        same cross-check and is never the reported accuracy: the driver computes accuracy from
        the dumped ids against the shipped ground truth.
        """
        return {
            "sindi_self_qps": self._self_qps,
            "sindi_self_recall": self._self_recall,
            "sindi_build": SINDI_BUILD,
            "patched_for_observability": "sparse/patches/vsag-93e034f.patch",
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
          f"{'OK' if ok else 'OUT OF BOUNDS, do not build'}")
    return ok
