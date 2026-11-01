"""LSP under this harness: Lane B, because it has no per-query API.

Lightweight Superblock Pruning (SIGIR '26, Carlson/Xie/Shah/Yang), `thefxperson/hierarchical_pruning`
(commit in sparse/env.example.sh). Its predecessor, Dynamic Superblock Pruning (DSP, SIGIR '25),
shares its input pipeline.

**LSP takes CIFF as input, not the Seismic format.** LSP can compress its document index with
Seismic's forward index (`--compress-document-index seismic`), but that is an internal option,
not an input format. The README requires that "the impact scores of the CIFF files have to be
quantized to 8 bits".

**The preparation is a property of the collection and is shared by all of its builds.** The
collection is converted to JSONL, quantized to 8-bit CIFF (`jsonl2ciff --quantize`), reordered
by recursive graph bisection (`create-rgb`), and the queries are encoded for PISA, each step
timed (sparse/run_sparse.py runs this once per collection with replicate-dsp, patched in
sparse/patches). `SPARSE_CIFF_PREP` names that directory and has no default, so a second copy
with a second `measured_by` is never prepared silently. The quantization and the bisection are
charged in full to every build through `shared_preparation`, which carries the sidecar's
`measured_by` onto every row; the conversions are format conversion and are not charged.

**Knobs, read from source rather than from the README.**

| knob | meaning |
|---|---|
| `gamma` | guaranteed number of superblocks evaluated, `usize`, default 100 |
| `beta` | query pruning factor |
| `mu` | superblock threshold overestimation; the README's k=10 configs use 0.0001 |
| `eta` | block-level probabilistic safeness |
| `alpha` | unused: declared `_alpha` (`src/search.rs:27,918`), so the compiler does not warn |

**`--n-searches` is a CLI flag (default 2); we pass 5**, one warm-up plus four. The driver
reports the two passes after the warm-up and records the rest as `unreported_pass_seconds`.

**Three CLI traps, found by reading `--help` against the source.**

1. `--ciff-file`, `--compress-document-index` and `--compress-range` all declare `-c`. Long
   flags only; a short one is ambiguous.
2. `--compress-document-index` and `--compress-range` are required, with no defaults.
3. Unrecognised values do not error. `compress_range` falls through to `Compression::Raw` and
   `compress_document_index` to `DocumentIndexTypes::BMPInv` via `_ =>` arms
   (`bin/ciff2bmp.rs:41-55`), so a typo silently builds a different index.
   `_check_build_args` rejects unknown values.

**Ids.** `to_trec` prints `doc_lexicon[doc_id]` (`src/util.rs:35`) in the TREC format
`<qid> Q0 <docid> <rank> <score> BMP`, so the run comes back in the collection's positional id
space, the space `groundtruth.tsv` uses, even though the CIFF is reordered by graph bisection.
"""

import json
import os
import re
import sys

import numpy as np

from base import SparseCLI, SPARSE_ROOT

#: LSP's build tree. Nightly Rust is required (`src/lib.rs` enables five features).
LSP_REPO = os.environ.get(
    "LSP_REPO", os.path.join(SPARSE_ROOT, "repos", "hierarchical_pruning"))

#: The prepared CIFF directory, shared by a collection's builds. Required; see the docstring.
SHARED_CIFF_PREP = os.environ.get("SPARSE_CIFF_PREP", "")

#: Passes per invocation: one warm-up plus four; the driver reports the two after the warm-up.
N_SEARCHES = int(os.environ.get("LSP_N_SEARCHES", "5"))

_LATENCY = re.compile(r"^\s*(\d+)\s*$")
_TREC = re.compile(r"^(\S+)\s+Q0\s+(\S+)\s+(\d+)\s+(\S+)\s+\S+\s*$")

#: `bin/ciff2bmp.rs:41-55`. An unknown value silently selects the `_ =>` arm, so the set is closed
#: here instead. Raw/BMPInv are the fall-through defaults and are named explicitly to be selectable.
_COMPRESS_RANGE = {"simdbp", "superblock", "raw"}
_COMPRESS_DOCIDX = {"seismic", "flatinv", "compactinv", "bmpinv"}


class Lsp(SparseCLI):
    """LSP over a graph-bisected, 8-bit-quantized CIFF.

    Build arguments in the order sparse/grids.py declares them, which is the order the
    `result_file` stem carries and therefore the order `dominance.py`'s monotone indices use.
    """

    def __init__(self, superblock, subblock, compress_document_index="seismic",
                 compress_range="simdbp", quant=4):
        _check_build_args(compress_document_index, compress_range, quant)
        self.superblock = superblock
        self.subblock = subblock
        self.compress_document_index = compress_document_index
        self.compress_range = compress_range
        self.quant = int(quant)
        self.name = (f"Lsp(superblock={superblock}, subblock={subblock}, "
                     f"docidx={compress_document_index}, range={compress_range}, "
                     f"quant={self.quant})")
        self.mu = self.eta = self.beta = self.gamma = None

    # -- build ---------------------------------------------------------------------------------

    @staticmethod
    def _prep_dir(index_path=None):
        """The prepared CIFF directory, shared across systems and across a collection's builds."""
        if not SHARED_CIFF_PREP:
            raise SystemExit(
                "SPARSE_CIFF_PREP is unset. It must name the prepared CIFF directory -- the one "
                "holding docs_reordered.ciff, queries.pisa and prep_timings.json. There is no "
                "default on purpose: preparing a second copy would waste an hour and mint a "
                "second `measured_by` for a computation that was done once.")
        return SHARED_CIFF_PREP

    def build_stages(self, documents_path, queries_path, index_path):
        """Only `ciff2bmp`, and it is charged.

        Everything upstream of it -- seismic → JSONL, `jsonl2ciff` with its conversion and its
        8-bit quantization, `create-rgb` graph bisection, the PISA query encoding -- is the shared
        preparation, measured once per collection. Its charged parts come back through
        `shared_preparation`; the uncharged parts are format conversion. Nothing is re-run.

        `documents_path` and `queries_path` are unused: LSP consumes the prepared CIFF, never the
        collection's own `.bin`.
        """
        prep = self._prep_dir()
        ciff = os.path.join(prep, "docs_reordered.ciff")
        if not os.path.exists(ciff):
            raise SystemExit(
                f"missing {ciff}. LSP builds from the graph-bisected, 8-bit-quantized CIFF; "
                "prepare it with replicate-dsp/scripts/prepare_measured.py, which measures each "
                "step and writes the prep_timings.json sidecar shared_preparation reads.")
        return [
            ("ciff2bmp", [os.path.join(LSP_REPO, "target", "release", "ciff2bmp"),
                          "--ciff-file", ciff,
                          "--output", index_path,
                          "--bsize", str(int(self.superblock)),
                          "--bsize", str(int(self.subblock)),
                          "--compress-document-index", self.compress_document_index,
                          "--compress-range", self.compress_range,
                          "--quant", str(self.quant)], True),
        ]

    def shared_preparation(self, index_path):
        """The charged parts of the shared preparation: 8-bit quantization and graph bisection.

        Read from the preparation's `prep_timings.json` sidecar: a real timing of that
        computation on that input, measured once and charged in full to every build that uses
        it. `measured_by` goes onto every row, so the row shows that this build did not pay it
        afresh.
        """
        sidecar = os.path.join(self._prep_dir(), "prep_timings.json")
        if not os.path.exists(sidecar):
            raise SystemExit(
                f"missing {sidecar}. Without it the charged quantization and bisection costs are "
                "unknown, and a build reported without them would understate LSP against every "
                "system that pays its preparation.")
        with open(sidecar) as fh:
            t = json.load(fh)
        if not t["subtraction_evidence"]["valid"]:
            raise SystemExit(
                f"{sidecar} reports the conversion/quantization subtraction as invalid "
                f"(unquantized brackets differ by "
                f"{100 * t['subtraction_evidence']['bracket_spread_frac']:.1f}%). Separate the "
                "step physically before charging anything.")
        by = t["measured_by"]
        return [("ciff-quantization", t["charged"]["ciff_quantization_s"], by),
                ("graph-bisection", t["charged"]["graph_bisection_s"], by)]

    # -- search --------------------------------------------------------------------------------

    def search_argv(self, index_path, k):
        queries_pisa = os.path.join(self._prep_dir(), "queries.pisa")
        if not os.path.exists(queries_pisa):
            raise SystemExit(f"missing {queries_pisa}; it is part of the shared preparation.")
        return [
            os.path.join(LSP_REPO, "target", "release", "search"),
            "--index", index_path,
            "--queries", queries_pisa,
            "--k", str(int(k)),
            "--mu", str(float(self.mu)),
            "--eta", str(float(self.eta)),
            "--beta", str(float(self.beta)),
            "--gamma", str(int(self.gamma)),
            "--n-searches", str(N_SEARCHES),
        ]

    def set_query_arguments(self, mu, eta, beta, gamma):
        self.mu = mu
        self.eta = eta
        self.beta = beta
        self.gamma = gamma

    def pass_boundary(self, line):
        """A bare integer line: LSP's own mean microseconds per query for the pass that ended.

        `src/search.rs:432` prints one per pass inside the `for i_search in 0..n_searches` loop,
        and `bin/search.rs` calls `to_trec` only after `b_search` returns, so every boundary
        arrives before any TREC line. Recorded as the self-report, never used as the measurement.
        """
        m = _LATENCY.match(line)
        return float(m.group(1)) if m else None

    def parse_ids(self, stdout_text, n_queries, k):
        """The TREC run: `<qid> Q0 <docid> <rank> <score> BMP`."""
        ids = np.full((n_queries, k), -1, dtype=np.int64)
        seen = set()
        for line in stdout_text.splitlines():
            m = _TREC.match(line)
            if not m:
                continue
            try:
                q, doc, rank = int(m.group(1)), int(m.group(2)), int(m.group(3))
            except ValueError:
                raise SystemExit(
                    f"LSP returned a non-integer id in {line!r}. The TREC output is expected to "
                    "carry the positional ids convert_seismic_to_jsonl.py wrote; if it does not, "
                    "the id-space gate has failed and no accuracy number here is meaningful.")
            if q < n_queries and 1 <= rank <= k:
                ids[q, rank - 1] = doc
                seen.add(q)
        if not seen:
            return None
        # A query LSP answers with no results is scored as a miss (0/10), not dropped, which would
        # change the query set. Such queries are counted on the row by driver.answer_counts.
        # More than 1 % missing is refused: that looks like a truncated run, not a result.
        self._empty_queries = sorted(set(range(n_queries)) - seen)
        if len(self._empty_queries) > n_queries // 100:
            raise SystemExit(
                f"LSP returned results for {len(seen)} of {n_queries} queries. A partial run "
                "would compute accuracy over a query set that is not the collection's.")
        if self._empty_queries:
            print(f"WARNING: LSP returned no results for {len(self._empty_queries)} queries "
                  f"{self._empty_queries[:20]}; scored as misses.", file=sys.stderr)
        return ids

    def index_bytes(self, index_path):
        return os.path.getsize(index_path) if os.path.exists(index_path) else None

    def get_additional(self):
        return {
            "lsp_compress_document_index": self.compress_document_index,
            "lsp_compress_range": self.compress_range,
            "lsp_quant_upper_bounds_bits": self.quant,
            "lsp_impact_bits": 8,          # the CIFF requirement; LSP is wrong without it
            "lsp_max_term_weight": 32,     # query-side renormalisation
            "lsp_n_searches": N_SEARCHES,
            "lsp_alpha_is_dead": True,     # `_alpha`, src/search.rs:27,918
            "lsp_shared_ciff_prep": SHARED_CIFF_PREP,
        }


def _check_build_args(compress_document_index, compress_range, quant):
    """Reject values `ciff2bmp` would silently reinterpret.

    `bin/ciff2bmp.rs` matches these strings with a `_ =>` fall-through -- unknown `--compress-range`
    becomes `Raw` and unknown `--compress-document-index` becomes `BMPInv`. A typo would therefore
    build a valid index of a configuration nobody selected, with every row mislabelled and no
    error. Fail here instead.
    """
    if compress_range not in _COMPRESS_RANGE:
        raise SystemExit(
            f"compress_range={compress_range!r} is not one of {sorted(_COMPRESS_RANGE)}. "
            "ciff2bmp would silently fall through to Raw and build a different index.")
    if compress_document_index not in _COMPRESS_DOCIDX:
        raise SystemExit(
            f"compress_document_index={compress_document_index!r} is not one of "
            f"{sorted(_COMPRESS_DOCIDX)}. ciff2bmp would silently fall through to BMPInv.")
    if int(quant) not in (2, 4, 8, 16):
        raise SystemExit(f"quant={quant!r} is not a plausible upper-bound width (2/4/8/16).")
