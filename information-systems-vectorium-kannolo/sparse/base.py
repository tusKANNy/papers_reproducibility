"""The contract every sparse module implements.

The same shape as VIBE's `BaseANN` (`vibe/algorithms/base/module.py`), so the JSONL rows the
sparse driver writes are read unchanged by the same analysis scripts as the dense ones
(`analysis/select_configs.py`, `dense/dominance.py`, `analysis/budget_sensitivity.py`). The
sparse driver writes those rows itself, so VIBE's result-tree reader (`dense/collect_results.py`)
has no counterpart here.

The only difference comes from the data: a sparse query is a pair of arrays, so `query` takes
`(components, values, k)` where VIBE's takes a single vector.

`get_memory_usage` is the same as VIBE's: the reported footprint is the USS delta of the
process around construction, which is defined identically for every library.
"""

import os

#: Where the sparse systems' source trees, builds and caches live (see sparse/env.example.sh).
SPARSE_ROOT = os.environ.get("SPARSE_ROOT", "")  # run_sparse.py --sparse-root sets it

import psutil


class SparseANN:
    """Base class for a sparse index under this harness."""

    #: Set by the driver from the module's constructor arguments; used in the row's `name`.
    name = "unnamed"

    def fit(self, documents_path):
        """Build the index from the collection's documents file.

        Build time is measured around this call, so any preparation the system needs happens
        inside it and is charged to the build.
        """
        raise NotImplementedError

    def set_query_arguments(self, **kwargs):
        """Apply one point of the search grid. Never rebuilds."""
        raise NotImplementedError

    def query(self, components, values, k):
        """Return up to `k` document ids for one sparse query, most similar first."""
        raise NotImplementedError

    def get_memory_usage(self):
        """Resident-and-private memory of this process, in KB.

        Verbatim from VIBE's `BaseANN`. The driver differences it around `fit`.
        """
        return psutil.Process().memory_full_info().uss / 1024

    def get_additional(self):
        """Extra fields to record with every row of this configuration."""
        return {}

    def index_bytes(self):
        """On-disk or in-library serialized size, when the system exposes one; else `None`.

        Recorded beside the USS delta so the two can be compared.
        """
        return None

    def __str__(self):
        return self.name


class SparseCLI:
    """The contract for command-line timing: an index built by a CLI, searched by a CLI.

    Some systems (LSP, SINDI) ship no per-query API, only binaries that take a whole query
    file, and every system is timed this way for the reported numbers. The row schema is the
    same as `SparseANN`'s, but the measurement differs.

    The unit of measurement is one pass over the query file, on our clock. The CLIs make
    several passes per invocation (LSP five, SINDI three), so timing the whole process and
    dividing by the query count would understate QPS by a different factor per system and
    include start-up and index loading. A module therefore declares where its CLI's passes end
    (`pass_boundary`) and the driver times the intervals between them; see
    `driver.measure_cli`.

    A system's self-reported timing is recorded beside our measurement, never used in its
    place: self-reports can exclude part of the per-query work (e.g. query preprocessing).
    """

    #: Set by the driver from the module's constructor arguments; used in the row's `name`.
    name = "unnamed"

    def build_stages(self, documents_path, queries_path, index_path):
        """`[(label, argv, charged), ...]`, run in order; the chain is this system's build.

        `charged` says whether the stage's time counts toward `build_time_s`: format conversion
        is excluded, data preparation is included. A conversion cost depends on the format the
        collection ships in (Seismic's binary format), not on the method, and charging it would
        penalize every system that reads another format. A step that transforms the data to make
        the index what it is (graph bisection, quantization, a kNN graph, an L1 prune) is part
        of the method and is charged.

        Excluded stages still run and are timed and recorded. A stage that is not clearly a
        pure conversion should be charged. For this reason `charged` is a required element, with
        no default of False.

        `index_path` is where the last stage must leave the index. Anything else the system
        needs at search time (a converted query file, an id map) belongs beside it, under
        `os.path.dirname(index_path)`; this module names it and finds it again.

        Unlike in-process tuning, serialization cannot be excluded: a build CLI's only output is
        the serialized index. The row records this.
        """
        raise NotImplementedError

    def shared_preparation(self, index_path):
        """`[(label, seconds, measured_by), ...]`: charged work measured once and reused here.

        For a preparation step shared by all of a collection's builds and too expensive to
        repeat per build: LSP's recursive graph bisection depends only on the CIFF, not on
        `superblock`/`subblock`, so one bisection serves every build. It is measured once and
        charged under three conditions:

        * the value is a real timing of that computation on that input on this machine, never
          scaled or modelled from another collection;
        * it is charged in full to every build that depends on it;
        * `measured_by` records which run produced it, and the driver puts it on the row, so a
          shared measurement is never mistaken for one made during this build.

        Returns an empty list for a system with no shared charged step, which is most of them.
        """
        return []

    def search_argv(self, index_path, k):
        """The search CLI's argv, for the current query arguments. The driver pins it."""
        raise NotImplementedError

    def set_query_arguments(self, **kwargs):
        """Apply one point of the search grid. Never rebuilds."""
        raise NotImplementedError

    def pass_boundary(self, line):
        """`None` if `line` is not the end of a pass over the query file.

        Otherwise the system's own reported microseconds per query for the pass that just
        ended, or `True` where it reports none. Called on every stdout line in order, and the
        driver timestamps the ones that match on its own clock.
        """
        raise NotImplementedError

    def parse_ids(self, stdout_text, n_queries, k):
        """The ids the CLI returned, `(n_queries, k)`, in the collection's own id space.

        `None` where the CLI does not emit them. A system's self-reported recall is never used,
        so the driver quarantines such a row instead of falling back to one.

        A system that reorders document ids (graph bisection does) maps them back here; the
        driver does not.
        """
        raise NotImplementedError

    def index_bytes(self, index_path):
        """On-disk serialized size, when the system exposes one; else `None`."""
        return None

    def get_additional(self):
        """Extra fields to record with every row of this configuration."""
        return {}

    def __str__(self):
        return self.name
