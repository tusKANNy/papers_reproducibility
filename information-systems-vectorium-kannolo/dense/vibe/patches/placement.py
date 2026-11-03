"""Build-time CPU and page placement for the benchmark process (installed into VIBE as vibe/placement.py).

Upstream VIBE runs the whole `run_algorithm.py` process under
`numactl --membind=<node> --physcpubind=<cpu>`. Index construction is therefore single-core, and
the `build_time` it reports is a single-core time. This module lets the *build* use every core,
while the *search* stays pinned to one physical core, as upstream. `build_time` then becomes the
64-thread time, measured on the same build that then serves the search sweep.

Search is unchanged: `pin_for_search()` restores the affinity upstream would have had, and it is
called before the first query.

The page-placement policy is a switch so that its effect on search is measured. Pages go where
they are *allocated*. A build spread over 64 cores under `--interleave=all` therefore leaves the
pinned search core reading half its index across the interconnect. This lasts for the whole
search, and it hurts most the random-access graph traversal we benchmark.

    policy         numactl options                            threads  pages after build
    migrate        --interleave=all, then migrate to node     64       local
    interleave     --interleave=all                           64       interleaved
    membind        --membind=<n>                              64       local (cross-socket build traffic)
    membind-local  --cpunodebind=<n> --membind=<n>            32       local (no interconnect traffic)
    upstream       --membind=<n> --physcpubind=<c>             1       local

All settings come from the environment. The worker that builds the `numactl` command line and
the child process that runs the build therefore read the same values, with no other channel:

    VIBE_PLACEMENT       upstream | membind | membind-local | interleave | migrate   (default upstream)
    VIBE_SEARCH_CPU      the physical core searches run on
    VIBE_SEARCH_NODE     that core's NUMA node, the migration target of the `migrate` policy
    VIBE_BUILD_CPUS      comma/range list of cores the build may use
"""

import ctypes
import os
import shutil
import subprocess
import threading
import time

PLACEMENT = os.environ.get("VIBE_PLACEMENT", "upstream")
UPSTREAM = "upstream"


def _expand_ranges(spec):
    out = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = (int(x) for x in tok.split("-", 1))
            out.update(range(min(a, b), max(a, b) + 1))
        else:
            out.add(int(tok))
    return out


def _search_cpu():
    v = os.environ.get("VIBE_SEARCH_CPU")
    return int(v) if v is not None else None


def _search_node():
    v = os.environ.get("VIBE_SEARCH_NODE")
    return int(v) if v is not None else None


def _build_cpus():
    v = os.environ.get("VIBE_BUILD_CPUS")
    return _expand_ranges(v) if v else None


def build_threads():
    """How many threads a module may use for *construction*.

    Modules call this to size their build. Most of VIBE's modules pass a literal `1` to their
    library's build call. Upstream this is consistent, because the whole process is pinned to one
    core. Under this patch it would leave every competitor building on a single thread while ours
    builds on 64, and the two build times could not be compared.

    Returns 1 under the `upstream` policy, so a module that calls this behaves as before when the
    patch is switched off.
    """
    if PLACEMENT == UPSTREAM:
        return 1
    cpus = _build_cpus()
    return len(cpus) if cpus else 1


def search_threads():
    """Always 1. The name makes the single search thread explicit in each module's search call."""
    return 1


def release_free_heap():
    """Return freed heap to the OS before the footprint is measured. Applied to every library.

    VIBE's cross-library footprint is the USS delta around `fit()`, and USS counts pages the
    process still holds whether or not anything is using them. glibc's malloc keeps a per-thread
    arena and does not return large frees by default. A build that peaks at twice its final size
    and runs on 64 threads therefore leaves roughly the peak resident. Measured on kANNolo on
    AgNews, USS came to 2.05x the index's own `SpaceUsage` after a 64-thread build, against 1.05x
    after a single-threaded one and 1.0015x once the heap was trimmed.

    This is an allocator artifact that grows with build parallelism; it is not a property of any
    index. The harness therefore trims the heap before the measurement, for every module. Trimming
    in one module only would make the footprint figure asymmetric. The trim does not affect
    libraries that use their own allocator, or memory already returned to the OS.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_size_t(0))
        return True
    except (OSError, AttributeError):
        return False


def _set_affinity_all_threads(cpus):
    """Pin every thread of this process, not only the calling one.

    Linux `sched_setaffinity` is per-thread, and by the time the search starts the library
    may already have spawned a build thread pool. Those threads would otherwise keep running on
    other cores. Walking `/proc/self/task` enforces the single-core policy on the whole process.
    """
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except (OSError, ValueError):
            pass


def unpin_for_build():
    """Widen affinity to the build core set. No-op under the `upstream` policy."""
    if PLACEMENT == UPSTREAM:
        return
    cpus = _build_cpus()
    if cpus:
        _set_affinity_all_threads(cpus)


def pin_for_search():
    """Restore the upstream single-core pinning before any query runs."""
    cpu = _search_cpu()
    if cpu is None:
        return
    _set_affinity_all_threads({cpu})


def _anon_pages_by_node(pid="self"):
    """Anonymous pages per NUMA node, from /proc/<pid>/numa_maps."""
    by_node = {}
    try:
        with open(f"/proc/{pid}/numa_maps") as fh:
            for line in fh:
                if "anon=" not in line:
                    continue
                for field in line.split():
                    if field.startswith("N") and "=" in field:
                        node, _, count = field[1:].partition("=")
                        if node.isdigit():
                            by_node[int(node)] = by_node.get(int(node), 0) + int(count)
    except OSError:
        return {}
    return by_node


def _anon_huge_pages_kb(pid="self"):
    """Anonymous memory backed by transparent huge pages, in KB, from /proc/<pid>/smaps_rollup.

    Recorded because `migratepages` splits huge pages as it moves them. A run can move 100% of
    its pages to the target node and still lose TLB coverage, which affects random graph
    traversal. This number explains why a migrated build can be slightly slower than one whose
    pages were allocated locally.
    """
    out = {}
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            for line in fh:
                for key in ("AnonHugePages:", "Anonymous:", "Rss:"):
                    if line.startswith(key):
                        out[key.rstrip(":")] = int(line.split()[1])
    except OSError:
        pass
    return out


def finalize_placement():
    """`migrate` policy: move this process's pages to the search node, then *verify* the move.

    `move_pages` is best effort. It leaves pages behind without an error when they are pinned or
    shared, or when the target node is short on memory. A run whose migration does not verify has
    a different page placement, so the caller must discard it. The measured local fraction is
    therefore returned and recorded.
    """
    node = _search_node()
    report = {
        "placement": PLACEMENT,
        "migrated": False,
        "anon_pages_before": None,
        "anon_pages_after": None,
        "local_fraction": None,
        "migrate_seconds": None,
    }

    before = _anon_pages_by_node()
    report["anon_pages_before"] = before
    report["smaps_before"] = _anon_huge_pages_kb()

    if PLACEMENT != "migrate" or node is None:
        total = sum(before.values())
        report["local_fraction"] = (before.get(node, 0) / total) if total else None
        return report

    if not shutil.which("migratepages"):
        report["error"] = "migratepages not found"
        return report

    other = ",".join(str(n) for n in sorted(before) if n != node) or "all"
    t0 = time.time()
    proc = subprocess.run(
        ["migratepages", str(os.getpid()), other, str(node)],
        capture_output=True,
        text=True,
    )
    report["migrate_seconds"] = time.time() - t0
    if proc.returncode != 0:
        report["error"] = f"migratepages rc={proc.returncode}: {proc.stderr.strip()}"
        return report

    after = _anon_pages_by_node()
    total = sum(after.values())
    report["migrated"] = True
    report["anon_pages_after"] = after
    report["smaps_after"] = _anon_huge_pages_kb()
    report["local_fraction"] = (after.get(node, 0) / total) if total else None
    return report


class PeakRssPoller:
    """Sample resident size during `fit()`.

    VIBE's memory hook brackets the call, so it reports the index that survives and never the
    peak. On DPR one node holds 257 GB and the f32 training array alone is ~64 GB. The gap
    between peak and final size therefore decides whether a configuration fits in memory.
    """

    def __init__(self, interval=0.25):
        self.interval = interval
        self.peak_pages = 0
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        page = os.sysconf("SC_PAGE_SIZE")
        while not self._stop.is_set():
            try:
                with open("/proc/self/statm") as fh:
                    resident = int(fh.read().split()[1])
                self.peak_pages = max(self.peak_pages, resident)
            except (OSError, IndexError, ValueError):
                pass
            self._stop.wait(self.interval)
        self.page_size = page

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.peak_bytes

    @property
    def peak_bytes(self):
        return self.peak_pages * os.sysconf("SC_PAGE_SIZE")
