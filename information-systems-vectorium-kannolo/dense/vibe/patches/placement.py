"""Build-time CPU and page placement for the benchmark process (installed into VIBE as vibe/placement.py).

Upstream VIBE runs the whole `run_algorithm.py` process under
`numactl --membind=<node> --physcpubind=<cpu>`, so index construction is single-core and the
`build_time` it reports is a single-core number. This module lets the *build* use every core
while the *search* stays pinned to exactly one physical core, as upstream — so `build_time`
becomes the 64-thread number, from the same build that then serves the search sweep.

Nothing here changes search: `pin_for_search()` restores precisely the affinity upstream
would have had, and it is called before the first query.

The page-placement arm is a switch because where the index pages land is a measured
question, not an assumed one. Pages go where they are *allocated*, so a build spread over
64 cores under `--interleave=all` leaves the pinned search core reading half its index
across the interconnect — permanently, and worst for exactly the random-access graph
traversal we are benchmarking.

    arm  build policy                              threads  pages after build
    D    --interleave=all, then migrate to node    64       local
    C    --interleave=all                          64       interleaved
    B    --membind=<n>                             64       local (cross-socket build traffic)
    B'   --cpunodebind=<n> --membind=<n>           32       local (no interconnect traffic)
    A    upstream: --membind=<n> --physcpubind=<c>  1       local

Configured entirely through the environment, so the worker that builds the `numactl` command
line and the child process that runs the build agree without a side channel:

    VIBE_PLACEMENT_ARM   A | B | Bprime | C | D          (default A = upstream)
    VIBE_SEARCH_CPU      the physical core searches run on
    VIBE_SEARCH_NODE     that core's NUMA node -- the migration target for arm D
    VIBE_BUILD_CPUS      comma/range list of cores the build may use
"""

import ctypes
import os
import shutil
import subprocess
import threading
import time

ARM = os.environ.get("VIBE_PLACEMENT_ARM", "A")
UPSTREAM_ARM = "A"


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

    Modules ask for this instead of hardcoding a number. Most of VIBE's modules pass a literal
    `1` to their library's build call — which is consistent upstream, where the whole process is
    pinned to one core anyway, but under this patch it would leave every competitor building on
    a single thread while ours builds on 64. A build-time comparison between those is not a
    comparison.

    Returns 1 on the upstream arm, so a module that calls this behaves exactly as before when
    the patch is switched off.
    """
    if ARM == UPSTREAM_ARM:
        return 1
    cpus = _build_cpus()
    return len(cpus) if cpus else 1


def search_threads():
    """Always 1. Named so a module's search call reads as a deliberate choice, not an oversight."""
    return 1


def release_free_heap():
    """Return freed heap to the OS before the footprint is measured. Applied to every library.

    VIBE's cross-library footprint is the USS delta around `fit()`, and USS counts pages the
    process still holds whether or not anything is using them. glibc's malloc keeps a per-thread
    arena and does not return large frees by default, so a build that peaks at twice its final
    size and runs on 64 threads leaves roughly the peak resident: measured on kANNolo at D3, USS
    came to 2.05x the index's own `SpaceUsage` after a 64-thread build, against 1.05x after a
    single-threaded one and 1.0015x once the heap was trimmed.

    That is an allocator artifact scaling with build parallelism, not a property of any index, so
    it is removed here -- in the harness, before the measurement, for every module -- rather than
    in one module, which would make the footprint figure asymmetric. Libraries that use their own
    allocator are unaffected either way, and so is anything already returned to the OS.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(ctypes.c_size_t(0))
        return True
    except (OSError, AttributeError):
        return False


def _set_affinity_all_threads(cpus):
    """Pin every thread of this process, not only the calling one.

    Linux `sched_setaffinity` is per-thread, and by the time the search starts the library
    may already have spawned a build thread pool. Leaving those free to roam would let them
    keep running on other cores; walking `/proc/self/task` is what actually enforces the
    single-core policy on the whole process.
    """
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), cpus)
        except (OSError, ValueError):
            pass


def unpin_for_build():
    """Widen affinity to the build core set. No-op on the upstream arm."""
    if ARM == UPSTREAM_ARM:
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

    Recorded because `migratepages` splits huge pages as it moves them: a run can land 100% of
    its pages on the target node and still lose TLB coverage, which random graph traversal feels.
    Without this number a migrated arm that is slightly slower than a natively-local one has no
    explanation, only a discrepancy.
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
    """Arm D: migrate this process's pages to the search node, then *verify* it happened.

    `move_pages` is best effort -- it silently leaves pages behind when they are pinned,
    shared, or the target is short on memory. A run whose migration does not verify is not a
    slower run, it is a differently-placed run, so the caller must discard it rather than
    report it. Hence the measured fraction is returned and recorded, never assumed.
    """
    node = _search_node()
    report = {
        "placement_arm": ARM,
        "migrated": False,
        "anon_pages_before": None,
        "anon_pages_after": None,
        "local_fraction": None,
        "migrate_seconds": None,
    }

    before = _anon_pages_by_node()
    report["anon_pages_before"] = before
    report["smaps_before"] = _anon_huge_pages_kb()

    if ARM != "D" or node is None:
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

    VIBE's memory hook brackets the call, so it reports the index that survives, never the
    peak. At D2 one node holds 257 GB and the f32 train array alone is ~64 GB, so the
    difference decides whether a configuration is droppable-by-OOM or fine.
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
