"""Measurement driver for the sparse benchmarks: one system, one collection, one grid.

Two ways of timing a search, never mixed in one row:

  In-process tuning (`--timing in-process`): one query at a time on one pinned core, `--runs 3`
      with the first pass a warm-up. `--module` is a `SparseANN` (sparse/base.py). Used to tune
      the systems that have a per-query Python API (kANNolo, PyANN).
  Command-line timing (`--timing cli`): a search CLI over the whole query file on one pinned
      core, timed between the CLI's own pass markers. The best of the two passes after the first
      is reported. `--module` is a `SparseCLI`. Every reported sparse number comes from
      command-line timing.

Construction runs on all cores with memory interleaved across nodes. The index is then migrated
to the search node, and the run is discarded if fewer than 99 % of its pages are local
(placement policy `migrate`, dense/vibe/patches/placement.py). A build over `--budget-s` is
recorded but not searched, and builds that dominate it are skipped (see `MONOTONE`). The
pipeline that calls this driver for every system is sparse/run_sparse.py; the grids are in
sparse/grids.py.
"""

import argparse
import gc
import importlib
import json
import os
import resource
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(BENCH_ROOT, "dense", "vibe", "patches"))

import placement  # noqa: E402  shared with the dense harness
import sparse_data  # noqa: E402


def accuracy_at_k(returned_ids, groundtruth, k):
    """Accuracy@k in percent: |R n G| / k, averaged over queries.

    Computed here for every system alike from the ids the system returned; no system's
    self-reported recall is used. The ground truth is the file shipped with the collection.

    A ground-truth entry of -1 marks a query whose shipped row is shorter than k; it matches
    nothing, so such a query can only lower the score, never inflate it.
    """
    hits = 0
    for returned, truth in zip(returned_ids, groundtruth):
        truth_set = {int(t) for t in truth if t >= 0}
        hits += sum(1 for r in returned[:k] if int(r) in truth_set)
    return 100.0 * hits / (len(groundtruth) * k)


def answer_counts(returned_ids, k):
    """Robustness, counted for every system alike: how many queries got no answer at all, and
    how many got fewer than k. A query with no results scores 0 in `accuracy_at_k` and is also
    reported separately, since a deployed system would return nothing for it. Padding (-1 /
    negative ids) counts as no result."""
    counts = [sum(1 for r in list(returned)[:k] if r is not None and int(r) >= 0)
              for returned in returned_ids]
    empty = [q for q, c in enumerate(counts) if c == 0]
    return {
        "n_empty_queries": len(empty),
        "n_short_queries": sum(1 for c in counts if c < k),
        "empty_query_ids": empty[:50],
    }


def machine_state():
    """Host, timestamp, load, other logged-in users and kernel release at this instant.

    Recorded on every row as provenance. The kernel release matters because a kernel change
    alters the scheduler, allocator and mitigations, and nothing else on the row would show it.
    """
    state = {"host": socket.gethostname(), "timestamp": datetime.now(timezone.utc).isoformat()}
    try:
        state["kernel"] = os.uname().release
    except OSError:
        state["kernel"] = None
    try:
        with open("/proc/loadavg") as fh:
            l1, l5, l15 = fh.read().split()[:3]
        state["loadavg"] = [float(l1), float(l5), float(l15)]
    except OSError:
        state["loadavg"] = None
    try:
        who = subprocess.run(["who"], capture_output=True, text=True, timeout=10).stdout
        me = os.environ.get("USER", "")
        state["other_users"] = sorted({ln.split()[0] for ln in who.splitlines() if ln.split() and ln.split()[0] != me})
    except (OSError, subprocess.SubprocessError):
        state["other_users"] = None
    return state


def time_one_pass(index, queries, k):
    """One timed pass: every query, one at a time, in order.

    Returns `(elapsed_seconds, returned_ids)`. `perf_counter` brackets only the search calls.
    Building the result list is inside the bracket: it is part of every library's call path, and
    excluding it for some systems and not others would not be symmetric.
    """
    n = len(queries)
    results = [None] * n
    start = time.perf_counter()
    for i in range(n):
        components, values = queries.get(i)
        results[i] = index.query(components, values, k)
    elapsed = time.perf_counter() - start
    return elapsed, results


def measure_in_process(index, queries, groundtruth, k, runs):
    """`runs` passes; drop the first as warm-up when there are three or more.

    Reports the maximum QPS over the timed passes and computes the spread over those only:
    under `--runs 3` pass 1 is the cold warm-up, and including it would report warm-up cost as
    spread.
    """
    pass_qps = []
    pass_seconds = []
    last_results = None
    for _ in range(runs):
        elapsed, results = time_one_pass(index, queries, k)
        pass_seconds.append(elapsed)
        pass_qps.append(len(queries) / elapsed)
        last_results = results

    timed = pass_qps[1:] if runs >= 3 else pass_qps[:]
    n_timed = len(timed)
    best = max(timed)
    spread = 100.0 * (max(timed) - min(timed)) / max(timed) if n_timed > 1 else 0.0

    return {
        "qps": best,
        "mean_us": 1e6 / best,
        "pass_qps": pass_qps,
        "pass_seconds": pass_seconds,
        "pass_spread_pct": spread,
        "n_timed_passes": n_timed,
        "run_count": runs,
        "accuracy_at_k": accuracy_at_k(last_results, groundtruth, k),
        **answer_counts(last_results, k),
    }


#: Timed passes per reported row under command-line timing: the first pass is a warm-up, the
#: next two are timed.
#:
#: The count is fixed so every system gets the same statistic. Some CLIs run more passes than
#: others (SINDI runs three in total), and a best-of-four for one system against a best-of-two
#: for another would favour the first. Passes beyond the second are still recorded on the row
#: under `unreported_pass_seconds`; `qps` and `pass_spread_pct` are computed from the two only.
CLI_TIMED_PASSES = 2


def _mapped_pages_by_node(pid):
    """Every mapped page of `pid` per NUMA node, both anonymous and file-backed.

    `placement._anon_pages_by_node` counts only `numa_maps` lines carrying `anon=`. That is the
    right count for an index built on this process's heap during in-process tuning. Under
    command-line timing, the search CLI may instead `mmap` an index file written by another
    process. Those pages are file-backed and missing from the anonymous count, so an index read
    entirely from a remote node would still show a perfect `local_fraction`. Counting every
    `N<node>=` field on every line avoids that.
    """
    by_node = {}
    try:
        with open(f"/proc/{pid}/numa_maps") as fh:
            for line in fh:
                for field in line.split():
                    if field.startswith("N") and "=" in field:
                        node, _, count = field[1:].partition("=")
                        if node.isdigit():
                            by_node[int(node)] = by_node.get(int(node), 0) + int(count)
    except OSError:
        return {}
    return by_node


def _smaps_rollup_kb(pid):
    """`(uss_kb, rss_kb)` for `pid`, or `(None, None)` if it exited first.

    USS here is the same quantity that `psutil`'s `memory_full_info().uss` reports and that
    `SparseANN.get_memory_usage` returns (the sum of the private mappings). It is read from
    `smaps_rollup` because the measured process is a child of this one.
    """
    private, rss = 0, None
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                if key in ("Private_Clean", "Private_Dirty", "Private_Hugetlb"):
                    private += int(rest.split()[0])
                elif key == "Rss":
                    rss = int(rest.split()[0])
    except (OSError, IndexError, ValueError):
        return None, None
    return private, rss


def search_child_state(pid, node):
    """`(build_placement, uss_kb, rss_kb)` for the search child, sampled while it runs.

    During in-process tuning, this process builds the index on all cores under
    `--interleave=all` and then migrates it to the search node (placement policy `migrate`).
    Under command-line timing, one process builds the index and another reads it. The search
    child therefore allocates node-local from the start under
    `numactl --membind=<node> --physcpubind=<cpu>` (the `upstream` policy) and needs no
    migration. The policy is recorded as `upstream-child` to distinguish it from an in-process
    run under `upstream`. `analysis/select_configs.py` accepts it on its measured
    `local_fraction`, like any other row.

    `local_fraction` is over every mapped page, including the CLI's binary, shared libraries
    and stack, which `--membind` cannot move. For a tiny index these dominate and lower the
    fraction; for a real index of hundreds of MB they are negligible, so a fraction below the
    floor means the index itself is not local. `anon_pages_after` holds the per-node counts.
    """
    pages = _mapped_pages_by_node(pid)
    total = sum(pages.values())
    uss_kb, rss_kb = _smaps_rollup_kb(pid)
    report = {
        "placement": "upstream-child",
        "policy": "numactl --membind=%s --physcpubind=<search cpu> on the search child" % node,
        "migrated": False,
        "anon_pages_before": None,
        "anon_pages_after": pages,
        "local_fraction": (pages.get(node, 0) / total) if total else None,
        "migrate_seconds": None,
        "pages_counted": "all mapped pages, anonymous and file-backed",
        "measured_on": "search child",
    }
    return report, uss_kb, rss_kb


def _thread_siblings(cpu):
    """The logical cpus sharing `cpu`'s physical core, as listed in sysfs."""
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as fh:
            spec = fh.read().strip()
    except OSError:
        return {cpu}
    siblings = set()
    for tok in spec.split(","):
        if "-" in tok:
            a, b = (int(x) for x in tok.split("-", 1))
            siblings.update(range(a, b + 1))
        elif tok:
            siblings.add(int(tok))
    return siblings or {cpu}


def pin_driver_off_search_core(cpu):
    """Keep this process off the core the child is measured on. Returns the cpu set applied.

    Under command-line timing, the driver and the search are two processes. If the driver stayed
    on the search core, every pass boundary would add a wakeup and a context switch on the core
    being measured.
    The driver takes every build core except the search core and its SMT siblings (read from
    sysfs).

    This must run after the child is spawned: `numactl --physcpubind` is checked against the
    caller's affinity mask, so a driver already moved off the search cpu makes `numactl` refuse
    with "cpu argument 0 is out of range" and the child never starts.
    """
    excluded = _thread_siblings(cpu)
    available = placement._build_cpus() or set(range(os.cpu_count() or 1))
    keep = set(available) - excluded
    if not keep:  # a one-core build set: there is no other core to move to, so report None
        return None
    try:
        os.sched_setaffinity(0, keep)
    except OSError:
        return None
    return sorted(keep)


def _child_cpus_allowed(pid):
    """The child's own affinity, read back from `/proc/<pid>/status` to verify the pinning."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def measure_cli(cli, index_path, ctx, stderr_log_path):
    """One search invocation: pin it, stream it, and time its passes on our clock.

    Returns the measurement fields for the row. The CLI is invoked once: the repeated passes
    are the CLI's own, and each pass is timed on our clock as the interval between consecutive
    pass markers on its stdout. This excludes process start-up and index loading, which precede
    the first marker.
    """
    node, cpu = placement._search_node(), placement._search_cpu()
    if node is None or cpu is None:
        raise SystemExit(
            "Command-line timing needs VIBE_SEARCH_CPU and VIBE_SEARCH_NODE set. Otherwise the "
            "search child is not pinned and the row would claim a placement it never had. Launch through "
            "sparse/run_sparse.py, which sets both for every system alike."
        )

    argv = list(cli.search_argv(index_path, ctx["k"]))
    pinned = ["numactl", f"--membind={node}", f"--physcpubind={cpu}", *argv]

    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["RAYON_NUM_THREADS"] = "1"

    boundaries = []      # (seconds since exec, the system's own us/query for the pass, or None)
    out_chunks = []
    child_placement, uss_kb, rss_kb = None, None, None
    child_cpus = None

    with open(stderr_log_path, "wb") as errlog:
        placement.unpin_for_build()
        t0 = time.perf_counter()
        proc = subprocess.Popen(pinned, stdout=subprocess.PIPE, stderr=errlog, env=env)
        driver_cpus = pin_driver_off_search_core(cpu)
        for raw in iter(proc.stdout.readline, b""):
            now = time.perf_counter()
            out_chunks.append(raw)
            marker = cli.pass_boundary(raw.decode("utf-8", "replace"))
            if marker is None:
                continue
            boundaries.append((now - t0, None if marker is True else marker))
            if child_placement is None:
                child_placement, uss_kb, rss_kb = search_child_state(proc.pid, node)
                child_cpus = _child_cpus_allowed(proc.pid)
        proc.stdout.close()
        rc = proc.wait()
        wall = time.perf_counter() - t0

    stdout_text = b"".join(out_chunks).decode("utf-8", "replace")
    measured = {
        "cli_method": "pass boundaries on our clock",
        "search_argv": pinned,
        "search_rc": rc,
        "search_wall_s": wall,
        "search_stderr_log": stderr_log_path,
        "n_passes_observed": len(boundaries),
        "self_reported_us_per_query": [b[1] for b in boundaries],
        "search_child_cpus_allowed": child_cpus,
        "driver_cpus": driver_cpus,
        "build_placement": child_placement,
        "uss_kb_search_child": uss_kb,
        "rss_kb_search_child": rss_kb,
        "uss_delta_kb": None,
        "uss_delta_note": "Command-line timing builds and searches in different processes, so there is no "
                          "in-process construction to bracket; uss_kb_search_child is the search "
                          "process's own private footprint, sampled after its first pass.",
    }

    if rc != 0:
        return {**measured, "quarantined": f"search CLI exited rc={rc}; see {stderr_log_path}"}

    needed = CLI_TIMED_PASSES + 1
    if len(boundaries) < needed:
        return {
            **measured,
            "quarantined": (
                f"{len(boundaries)} pass boundaries observed, {needed} needed "
                f"(1 warm-up + {CLI_TIMED_PASSES} timed). This runner times the intervals "
                "between a CLI's own pass boundaries and has no fallback: see the command-line timing "
                "section of this module's docstring."
            ),
        }

    offsets = [b[0] for b in boundaries]
    durations = [offsets[i] - offsets[i - 1] for i in range(1, len(offsets))]
    timed = durations[:CLI_TIMED_PASSES]
    unreported = durations[CLI_TIMED_PASSES:]

    n = ctx["n_queries"]
    timed_qps = [n / d for d in timed]
    best = max(timed_qps)
    spread = 100.0 * (max(timed_qps) - min(timed_qps)) / max(timed_qps) if len(timed_qps) > 1 else 0.0

    ids = cli.parse_ids(stdout_text, n, ctx["k"])
    measured.update({
        "qps": best,
        "mean_us": 1e6 / best,
        "pass_qps": [None] + timed_qps,
        "pass_seconds": [None] + timed,
        "load_and_warmup_s": offsets[0],
        "unreported_pass_seconds": unreported,
        "pass_spread_pct": spread,
        "n_timed_passes": len(timed),
        "run_count": len(boundaries),
        "accuracy_at_k": None if ids is None else accuracy_at_k(ids, ctx["groundtruth"], ctx["k"]),
        **({} if ids is None else answer_counts(ids, ctx["k"])),
    })
    if ids is None:
        measured["quarantined"] = (
            "the search CLI returned no document ids, so accuracy cannot be computed; a system's "
            "self-reported recall is never used."
        )
    return measured


def index_path_for(directory, result_file):
    """Where a build configuration's index lives. One path per `result_file`, so save and reuse
    agree without a second naming convention."""
    return os.path.join(directory, result_file.replace("/", "__").replace(".jsonl", ".idx"))


def save_built_index(index, save_dir, result_file):
    """Save a built index, so that each configuration is built only once.

    Command-line timing re-times the configurations selected during in-process tuning from the
    saved index files, without rebuilding them. Serialization is outside the build-time bracket:
    under in-process tuning, build time is wall-clock time excluding serialization.
    """
    if not save_dir:
        return None
    path = index_path_for(save_dir, result_file)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        index.save(path)
        return path
    except Exception as exc:  # noqa: BLE001 -- a failed save must not discard a finished build
        print(f"WARNING: could not save index: {type(exc).__name__}: {exc}", file=sys.stderr)
        return None


def build_index(index, documents_path, load_from=None):
    """Build (or load) under the `migrate` placement policy, bracketing the USS and sampling
    peak RSS.

    `load_from` reads a previously saved index instead of constructing one. Placement still
    happens: a loaded index must be migrated and verified like a built one, otherwise the search
    would be measured somewhere other than where the row claims. `build_time_s` is `None`,
    because the build time of a loaded index belongs to the run that built it. A loaded row is
    therefore usable for search measurements and useless for build ones, and its own fields say
    so.

    The order matters and matches the dense benchmarks:

      1. widen affinity to the build cores (the `migrate` policy builds on all cores);
      2. record USS before;
      3. build, with the RSS poller running;
      4. trim the heap, so the footprint excludes the allocator's arenas;
      5. record USS after;
      6. migrate pages to the search node and *verify* the migration;
      7. re-pin to the single search core, before any query runs.

    A run whose migration does not verify is invalid: the caller discards it.
    """
    placement.unpin_for_build()
    gc.collect()
    uss_before = index.get_memory_usage()

    poller = placement.PeakRssPoller().start()
    start = time.perf_counter()
    if load_from:
        index.load(load_from)
        build_seconds = None
    else:
        index.fit(documents_path)
        build_seconds = time.perf_counter() - start
    load_seconds = (time.perf_counter() - start) if load_from else None
    peak_rss = poller.stop()

    trimmed = placement.release_free_heap()
    gc.collect()
    uss_after = index.get_memory_usage()

    report = placement.finalize_placement()
    report["heap_trimmed"] = trimmed
    report["peak_rss_bytes"] = peak_rss

    placement.pin_for_search()

    return {
        "build_time_s": build_seconds,
        "loaded_from": load_from,
        "load_time_s": load_seconds,
        "uss_delta_kb": uss_after - uss_before,
        "build_placement": report,
    }


def migration_verified(report, floor=0.99):
    """Whether a run's pages are on the node that its placement policy requires.

    `migrate` is the only policy that moves pages. Under any other policy there is nothing to
    verify, and the run is kept; its `local_fraction` is recorded in both cases.
    """
    if report.get("placement") != "migrate":
        return True
    if not report.get("migrated"):
        return False
    fraction = report.get("local_fraction")
    return fraction is not None and fraction >= floor


def result_file_key(dataset, k, algo, build_kwargs):
    """A stable identifier for one *built index*, in the shape the reused scripts expect.

    `analysis/select_configs.py` uses `result_file` as its build key (one built index per value),
    and `dense/dominance.py --replay` parses the filename stem back into a tuple of build
    arguments, splitting on underscores and reading each token as int, then float, then string.
    VIBE gets this from its one-hdf5-per-build layout; here the same key is synthesized in the
    same shape.

    The argument order is the grid's own declaration order. This keeps the parsed tuples
    comparable for monotonicity: a cost-monotone parameter must be in the same position on every
    row of a system.
    """
    def token(v):
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        text = str(v)
        if "." in text:
            if not isinstance(v, float):
                raise ValueError(
                    f"build argument {v!r} would put a '.' in the result_file stem and is not a "
                    "float, so dominance.py --replay would parse it as a string and order it "
                    "lexicographically. Use an int or a float."
                )
        return text

    stem = "_".join(token(v) for v in build_kwargs.values()) or "default"
    return f"results/{dataset}/{k}/{algo}/{stem}.jsonl"


def load_module(spec):
    """`package.module:ClassName` -> the class."""
    module_name, _, class_name = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def run_configuration(cls, build_kwargs, query_arg_grid, ctx):
    """Build once, sweep the query grid over that build, and yield one row per point."""
    index = cls(**build_kwargs)
    result_file = result_file_key(
        ctx["row_base"]["dataset"], ctx["k"], ctx["algo"], build_kwargs
    )
    reuse_dir = ctx.get("reuse_indexes")
    load_from = None
    if reuse_dir:
        candidate = index_path_for(reuse_dir, result_file)
        if os.path.exists(candidate):
            load_from = candidate
            print(f"reusing index {candidate}", flush=True)

    build_row = build_index(index, ctx["documents_path"], load_from=load_from)
    saved_to = load_from or save_built_index(index, ctx.get("save_indexes"), result_file)

    report = build_row["build_placement"]
    if not migration_verified(report):
        yield {
            **ctx["row_base"],
            "algo": ctx["algo"],
            "name": str(index),
            "build_args": build_kwargs,
            "result_file": result_file,
            "quarantined": "migration did not verify",
            **build_row,
        }
        return

    if (ctx["budget_s"] is not None
            and build_row["build_time_s"] is not None
            and build_row["build_time_s"] > ctx["budget_s"]):
        yield {**ctx["row_base"], "algo": ctx["algo"], "name": str(index), "build_args": build_kwargs,
               "result_file": result_file, "over_budget": True, "index_path": saved_to, **build_row}
        return
        over_budget = True
    else:
        over_budget = False

    try:
        index_bytes = index.index_bytes()
        index_bytes_error = None
    except Exception as exc:  # noqa: BLE001 -- any library-side failure, none of them fatal here
        index_bytes = None
        index_bytes_error = f"{type(exc).__name__}: {exc}"
        print(f"WARNING: index_bytes() unavailable: {index_bytes_error}", file=sys.stderr)

    for query_args in query_arg_grid:
        index.set_query_arguments(**query_args)
        measured = measure_in_process(index, ctx["queries"], ctx["groundtruth"], ctx["k"], ctx["runs"])
        row = {
            **ctx["row_base"],
            "algo": ctx["algo"],
            "name": str(index),
            "build_args": build_kwargs,
            "result_file": result_file,
            "query_args": json.dumps(query_args, sort_keys=True),
            "query_args_dict": query_args,
            "over_budget": over_budget,
            "index_bytes": index_bytes,
            "index_bytes_error": index_bytes_error,
            "index_path": saved_to,
            **build_row,
            **measured,
            **index.get_additional(),
            "machine": machine_state(),
        }
        yield row

    del index
    gc.collect()


def build_index_cli(cli, documents_path, queries_path, index_path):
    """Run the build chain of a command-line module, timed as one number.

    Build time is measured from the collection's documents to a searchable index, for every
    system alike. Each stage the module marks as charged counts toward `build_time_s`; stages
    marked uncharged (pure format conversions) are timed and recorded but not charged.
    Preparation measured once and shared by a collection's builds (`shared_preparation`, e.g.
    LSP's CIFF quantization and graph bisection) is charged in full to every build.

    Here serialization is inside the bracket; under in-process tuning it is outside. A build
    CLI's only product is the index file, so the index never exists unserialized. The row
    records this in `build_includes_serialization`.

    The build children inherit this process's `numactl --interleave=all` and thread settings
    (set by sparse/run_sparse.py), as an in-process build does. Where those pages land does not
    matter: the search runs in a different process that allocates its own pages node-local.
    """
    placement.unpin_for_build()
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    rss_children_before_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss

    if os.path.exists(index_path):
        os.remove(index_path)

    stages, t0 = [], time.perf_counter()
    charged_seconds = 0.0
    for label, argv, charged in cli.build_stages(documents_path, queries_path, index_path):
        log_path = f"{index_path}.{label}.buildlog"
        s0 = time.perf_counter()
        with open(log_path, "wb") as log:
            rc = subprocess.call(list(argv), stdout=log, stderr=subprocess.STDOUT)
        seconds = time.perf_counter() - s0
        stages.append({"stage": label, "seconds": seconds, "rc": rc, "charged": bool(charged),
                       "argv": list(argv), "log": log_path})
        if charged:
            charged_seconds += seconds
        print(f"  build stage {label}: {seconds:.1f}s rc={rc} "
              f"{'charged' if charged else 'NOT charged (format conversion)'}", flush=True)
        if rc != 0:
            return {"build_time_s": None, "build_stages": stages,
                    "build_error": f"stage {label} exited rc={rc}; see {log_path}"}
    wall_seconds = time.perf_counter() - t0

    shared = list(cli.shared_preparation(index_path))
    for label, seconds, measured_by in shared:
        charged_seconds += seconds
        print(f"  shared preparation {label}: {seconds:.1f}s charged (measured by {measured_by})",
              flush=True)

    return {
        "build_time_s": charged_seconds,
        "build_wall_s": wall_seconds,
        "build_conversion_s": sum(s["seconds"] for s in stages if not s["charged"]),
        "build_shared_preparation": [
            {"stage": l, "seconds": sec, "measured_by": by} for l, sec, by in shared],
        "build_stages": stages,
        "build_error": None,
        "loaded_from": None,
        "load_time_s": None,
        "build_includes_serialization": True,
        "build_threads_env": {k: os.environ.get(k) for k in
                              ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MALLOC_ARENA_MAX")},
        "peak_rss_children_before_kb": rss_children_before_kb,
        "peak_rss_children_after_kb": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }


def run_configuration_cli(cls, build_kwargs, query_arg_grid, ctx):
    """Build once through the CLI chain, then time the search CLI at every query point."""
    cli = cls(**build_kwargs)
    result_file = result_file_key(
        ctx["row_base"]["dataset"], ctx["k"], ctx["algo"], build_kwargs
    )

    reuse_dir, save_dir = ctx.get("reuse_indexes"), ctx.get("save_indexes")
    reuse_candidate = index_path_for(reuse_dir, result_file) if reuse_dir else None
    if reuse_candidate and os.path.exists(reuse_candidate):
        index_path = reuse_candidate
        print(f"reusing index {index_path}", flush=True)
        build_row = {"build_time_s": None, "build_stages": None, "build_error": None,
                     "loaded_from": index_path, "load_time_s": None}
    elif save_dir:
        index_path = index_path_for(save_dir, result_file)
        build_row = build_index_cli(cli, ctx["documents_path"], ctx["queries_path"], index_path)
    else:
        raise SystemExit("Command-line timing needs --save-indexes DIR (where the build CLI writes its index) "
                         "or --reuse-indexes DIR holding one already built.")

    if build_row.get("build_error"):
        yield {**ctx["row_base"], "algo": ctx["algo"], "name": str(cli),
               "build_args": build_kwargs, "result_file": result_file,
               "quarantined": build_row["build_error"], **build_row}
        return

    over_budget = (ctx["budget_s"] is not None
                   and build_row["build_time_s"] is not None
                   and build_row["build_time_s"] > ctx["budget_s"])

    try:
        index_bytes, index_bytes_error = cli.index_bytes(index_path), None
    except Exception as exc:  # noqa: BLE001 -- a missing size field must not discard a finished build
        index_bytes, index_bytes_error = None, f"{type(exc).__name__}: {exc}"
        print(f"WARNING: index_bytes() unavailable: {index_bytes_error}", file=sys.stderr)

    for i, query_args in enumerate(query_arg_grid):
        cli.set_query_arguments(**query_args)
        measured = measure_cli(cli, index_path, ctx, f"{index_path}.q{i:04d}.searchlog")
        placement_report = measured.get("build_placement") or {}
        frac = placement_report.get("local_fraction")
        if frac is not None and frac < 0.99:
            print(f"WARNING: search child local_fraction={frac:.4f} < 0.99: the index is not "
                  f"node-local to the search core. Row kept and flagged.", file=sys.stderr)
        row = {
            **ctx["row_base"],
            "algo": ctx["algo"],
            "name": str(cli),
            "build_args": build_kwargs,
            "result_file": result_file,
            "query_args": json.dumps(query_args, sort_keys=True),
            "query_args_dict": query_args,
            "over_budget": over_budget,
            "index_bytes": index_bytes,
            "index_bytes_error": index_bytes_error,
            "index_path": index_path,
            **build_row,
            **measured,
            **cli.get_additional(),
            "machine": machine_state(),
        }
        yield row


#: Build parameters along which build time grows, per system. A build over the budget is not
#: searched, and every build at least as large in all of them (and equal in the rest) is skipped.
MONOTONE = {
    "kannolo-hnsw": ("m", "ef_construction"),
    "kannolo-hnsw-rerank": ("m", "ef_construction", "prune_percentage"),
    "pyanns": ("r", "l"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--module", required=True, help='"package.module:ClassName" of a SparseANN')
    ap.add_argument("--algo", required=True, help="system name recorded in the row's `algo`")
    ap.add_argument("--dataset", required=True, help="collection id recorded in the row")
    ap.add_argument("--documents", required=True, help="path to the collection's documents.bin")
    ap.add_argument("--queries", required=True, help="path to the collection's queries.bin")
    ap.add_argument("--groundtruth", required=True, help="path to the shipped groundtruth.tsv")
    ap.add_argument("--grid", required=True, help='"package.module:function" returning the grid')
    ap.add_argument("--grid-args", default=None, metavar="JSON",
                    help='keyword arguments for the grid function, as a JSON object. Used for '
                         'values fixed by an earlier selection step (the L1 pruning factor, the '
                         'first-stage representation). They are not grid axes: the selected value '
                         'is written at the call site and recorded in every row\'s build_args.')
    ap.add_argument("--out", required=True, help="JSONL file to append rows to")
    ap.add_argument("-k", type=int, default=10)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--budget-s", type=float, default=None, help="build-time budget B for this collection (s)")
    ap.add_argument("--limit-queries", type=int, default=None,
                    help="testing only: a reported row never subsamples queries.")
    ap.add_argument("--timing", default="in-process", choices=["in-process", "cli"],
                    help="in-process: one query at a time; --module is a SparseANN. "
                         "cli: a search CLI over the whole query file, with each of its own passes "
                         "timed on our clock; --module is a SparseCLI. The two measure different "
                         "quantities, and no ratio is ever taken across them.")
    ap.add_argument("--reuse-indexes", default=None, metavar="DIR",
                    help="load a build's index from DIR when one is already there, instead of "
                         "rebuilding. Such a row carries build_time_s = null and loaded_from, so "
                         "it is usable for search measurements and never for build ones.")
    ap.add_argument("--save-indexes", default=None, metavar="DIR",
                    help="write each built index under DIR. Needed for command-line timing, which re-times "
                         "selected configurations without rebuilding, and for grid calibration, "
                         "which sweeps one build repeatedly.")
    args = ap.parse_args(argv)

    if args.timing == "cli":
        if args.runs != ap.get_default("runs"):
            print(f"WARNING: --runs {args.runs} is ignored under command-line timing: the search "
                  f"CLI makes its own passes. {CLI_TIMED_PASSES} of them are reported per row.",
                  file=sys.stderr)
        print(f"# command-line timing: search child pinned to cpu {placement._search_cpu()} / node "
              f"{placement._search_node()}, {CLI_TIMED_PASSES} timed passes reported per row",
              file=sys.stderr)
    elif args.runs not in (2, 3):
        print(
            f"WARNING: --runs {args.runs} gives {max(args.runs - 1, args.runs) if args.runs < 3 else args.runs - 1} "
            "timed passes. Every reported row has exactly two: --runs 3, with the first "
            "discarded as warm-up. Rows from this run are not comparable to the rest on spread.",
            file=sys.stderr,
        )

    if args.limit_queries is not None:
        print(
            "WARNING: --limit-queries subsamples the query set; such rows are tagged and never "
            "reported.",
            file=sys.stderr,
        )

    queries = sparse_data.read_seismic(args.queries, limit=args.limit_queries)
    groundtruth = sparse_data.read_groundtruth(args.groundtruth, args.k, n_queries=len(queries))

    grid_fn = load_module(args.grid)
    grid_kwargs = json.loads(args.grid_args) if args.grid_args else {}
    cls = load_module(args.module)

    row_base = {
        "dataset": args.dataset,
        "k": args.k,
        "timing": args.timing,
        "n_queries": len(queries),
        "subsampled_queries": args.limit_queries is not None,
        "budget_s": args.budget_s,
    }
    ctx = {
        "documents_path": args.documents,
        "queries_path": args.queries,
        "queries": queries,
        "n_queries": len(queries),
        "groundtruth": groundtruth,
        "k": args.k,
        "runs": args.runs,
        "algo": args.algo,
        "budget_s": args.budget_s,
        "row_base": row_base,
        "save_indexes": args.save_indexes,
        "reuse_indexes": args.reuse_indexes,
    }
    run_one = run_configuration if args.timing == "in-process" else run_configuration_cli

    monotone = MONOTONE.get(args.algo, ())
    builds = list(grid_fn(**grid_kwargs))
    if monotone:
        builds.sort(key=lambda bq: tuple(bq[0][k] for k in monotone))
    over = []  # build_kwargs of builds measured over budget

    def dominated(b):
        return any(all(b[k] >= o[k] for k in monotone)
                   and all(b[k] == o[k] for k in b if k not in monotone) for o in over)

    n_rows = 0
    with open(args.out, "a") as out:
        for build_kwargs, query_arg_grid in builds:
            if monotone and dominated(build_kwargs):
                print(f"skipped {build_kwargs}: dominates an over-budget build", flush=True)
                continue
            for row in run_one(cls, build_kwargs, query_arg_grid, ctx):
                if row.get("over_budget") and row.get("accuracy_at_k") is None:
                    over.append(build_kwargs)
                    print(f"over budget: {build_kwargs} ({row.get('build_time_s'):.0f} s)", flush=True)
                out.write(json.dumps(row) + "\n")
                out.flush()
                n_rows += 1
                acc = row.get("accuracy_at_k")
                qps = row.get("qps")
                if acc is not None:
                    print(f"{row['name']} {row['query_args']}: acc={acc:.2f}% qps={qps:.1f}", flush=True)
                elif row.get("quarantined"):
                    print(f"QUARANTINED {row['name']} {row.get('query_args', '')}: "
                          f"{row['quarantined']}", flush=True)
    print(f"wrote {n_rows} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
