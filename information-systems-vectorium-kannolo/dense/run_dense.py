#!/usr/bin/env python3
"""Run every system's grid on one dense collection inside VIBE, under the construction budget B.

Per system:

  1. Resolve its grid through VIBE's own definitions loader (`vibe/definitions/`).
  2. Order the builds cheapest first along the library's cost-monotone parameters (dominance.py).
  3. Build and search one configuration at a time (`run.py --algorithm <system>` on a definitions
     tree holding only that configuration), then read its build time back from VIBE's result file.
  4. A build over B is kept on disk but excluded from the selection, and every configuration at
     least as large in all cost-monotone parameters is skipped without being built.

A configuration whose predicted footprint does not fit on the search node is not built
(`predict_footprint`; only SymphonyQG is modelled). VIBE runs one search configuration `--runs`
times and keeps the best pass; the first pass acts as the warm-up.

    python dense/run_dense.py --vibe <vibe> --root <data> --dataset msmarco-qwen-1024-normalized --budget-s 2400
    python dense/run_dense.py --vibe <vibe> --root <data> --dataset agnews-mxbai-1024-euclidean --budget-s 300 \
        --systems kannolo-rerank

Paths: --vibe (the VIBE checkout) and --root (datasets/ and logs/), or KANNOLO_BENCH_VIBE and
KANNOLO_BENCH_ROOT;
DENSE_SMOKE=1 runs only the cheapest build of each system, on three search points.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dominance  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VIBE = os.environ.get("KANNOLO_BENCH_VIBE")
DEFS = os.path.join(HERE, "vibe", "definitions")
ROOT = os.environ.get("KANNOLO_BENCH_ROOT")
LOGS = SVS_STAGING = None  # set by set_paths()
SYSTEMS = ["kannolo", "kannolo-rerank", "vamana-leanvec(svs)", "lorann",
           "symphonyqg", "glass", "hnswlib", "rabitq-hnsw"]
# Over-budget builds still report a measured build time: the build allowance is 4B, plus
# SEARCH_S_PER_GROUP per search configuration, because VIBE's timeout covers build and search.
SEARCH_S_PER_GROUP = 10
MEMORY_MARGIN = 0.95  # share of the search node a build may occupy
RESIDENT_OVER_INDEX = 1.25  # SymphonyQG: measured resident set / (index + training array)


def add_path_args(ap):
    ap.add_argument("--vibe", default=VIBE, help="the VIBE checkout (or $KANNOLO_BENCH_VIBE)")
    ap.add_argument("--root", default=ROOT,
                    help="data directory: datasets/ and logs/ (or $KANNOLO_BENCH_ROOT)")


def set_paths(args):
    global VIBE, ROOT, LOGS, SVS_STAGING
    if not args.vibe or not args.root:
        raise SystemExit("give --vibe and --root (or set KANNOLO_BENCH_VIBE and KANNOLO_BENCH_ROOT)")
    VIBE, ROOT = os.path.abspath(args.vibe), os.path.abspath(args.root)
    LOGS = os.path.join(ROOT, "logs")
    SVS_STAGING = os.path.join(ROOT, "svs_staging")  # SVS writes the training set to disk before building


def vibe_import(name):
    sys.path.insert(0, VIBE)
    cwd = os.getcwd()
    try:
        os.chdir(VIBE)  # vibe.main reads logging.conf from the working directory at import time
        return __import__(name, fromlist=["*"])
    finally:
        os.chdir(cwd)


def dataset_properties(dataset):
    dimension, distance, point_type = vibe_import("vibe.main").parse_dataset_string(dataset)
    return dimension, point_type, distance


def load_grid(dataset, system, k):
    dimension, point_type, distance = dataset_properties(dataset)
    defs = vibe_import("vibe.definitions").get_definitions(
        base_dir=DEFS, dimension=dimension, point_type=point_type, distance_metric=distance, count=k)
    return [d for d in defs if d.algorithm == system]


def node_capacity_bytes():
    caps = []
    for entry in sorted(glob.glob("/sys/devices/system/node/node[0-9]*/meminfo")):
        with open(entry) as fh:
            caps.extend(int(l.split()[-2]) * 1024 for l in fh if "MemTotal" in l)
    return min(caps)


def predict_footprint(system, args, n, d):
    """Peak resident bytes of a SymphonyQG build, or None for every other system.

    One row per node: own vector, the neighbours' RaBitQ codes, factors and ids
    (`symqglib/qg/qg.hpp`), plus VIBE's f32 training array, times the measured allocator overhead.
    """
    if system != "symphonyqg":
        return None
    degree = int(args[1])
    padded = 1 << (d - 1).bit_length()
    row_floats = d + (padded // 64) * 2 * degree + 3 * degree + degree
    return int((n * row_floats * 4 + n * d * 4) * RESIDENT_OVER_INDEX)


def single_config_defs(system, definition, out_dir):
    """Write a definitions tree holding exactly one build configuration of `system`."""
    vdefs = vibe_import("vibe.definitions")
    from vibe.yaml import safe_load

    for entry in sorted(os.listdir(DEFS)):
        cfg = os.path.join(DEFS, entry, "config.yml")
        if not os.path.isfile(cfg):
            continue
        with open(cfg) as fh:
            data = safe_load(fh)
        for point_type in (data or {}).values():
            for metric in point_type.values():
                for algo in metric:
                    if algo.get("name") != system:
                        continue
                    # The run group this configuration came from: the one with its search grid.
                    for group in algo["run_groups"].values():
                        if vdefs.prepare_query_args(group) == definition.query_argument_groups:
                            break
                    else:
                        continue
                    n_base = len(algo.get("base_args", []))
                    one = {"float": {"any": [{
                        "base_args": algo.get("base_args", []),
                        "constructor": algo["constructor"],
                        "disabled": False,
                        "singularity_image": algo.get("singularity_image", system),
                        "module": algo["module"],
                        "name": system,
                        "ood": algo.get("ood", False),  # LoRANN on LAION trains on the learn set
                        "gpu": algo.get("gpu", False),
                        "run_groups": {"one": {
                            # A list-valued argument (an encoder spec) is wrapped one level deeper,
                            # so the cross product takes it whole.
                            "arg_groups": [[v] for v in definition.arguments[n_base:]],
                            **{k: _smoke(v) for k, v in group.items() if k in ("query_args", "query_arg_groups")},
                        }},
                    }]}}
                    os.makedirs(os.path.join(out_dir, entry), exist_ok=True)
                    with open(os.path.join(out_dir, entry, "config.yml"), "w") as fh:
                        json.dump(one, fh, indent=1)  # JSON is valid YAML
                    return
    raise SystemExit(f"no run group of {system!r} under {DEFS} matches {definition.arguments}")


def _smoke(query_args):
    """DENSE_SMOKE=1: three search points (the first three values of the first axis)."""
    if not os.environ.get("DENSE_SMOKE") or not isinstance(query_args, dict):
        return query_args
    keys = list(query_args)
    return {k: (v[:3] if i == 0 else v[:1]) if isinstance(v, list) else v
            for i, (k, v) in enumerate(query_args.items())}


def build_time_from_results(dataset, k, definition):
    import h5py
    path, _ = vibe_import("vibe.results").build_result_filepath(dataset, k, definition, None, False)
    path = path if os.path.isabs(path) else os.path.join(VIBE, path)
    if not os.path.isfile(path):
        return None
    with h5py.File(path, "r") as f:
        for group in f.keys():
            return float(f[group].attrs["build_time"])
    return None


def run_one(dataset, k, timeout, system, definition, log, runs):
    timeout += SEARCH_S_PER_GROUP * max(1, len(definition.query_argument_groups))
    with tempfile.TemporaryDirectory(prefix="onedef_") as tmp:
        single_config_defs(system, definition, tmp)
        cmd = [sys.executable, "-u", "run.py", "--dataset", dataset, "--definitions", tmp,
               "--algorithm", system, "-k", str(k), "--runs", str(runs), "--timeout", str(timeout),
               "--parallelism", "1", "--local"]
        env = dict(os.environ, VIBE_PLACEMENT_ARM="D", VIBE_SVS_TMPDIR=SVS_STAGING)
        with open(log, "a") as fh:
            fh.write(f"\n=== {system} {definition.arguments} {time.strftime('%FT%T')} ===\n")
            fh.flush()
            subprocess.run(cmd, cwd=VIBE, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return build_time_from_results(dataset, k, definition)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--budget-s", type=float, required=True, help="the collection's budget B")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--runs", type=int, default=3, help="timed passes; the best is kept")
    ap.add_argument("--systems", nargs="*", default=SYSTEMS)
    ap.add_argument("--dry-run", action="store_true",
                    help="build nothing; read build times already on disk")
    add_path_args(ap)
    args = ap.parse_args()
    set_paths(args)

    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(SVS_STAGING, exist_ok=True)
    dimension, _, _ = dataset_properties(args.dataset)
    with open(os.path.join(ROOT, "datasets", args.dataset, "meta.json")) as fh:
        n_points = json.load(fh)["n"]
    node_cap = node_capacity_bytes()
    log = os.path.join(LOGS, f"dense_{args.dataset}.log")
    ledger_path = os.path.join(LOGS, f"dense_{args.dataset}_ledger{'_dryrun' if args.dry_run else ''}.json")
    ledger = {"dataset": args.dataset, "budget_s": args.budget_s, "systems": {}}
    if os.path.exists(ledger_path) and not args.dry_run:
        with open(ledger_path) as fh:
            ledger["systems"].update(json.load(fh).get("systems", {}))

    for system in args.systems:
        grid = load_grid(args.dataset, system, args.k)
        by_args = {tuple(map(_hashable, d.arguments)): d for d in grid}
        configs = list(by_args)
        if os.environ.get("DENSE_SMOKE"):  # smoke test: the cheapest build only
            configs = dominance.order(configs, system)[:1]
        print(f"\n=== {system}: {len(configs)} configurations, B = {args.budget_s:.0f} s ===", flush=True)
        too_big = []
        for cfg in list(configs):
            predicted = predict_footprint(system, list(cfg), n_points, dimension)
            if predicted is not None and predicted > node_cap * MEMORY_MARGIN:
                too_big.append((cfg, predicted))
                configs.remove(cfg)
                print(f"  not built (memory) {list(cfg)[1:]}: predicted {predicted / 2**30:.0f} GiB", flush=True)

        def build_time_of(cfg, _system=system):
            if args.dry_run:
                return build_time_from_results(args.dataset, args.k, by_args[cfg])
            secs = run_one(args.dataset, args.k, int(4 * args.budget_s), _system, by_args[cfg], log, args.runs)
            print(f"  {_system} {list(cfg)[1:]}  build {'n/a' if secs is None else f'{secs:.1f} s'}", flush=True)
            return secs

        built, skipped = dominance.plan(configs, system, build_time_of, args.budget_s)
        for cfg, blocker in skipped:
            print(f"  skipped {list(cfg)[1:]}: dominates over-budget {list(blocker)[1:]}", flush=True)
        ledger["systems"][system] = {
            "built": [{"args": list(c), "build_time_s": t} for c, t, _ in built],
            "skipped_by_dominance": [{"args": list(c), "dominates": list(b)} for c, b in skipped],
            "skipped_by_memory": [{"args": list(c), "predicted_bytes": p} for c, p in too_big],
        }
        with open(ledger_path, "w") as fh:
            json.dump(ledger, fh, indent=1)
    return 0


def _hashable(x):
    return json.dumps(x, sort_keys=True) if isinstance(x, (list, dict)) else x


if __name__ == "__main__":
    sys.exit(main())
