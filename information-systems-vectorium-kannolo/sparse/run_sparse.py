#!/usr/bin/env python3
"""Run the sparse benchmarks of one collection: every system, its whole grid (sparse/grids.py).

  * kANNolo-HNSW, kANNolo-Rerank and PyANN are tuned in-process over their grid, with the
    indexes saved. The three fastest configurations at each target t = 90..99 are then timed
    from the command line on the same index files.
  * Seismic, Seismic-Wave, LSP and SINDI are built and searched with their command-line tools
    over their whole grid.
Only rows from command-line timing are reported. The systems built on Vectorium use the
blocked-sparse encoder.

    python sparse/run_sparse.py --sparse-root <systems> --collection nq --data <nq-dir> --out results/sparse/nq.jsonl
    python sparse/run_sparse.py ... --systems kannolo-hnsw-rerank

`--data` holds documents.bin, queries.bin and groundtruth.tsv. Paths of the systems' builds are
set in the environment (sparse/env.example.sh).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECTIONS = {  # name -> (dataset id, number of queries, budget B in s)
    "msmarco-splade": ("msmarco-splade-v3", 6980, 5400),
    "nq": ("nq-cocondenser", 3452, 1800),
    "msmarco-esplade": ("msmarco-esplade", 6980, 6000),
}
# system -> (in-process tuning module or None, command-line module, retime_grid function)
SYSTEMS = {
    "kannolo-hnsw": ("kannolo_sparse:KannoloSparseHNSW", "kannolo_blocked:KannoloHnswBlocked", "kannolo_hnsw"),
    "kannolo-hnsw-rerank": ("kannolo_sparse:KannoloSparseRerank", "kannolo_blocked:KannoloRerankBlocked",
                            "kannolo_rerank"),
    "pyanns": ("pyanns_sparse:PyannsSparseHNSW", "cli_retime:Pyanns", "pyanns"),
    "seismic": (None, "seismic_blocked:SeismicBlocked", None),
    "seismic-wave": (None, "seismic_blocked:SeismicWaveBlocked", None),
    "lsp": (None, "lsp_sparse:Lsp", None),
    "sindi": (None, "sindi_sparse:Sindi", None),
}
TOP = 3


def driver(args, extra_env, python=sys.executable):
    env = dict(os.environ, **extra_env,
               VIBE_PLACEMENT="migrate", VIBE_SEARCH_CPU=os.environ.get("VIBE_SEARCH_CPU", "0"),
               VIBE_SEARCH_NODE=os.environ.get("VIBE_SEARCH_NODE", "0"),
               VIBE_BUILD_CPUS=os.environ.get("VIBE_BUILD_CPUS", f"0-{os.cpu_count() - 1}"),
               RAYON_NUM_THREADS=os.environ.get("RAYON_NUM_THREADS", str(os.cpu_count())),
               MALLOC_ARENA_MAX="8", MALLOC_MMAP_THRESHOLD_="268435456",
               PYTHONPATH=os.pathsep.join([HERE, os.path.join(HERE, "modules"),
                                           os.environ.get("PYTHONPATH", "")]))
    cmd = ["numactl", "--interleave=all", python, os.path.join(HERE, "driver.py"), *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def prepare_lsp(data, work):
    """LSP's input, once per collection: the collection as JSONL, then 8-bit quantization, graph
    bisection and query encoding, each timed (sparse/tools/prepare_lsp_input.py)."""
    prep = os.path.join(work, "lsp_prep")
    if os.path.exists(os.path.join(prep, "prep_timings.json")):
        return prep
    os.makedirs(prep, exist_ok=True)
    jsonl = {}
    for part in ("documents", "queries"):
        jsonl[part] = os.path.join(prep, f"{part}.jsonl")
        subprocess.run([sys.executable, os.path.join(HERE, "tools", "convert_seismic_to_jsonl.py"),
                        "--seismic", data[part], "--out", jsonl[part]], check=True)
    subprocess.run([sys.executable, os.path.join(HERE, "tools", "prepare_lsp_input.py"),
                    "--docs-jsonl", jsonl["documents"], "--queries-jsonl", jsonl["queries"],
                    "--out-dir", prep], check=True, cwd=prep)  # the bisection tool logs to its cwd
    return prep


def charged_build_s(row):
    """Build time without the conversion of the collection into the library's own input format."""
    return row["build_time_s"] - (row.get("convert_s") or 0.0)


def top_configs(tune_rows_path, budget_s):
    rows = [json.loads(l) for l in open(tune_rows_path)]
    rows = [r for r in rows if r.get("accuracy_at_k") is not None and not r.get("over_budget")
            and charged_build_s(r) <= budget_s]
    chosen = {}
    for t in range(90, 100):
        for r in sorted((r for r in rows if r["accuracy_at_k"] >= t), key=lambda r: -r["qps"])[:TOP]:
            chosen[(r["result_file"], r["query_args"])] = r
    return chosen.values()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True, choices=sorted(COLLECTIONS))
    ap.add_argument("--data", required=True, help="directory with documents.bin, queries.bin, groundtruth.tsv")
    ap.add_argument("--out", required=True, help="JSONL of the reported rows (command-line timing)")
    ap.add_argument("--work", default=None, help="indexes and tuning rows (default: next to --out)")
    ap.add_argument("--sparse-root", default=os.environ.get("SPARSE_ROOT"),
                    help="directory of the systems' source trees and builds (or $SPARSE_ROOT); "
                         "individual paths can be overridden, see sparse/env.example.sh")
    ap.add_argument("--systems", nargs="*", default=list(SYSTEMS))
    args = ap.parse_args()
    if not args.sparse_root:
        raise SystemExit("give --sparse-root (or set SPARSE_ROOT)")
    os.environ["SPARSE_ROOT"] = os.path.abspath(args.sparse_root)

    dataset, _, budget = COLLECTIONS[args.collection]
    with open(os.path.join(args.data, "queries.bin"), "rb") as fh:
        n_queries = int.from_bytes(fh.read(4), "little")
    work = args.work or os.path.splitext(args.out)[0] + "_work"
    os.makedirs(work, exist_ok=True)
    data = {k: os.path.join(args.data, f) for k, f in
            (("documents", "documents.bin"), ("queries", "queries.bin"), ("groundtruth", "groundtruth.tsv"))}
    common = ["--dataset", dataset, "--documents", data["documents"], "--queries", data["queries"],
              "--groundtruth", data["groundtruth"], "-k", "10", "--budget-s", str(budget)]
    env = {"RETIME_QUERIES": data["queries"], "RETIME_N_QUERIES": str(n_queries)}
    grid_args = lambda system: json.dumps({"system": system, "collection": args.collection})

    for system in args.systems:
        tune_module, cli_module, retime_fn = SYSTEMS[system]
        indexes = os.path.join(work, "indexes", system)
        if system == "sindi":  # its search tool computes recall against a ground-truth file of its own
            os.makedirs(indexes, exist_ok=True)
            gt = os.path.join(indexes, "gt_top10.bin")
            if not os.path.exists(gt):
                subprocess.run([sys.executable, os.path.join(HERE, "tools", "gt_tsv_to_bin.py"),
                                "--tsv", data["groundtruth"], "--out", gt, "-k", "10"], check=True)
        if system == "lsp" and not os.environ.get("SPARSE_CIFF_PREP"):
            env["SPARSE_CIFF_PREP"] = prepare_lsp(data, work)
        if tune_module is None:
            driver(["--timing", "cli", "--module", cli_module, "--algo", system, "--grid", "grids:grid",
                    "--grid-args", grid_args(system), "--save-indexes", indexes,
                    "--out", args.out, *common], env)
            continue
        tune_out = os.path.join(work, f"tune_{system}.jsonl")
        # PyANN's Python bindings live in their own environment (PYANNS_PYTHON).
        python = os.environ.get("PYANNS_PYTHON", sys.executable) if system == "pyanns" else sys.executable
        driver(["--timing", "in-process", "--module", tune_module, "--algo", system, "--grid", "grids:grid",
                "--grid-args", grid_args(system), "--save-indexes", indexes, "--runs", "3",
                "--out", tune_out, *common], env, python)
        configs = [{"build_args": r["build_args"], "query_args": r["query_args_dict"]}
                   for r in top_configs(tune_out, budget)]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=work) as fh:
            json.dump({system: configs}, fh)
        retimed = os.path.join(work, f"retime_{system}.jsonl")
        driver(["--timing", "cli", "--module", cli_module, "--algo", system,
                "--grid", f"retime_grid:{retime_fn}", "--reuse-indexes", indexes,
                "--out", retimed, *common], dict(env, RETIME_CONFIGS=fh.name))
        # The re-timed rows load the saved indexes; their build time is the one measured in tuning.
        build_time = {r["result_file"]: charged_build_s(r) for r in map(json.loads, open(tune_out))}
        with open(args.out, "a") as out:
            for line in open(retimed):
                row = json.loads(line)
                row["build_time_s"] = build_time[row["result_file"]]
                row["over_budget"] = row["build_time_s"] > budget
                out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
