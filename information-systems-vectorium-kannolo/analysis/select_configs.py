#!/usr/bin/env python3
"""Select, per system, the configurations the paper reports, from one collection's rows.

Rules (Section 6 of the paper):
  * a row counts only if its index was placed on the search node (>= 99 % of its pages local).
    The exception is an index built interleaved across both nodes because it cannot fit on one
    (SymphonyQG on DPR, placement policy interleave);
  * a build whose construction time exceeds the budget B is excluded;
  * LoRANN rescores against the harness's copy of the collection, so the raw f32 collection is
    added to its footprint;
  * table: per system and target t in {90, 95, 99}, the fastest configuration with Accuracy@10 >= t;
  * frontier (figures): per system, the fastest configuration in each integer accuracy interval
    [t, t+1), t = 90..99, keeping a point only if it is faster than every more accurate one, and
    points at least MIN_ACC_SEPARATION apart.
Variants (ext-8 rescoring, LoRANN without its learn set) enter only the table.

    python analysis/select_configs.py --results results/dense/dpr.jsonl --budget-s 5400 --out selections/dense_dpr.json
"""

import argparse
import json
from collections import defaultdict

MIN_LOCAL_FRACTION = 0.99
TABLE_TARGETS = [90, 95, 99]
MIN_ACC_SEPARATION = 0.5
SERIES = {
    "dense": ["kannolo", "kannolo-rerank", "hnswlib", "glass", "symphonyqg", "lorann",
              "rabitq-hnsw", "vamana-leanvec(svs)"],
    "sparse": ["kannolo-hnsw", "kannolo-hnsw-rerank", "seismic", "seismic-wave", "pyanns",
               "sindi", "lsp"],
}
VARIANTS = ["kannolo-rerank-ext8", "kannolo-rerank-ext8-resid", "lorann-plain"]
BORROWS_COLLECTION = {"lorann", "lorann-plain"}
COLLECTION_SHAPE = {  # (n, d), for the f32 collection charged to LoRANN
    "agnews-mxbai-1024-euclidean": (769_382, 1024),
    "msmarco-qwen-1024-normalized": (8_840_823, 1024),
    "dpr-jina-768-normalized": (20_969_760, 768),
    "laion-clip-512-normalized": (1_000_448, 512),
}


def load(path):
    rows = []
    for line in open(path):
        row = json.loads(line)
        placement = row.get("build_placement") or {}
        interleaved = placement.get("placement") == "interleave" and not placement.get("migrated", False)
        if not interleaved and (placement.get("local_fraction") or 0) < MIN_LOCAL_FRACTION:
            continue
        if row["algo"] in BORROWS_COLLECTION and row.get("uss_delta_kb") is not None:
            n, d = COLLECTION_SHAPE[row["dataset"]]
            row["uss_delta_kb"] += n * d * 4 / 1024.0
        rows.append(row)
    return rows


def fastest_per_target(rows):
    out = {}
    for t in TABLE_TARGETS:
        ok = [r for r in rows if r["accuracy_at_k"] >= t]
        if ok:
            out[str(t)] = min(ok, key=lambda r: r["mean_us"])
    return out


def frontier(rows):
    bins = defaultdict(list)
    for r in rows:
        if 90.0 <= r["accuracy_at_k"] < 100.0:
            bins[int(r["accuracy_at_k"])].append(r)
    kept, best_above = [], 0.0
    for edge in sorted(bins, reverse=True):
        candidate = max(bins[edge], key=lambda r: r["qps"])
        if candidate["qps"] > best_above:
            kept.append(candidate)
            best_above = candidate["qps"]
    if not kept:
        return []
    top = max(kept, key=lambda r: r["accuracy_at_k"])
    spread = [top]
    for r in sorted(kept, key=lambda r: -r["qps"]):
        if r is not top and all(abs(k["accuracy_at_k"] - r["accuracy_at_k"]) >= MIN_ACC_SEPARATION
                                for k in spread):
            spread.append(r)
    return sorted(spread, key=lambda r: r["accuracy_at_k"])


def select(rows, budget_s, series):
    by_algo = defaultdict(list)
    for r in rows:
        if r["build_time_s"] is not None and r["build_time_s"] <= budget_s:
            by_algo[r["algo"]].append(r)
    spreads = sorted(r["pass_spread_pct"] for r in rows
                     if r["algo"] in series and r.get("pass_spread_pct") is not None)
    return {
        "dataset": rows[0]["dataset"],
        "budget_s": budget_s,
        "series_order": series,
        "per_accuracy": {a: frontier(by_algo[a]) for a in series if by_algo.get(a)},
        "per_target": {a: fastest_per_target(by_algo[a]) for a in series if by_algo.get(a)},
        "per_target_variants": {a: fastest_per_target(by_algo[a]) for a in VARIANTS if by_algo.get(a)},
        "pass_spread": {"median_pct": spreads[len(spreads) // 2] if spreads else None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    ap.add_argument("--series", choices=sorted(SERIES), default="dense")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sel = select(load(args.results), args.budget_s, SERIES[args.series])
    with open(args.out, "w") as fh:
        json.dump(sel, fh, indent=1)
    for algo, per_t in {**sel["per_target"], **sel["per_target_variants"]}.items():
        print(f"{algo:24s}", "  ".join(f"{t}: {r['qps']:8.0f}" for t, r in per_t.items()))


if __name__ == "__main__":
    main()
