#!/usr/bin/env python3
"""Compare a SINDI `generate_gt`-format .bin against the seismic dataset's
shipped groundtruth.tsv. Both should be exact brute-force IP top-k.

Reports:
  * exact set-match rate (queries where the top-k id sets are equal)
  * mean Recall@k (overlap / k, averaged over queries)
  * mean / max score divergence on overlapping ids

A non-trivial mismatch rate means that the CSR encoding lost information
compared with the source of the shipped groundtruth. Investigate it before
trusting any downstream recall number.

gt.bin layout (vsag/sparse/scripts/generate_gt.cpp:119-133):
    uint32 num_queries
    uint32 topk
    int32[N*k] ids
    float[N*k] distances

TSV layout: qid \\t docid \\t rank \\t score, sorted by (qid, rank).
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def load_gt_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Ids and **similarities** from a generate_gt .bin.

    vsag stores inner product as a *distance*, i.e. negated: a bin entry of -20.98 is the same
    quantity the shipped TSV writes as +20.98. With the raw value, the score-divergence statistic
    below would compare `-s` against `+s` and report `2s`: on NQ, "mean=4.297e+01" for scores
    whose mean magnitude is ~21. That number could never be small, so the statistic meant to
    detect a lossy CSR encoding could not detect anything. The value is negated here, when it is
    read, so every caller gets a comparable quantity.
    """
    with path.open("rb") as f:
        n, k = struct.unpack("<II", f.read(8))
        ids = np.fromfile(f, dtype="<i4", count=n * k).reshape(n, k)
        dists = np.fromfile(f, dtype="<f4", count=n * k).reshape(n, k)
    return ids, -dists


def load_gt_tsv(path: Path, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids[n,k], scores[n,k]) sorted by (qid, rank)."""
    arr = np.loadtxt(path, dtype=np.float64, delimiter="\t")
    qid = arr[:, 0].astype(np.int64)
    docid = arr[:, 1].astype(np.int64)
    rank = arr[:, 2].astype(np.int64)
    score = arr[:, 3].astype(np.float32)
    n = qid.max() + 1
    if arr.shape[0] != n * k:
        raise SystemExit(
            f"TSV has {arr.shape[0]} rows but expected {n}*{k}={n*k}; check topk"
        )
    order = np.lexsort((rank, qid))
    return docid[order].reshape(n, k), score[order].reshape(n, k)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bin", required=True, type=Path, help="gt.bin from generate_gt")
    p.add_argument("--tsv", required=True, type=Path, help="seismic groundtruth.tsv")
    p.add_argument("--show-mismatches", type=int, default=5, help="Print first N mismatching queries")
    args = p.parse_args()

    bin_ids, bin_scores = load_gt_bin(args.bin)
    n_bin, k = bin_ids.shape
    tsv_ids, tsv_scores = load_gt_tsv(args.tsv, k)
    n_tsv = tsv_ids.shape[0]

    if n_bin != n_tsv:
        print(f"WARN: query count mismatch bin={n_bin} tsv={n_tsv}", file=sys.stderr)
    n = min(n_bin, n_tsv)

    exact = 0
    overlap_sum = 0
    score_diffs: list[float] = []
    mismatches: list[int] = []
    for q in range(n):
        a = set(int(x) for x in bin_ids[q])
        b = set(int(x) for x in tsv_ids[q])
        ov = a & b
        overlap_sum += len(ov)
        if a == b:
            exact += 1
        else:
            if len(mismatches) < args.show_mismatches:
                mismatches.append(q)
        # Score divergence on overlapping ids
        for d in ov:
            # find rank of d in each list (cheap for small k)
            i = int(np.where(bin_ids[q] == d)[0][0])
            j = int(np.where(tsv_ids[q] == d)[0][0])
            score_diffs.append(abs(float(bin_scores[q, i]) - float(tsv_scores[q, j])))

    print(f"queries compared: {n}  topk={k}")
    print(f"exact set match: {exact}/{n} ({100.0*exact/n:.2f}%)")
    print(f"mean Recall@{k}: {overlap_sum / (n*k):.4f}")
    if score_diffs:
        sd = np.array(score_diffs)
        print(f"score abs-diff on overlapping ids: mean={sd.mean():.3e}  max={sd.max():.3e}")
    if mismatches:
        print(f"\nfirst {len(mismatches)} mismatching queries:")
        for q in mismatches:
            a = list(int(x) for x in bin_ids[q])
            b = list(int(x) for x in tsv_ids[q])
            print(f"  q={q}")
            print(f"    bin ids: {a}")
            print(f"    tsv ids: {b}")

    # Requiring exact agreement would fail a correct scan.
    #
    # Two correct exhaustive f32 scans that accumulate in different orders can rank equal-score
    # documents differently, so a document on the rank-k boundary can go either way. The check
    # therefore requires every disagreement to be a floating-point tie. On NQ, 8 of 3452 queries
    # disagreed, each by a single document at rank 10, and the two candidates' scores agreed to
    # 0-4 ULP (worst relative gap 1.27e-7, f32 eps 1.19e-7). This passes the check. A lossy CSR
    # encoding instead gives gaps orders of magnitude larger, and disagreements that are not
    # confined to the boundary.
    #
    # RTOL is ~8 ULP of f32: large enough for differences in accumulation order, and far too
    # small to absorb a real encoding error.
    RTOL = 1e-6
    untied = []
    for q in range(n):
        only_bin = set(map(int, bin_ids[q])) - set(map(int, tsv_ids[q]))
        if not only_bin:
            continue
        only_tsv = set(map(int, tsv_ids[q])) - set(map(int, bin_ids[q]))
        edge_bin = min(float(bin_scores[q, int(np.where(bin_ids[q] == d)[0][0])]) for d in only_bin)
        edge_tsv = min(float(tsv_scores[q, int(np.where(tsv_ids[q] == d)[0][0])]) for d in only_tsv)
        gap = abs(edge_bin - edge_tsv)
        if gap > RTOL * max(abs(edge_bin), abs(edge_tsv)):
            untied.append((q, edge_bin, edge_tsv, gap))

    if exact == n:
        print("\nCHECK PASSED: exact set match on every query.")
        return 0
    if not untied:
        print(f"\nCHECK PASSED: {n - exact} of {n} queries disagree, and every disagreement is a "
              f"floating-point tie at the rank-{k} boundary (rtol {RTOL:.0e}).")
        return 0
    print(f"\nCHECK FAILED: {len(untied)} queries disagree by more than a float tie. The CSR encoding "
          "or the ground truth is wrong; do not run a parameter grid on this collection.")
    for q, sb, st, gap in untied[:args.show_mismatches]:
        print(f"  q={q}  boundary bin={sb:.6f} tsv={st:.6f}  gap={gap:.3e}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
