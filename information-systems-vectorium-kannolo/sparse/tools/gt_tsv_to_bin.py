#!/usr/bin/env python3
"""Write SINDI's `gt_top<k>.bin` from the collection's shipped `groundtruth.tsv`.

`sindi_index_search` refuses to run without a ground-truth binary: it computes its own recall
and needs something to compute it against. That self-reported recall is recorded but never
reported; accuracy is computed by the driver from the returned ids against the shipped
`groundtruth.tsv`. Converting the shipped file (instead of running vsag's `generate_gt`, an
exhaustive scan) keeps SINDI's self-report consistent with the reported number and takes seconds.
Ground truth is a prerequisite, never a build stage, so it is not charged either way.

FORMAT, read from `sindi_index_search.cpp:knn_result_read` (with the PoC's
`patches/0001-generate_gt-int32-header.patch`):

    header : 2 x uint32 little-endian  (n queries, k)
    ids    : n*k x int32  little-endian
    dists  : n*k x float32 little-endian, NEGATED -- vsag stores inner product as a distance,
             so the TSV's +20.98 is written here as -20.98. With the wrong sign, compare_gt's
             score check would compare -s against +s and could not detect a lossy encoding.
"""
import argparse
import struct
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsv", required=True, type=Path, help="shipped groundtruth.tsv")
    p.add_argument("--out", required=True, type=Path, help="gt_top<k>.bin to write")
    p.add_argument("-k", type=int, default=10)
    a = p.parse_args()

    arr = np.loadtxt(a.tsv, dtype=np.float64, delimiter="\t")
    qid = arr[:, 0].astype(np.int64)
    docid = arr[:, 1].astype(np.int64)
    rank = arr[:, 2].astype(np.int64)
    score = arr[:, 3].astype(np.float32)

    n = int(qid.max()) + 1
    if arr.shape[0] != n * a.k:
        raise SystemExit(f"TSV has {arr.shape[0]} rows, expected {n}*{a.k}={n*a.k}")
    if docid.max() > np.iinfo(np.int32).max:
        raise SystemExit(f"doc id {docid.max()} exceeds int32; the format cannot carry it")

    order = np.lexsort((rank, qid))
    ids = docid[order].reshape(n, a.k).astype("<i4")
    dists = (-score[order].reshape(n, a.k)).astype("<f4")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("wb") as f:
        f.write(struct.pack("<II", n, a.k))
        ids.tofile(f)
        dists.tofile(f)

    # Read it back the way compare_gt.py reads it and assert it round-trips, so a format error
    # fails here rather than inside the SINDI binary.
    with a.out.open("rb") as f:
        rn, rk = struct.unpack("<II", f.read(8))
        rids = np.fromfile(f, dtype="<i4", count=rn * rk).reshape(rn, rk)
        rd = -np.fromfile(f, dtype="<f4", count=rn * rk).reshape(rn, rk)
    assert (rn, rk) == (n, a.k), (rn, rk)
    assert np.array_equal(rids, ids), "id round-trip failed"
    assert np.allclose(rd, score[order].reshape(n, a.k)), "score round-trip failed"

    print(f"wrote {a.out}  n={n} k={a.k}  {a.out.stat().st_size} bytes")
    print(f"  ids  [0][:5] = {ids[0][:5].tolist()}")
    print(f"  score[0][:3] = {score[order].reshape(n, a.k)[0][:3].tolist()} "
          f"(stored negated)")
    print("  round-trip through the generate_gt format: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
