#!/usr/bin/env python3
"""Per-query latency percentiles of the reported dense configurations, from VIBE's stored times.

    python analysis/tail_dense.py $VIBE selections/dense_*.json > results/tail/dense.tsv
"""
import json
import os
import sys

import h5py
import numpy as np

vibe = sys.argv[1]
print("dataset\talgo\ttarget\tname\tacc\tqps\tn\tmean_us\tp50_us\tp95_us\tp99_us\tmax_us\tp99_over_p50")
for path in sys.argv[2:]:
    sel = json.load(open(path))
    for group in (sel["table2"], sel["table2_control_arms"]):
        for algo, per_t in group.items():
            for t, row in sorted(per_t.items()):
                with h5py.File(os.path.join(vibe, row["result_file"]), "r") as h:
                    us = np.asarray(h[str(row["query_args"])]["times"]) * 1e6
                p50, p95, p99 = np.percentile(us, [50, 95, 99])
                print(f"{row['dataset']}\t{algo}\t{t}\t{row['name']}\t{row['accuracy_at_k']:.2f}\t"
                      f"{row['qps']:.0f}\t{len(us)}\t{us.mean():.1f}\t{p50:.1f}\t{p95:.1f}\t{p99:.1f}\t"
                      f"{us.max():.1f}\t{p99 / p50:.2f}")
