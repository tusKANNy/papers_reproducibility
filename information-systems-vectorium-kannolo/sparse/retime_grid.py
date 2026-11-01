"""Re-timing grid: the configurations listed in $RETIME_CONFIGS, which run_sparse.py writes from
the in-process tuning rows (the three fastest configurations at each accuracy target).
"""
import json
import os

CONFIGS = os.environ.get("RETIME_CONFIGS")


def _load(system):
    with open(CONFIGS) as fh:
        cfg = json.load(fh)
    if system not in cfg:
        raise SystemExit(f"{system!r} not in {CONFIGS}; have {sorted(cfg)}")
    return [(c["build_args"], [c["query_args"]]) for c in cfg[system]]


def seismic():
    return _load("seismic")


def seismic_wave():
    return _load("seismic-wave")


def kannolo_hnsw():
    return _load("kannolo-hnsw")


def kannolo_rerank():
    return _load("kannolo-hnsw-rerank")


def pyanns():
    return _load("pyanns")
