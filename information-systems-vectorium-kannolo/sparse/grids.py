"""The parameter grid of every sparse system, per collection.

`grid(system, collection)` returns [(build_kwargs, [query_kwargs, ...]), ...], in the order the
driver runs them. Collections: "msmarco-splade", "nq", "msmarco-esplade".

Seismic, Seismic-Wave and SINDI use their authors' released configurations; LSP's grid sweeps
around its documented settings with 8-bit weights; PyANN uses our HNSW construction parameters
and a wider beam than its published search points. The kANNolo grids are ours. An axis we
constructed is extended on a collection where a selected configuration sits at its end.
"""

import os
from itertools import product as _product

COLLECTIONS = ["msmarco-splade", "nq", "msmarco-esplade"]


def product(**axes):
    keys = list(axes)
    return [dict(zip(keys, values)) for values in _product(*(axes[k] for k in keys))]


def builds(keys, tuples, **fixed):
    return [dict(zip(keys, t), **fixed) for t in tuples]


# --- kANNolo-HNSW: single-stage HNSW over the blocked-sparse encoder ----------------------------

HNSW_BUILDS = {
    "msmarco-splade": [(16, 200), (16, 500), (16, 1000), (16, 2000), (32, 500), (32, 1000),
                       (32, 2000), (64, 1000)],
    "nq": [(16, 200), (16, 500), (16, 1000), (32, 200), (32, 500)],
    "msmarco-esplade": [(16, 200), (16, 500), (16, 1000), (32, 500)],
}
HNSW_EF_NO_LAMBDA = [50, 100, 200, 400, 800, 1600, 3200]
HNSW_LAMBDA = [0.005, 0.01, 0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.21, 0.25]
HNSW_EF_WITH_LAMBDA = {
    "msmarco-splade": [10, 12, 14, 16, 18, 20, 22, 25, 30, 50, 75, 100, 150, 200],
    "nq": [10, 12, 14, 16, 18, 20, 22, 25, 30, 50],
    "msmarco-esplade": [10, 12, 14, 16, 18, 20, 22, 25, 30, 50, 75, 100, 150, 200],
}
# lambda extended above 0.25 on MsMarco-SPLADE, where the selection reached the end of the axis
HNSW_LAMBDA_EXT = {"msmarco-splade": ([10, 12, 14, 16, 18, 20, 22, 25, 30, 50, 75], [0.28, 0.32, 0.36, 0.40])}


def kannolo_hnsw(c):
    queries = product(ef_search=HNSW_EF_NO_LAMBDA, early_exit_threshold=[None])
    queries += product(ef_search=HNSW_EF_WITH_LAMBDA[c], early_exit_threshold=HNSW_LAMBDA)
    if c in HNSW_LAMBDA_EXT:
        efs, lams = HNSW_LAMBDA_EXT[c]
        queries += product(ef_search=efs, early_exit_threshold=lams)
    return [(dict(m=m, ef_construction=efc, graph_type="standard"), queries) for m, efc in HNSW_BUILDS[c]]


# --- kANNolo-Rerank: HNSW over the largest entries (a % of the l1 norm), residual rescoring -----

RERANK_K_CANDIDATES = {
    "msmarco-splade": [20, 30, 40, 50, 75, 100, 150],
    "nq": [20, 30, 40, 50, 75, 100, 150],
    "msmarco-esplade": [20, 30, 40, 50, 75, 100, 150, 200, 300, 500],
}
RERANK_EF_MULTIPLIER = [1.0, 1.25, 1.5]
RERANK_LAMBDA = [None, 0.01, 0.03, 0.06, 0.09, 0.12, 0.18, 0.25]


def kannolo_hnsw_rerank(c):
    queries = [dict(k_candidates=kc, ef_search=int(round(mult * kc)), alpha=None, beta=None,
                    early_exit_threshold=lam)
               for mult in RERANK_EF_MULTIPLIER for lam in RERANK_LAMBDA for kc in RERANK_K_CANDIDATES[c]]
    return [(dict(m=m, ef_construction=efc, prune_percentage=float(a),
                  first_stage_value_type="blocked-sparse", graph_type="standard"), queries)
            for m in (16, 32) for efc in (1000, 2000) for a in (60, 70, 80)]


# --- Seismic and Seismic-Wave: the authors' released configurations -----------------------------

SEISMIC_KEYS = ("n_postings", "centroid_fraction", "min_cluster_size", "summary_energy", "max_fraction")
SEISMIC_BUILDS = {
    "msmarco-splade": [(2000, .2, 2, .4, 2.), (2000, .2, 2, .5, 2.), (2000, .2, 2, .5, 3.), (2000, .2, 2, .5, 4.),
                       (2000, .2, 2, .5, 6.), (3000, .2, 2, .4, 4.), (3000, .2, 2, .5, 2.), (3000, .2, 2, .5, 3.),
                       (3000, .2, 2, .5, 4.), (3000, .2, 5, .4, 6.), (4000, .2, 2, .5, 3.), (4000, .2, 5, .4, 6.)],
    "nq": [(2000, .1, 2, .5, 3.), (2000, .1, 5, .4, 3.), (2000, .1, 10, .4, 4.), (2000, .1, 10, .5, 6.),
           (2000, .2, 2, .4, 3.), (2000, .2, 2, .4, 4.), (2000, .2, 2, .5, 3.), (2000, .2, 2, .5, 6.),
           (2000, .2, 5, .4, 3.), (2000, .2, 5, .4, 6.), (2000, .2, 10, .4, 6.), (3000, .1, 5, .5, 6.),
           (3000, .1, 10, .4, 6.), (3000, .2, 2, .5, 2.), (3000, .2, 2, .5, 3.), (3000, .2, 2, .5, 4.),
           (3000, .2, 2, .5, 6.), (3000, .2, 5, .4, 6.), (3000, .2, 5, .5, 3.), (3000, .2, 10, .4, 4.),
           (4000, .2, 2, .5, 2.), (4000, .2, 2, .5, 3.), (4000, .2, 2, .5, 4.)],
    "msmarco-esplade": [(2000, .2, 2, .5, 2.), (2000, .2, 2, .5, 4.), (2000, .2, 2, .5, 6.), (3000, .1, 2, .4, 6.),
                        (3000, .2, 2, .5, 4.), (4000, .2, 2, .5, 2.), (4000, .2, 2, .5, 4.), (4000, .2, 2, .5, 6.)],
}
# Seismic-Wave on MsMarco-SPLADE: the Seismic builds with min_cluster_size 2 and summary_energy 0.5.
WAVE_BUILDS = dict(SEISMIC_BUILDS, **{"msmarco-splade": [
    (4000, .2, 2, .5, 3.), (3000, .2, 2, .5, 4.), (2000, .2, 2, .5, 6.), (3000, .2, 2, .5, 3.),
    (3000, .2, 2, .5, 2.), (2000, .2, 2, .5, 4.), (2000, .2, 2, .5, 3.), (2000, .2, 2, .5, 2.)]})
SEISMIC_SEARCH = dict(query_cut=[2, 4, 6, 7, 8, 10, 12], heap_factor=[0.7, 0.8, 0.9, 1.0],
                      first_sorted=[False, True])


def seismic(c):
    queries = product(**SEISMIC_SEARCH, n_knn=[0])
    return [(b, queries) for b in builds(SEISMIC_KEYS, SEISMIC_BUILDS[c], doc_cut=15, nknn=0)]


def seismic_wave(c):
    queries = product(**SEISMIC_SEARCH, n_knn=[10, 25, 50])
    return [(b, queries) for b in builds(SEISMIC_KEYS, WAVE_BUILDS[c], doc_cut=15, nknn=50)]


# --- PyANN: our HNSW construction parameters, a wider beam than its published points ------------

PYANN_BUILDS = {
    "msmarco-splade": [(16, 200), (32, 200), (16, 500), (64, 200)],
    "nq": [(r, l) for r in (16, 32, 64) for l in (200, 500, 1000, 2000)],
    "msmarco-esplade": [(16, 200), (32, 200), (16, 500), (64, 200)],
}
PYANN_SEARCH = (product(ef=[50, 55, 60, 65, 70, 75, 80, 100, 200, 400, 800, 1600, 3200], budget=[0.1])
                + product(ef=[50, 65, 70, 100, 200, 400, 800, 1600, 3200], budget=[0.08]))


def pyanns(c):
    return [(dict(r=r, l=l, drop_ratio=0.0), PYANN_SEARCH) for r, l in PYANN_BUILDS[c]]


# --- LSP: around its documented settings, 8- and 4-bit weights ----------------------------------

LSP_KEYS = ("superblock", "subblock", "compress_document_index", "quant")
LSP_BUILDS_MSMARCO = [(64, 4, "seismic", 4), (64, 4, "seismic", 8), (128, 4, "seismic", 4), (128, 4, "seismic", 8),
                      (128, 8, "seismic", 4), (128, 8, "seismic", 8), (128, 8, "bmpinv", 8)]
LSP_BUILDS = {
    "msmarco-splade": LSP_BUILDS_MSMARCO,
    "nq": [(sb, b, cdi, q) for sb, b in ((128, 8), (256, 16), (512, 16)) for cdi in ("seismic", "bmpinv") for q in (4, 8)]
          + [(sb, b, "seismic", q) for sb, b in ((64, 4), (128, 4), (256, 8)) for q in (4, 8)],
    "msmarco-esplade": LSP_BUILDS_MSMARCO,
}
LSP_GAMMA = {
    "msmarco-splade": [100, 250, 500, 1000, 2000, 4000],
    "nq": [100, 250, 500, 1000, 2000, 4000, 8000],
    "msmarco-esplade": [25, 50, 75, 100, 250, 500, 1000, 2000, 4000],
}


def lsp(c):
    queries = product(gamma=LSP_GAMMA[c], beta=[0.33, 0.5, 0.75, 1.0], mu=[0.0001], eta=[0.9, 1.0])
    return [(b, queries) for b in builds(LSP_KEYS, LSP_BUILDS[c], compress_range="simdbp")]


# --- SINDI: around the configuration recommended in its paper ------------------------------------

SINDI_ALPHA = [0.2, 0.4, 0.6, 0.8, 1.0]
SINDI_BETA = [0.2, 0.4, 0.6, 0.8, 1.0]
SINDI_GAMMA = [50, 100, 150, 200, 300, 500, 700, 1000, 1500]


def sindi(c):
    beta = SINDI_BETA if c != "nq" else [0.08, 0.10, 0.15] + SINDI_BETA  # extended downward on NQ
    out = [(dict(lambda_window=60000, alpha=a, use_reorder=True), product(beta=beta, gamma=SINDI_GAMMA))
           for a in SINDI_ALPHA]
    if c == "nq":  # the window size, at the recommended alpha
        out += [(dict(lambda_window=w, alpha=0.6, use_reorder=True),
                 product(beta=SINDI_BETA, gamma=[100, 300, 1000])) for w in (18000, 30000, 120000)]
    return out


GRIDS = {"kannolo-hnsw": kannolo_hnsw, "kannolo-hnsw-rerank": kannolo_hnsw_rerank, "seismic": seismic,
         "seismic-wave": seismic_wave, "pyanns": pyanns, "lsp": lsp, "sindi": sindi}


def grid(system, collection):
    g = GRIDS[system](collection)
    if os.environ.get("SPARSE_SMOKE"):  # smoke test: the first build, three search points
        g = [(g[0][0], g[0][1][:3])]
    return g


# --- for analysis/grids_table.py ----------------------------------------------------------------

TABLE_ORDER = [("kannolo-hnsw", r"\kannolohnsw"), ("kannolo-hnsw-rerank", r"\kannolorerank"),
               ("seismic", r"\seismic"), ("seismic-wave", r"\seismicw"), ("sindi", r"\sindi"),
               ("lsp", r"\lsp"), ("pyanns", r"\pyann")]
SHORT = {"msmarco-splade": r"\msmarcosplade", "nq": r"\nq", "msmarco-esplade": r"\msmarcoesplade"}


def _sorted(values):
    return sorted(set(values), key=lambda v: (v is not None, v if not isinstance(v, tuple) else v))


def _arms(dicts):
    """Split a set of configurations into the fewest full products ("arms") of per-key value lists."""
    keys = list(dicts[0])
    axes = {k: _sorted(d[k] for d in dicts) for k in keys}
    n = 1
    for v in axes.values():
        n *= len(v)
    if n == len({tuple(d[k] for k in keys) for d in dicts}):
        return [axes]
    best = None
    for k in keys:
        if len(axes[k]) < 2:
            continue
        groups = {}
        for d in dicts:
            groups.setdefault(d[k], []).append({x: d[x] for x in keys if x != k})
        merged = []  # (rest-arms, values of k)
        for value, rows in groups.items():
            rest = _arms(rows)
            for m in merged:
                if m[0] == rest:
                    m[1].append(value)
                    break
            else:
                merged.append((rest, [value]))
        arms = [dict({k: _sorted(vals)}, **arm) for rest, vals in merged for arm in rest]
        if best is None or len(arms) < len(best):
            best = arms
    return best


def _builds_text_arms(builds):
    """Build sets as product arms, or as one tuple list over the varying parameters when the
    set is irregular (more than three arms)."""
    arms = _arms(builds)
    if len(arms) <= 3:
        return arms
    keys = list(builds[0])
    varying = [k for k in keys if len({b[k] for b in builds}) > 1]
    fixed = {k: [builds[0][k]] for k in keys if k not in varying}
    return [dict({"(" + ", ".join(varying) + ")": sorted(tuple(b[k] for k in varying) for b in builds)}, **fixed)]


def table_blocks(system):
    """[(collections label, build arms, search arms)]: one entry per group of builds sharing a search
    grid, collections with identical grids merged."""
    per_c = []
    for c in COLLECTIONS:
        groups = []  # (builds, queries)
        for b, qq in grid(system, c):
            for g in groups:
                if g[1] == qq:
                    g[0].append(b)
                    break
            else:
                groups.append(([b], qq))
        per_c.append((c, [(_builds_text_arms(bs), _arms(qq)) for bs, qq in groups]))
    merged = []
    for c, entries in per_c:
        for m in merged:
            if m[1] == entries:
                m[0].append(c)
                break
        else:
            merged.append(([c], entries))
    return [(", ".join(SHORT[c] for c in cs) if i == 0 else "", ba, sa)
            for cs, entries in merged for i, (ba, sa) in enumerate(entries)]
