"""Budget pruning for the dense driver.

Each system's builds run in ascending order of its cost-monotone parameters. When a build exceeds
the budget B, every configuration that is at least as large in all of those parameters, and equal
in every other parameter, is skipped without being built. A parameter is listed as monotone only
where build time was measured to increase with it; everything else must match for one
configuration to dominate another, so an unlisted parameter can never widen the pruning.
"""

import json

# Argument positions after VIBE substitutes `base_args`, so index 0 is always the metric.
SYSTEMS = {
    # [metric, m, ef_construction, graph_type]
    "kannolo": {"monotone": (1, 2), "labels": ("m", "ef_construction")},
    # [metric, m, ef_construction, graph_type, encoder_spec(, rerank_bits)]
    "kannolo-rerank": {"monotone": (1, 2), "labels": ("m", "ef_construction")},
    "kannolo-rerank-ext8": {"monotone": (1, 2), "labels": ("m", "ef_construction")},
    # [metric, M, efConstruction]
    "hnswlib": {"monotone": (1, 2), "labels": ("M", "efConstruction")},
    # [metric, L, R, quant, search_quant, refine_quant]
    "glass": {"monotone": (1, 2), "labels": ("L", "R")},
    # [metric, degree_bound, EF, num_iter]
    "symphonyqg": {"monotone": (1, 2), "labels": ("degree_bound", "EF")},
    # [metric, graph_max_degree, alpha, window_size, reduced_dim]
    "vamana-leanvec(svs)": {"monotone": (1, 3), "labels": ("graph_max_degree", "window_size")},
    # [metric, num_clusters, M, efConstruction, total_bits, fast_quantization]; total_bits and
    # num_clusters leave the build time flat, so they are not a cost ordering.
    "rabitq-hnsw": {"monotone": (2, 3), "labels": ("M", "efConstruction")},
    # [metric, quantization_bits, n_clusters, global_dim, rank, train_size]; rank and n_clusters
    # were measured to invert, so only global_dim is monotone.
    "lorann": {"monotone": (3,), "labels": ("global_dim",)},
    "lorann-plain": {"monotone": (3,), "labels": ("global_dim",)},
}


def _key(args, idx):
    return tuple(args[i] for i in idx)


def _fixed_key(args, monotone):
    return tuple(json.dumps(a, sort_keys=True) if isinstance(a, (list, dict)) else a
                 for i, a in enumerate(args) if i not in monotone)


def dominates(x, y, monotone):
    """True if x is at least as large as y in every monotone parameter, and larger in one."""
    xs, ys = _key(x, monotone), _key(y, monotone)
    return all(a >= b for a, b in zip(xs, ys)) and xs != ys


def order(configs, system):
    monotone = SYSTEMS[system]["monotone"]
    return sorted(configs, key=lambda c: (_key(c, monotone), str(c)))


def plan(configs, system, build_time_of, budget_s):
    """Build cheapest first; skip the dominators of any build over budget.

    `build_time_of(config)` builds the configuration and returns its build time (or None).
    Returns (built, skipped): built = [(config, seconds, None)], skipped = [(config, blocker)].
    """
    monotone = SYSTEMS[system]["monotone"]
    built, skipped, over = [], [], []
    for cfg in order(configs, system):
        blocker = next((b for b in over if _fixed_key(cfg, monotone) == _fixed_key(b, monotone)
                        and dominates(cfg, b, monotone)), None)
        if blocker is not None:
            skipped.append((cfg, blocker))
            continue
        secs = build_time_of(cfg)
        built.append((cfg, secs, None))
        if secs is not None and secs > budget_s:
            over.append(cfg)
    return built, skipped
