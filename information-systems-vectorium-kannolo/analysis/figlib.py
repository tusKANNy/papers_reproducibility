"""Helpers shared by the figure and table scripts."""

import matplotlib.ticker as mticker

MIB = 1024.0
# Plain-number tick ladder for log axes, and the minimum number of labels an axis needs.
LOG_TICKS = [100, 200, 300, 500, 1000, 2000, 3000, 5000, 10000, 20000, 30000, 50000]
MIN_LOG_TICKS = 4


def name_log_ticks(ax, axis, values, pad_lo=0.8, pad_hi=1.25, ladder=None, fine=None):
    """Put plain-number ticks on a log axis, only inside the data range.

    `fine` is a denser ladder. It is used only when the primary ladder puts fewer than
    MIN_LOG_TICKS labels inside the data range, that is, when the axis is narrow. One fixed ladder
    does not fit both a 2.7-decade dense axis, where four labels per decade collide, and a
    0.7-decade sparse axis, where a 1-3 ladder gives two labels for the whole figure.
    """
    lo, hi = min(values) * pad_lo, max(values) * pad_hi
    ticks = [t for t in (ladder or LOG_TICKS) if lo <= t <= hi]
    if fine and len(ticks) < MIN_LOG_TICKS:
        finer = [t for t in fine if lo <= t <= hi]
        if len(finer) > len(ticks):
            ticks = finer
    if len(ticks) < 2:
        return
    setter, minor = ((ax.set_xticks, ax.xaxis) if axis == 'x'
                     else (ax.set_yticks, ax.yaxis))
    setter(ticks)
    (ax.set_xticklabels if axis == 'x' else ax.set_yticklabels)([str(t) for t in ticks])
    minor.set_minor_formatter(mticker.NullFormatter())


def _footprint_field(table, algos, targets):
    have_delta = have_child = False
    for algo in algos:
        for t in targets:
            row = table[algo].get(str(t))
            if row is None:
                continue
            if row.get("uss_delta_kb") is not None:
                have_delta = True
            elif row.get("uss_kb_search_child") is not None:
                have_child = True
    if have_delta and have_child:
        raise SystemExit(
            "the selection mixes footprints from in-process tuning (uss_delta_kb, a delta "
            "around construction) with footprints from command-line timing (uss_kb_search_child, "
            "an absolute RSS that includes the CLI runtime). On one axis, a difference in timing "
            "method would look like a footprint difference. Select rows of one timing method "
            "only, or re-time the in-process systems with command-line timing.")
    if have_delta:
        return "uss_delta_kb", "USS delta around construction"
    if have_child:
        return "uss_kb_search_child", "search-process RSS (absolute, includes the CLI runtime)"
    raise SystemExit("no row carries a footprint field.")
