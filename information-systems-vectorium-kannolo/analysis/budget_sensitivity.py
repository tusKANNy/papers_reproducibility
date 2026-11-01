#!/usr/bin/env python3
"""Re-select one collection at several budgets (Appendix A.1 of the paper).

Configurations skipped without being built at the nominal B (they dominate an over-budget build)
were never measured, so a larger budget can only re-admit builds that were measured over B.

    python analysis/budget_sensitivity.py --results results/dense/dpr.jsonl --budget-s 5400
"""

import argparse

import select_configs as sc
import style as st


def order_at(sel, t):
    table = sel["table2"]
    return sorted((a for a in table if t in table[a]), key=lambda a: -table[a][t]["qps"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--budget-s", type=float, required=True)
    ap.add_argument("--multipliers", type=float, nargs="+", default=[0.5, 1, 2, 4])
    ap.add_argument("--series", choices=sorted(sc.SERIES), default="dense")
    args = ap.parse_args()

    rows = sc.load(args.results)
    budgets = [args.budget_s * m for m in args.multipliers]
    sels = {b: sc.select(rows, b, sc.SERIES[args.series]) for b in budgets}
    algos = [a for a in sc.SERIES[args.series] if any(a in s["table2"] for s in sels.values())]
    print(f"# Budget sensitivity -- {rows[0]['dataset']}, nominal B = {args.budget_s:.0f} s\n")
    for t in map(str, sc.TABLE_TARGETS):
        print(f"## Accuracy@10 >= {t}%\n")
        print("| system | " + " | ".join(f"{m:g}B" for m in args.multipliers) + " |")
        print("|---|" + "---:|" * len(budgets))
        for a in algos:
            cells = [f"{sels[b]['table2'].get(a, {})[t]['qps']:.0f}" if t in sels[b]["table2"].get(a, {})
                     else "--" for b in budgets]
            print(f"| {st.DISPLAY[a]} | " + " | ".join(cells) + " |")
        base = order_at(sels[args.budget_s], t)
        for b in budgets:
            o = order_at(sels[b], t)
            print(f"- B = {b:.0f} s ({'same order as B' if o == base else 'ORDER CHANGES'}): "
                  + " > ".join(st.DISPLAY[a] for a in o))
        print()


if __name__ == "__main__":
    main()
