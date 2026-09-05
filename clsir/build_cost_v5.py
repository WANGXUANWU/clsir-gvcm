"""Cost-scaling table for Section S3, from results/v5/cost_runs.csv.

The same table build_cost.py produced from the v2 study, but reading the v5 run
and using the example numbering of build_v3, in which the paper's four examples
come first.  The final column is the ratio of the warm started fully iterated
estimator, the faster of its two implementations, to the linear update.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_v3 import EX, HEAD

SRC = Path("../results/v5/cost_runs.csv")
OUT = Path("../draft/tables")
MS = ["CLSIR-QS", "CLSIR-OS", "OSL", "LMLE-warm", "LMLE", "FZ", "FIRTH", "CGA"]


def main() -> None:
    cost = pd.read_csv(SRC)
    sizes = sorted(cost.grid_size.unique())
    present = [m for m in MS if m in set(cost.method)]

    ns = sorted(cost.n.unique()) if "n" in cost.columns else [None]
    L = [r"\begin{tabular}{lrr" + "r" * len(present) + "r}", r"\toprule",
         "Example & $n$ & $n_{\\rm grid}$ & " + " & ".join(HEAD[m] for m in present)
         + " & ratio" + r"\\", r"\midrule"]
    for dz in EX:
        if cost[cost.design == dz].empty:
            continue
        first = True
        for n in ns:
            sub = cost[(cost.design == dz)]
            if n is not None:
                sub = sub[sub.n == n]
            for k, g in enumerate(sizes):
                vals = {}
                for m in present:
                    v = sub[(sub.method == m) & (sub.grid_size == g)]["runtime"]
                    vals[m] = 1000 * float(v.mean()) if len(v) else np.nan
                cells = ["--" if not np.isfinite(vals[m]) else f"{vals[m]:.1f}"
                         for m in present]
                ratio = vals.get("LMLE-warm", np.nan) / vals.get("CLSIR-OS", np.nan)
                lead = f"{EX[dz][0]} ({EX[dz][1]})" if first else ""
                nlab = f"${n}$" if (k == 0 and n is not None) else ""
                L.append(f"{lead} & {nlab} & {g} & " + " & ".join(cells)
                         + (f" & {ratio:.2f}" if np.isfinite(ratio) else " & --")
                         + r"\\")
                first = False
        L.append(r"\addlinespace")
    if L[-1] == r"\addlinespace":
        L.pop()
    L += [r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v5_cost.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote v5_cost.tex")

    print("\n== ratio of LMLE-warm to CLSIR-OS, by example and grid size ==")
    for dz in EX:
        sub = cost[cost.design == dz]
        if sub.empty:
            continue
        r = []
        for g in sizes:
            a = sub[(sub.method == "LMLE-warm") & (sub.grid_size == g)]["runtime"].mean()
            b = sub[(sub.method == "CLSIR-OS") & (sub.grid_size == g)]["runtime"].mean()
            r.append(a / b)
        print(f"  {EX[dz][0]:11s} " + "  ".join(f"{g}:{x:.2f}" for g, x in zip(sizes, r)))

    print("\n== absolute milliseconds at the largest grid ==")
    g = sizes[-1]
    for dz in EX:
        sub = cost[(cost.design == dz) & (cost.grid_size == g)]
        if sub.empty:
            continue
        m = 1000 * sub.groupby("method")["runtime"].mean()
        print(f"  {EX[dz][0]:11s} " + "  ".join(f"{k} {v:.0f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
