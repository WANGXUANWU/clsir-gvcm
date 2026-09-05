"""Tables for the two studies this revision adds: coverage, and size and power."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("../draft/tables")
EX = {"bin2": "Example 1", "bin10": "Example 2", "poi10": "Example 3",
      "gau2": "Example 4", "poi2": "Example 5", "binskew": "Example 6"}


def coverage() -> None:
    src = Path("../results/v6_coverage/coverage.csv")
    if not src.exists():
        print("  (no coverage results yet)")
        return
    df = pd.read_csv(src)
    ns = sorted(df.n.unique())
    L = [r"\begin{tabular}{l" + "rr" * len(ns) + "}", r"\toprule",
         "& " + " & ".join(r"\multicolumn{2}{c}{$n=%d$}" % n for n in ns) + r"\\"]
    L.append("".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * k, 3 + 2 * k)
                     for k in range(len(ns))))
    L.append("Example & " + " & ".join(["CLSIR-OS & CLSIR-QS"] * len(ns)) + r"\\")
    L.append(r"\midrule")
    for dz in EX:
        sub = df[df.design == dz]
        if sub.empty:
            continue
        cells = []
        for n in ns:
            s = sub[sub.n == n]
            cells += [f"{s.cov_os.mean():.3f}", f"{s.cov_qs.mean():.3f}"]
        L.append(f"{EX[dz]} & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "v5_coverage.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote v5_coverage.tex")
    print(df.groupby(["design", "n"])[["cov_os", "cov_qs"]].mean().round(3).to_string())


def power() -> None:
    src = Path("../results/v6_power/power.csv")
    if not src.exists():
        print("  (no power results yet)")
        return
    df = pd.read_csv(src)
    avals = sorted(df.a.unique())
    L = [r"\begin{tabular}{l" + "r" * len(avals) + "}", r"\toprule",
         r"& \multicolumn{%d}{c}{departure $a$}\\" % len(avals),
         r"\cmidrule(lr){2-%d}" % (1 + len(avals)),
         "Coefficient & " + " & ".join(f"${a:g}$" for a in avals) + r"\\",
         r"\midrule"]
    for lvl, tag in ((0.05, "0.05"), (0.10, "0.10")):
        L.append(r"\multicolumn{%d}{l}{\emph{nominal level %s}}\\"
                 % (1 + len(avals), tag))
        for k in sorted(df.coef.unique()):
            s = df[df.coef == k]
            cells = [f"{(s[s.a == a].p <= lvl).mean():.3f}" for a in avals]
            name = f"$a_{k}$" + (r" (under test)" if k == 2 else "")
            L.append(f"{name} & " + " & ".join(cells) + r"\\")
        L.append(r"\addlinespace")
    if L[-1] == r"\addlinespace":
        L.pop()
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "v5_power.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote v5_power.tex")
    print((df.assign(rej=df.p <= 0.05).groupby(["a", "coef"])["rej"].mean()
           .unstack().round(3)).to_string())


if __name__ == "__main__":
    coverage()
    power()
