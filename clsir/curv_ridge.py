"""Two robustness checks on the quadratic update of Section 2.5.

(1) The bandwidth at which the curvature row of the smoothing pass is read.
    Theorem 4 says it should not matter, the curvature block being multiplied
    by h^2 before it enters the initial value condition.

(2) The ridge.  Both updates solve a ridged linear system, and the quadratic
    one has a larger parameter vector, so the comparison is repeated with the
    ridge switched off in the designs where the unridged local fit is defined.

Writes ../draft/tables/v5_curv.tex and prints the numbers.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PY = sys.executable
OUT = HERE / "../draft/tables"
EXN = {"bin2": "Example 1", "bin10": "Example 2", "poi2": "Example 3",
       "poi10": "Example 4"}


def run(tag, designs, methods, reps, no_ridge):
    out = HERE / f"../results/{tag}"
    if (out / "grid_runs.csv").exists():
        print(f"  {tag}: using existing results")
        return pd.read_csv(out / "grid_runs.csv")
    cmd = [PY, "clsir_study.py", "grid", "--reps", str(reps),
           "--n", "500", "2000", "--designs", *designs,
           "--c-grid", "0.5", "0.8", "1.3", "--grid-size", "101",
           "--methods", *methods, "--workers", "6", "--out", str(out)]
    if no_ridge:
        cmd.append("--no-ridge")
    print("  " + " ".join(cmd[1:]))
    env = dict(os.environ)
    if no_ridge:
        env["CLSIR_NO_RIDGE"] = "1"
    subprocess.run(cmd, cwd=HERE, check=True, env=env)
    return pd.read_csv(out / "grid_runs.csv")


def block(g, designs, methods, labels):
    ns = sorted(g.n.unique())
    cs = sorted(g.c.unique())
    rows = []
    for dz in designs:
        for k, m in enumerate(methods):
            cells = []
            for n in ns:
                for c in cs:
                    v = g[(g.design == dz) & (g.n == n) & (g.c == c)
                          & (g.method == m)]["ise"]
                    v = v[np.isfinite(v)]
                    cells.append(f"{v.mean():.4g}" if len(v) else "--")
            lead = EXN.get(dz, dz) if k == 0 else ""
            rows.append(f"{lead} & {labels[m]} & " + " & ".join(cells) + r"\\")
        rows.append(r"\addlinespace")
    if rows[-1] == r"\addlinespace":
        rows.pop()
    return rows, ns, cs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1/2] curvature bandwidth")
    d1 = ["bin2", "bin10", "poi2", "poi10"]
    g1 = run("v4_curv", d1, ["CLSIR-QS", "CLSIR-QR20", "CLSIR-QR30"], a.reps, False)
    lab1 = {"CLSIR-QS": "$b_2=b$", "CLSIR-QR20": "$b_2=2b$",
            "CLSIR-QR30": "$b_2=3b$"}

    print("[2/2] ridge switched off")
    d2 = ["poi2", "poi10"]
    g2 = run("v4_noridge", d2, ["CLSIR-OS", "CLSIR-QS"], a.reps, True)
    lab2 = {"CLSIR-OS": "CLSIR-OS", "CLSIR-QS": "CLSIR-QS"}

    r1, ns, cs = block(g1, d1, list(lab1), lab1)
    r2, _, _ = block(g2, d2, list(lab2), lab2)
    ncol = len(ns) * len(cs)
    L = [r"\begin{tabular}{ll" + "r" * ncol + "}", r"\toprule",
         "& & " + " & ".join(r"\multicolumn{%d}{c}{$n=%d$}" % (len(cs), n)
                             for n in ns) + r"\\",
         "".join(r"\cmidrule(lr){%d-%d}" % (3 + len(cs) * k, 2 + len(cs) * (k + 1))
                 for k in range(len(ns))),
         "Example & Setting & " + " & ".join(
             f"${c:.2f}$" for _ in ns for c in cs) + r"\\",
         r"\midrule",
         r"\multicolumn{%d}{l}{\emph{Bandwidth at which the curvature is read}}\\"
         % (ncol + 2)]
    L += r1
    L += [r"\midrule",
          r"\multicolumn{%d}{l}{\emph{Ridge switched off}}\\" % (ncol + 2)]
    L += r2
    L += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "v5_curv.tex").write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT / 'v5_curv.tex'}")

    print("\n== curvature bandwidth, mean ISE ==")
    print(g1.pivot_table(index=["design", "n", "c"], columns="method",
                         values="ise", aggfunc="mean").round(5).to_string())
    print("\n== ridge off, mean ISE ==")
    print(g2.pivot_table(index=["design", "n", "c"], columns="method",
                         values="ise", aggfunc="mean").round(5).to_string())
    print("\n== ridge off, failure rate ==")
    print(g2.pivot_table(index=["design", "n"], columns="method",
                         values="blown", aggfunc="mean").round(3).to_string())


if __name__ == "__main__":
    main()
