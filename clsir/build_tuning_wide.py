"""The pilot-ratio sensitivity table for Section S3, from tuning_wide.csv.

Rows are example by sample size, columns the candidate pilot ratios.  The
column headed by the adopted default carries the mean integrated squared error
itself; every other column carries the percentage change from it, so a positive
entry is a loss.  The last column is the diagnostic that explains the pattern:
the Fisher information the pilot window carries per parameter it must estimate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clsir_study as S
from clsir_core import default_pilot_ratio

EX = {"bin2": "Example 1", "bin10": "Example 2", "poi10": "Example 3",
      "gau2": "Example 4", "poi2": "Example 5", "binskew": "Example 6"}
DESC = {"bin2": "Bernoulli, $p=2$", "bin10": "Bernoulli, $p=10$",
        "poi10": "Poisson, $p=10$", "gau2": "Gaussian, $p=2$",
        "poi2": "Poisson, $p=2$", "binskew": "Bernoulli, non-elliptical"}
ORDER = list(EX)
C1 = [1.5, 2.2, 3.0, 4.0, 5.5]
SRC = Path("../results/v5_tuning/tuning_wide.csv")
OUT = Path("../draft/tables/v5_tuning.tex")


def info_per_parameter(dz: str, n: int, c1: float = 1.5) -> float:
    """Fisher information in the pilot window, per pilot parameter, at c = 0.8.

    Evaluated at a common reference ratio rather than at each family's adopted
    default, so that the column is a property of the design alone and the rows
    may be compared with one another.
    """
    d = S.DESIGNS[dz]
    rng = np.random.default_rng(5)
    u, x, _ = S.simulate(rng, n, d)
    eta = np.sum(x * S.coefficients(u, d), axis=1)
    if d.family == "binomial":
        mu = 1.0 / (1.0 + np.exp(-eta))
        rho = mu * (1.0 - mu)
    elif d.family == "poisson":
        rho = np.exp(eta)
    else:
        rho = np.ones_like(eta)
    h1 = c1 * 0.80 * S.base_bandwidth(n, d)
    return 2.0 * n * h1 * float(rho.mean()) / (d.p + 2)


def fmt_abs(v: float) -> str:
    a = abs(v)
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.05:
        return f"{v:.3f}"
    return f"{v:.4f}"


def main() -> None:
    df = pd.read_csv(SRC)
    piv = df.pivot_table(index=["method", "design", "n"], columns="c1", values="ise")

    L = [r"\begin{tabular}{llr" + "r" * len(C1) + r"r}", r"\toprule"]
    L.append(r"& & & \multicolumn{%d}{c}{pilot ratio $c_1$} & \\" % len(C1))
    L.append(r"\cmidrule(lr){4-%d}" % (3 + len(C1)))
    L.append(r"Example & Method & $n$ & "
             + " & ".join(f"${c}$" for c in C1)
             + r" & info./par.\\")
    L.append(r"\midrule")
    for dz in ORDER:
        d = S.DESIGNS[dz]
        c1_star = default_pilot_ratio(d.family)
        first = True
        for meth in ("CLSIR-OS", "CLSIR-QS"):
            for n in (500, 2000):
                key = (meth, dz, n)
                if key not in piv.index:
                    continue
                row = piv.loc[key]
                base = row[c1_star]
                cells = []
                for c in C1:
                    v = row.get(c, np.nan)
                    if not np.isfinite(v):
                        cells.append("--")
                    elif abs(c - c1_star) < 1e-9:
                        cells.append(r"\textbf{" + fmt_abs(v) + "}")
                    else:
                        cells.append(f"{100.0 * (v / base - 1.0):+.0f}")
                lead = f"{EX[dz]} ({DESC[dz]})" if first else ""
                mlab = meth if n == 500 else ""
                cells.append(f"{info_per_parameter(dz, n):.0f}"
                             if meth == "CLSIR-OS" else "")
                L.append(f"{lead} & {mlab} & ${n}$ & " + " & ".join(cells) + r"\\")
                first = False
        L.append(r"\addlinespace")
    if L[-1] == r"\addlinespace":
        L.pop()
    L += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")

    # the summary the supplement quotes in prose
    print("\n== change from the adopted default, per family ==")
    for meth in ("CLSIR-OS", "CLSIR-QS"):
        for dz in ORDER:
            c1_star = default_pilot_ratio(S.DESIGNS[dz].family)
            for n in (500, 2000):
                key = (meth, dz, n)
                if key not in piv.index:
                    continue
                row = piv.loc[key]
                alt = row[1.5]
                print(f"  {meth} {EX[dz]:10s} n={n:5d}: c1*={c1_star} "
                      f"ISE {row[c1_star]:.4f}; against a flat c1=1.5 "
                      f"{100 * (row[c1_star] / alt - 1):+.1f}%")


if __name__ == "__main__":
    main()
