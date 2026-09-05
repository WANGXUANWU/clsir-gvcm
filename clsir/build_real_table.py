"""The data-application table for Section 5.2, from results/real/saheart.npz.

This replaces the four diagnostic panels that used to sit under the fitted
coefficients: they had four different vertical scales and no common horizontal
axis, and everything they carried is a single number per estimator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SRC = Path("../results/real/saheart.npz")
OUT = Path("../draft/tables/v5_real.tex")
HEAD = {"CLSIR-QS": "CLSIR-QS", "CLSIR-OS": "CLSIR-OS", "LMLE": "LMLE",
        "OSL": "OSL", "FZ": "FZ", "FIRTH": "Firth", "CGA": "CGA"}
ORDER = ["CLSIR-OS", "CLSIR-QS", "FIRTH", "OSL", "LMLE", "FZ", "CGA"]


def main() -> None:
    d = dict(np.load(SRC, allow_pickle=True))
    methods = [str(s) for s in d["methods"]]
    hg = np.asarray(d["hgrid"], float)
    sel = np.asarray(d["sel"], float)
    times = 1000.0 * np.asarray(d["times"], float)

    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Method & selected $h$ & deviance & ages with no fit & ms per fit\\",
         r"\midrule"]
    best = min(float(np.nanmin(d[f"cv_{m}"])) for m in methods)
    for m in ORDER:
        if m not in methods:
            continue
        i = methods.index(m)
        cv = np.asarray(d[f"cv_{m}"], float)
        dev = float(np.nanmin(cv))
        div = np.asarray(d.get(f"div8_{m}", np.zeros_like(hg)), float)
        worst = int(np.nanmax(div))
        devs = f"{dev:.4f}"
        if abs(dev - best) < 1e-12:
            devs = r"\textbf{" + devs + "}"
        L.append(f"{HEAD[m]} & ${sel[i]:.2f}$ & {devs} & "
                 f"{'0' if worst == 0 else worst} & {times[i]:.0f}" + r"\\")
    L.append(r"\addlinespace")
    L.append(r"Constant coefficients & -- & "
             + f"{float(d['base']):.4f}" + r" & 0 & --\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")

    print("\n== five fold deviance at the selected bandwidth ==")
    for m in ORDER:
        if m in methods:
            print(f"  {m:9s} {float(np.nanmin(d[f'cv_{m}'])):.4f}")
    print(f"  constant  {float(d['base']):.4f}")
    print("\n== worst number of ages with no fit, eight covariate model ==")
    for m in ORDER:
        if m in methods:
            print(f"  {m:9s} {int(np.nanmax(d[f'div8_{m}']))}")


if __name__ == "__main__":
    main()
