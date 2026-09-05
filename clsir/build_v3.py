"""LaTeX tables for the paper and its supplement.

Reads the Monte Carlo output in ../results/v3 and writes ../draft/tables/v3_*.tex.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("../draft/tables")
R = Path("../results")

HEAD = {"CLSIR-QS": "CLSIR-QS", "CLSIR-OS": "CLSIR-OS", "LMLE": "LMLE",
        "OSL": "OSL", "FZ": "FZ", "FIRTH": "Firth", "CGA": "CGA", "GA": "GA",
        "LC-OS": "LC-OS", "LMLE-warm": "LMLE$^{\\ast}$"}
# The four examples carried in the paper are numbered first; the two reported
# only in the supplement follow.  Dict order drives every table and figure.
EX = {"bin2": ("Example 1", "Bernoulli, $p=2$"),
      "bin10": ("Example 2", "Bernoulli, $p=10$"),
      "poi10": ("Example 3", "Poisson, $p=10$"),
      "gau2": ("Example 4", "Gaussian, $p=2$"),
      "poi2": ("Example 5", "Poisson, $p=2$"),
      "binskew": ("Example 6", "Bernoulli, non-elliptical")}
MAIN_DESIGNS = ["bin2", "bin10", "poi10", "gau2"]
SUPP_DESIGNS = ["poi2", "binskew"]
FAMILY = {"bin2": "binomial", "bin10": "binomial", "binskew": "binomial",
          "poi2": "poisson", "poi10": "poisson", "gau2": "gaussian"}

MAIN_M = ["CLSIR-QS", "CLSIR-OS", "LMLE", "OSL", "FZ", "FIRTH", "CGA"]
# the paired subset: the estimators that solve the same local likelihood and so
# may be read as paired differences.  The grouped estimators fail on almost
# every binary replication and are averaged over their own successes instead.
PAIR_M = ["CLSIR-QS", "CLSIR-OS", "LMLE", "OSL", "FZ"]
PAIRED_AVG = set(PAIR_M) | {"LC-OS", "FIRTH"}
# the Jeffreys penalty is a device for the binary response; for the other
# families the penalized and unpenalized local fits coincide and the row is
# suppressed rather than duplicated
BINARY_ONLY = {"FIRTH"}


def fmt(v, digits=3):
    if v is None or not np.isfinite(v):
        return "--"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.05:
        return f"{v:.3f}"
    return f"{v:.4f}"


def paired_keep(cell, methods=None):
    methods = PAIR_M if methods is None else methods
    piv = cell.pivot_table(index="seed", columns="method", values="blown", aggfunc="max")
    have = [m for m in methods if m in piv.columns]
    if not have:
        return set(piv.index)
    return set(piv.index[piv[have].max(axis=1) == 0])


def cellmean(sub, c, m, keep):
    v = sub[(sub.c == c) & (sub.method == m)]
    if m in PAIRED_AVG:
        v = v[v.seed.isin(keep)]
    v = v["ise"]
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def rows_for(dz, methods, present):
    out = []
    for m in methods:
        if m not in present:
            continue
        if m in BINARY_ONLY and FAMILY[dz] != "binomial":
            continue
        out.append(m)
    return out


def tab_ise(g, designs, methods, ns, cs, show_fail=True):
    """Rows: example x method.  Columns: sample size x bandwidth, plus failures."""
    present = set(g.method)
    ncol = len(ns) * len(cs) + (len(ns) if show_fail else 0)
    L = [r"\begin{tabular}{" + "ll" + "r" * ncol + "}", r"\toprule"]
    grp = [r"\multicolumn{%d}{c}{$n=%d$}" % (len(cs) + show_fail, n) for n in ns]
    L.append("& & " + " & ".join(grp) + r"\\")
    rule, start = [], 3
    for _ in ns:
        rule.append(r"\cmidrule(lr){%d-%d}" % (start, start + len(cs) - 1 + show_fail))
        start += len(cs) + show_fail
    L.append("".join(rule))
    hdr = []
    for _ in ns:
        hdr += [f"${c:.2f}$" for c in cs] + (["fail"] if show_fail else [])
    L.append("Example & Method & " + " & ".join(hdr) + r"\\")
    L.append(r"\midrule")
    for dz in designs:
        ms = rows_for(dz, methods, present)
        vals, fails = {}, {}
        for n in ns:
            sub = g[(g.design == dz) & (g.n == n)]
            for c in cs:
                keep = paired_keep(sub[sub.c == c])
                for m in ms:
                    vals[(n, c, m)] = cellmean(sub, c, m, keep)
            for m in ms:
                w = sub[sub.method == m]["blown"]
                fails[(n, m)] = 100.0 * float(w.mean()) if len(w) else np.nan
        best = {}
        for n in ns:
            for c in cs:
                fin = [vals[(n, c, m)] for m in ms if np.isfinite(vals[(n, c, m)])]
                best[(n, c)] = min(fin) if fin else np.nan
        for k, m in enumerate(ms):
            cells = []
            for n in ns:
                for c in cs:
                    v = vals[(n, c, m)]
                    s = fmt(v)
                    if np.isfinite(v) and np.isfinite(best[(n, c)]) \
                            and abs(v - best[(n, c)]) < 1e-12:
                        s = r"\textbf{" + s + "}"
                    cells.append(s)
                if show_fail:
                    f = fails[(n, m)]
                    cells.append("--" if not np.isfinite(f)
                                 else ("0" if f == 0 else f"{f:.0f}"))
            lead = f"{EX[dz][0]} ({EX[dz][1]})" if k == 0 else ""
            L.append(f"{lead} & {HEAD[m]} & " + " & ".join(cells) + r"\\")
        L.append(r"\addlinespace")
    if L[-1] == r"\addlinespace":
        L.pop()
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tab_reliability(g, designs, methods):
    """Failure rate and worst-case error, the evidence behind Proposition 1."""
    present = set(g.method)
    ns = sorted(g.n.unique())
    L = [r"\begin{tabular}{ll" + "rr" * len(ns) + "}", r"\toprule"]
    L.append("& & " + " & ".join(r"\multicolumn{2}{c}{$n=%d$}" % n for n in ns) + r"\\")
    L.append("".join(r"\cmidrule(lr){%d-%d}" % (3 + 2 * k, 4 + 2 * k)
                     for k in range(len(ns))))
    L.append("Example & Method & " + " & ".join(
        ["failures (\\%) & worst ISE"] * len(ns)) + r"\\")
    L.append(r"\midrule")
    for dz in designs:
        ms = rows_for(dz, methods, present)
        for k, m in enumerate(ms):
            cells = []
            for n in ns:
                s = g[(g.design == dz) & (g.n == n) & (g.method == m)]
                fail = 100.0 * float(s["blown"].mean()) if len(s) else np.nan
                worst = float(np.nanmax(s["ise"])) if len(s) else np.nan
                cells += ["--" if not np.isfinite(fail) else f"{fail:.1f}", fmt(worst)]
            lead = f"{EX[dz][0]}" if k == 0 else ""
            L.append(f"{lead} & {HEAD[m]} & " + " & ".join(cells) + r"\\")
        L.append(r"\addlinespace")
    if L[-1] == r"\addlinespace":
        L.pop()
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def tab_cv(cv, designs, methods, only_n=None):
    """One row per estimator: accuracy, selected bandwidth and wall clock cost."""
    present = set(cv.method)
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         "Example & Method & mean ISE & selected $c$ & final fit (s) "
         "& whole analysis (s) & fail" + r"\\",
         r"\midrule"]
    for dz in designs:
        for n in sorted(cv.n.unique()):
            if only_n is not None and n != only_n:
                continue
            sub = cv[(cv.design == dz) & (cv.n == n)]
            if sub.empty:
                continue
            ms = rows_for(dz, methods, present)
            keep = paired_keep(sub)
            ise = {}
            for m in ms:
                v = (sub[(sub.method == m) & (sub.seed.isin(keep))]["ise"]
                     if m in PAIRED_AVG or m == "LMLE-warm"
                     else sub[sub.method == m]["ise"].dropna())
                ise[m] = float(v.mean()) if len(v) else np.nan
            fin = [v for v in ise.values() if np.isfinite(v)]
            bi = min(fin) if fin else np.nan
            for k, m in enumerate(ms):
                b = sub[sub.method == m]
                cell = fmt(ise[m])
                if np.isfinite(ise[m]) and np.isfinite(bi) and abs(ise[m] - bi) < 1e-12:
                    cell = r"\textbf{" + cell + "}"
                fail = 100.0 * float(b["blown"].mean())
                lead = f"{EX[dz][0]} ({EX[dz][1]}), $n={n}$" if k == 0 else ""
                L.append(f"{lead} & {HEAD[m]} & {cell} & {b.c.mean():.2f} & "
                         f"{b.final_runtime.mean():.3f} & {b.runtime.mean():.2f} & "
                         f"{fail:.0f}" + r"\\")
            L.append(r"\addlinespace")
    if L[-1] == r"\addlinespace":
        L.pop()
    L += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(L)


def report(g, designs, ns, cs_all, methods):
    """Console summary of the claims the paper makes about the tables."""
    present = set(g.method)
    print("\n== best-in-cell counts ==")
    tally = {m: 0 for m in methods}
    tot = 0
    ours = 0
    for dz in designs:
        ms = rows_for(dz, methods, present)
        for n in ns:
            sub = g[(g.design == dz) & (g.n == n)]
            for c in cs_all:
                keep = paired_keep(sub[sub.c == c])
                v = {m: cellmean(sub, c, m, keep) for m in ms}
                fin = {m: x for m, x in v.items() if np.isfinite(x)}
                if not fin:
                    continue
                tot += 1
                win = min(fin, key=fin.get)
                tally[win] += 1
                if min(v.get("CLSIR-QS", np.inf), v.get("CLSIR-OS", np.inf)) \
                        <= min(fin.values()) + 1e-12:
                    ours += 1
    for m, k in sorted(tally.items(), key=lambda kv: -kv[1]):
        if k:
            print(f"   {m:10s} {k:3d} of {tot}")
    print(f"   one of the two proposed estimators is best in {ours} of {tot} cells "
          f"({100 * ours / max(tot, 1):.0f} per cent)")
    print("\n== failure rate (%) ==")
    print((100 * g.groupby(["design", "n", "method"])["blown"].mean())
          .unstack().round(1).to_string())
    print("\n== mean runtime per fit (ms) ==")
    print((1000 * g.groupby(["design", "n", "method"])["runtime"].mean())
          .unstack().round(1).to_string())


def load_grid(results: Path, extra: list[Path] | None = None) -> pd.DataFrame:
    """The bandwidth grid, with any separately run estimators merged in.

    The penalized fit is run on the Bernoulli designs only and lands in its own
    directory; the seeds are keyed on the design name rather than on its
    position in --designs, so the two runs are paired and can simply be
    concatenated.
    """
    frames = [pd.read_csv(results / "grid_runs.csv")]
    for e in extra or []:
        p = e / "grid_runs.csv"
        if p.exists():
            frames.append(pd.read_csv(p))
            print(f"  merged {p}")
    g = pd.concat(frames, ignore_index=True)
    # --results is read first and wins: a re-run of a subset of the estimators
    # lands in its own directory and supersedes the same rows in the earlier one
    before = len(g)
    g = g.drop_duplicates(subset=["design", "n", "seed", "c", "method"], keep="first")
    if len(g) < before:
        print(f"  dropped {before - len(g)} superseded rows")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=R / "v4")
    ap.add_argument("--extra", type=Path, nargs="*", default=[R / "v4_firth"])
    ap.add_argument("--prefix", default="v4", help="stem of the table files written")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gp = a.results / "grid_runs.csv"
    if not gp.exists():
        print("no grid results yet")
        return
    g = load_grid(a.results, a.extra)
    ns = sorted(g.n.unique())
    cs_main = [0.65, 0.80, 1.00, 1.30]
    cs_all = sorted(g.c.unique())
    designs_main = [d for d in MAIN_DESIGNS if d in set(g.design)]
    designs_all = [d for d in EX if d in set(g.design)]

    (OUT / f"{a.prefix}_ise_main.tex").write_text(
        # the failure counts live in the supplement's reliability table, so
        # Table 1 of the paper carries accuracy alone
        tab_ise(g, designs_main, MAIN_M, ns, cs_main, show_fail=False),
        encoding="utf-8")
    (OUT / f"{a.prefix}_ise_full.tex").write_text(
        tab_ise(g, designs_all, MAIN_M + ["GA", "LC-OS"], ns, cs_all),
        encoding="utf-8")
    (OUT / f"{a.prefix}_ise_extra.tex").write_text(
        tab_ise(g, [d for d in SUPP_DESIGNS if d in set(g.design)],
                MAIN_M + ["LC-OS"], ns, cs_main), encoding="utf-8")
    (OUT / f"{a.prefix}_reliability.tex").write_text(
        tab_reliability(g, designs_all, MAIN_M), encoding="utf-8")
    print(f"wrote {a.prefix}_ise_main / _ise_full / _ise_extra / _reliability")

    report(g, designs_main, ns, cs_all, MAIN_M)

    cp = a.results / "cv_runs.csv"
    if cp.exists():
        cv = pd.read_csv(cp)
        ms = [m for m in ["CLSIR-QS", "CLSIR-OS", "LMLE", "LMLE-warm", "OSL",
                          "FZ", "FIRTH", "CGA"] if m in set(cv.method)]
        designs = [d for d in EX if d in set(cv.design)]
        for n in sorted(cv.n.unique()):
            (OUT / f"{a.prefix}_cv_{n}.tex").write_text(
                tab_cv(cv, designs, ms, only_n=int(n)), encoding="utf-8")
        print("wrote " + ", ".join(f"{a.prefix}_cv_{n}.tex" for n in sorted(cv.n.unique())))
        print("\n== complete analysis, seconds and speed-up ==")
        for dz in designs:
            for n in sorted(cv.n.unique()):
                s = cv[(cv.design == dz) & (cv.n == n)]
                ours = float(s[s.method == "CLSIR-OS"]["runtime"].mean())
                cand = [float(s[s.method == m]["runtime"].mean())
                        for m in ("LMLE", "LMLE-warm") if m in set(s.method)]
                if cand and ours > 0:
                    print(f"  {EX[dz][0]}, n={n:5d}: {ours:5.2f} vs {min(cand):6.2f} "
                          f"-> {min(cand) / ours:5.2f}x")


if __name__ == "__main__":
    main()
