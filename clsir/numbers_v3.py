"""Every number the text of Sections 5 and S3 quotes, printed in one place.

Run after the studies in ../results/v4 are complete.  Nothing here is written
into the draft automatically: the point is to have the quantities in front of
one while writing, so that no sentence claims something the runs do not show.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import build_v3 as B

R = Path("../results/v4")
EXN = {k: v[0] for k, v in B.EX.items()}


def sec(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def grid_numbers(g):
    ns = sorted(g.n.unique())
    cs = sorted(g.c.unique())
    designs = [d for d in B.EX if d in set(g.design)]
    present = set(g.method)

    sec("1. mean ISE, every cell")
    for dz in designs:
        ms = B.rows_for(dz, B.MAIN_M, present)
        for n in ns:
            sub = g[(g.design == dz) & (g.n == n)]
            row = {}
            for c in cs:
                keep = B.paired_keep(sub[sub.c == c])
                row[c] = {m: B.cellmean(sub, c, m, keep) for m in ms}
            df = pd.DataFrame(row).T
            print(f"\n-- {EXN[dz]} ({B.EX[dz][1]}), n={n}")
            print(df.round(4).to_string())
            best = df.idxmin(axis=1)
            print("   best: " + ", ".join(f"c={c:g}:{best[c]}" for c in cs))

    sec("2. who is best, and by how much")
    tally = {}
    tot = 0
    ours_cells = 0
    for dz in designs:
        ms = B.rows_for(dz, B.MAIN_M, present)
        for n in ns:
            sub = g[(g.design == dz) & (g.n == n)]
            for c in cs:
                keep = B.paired_keep(sub[sub.c == c])
                v = {m: B.cellmean(sub, c, m, keep) for m in ms}
                fin = {m: x for m, x in v.items() if np.isfinite(x)}
                if not fin:
                    continue
                tot += 1
                w = min(fin, key=fin.get)
                tally[w] = tally.get(w, 0) + 1
                best_ours = min(v.get("CLSIR-QS", np.inf), v.get("CLSIR-OS", np.inf))
                if best_ours <= min(fin.values()) + 1e-12:
                    ours_cells += 1
    print(f"  cells: {tot}")
    for m, k in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"    {m:10s} best in {k:3d}  ({100*k/tot:.0f}%)")
    print(f"  one of ours best in {ours_cells} of {tot} ({100*ours_cells/tot:.0f}%)")

    for sel, lab in ((["bin2", "bin10", "binskew"], "Bernoulli"),
                     (["poi2", "poi10"], "Poisson")):
        sel = [d for d in sel if d in designs]
        t = o = 0
        for dz in sel:
            ms = B.rows_for(dz, B.MAIN_M, present)
            for n in ns:
                sub = g[(g.design == dz) & (g.n == n)]
                for c in cs:
                    keep = B.paired_keep(sub[sub.c == c])
                    v = {m: B.cellmean(sub, c, m, keep) for m in ms}
                    fin = {m: x for m, x in v.items() if np.isfinite(x)}
                    if not fin:
                        continue
                    t += 1
                    if min(v.get("CLSIR-QS", np.inf),
                           v.get("CLSIR-OS", np.inf)) <= min(fin.values()) + 1e-12:
                        o += 1
        print(f"  {lab}: ours best in {o} of {t}")

    sec("3. the crossing between the linear and the quadratic update")
    for dz in designs:
        for n in ns:
            sub = g[(g.design == dz) & (g.n == n)]
            r = []
            for c in cs:
                keep = B.paired_keep(sub[sub.c == c])
                a = B.cellmean(sub, c, "CLSIR-QS", keep)
                b = B.cellmean(sub, c, "CLSIR-OS", keep)
                r.append(a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else np.nan)
            r = np.array(r)
            cross = next((f"{cs[i]:g}" for i in range(len(cs)) if r[i] < 1.0), "never")
            print(f"  {EXN[dz]:10s} n={n:5d}  QS/OS = " +
                  " ".join(f"{x:.2f}" for x in r) + f"   first c with QS better: {cross}")

    sec("4. failure rate (per cent of replications without a finite curve)")
    print((100 * g.groupby(["design", "n", "method"])["blown"].mean())
          .unstack().round(2).to_string())

    sec("5. worst observed ISE")
    print(g.groupby(["design", "n", "method"])["ise"].max().unstack().round(2).to_string())

    sec("6. mean runtime per fit, milliseconds")
    print((1000 * g.groupby(["design", "n", "method"])["runtime"].mean())
          .unstack().round(1).to_string())

    sec("7. total fits behind the study, and failures of ours")
    for m in ("CLSIR-QS", "CLSIR-OS"):
        s = g[g.method == m]
        print(f"  {m}: {len(s)} fits, {int(s['blown'].sum())} failures")


def cv_numbers(cv):
    sec("8. complete analysis with a cross-validated bandwidth")
    designs = [d for d in B.EX if d in set(cv.design)]
    ms = [m for m in ["CLSIR-QS", "CLSIR-OS", "LMLE", "LMLE-warm", "OSL", "FZ",
                      "FIRTH", "CGA"] if m in set(cv.method)]
    best_count = {}
    for dz in designs:
        for n in sorted(cv.n.unique()):
            sub = cv[(cv.design == dz) & (cv.n == n)]
            if sub.empty:
                continue
            keep = B.paired_keep(sub)
            ise = {}
            for m in ms:
                v = (sub[(sub.method == m) & (sub.seed.isin(keep))]["ise"]
                     if m in B.PAIRED_AVG or m == "LMLE-warm"
                     else sub[sub.method == m]["ise"].dropna())
                ise[m] = float(v.mean()) if len(v) else np.nan
            fin = {m: v for m, v in ise.items() if np.isfinite(v)}
            w = min(fin, key=fin.get) if fin else "--"
            best_count[w] = best_count.get(w, 0) + 1
            ours = float(sub[sub.method == "CLSIR-OS"]["runtime"].mean())
            cand = [float(sub[sub.method == m]["runtime"].mean())
                    for m in ("LMLE", "LMLE-warm") if m in set(sub.method)]
            sp = min(cand) / ours if cand and ours > 0 else np.nan
            print(f"  {EXN[dz]:10s} n={n:5d} best={w:9s} "
                  + " ".join(f"{m}={ise[m]:.4g}" for m in ms if np.isfinite(ise[m]))
                  + f" | analysis {ours:.2f}s vs {min(cand):.2f}s -> {sp:.2f}x")
    print("  best-in-cell: " + ", ".join(f"{k} {v}" for k, v in best_count.items()))
    print("\n  selected bandwidth constant, mean")
    print(cv.groupby(["design", "n", "method"])["c"].mean().unstack().round(2).to_string())
    print("\n  whole-analysis seconds")
    print(cv.groupby(["design", "n", "method"])["runtime"].mean().unstack().round(2).to_string())


def cost_numbers(cost):
    sec("9. cost scaling in the number of evaluation points (ms)")
    t = (1000 * cost.groupby(["design", "grid_size", "method"])["runtime"].mean()).unstack()
    print(t.round(1).to_string())
    print("\n  ratio of the warm started fully iterated estimator to CLSIR-OS")
    for dz in sorted(set(cost.design)):
        s = t.loc[dz]
        if "LMLE-warm" in s.columns and "CLSIR-OS" in s.columns:
            print(f"   {dz}: " + " ".join(
                f"{g}:{s.loc[g,'LMLE-warm']/s.loc[g,'CLSIR-OS']:.2f}" for g in s.index))
    print("\n  iterations per fit")
    print(cost.groupby(["design", "grid_size", "method"])["iterations"].mean()
          .unstack().round(0).to_string())


def parallel_numbers(par):
    sec("10. parallel scaling of the update")
    for dz in sorted(set(par.design)):
        s = par[par.design == dz].groupby("workers")["runtime"].mean()
        base = float(s.iloc[0])
        print(f"  {dz}: " + "  ".join(
            f"w={int(w)} {base/v:.2f}x" for w, v in s.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=R)
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="further result directories to merge, --results winning")
    a = ap.parse_args()
    if (a.results / "grid_runs.csv").exists():
        grid_numbers(B.load_grid(a.results, a.extra))
    if (a.results / "cv_runs.csv").exists():
        cv_numbers(pd.read_csv(a.results / "cv_runs.csv"))
    if (a.results / "cost_runs.csv").exists():
        cost_numbers(pd.read_csv(a.results / "cost_runs.csv"))
    if (a.results / "parallel_runs.csv").exists():
        parallel_numbers(pd.read_csv(a.results / "parallel_runs.csv"))


if __name__ == "__main__":
    main()
