"""Locate the best pilot ratio c1 in every design, over a wider grid than tuning_ext.

tuning_ext stopped at c1 = 3.5 and covered three designs.  The integrated squared
error was still falling at the end of that grid in both Bernoulli designs and
rising in the Poisson one, which is the pattern a Fisher-information argument
predicts: a Bernoulli observation carries at most a quarter of a unit of
information about the linear predictor, a count carries the mean, so the pilot
window has to be wider for a binary response to reach the same precision.  This
run checks that reading on all six designs and both sample sizes, and finds where
each family turns.

Run:  python tuning_wide.py [reps] [workers]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clsir_study as S
from clsir_core import LocalIndex, fit_clsir_os, slice_labels

C1_GRID = [1.5, 2.2, 3.0, 4.0, 5.5]
C2 = 1.2
CS = [0.65, 0.80, 1.00, 1.30]
DESIGNS = ["bin2", "bin10", "poi2", "poi10", "gau2", "binskew"]
NS = [500, 2000]
OUT = Path("../results/v5_tuning")


def one(job):
    dz, n, rep = job
    d = S.DESIGNS[dz]
    h0 = S.base_bandwidth(n, d)
    grid = np.linspace(0.10, 0.90, 101)
    rng = np.random.default_rng(91_000 + rep)
    u, x, y = S.simulate(rng, n, d)
    idx = LocalIndex(u, x, y, slice_labels(y, d.family, 5))
    out = []
    for c in CS:
        for c1 in C1_GRID:
            for quad in (False, True):
                e = fit_clsir_os(idx, grid, c * h0, d.family, pilot_ratio=c1,
                                 smooth_ratio=C2, quadratic=quad).estimate
                out.append({"design": dz, "n": n, "rep": rep, "c": c, "c1": c1,
                            "method": "CLSIR-QS" if quad else "CLSIR-OS",
                            "ise": S.evaluate(e, grid, d, 7)["ise"]})
    return out


def main(reps: int = 100, workers: int = 6) -> None:
    jobs = [(dz, n, r) for dz in DESIGNS for n in NS for r in range(reps)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(one, jobs, chunksize=2), 1):
            rows.extend(res)
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} replications", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "tuning_wide.csv", index=False)

    pd.set_option("display.width", 250)
    for meth in ("CLSIR-OS", "CLSIR-QS"):
        piv = (df[df.method == meth]
               .pivot_table(index=["design", "n"], columns="c1", values="ise"))
        print(f"\n=== {meth}: mean ISE by c1 (averaged over the four bandwidths) ===")
        print(piv.round(5).to_string())
        print("  relative to c1 = 1.5 (values below 1 are better):")
        print(piv.div(piv[1.5], axis=0).round(3).to_string())


if __name__ == "__main__":
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 100,
         workers=int(sys.argv[2]) if len(sys.argv) > 2 else 6)
