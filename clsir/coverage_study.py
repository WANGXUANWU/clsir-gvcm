"""Pointwise coverage of the two intervals, in every example.

Section 3 of the paper builds a pointwise interval on the limit distribution and
notes that at a mean squared error optimal bandwidth the leading bias and the
standard error are of the same order, so the interval centred on the linear
update must undercover.  Theorem S1 offers the repair at no extra cost: centre
on the quadratic update and replace nu0 by nu0*, which leaves a centring error
of order h^4.  verify_theory.py checks this at one design and one evaluation
point; this script checks it in every example, at both sample sizes, and
averaged over the interior of the index range.

Run:  python coverage_study.py [reps] [workers]
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
from clsir_core import (LocalIndex, NU0_EPAN, NU0Q_EPAN, fit_clsir_os,
                        sandwich_variance, slice_labels)

DESIGNS = ["bin2", "bin10", "poi10", "gau2", "poi2", "binskew"]
NS = [500, 2000]
C = 0.80                      # the bandwidth constant the paper displays
GRID = np.linspace(0.15, 0.85, 21)
Z = 1.959963985
OUT = Path("../results/v6_coverage")


def one(job):
    dz, n, rep = job
    d = S.DESIGNS[dz]
    h = C * S.base_bandwidth(n, d)
    q = d.p + 1
    rng = np.random.default_rng(31_000 + rep)
    u, x, y = S.simulate(rng, n, d)
    idx = LocalIndex(u, x, y, slice_labels(y, d.family, 5))
    truth = S.coefficients(GRID, d)

    os_fit = fit_clsir_os(idx, GRID, h, d.family).estimate
    qs_fit = fit_clsir_os(idx, GRID, h, d.family, quadratic=True).estimate
    if not (np.all(np.isfinite(os_fit)) and np.all(np.isfinite(qs_fit))):
        return []

    rows = []
    for j, g in enumerate(GRID):
        try:
            v_os = sandwich_variance(idx, g, h, d.family, os_fit[j],
                                     np.zeros(q), nu0=NU0_EPAN)
            v_qs = sandwich_variance(idx, g, h, d.family, qs_fit[j],
                                     np.zeros(q), nu0=NU0Q_EPAN)
        except np.linalg.LinAlgError:
            continue
        se_os = np.sqrt(np.maximum(np.diag(v_os), 0.0))
        se_qs = np.sqrt(np.maximum(np.diag(v_qs), 0.0))
        for k in range(q):
            rows.append({"design": dz, "n": n, "rep": rep, "u": g, "coef": k,
                         "cov_os": float(abs(os_fit[j, k] - truth[j, k])
                                         <= Z * se_os[k]),
                         "cov_qs": float(abs(qs_fit[j, k] - truth[j, k])
                                         <= Z * se_qs[k]),
                         "len_os": float(2 * Z * se_os[k]),
                         "len_qs": float(2 * Z * se_qs[k])})
    return rows


def main(reps: int = 400, workers: int = 6) -> None:
    jobs = [(dz, n, r) for dz in DESIGNS for n in NS for r in range(reps)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(one, jobs, chunksize=4), 1):
            rows.extend(res)
            if k % 200 == 0:
                print(f"  {k}/{len(jobs)} replications", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "coverage.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== coverage of the nominal 95 per cent interval, averaged over "
          "the interior and the coefficients ===")
    t = df.groupby(["design", "n"])[["cov_os", "cov_qs"]].mean().round(3)
    t["len ratio qs/os"] = (df.groupby(["design", "n"])["len_qs"].mean()
                            / df.groupby(["design", "n"])["len_os"].mean()).round(3)
    print(t.to_string())
    print("\n=== by coefficient, n = 500 ===")
    print(df[df.n == 500].groupby(["design", "coef"])[["cov_os", "cov_qs"]]
          .mean().round(3).unstack().to_string())


if __name__ == "__main__":
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 400,
         workers=int(sys.argv[2]) if len(sys.argv) > 2 else 6)
