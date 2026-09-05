"""Size and power of the constancy test used in the data application.

Section 5.2 of the paper tests whether a coefficient is constant, taking the
integrated squared deviation of the fitted curve from its own mean as the
statistic and calibrating it by a parametric bootstrap from the fitted constant
coefficient model.  That test carries a substantive conclusion -- only
cumulative tobacco varies with age -- and had not been validated, so this is the
size and power study for it, in the design of \\citet{peng2021}: the alternative
is indexed by a departure parameter, and the null sits at one end of the family.

The coefficient under test is

    a_2^{(a)}(u) = abar + a { a_2(u) - abar },      abar = int_0^1 a_2 = 1,

so a = 0 gives the constant coefficient of the null and a = 1 the coefficient
function used everywhere else in the paper.  The response is Bernoulli with
p = 2, that is Example 1.

Run:  python power_study.py [reps] [boot] [workers]
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
from clsir_core import LocalIndex, fit_clsir_os, fit_wglm, slice_labels

A_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
N = 500
C = 0.80
GRID = np.linspace(0.10, 0.90, 101)
A2BAR = 1.0                    # mean of 2 sin^2(2 pi u) over [0, 1]
OUT = Path("../results/v6_power")


def coefficients(u, a):
    """The Example 1 coefficients with a_2 shrunk towards its own mean."""
    a0 = np.exp(2 * u - 1)
    a1 = 8 * u * (1 - u)
    a2 = A2BAR + a * (2 * np.sin(2 * np.pi * u) ** 2 - A2BAR)
    return np.column_stack([a0, a1, a2])


def simulate(rng, n, a):
    """Example 1's design, with a_2 replaced by its shrunken version."""
    u = rng.uniform(0.0, 1.0, n)
    rho = 2.0 ** -0.5
    corr = (1.0 - rho) * np.eye(2) + rho * np.ones((2, 2))
    z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(corr).T
    x = np.column_stack((np.ones(n), z))
    eta = np.sum(x * coefficients(u, a), axis=1)
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta)), n).astype(float)
    return u, x, y


def variation(est):
    return np.mean((est - est.mean(axis=0)) ** 2, axis=0)


def build(u, x, y):
    return LocalIndex(u, x, y, slice_labels(y, "binomial", 2))


def one(job):
    a, rep, boot = job
    rng = np.random.default_rng(64_000 + rep)
    u, x, y = simulate(rng, N, a)
    h = C * 0.55 * N ** -0.2
    try:
        obs = variation(fit_clsir_os(build(u, x, y), GRID, h, "binomial",
                                     quadratic=True).estimate)
        beta, _, _ = fit_wglm(x, y, np.ones(N), "binomial", max_iter=100)
    except Exception:
        return []
    prob = 1.0 / (1.0 + np.exp(-(x @ beta)))
    brng = np.random.default_rng(900_000 + rep)
    null = np.full((boot, obs.size), np.nan)
    for b in range(boot):
        ys = (brng.random(N) < prob).astype(float)
        if ys.sum() < 5 or ys.sum() > N - 5:
            continue
        try:
            null[b] = variation(fit_clsir_os(build(u, x, ys), GRID, h,
                                             "binomial", quadratic=True).estimate)
        except Exception:
            continue
    ok = np.isfinite(null[:, 0])
    if ok.sum() < boot // 2:
        return []
    p = (1.0 + (null[ok] >= obs).sum(axis=0)) / (1.0 + ok.sum())
    return [{"a": a, "rep": rep, "coef": k, "p": float(p[k])}
            for k in range(obs.size)]


def main(reps: int = 200, boot: int = 99, workers: int = 6) -> None:
    jobs = [(a, r, boot) for a in A_GRID for r in range(reps)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(one, jobs, chunksize=2), 1):
            rows.extend(res)
            if k % 50 == 0:
                print(f"  {k}/{len(jobs)} replications", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "power.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== rejection rate of the constancy test, nominal 0.05 ===")
    for lvl in (0.05, 0.10):
        t = (df.assign(rej=df.p <= lvl)
               .groupby(["a", "coef"])["rej"].mean().unstack().round(3))
        t.columns = [f"a_{k}" for k in t.columns]
        print(f"\n  level {lvl}:")
        print(t.to_string())
    print("\n  (coefficient 2 is the one under test; a = 0 is the null, so its"
          " column at a = 0 is the empirical size)")


if __name__ == "__main__":
    main(reps=int(sys.argv[1]) if len(sys.argv) > 1 else 200,
         boot=int(sys.argv[2]) if len(sys.argv) > 2 else 99,
         workers=int(sys.argv[3]) if len(sys.argv) > 3 else 6)
