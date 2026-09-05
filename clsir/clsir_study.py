"""Monte Carlo studies for the calibrated local SIR one-step estimator.

Sub-commands
------------
grid   accuracy and reliability over a common bandwidth grid
cv     complete analysis: K-fold likelihood cross-validation + final fit
cost   wall-clock scaling in the number of evaluation points and in p
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clsir_core import (  # noqa: E402
    LocalIndex,
    expit,
    fit_clsir_os,
    fit_fan_zhang,
    fit_gla,
    fit_lc_os,
    fit_local_firth,
    fit_local_mle,
    fit_os_local_mle,
    slice_labels,
)

METHODS = ["CLSIR-OS", "CLSIR-QS", "LMLE", "OSL", "FZ", "FIRTH", "CGA", "GA"]
ALL_METHODS = METHODS + ["LC-OS", "CLSIR-BC", "CLSIR-TS", "TSL", "LMLE-warm"]
BLOWUP = 50.0  # pre-specified bound: an estimate exceeding it counts as a failure


# --------------------------------------------------------------------------
# designs
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    name: str
    family: str
    p: int
    omega: float = 1.0          # oscillation multiplier of the turning pair
    rho: float = 0.5            # AR(1) correlation of Z given U
    elliptical: bool = True     # False violates the SIR linearity condition
    sigma: float = 0.7          # Gaussian noise scale


DESIGNS: dict[str, Design] = {
    "bin2": Design("bin2", "binomial", 2),
    "bin10": Design("bin10", "binomial", 10),
    "poi2": Design("poi2", "poisson", 2),
    "poi10": Design("poi10", "poisson", 10),
    "gau2": Design("gau2", "gaussian", 2),
    "binskew": Design("binskew", "binomial", 2, elliptical=False),
    "bin10skew": Design("bin10skew", "binomial", 10, elliptical=False),
}


def coefficients(u: np.ndarray, d: Design) -> np.ndarray:
    """Coefficient functions of Cai, Fan and Li (2000, eq. 4.5), padded with zeros.

    a_0(u) = exp(2u - 1), a_1(u) = 8u(1 - u), a_2(u) = 2 sin^2(2 pi u), and
    a_3 = ... = a_p = 0 when p > 2.  The Poisson designs rescale the predictor as
    in their Example 2 so that the counts have a realistic magnitude.
    """
    u = np.atleast_1d(np.asarray(u, float))
    a = np.zeros((u.size, d.p + 1))
    a0 = np.exp(2.0 * u - 1.0)
    a1 = 8.0 * u * (1.0 - u)
    a2 = 2.0 * np.sin(2.0 * np.pi * u) ** 2
    if d.family == "poisson":
        b0, sc = 1.20, 0.40
    else:
        b0, sc = 0.00, 1.00
    a[:, 0] = b0 + sc * a0
    a[:, 1] = sc * a1
    a[:, 2] = sc * a2
    return a


def ar1(p: int, rho: float) -> np.ndarray:
    i = np.arange(p)
    return rho ** np.abs(i[:, None] - i[None, :])


def simulate(rng: np.random.Generator, n: int, d: Design):
    """U uniform on [0,1]; Z standard normal with equicorrelation 2^{-1/2}."""
    u = rng.uniform(0.0, 1.0, n)
    rho = 2.0 ** -0.5
    corr = (1.0 - rho) * np.eye(d.p) + rho * np.ones((d.p, d.p))
    chol = np.linalg.cholesky(corr)
    if d.elliptical:
        z = rng.standard_normal((n, d.p)) @ chol.T
    else:
        raw = (rng.chisquare(3.0, size=(n, d.p)) - 3.0) / math.sqrt(6.0)
        z = raw @ chol.T
        z[:, 0] = z[:, 0] + 0.35 * (z[:, 1] ** 2 - 1.0)
    x = np.column_stack((np.ones(n), z))
    eta = np.sum(x * coefficients(u, d), axis=1)
    if d.family == "gaussian":
        y = eta + rng.normal(0.0, d.sigma, n)
    elif d.family == "binomial":
        y = rng.binomial(1, expit(eta), n).astype(float)
    else:
        y = rng.poisson(np.exp(np.clip(eta, -6.0, 6.0)), n).astype(float)
    return u, x, y


def base_bandwidth(n: int, d: Design) -> float:
    return 0.55 * n ** (-0.2)


def group_size(n: int, q: int) -> int:
    """Group size inside the admissible window [n^{1/4}, n^{3/5}]."""
    lo, hi = n**0.25, n**0.6
    return int(min(max(math.ceil(math.sqrt(n)), math.ceil(4 * q), math.ceil(lo)), math.floor(hi)))


# --------------------------------------------------------------------------
# fitting front-end
# --------------------------------------------------------------------------


def fit_method(method, idx, grid, h, family, m):
    if method == "CLSIR-OS":
        return fit_clsir_os(idx, grid, h, family)
    if method == "CLSIR-BC":
        return fit_clsir_os(idx, grid, h, family, bias_correct=True)
    if method == "CLSIR-QS":
        return fit_clsir_os(idx, grid, h, family, quadratic=True)
    if method == "CLSIR-QSS":
        return fit_clsir_os(idx, grid, h, family, quadratic=True, steps=2)
    if method.startswith("CLSIR-QR"):          # CLSIR-QR<r>: curvature ratio r/10
        r = float(method[len("CLSIR-QR"):]) / 10.0
        return fit_clsir_os(idx, grid, h, family, quadratic=True, curv_ratio=r)
    if method == "FIRTH":
        return fit_local_firth(idx, grid, h, family)
    if method == "CLSIR-TS":
        return fit_clsir_os(idx, grid, h, family, steps=2)
    if method == "TSL":
        return fit_os_local_mle(idx, grid, h, family, steps=2)
    if method == "LC-OS":
        return fit_lc_os(idx, grid, h, family)
    if method == "OSL":
        return fit_os_local_mle(idx, grid, h, family)
    if method == "FZ":
        return fit_fan_zhang(idx, grid, h, family)
    if method == "LMLE":
        return fit_local_mle(idx, grid, h, family)
    if method == "LMLE-warm":
        return fit_local_mle(idx, grid, h, family, warm_start=True)
    if method == "GA":
        return fit_gla(idx, grid, h, family, m, bias_correct=False)
    if method == "CGA":
        return fit_gla(idx, grid, h, family, m, bias_correct=True)
    raise ValueError(method)


def deviance(y: np.ndarray, eta: np.ndarray, family: str) -> float:
    eta = np.clip(eta, -30.0, 30.0)
    if family == "gaussian":
        return float(np.mean((y - eta) ** 2))
    if family == "binomial":
        return float(np.mean(np.logaddexp(0.0, eta) - y * eta))
    return float(np.mean(np.exp(eta) - y * eta))


def curve_predict(est: np.ndarray, grid: np.ndarray, u: np.ndarray, x: np.ndarray) -> np.ndarray:
    a = np.column_stack([np.interp(u, grid, est[:, j]) for j in range(est.shape[1])])
    return np.sum(x * a, axis=1)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


_TEST_CACHE: dict = {}


def test_sample(d: Design, seed: int, grid: np.ndarray, n_test: int = 4000):
    """One independent test sample per (design, seed), reused across methods."""
    key = (d.name, seed, float(grid[0]), float(grid[-1]))
    hit = _TEST_CACHE.get(key)
    if hit is None:
        rng = np.random.default_rng(seed)
        u, x, y = simulate(rng, n_test, d)
        keep = (u >= grid[0]) & (u <= grid[-1])
        hit = (u[keep], x[keep], y[keep])
        if len(_TEST_CACHE) > 8:
            _TEST_CACHE.clear()
        _TEST_CACHE[key] = hit
    return hit


def evaluate(est: np.ndarray, grid: np.ndarray, d: Design, seed: int) -> dict:
    truth = coefficients(grid, d)
    finite = bool(np.all(np.isfinite(est)))
    blown = (not finite) or float(np.max(np.abs(est))) > BLOWUP
    if blown:
        return {"ise": float("nan"), "ise_slope": float("nan"),
                "ise_intercept": float("nan"), "pred": float("nan"), "blown": 1.0}
    err = est - truth
    ut, xt, yt = test_sample(d, seed, grid)
    pred = deviance(yt, curve_predict(est, grid, ut, xt), d.family)
    return {
        "ise": float(np.mean(np.sum(err**2, axis=1))),
        "ise_slope": float(np.mean(np.sum(err[:, 1:] ** 2, axis=1))),
        "ise_intercept": float(np.mean(err[:, 0] ** 2)),
        "pred": pred,
        "blown": 0.0,
    }


# --------------------------------------------------------------------------
# study "grid"
# --------------------------------------------------------------------------


def run_grid(job: tuple) -> list[dict]:
    seed, n, name, c_grid, grid_size, methods = job
    d = DESIGNS[name]
    rng = np.random.default_rng(seed)
    u, x, y = simulate(rng, n, d)
    idx = LocalIndex(u, x, y, slice_labels(y, d.family, 5))
    grid = np.linspace(0.10, 0.90, grid_size)
    q = x.shape[1]
    h0 = base_bandwidth(n, d)
    m = group_size(n, q)
    rows = []
    for c in c_grid:
        h = c * h0
        for method in methods:
            res = fit_method(method, idx, grid, h, d.family, m)
            met = evaluate(res.estimate, grid, d, seed + 7_000_000)
            rows.append({"study": "grid", "design": name, "family": d.family, "p": d.p,
                         "n": n, "seed": seed, "c": c, "h": h, "m": m, "method": method,
                         "runtime": res.runtime, "pilot_failures": float(res.pilot_failures),
                         "update_failures": float(res.update_failures),
                         "iterations": float(res.iterations), **met})
    return rows


# --------------------------------------------------------------------------
# study "cv": complete analysis with K-fold likelihood cross-validation
# --------------------------------------------------------------------------


def run_cv(job: tuple) -> list[dict]:
    seed, n, name, c_grid, grid_size, methods, folds = job
    d = DESIGNS[name]
    rng = np.random.default_rng(seed)
    u, x, y = simulate(rng, n, d)
    grid = np.linspace(0.10, 0.90, grid_size)
    q = x.shape[1]
    h0 = base_bandwidth(n, d)
    m_full = group_size(n, q)
    part = rng.permutation(n) % folds
    rows = []
    for method in methods:
        t0 = time.perf_counter()
        scores = np.full(len(c_grid), np.inf)
        for ci, c in enumerate(c_grid):
            tot, ok = 0.0, True
            for k in range(folds):
                tr, te = part != k, part == k
                idx_tr = LocalIndex(u[tr], x[tr], y[tr], slice_labels(y[tr], d.family, 5))
                m_tr = group_size(int(tr.sum()), q)
                res = fit_method(method, idx_tr, grid, c * h0, d.family, m_tr)
                if not np.all(np.isfinite(res.estimate)):
                    ok = False
                    break
                inside = te & (u >= grid[0]) & (u <= grid[-1])
                if not np.any(inside):
                    continue
                tot += deviance(y[inside], curve_predict(res.estimate, grid,
                                                         u[inside], x[inside]), d.family)
            if ok:
                scores[ci] = tot / folds
        if np.all(~np.isfinite(scores)):
            c_hat = c_grid[len(c_grid) // 2]
        else:
            c_hat = c_grid[int(np.nanargmin(scores))]
        idx_full = LocalIndex(u, x, y, slice_labels(y, d.family, 5))
        res = fit_method(method, idx_full, grid, c_hat * h0, d.family, m_full)
        total_time = time.perf_counter() - t0
        met = evaluate(res.estimate, grid, d, seed + 7_000_000)
        rows.append({"study": "cv", "design": name, "family": d.family, "p": d.p, "n": n,
                     "seed": seed, "c": c_hat, "h": c_hat * h0, "m": m_full, "method": method,
                     # the selected criterion value, so that the same
                     # cross-validation may also choose between the linear and
                     # the quadratic update after the fact
                     "cv_score": float(np.min(scores)) if np.any(np.isfinite(scores))
                     else float("nan"),
                     "runtime": total_time, "final_runtime": res.runtime,
                     "pilot_failures": float(res.pilot_failures),
                     "update_failures": float(res.update_failures),
                     "iterations": float(res.iterations), **met})
    return rows


# --------------------------------------------------------------------------
# study "cost"
# --------------------------------------------------------------------------


def run_cost(job: tuple) -> list[dict]:
    seed, n, name, sizes, methods = job
    d = DESIGNS[name]
    rng = np.random.default_rng(seed)
    u, x, y = simulate(rng, n, d)
    idx = LocalIndex(u, x, y, slice_labels(y, d.family, 5))
    q = x.shape[1]
    h = base_bandwidth(n, d)
    m = group_size(n, q)
    rows = []
    for g in sizes:
        grid = np.linspace(0.10, 0.90, g)
        for method in methods:
            res = fit_method(method, idx, grid, h, d.family, m)
            rows.append({"study": "cost", "design": name, "family": d.family, "p": d.p,
                         "n": n, "seed": seed, "grid_size": g, "method": method,
                         "runtime": res.runtime, "iterations": float(res.iterations)})
    return rows


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def dispatch(job):
    kind = job[0]
    return {"grid": run_grid, "cv": run_cv, "cost": run_cost}[kind](job[1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["grid", "cv", "cost"])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, nargs="+", default=[500, 1000])
    ap.add_argument("--designs", nargs="+", default=["logit4", "logit10", "pois4"])
    ap.add_argument("--c-grid", type=float, nargs="+",
                    default=[0.5, 0.65, 0.8, 1.0, 1.3, 1.7])
    ap.add_argument("--grid-size", type=int, default=101)
    ap.add_argument("--cost-sizes", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--no-ridge", action="store_true",
                    help="switch off the Cai-Fan-Li ridge (exact Gaussian control)")
    ap.add_argument("--out", type=Path, default=Path("../results/main"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.no_ridge:
        os.environ["CLSIR_NO_RIDGE"] = "1"
        import clsir_core
        clsir_core.RIDGE_ON = False

    jobs = []
    order = list(DESIGNS)
    for name in args.designs:
        # the offset is keyed on the design itself, not on its position in
        # --designs, so that a run over a subset of the designs reproduces the
        # seeds of the full run and the two remain paired
        di = order.index(name)
        for n in args.n:
            for r in range(args.reps):
                seed = args.seed + 1_000_000 * di + 1_000 * (n % 100_000) + r
                if args.study == "grid":
                    jobs.append(("grid", seed, n, name, tuple(args.c_grid),
                                 args.grid_size, tuple(args.methods)))
                elif args.study == "cv":
                    jobs.append(("cv", seed, n, name, tuple(args.c_grid),
                                 args.grid_size, tuple(args.methods), args.folds))
                else:
                    jobs.append(("cost", seed, n, name, tuple(args.cost_sizes),
                                 tuple(args.methods)))

    t0 = time.perf_counter()
    rows: list[dict] = []
    if args.workers <= 1:
        for k, job in enumerate(jobs, 1):
            rows.extend(dispatch(job))
            if k % 10 == 0 or k == len(jobs):
                print(json.dumps({"done": k, "total": len(jobs),
                                  "elapsed": round(time.perf_counter() - t0, 1)}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(dispatch, job) for job in jobs]
            for k, fut in enumerate(as_completed(futures), 1):
                rows.extend(fut.result())
                if k % 50 == 0 or k == len(jobs):
                    print(json.dumps({"done": k, "total": len(jobs),
                                      "elapsed": round(time.perf_counter() - t0, 1)}), flush=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    tag = args.study
    df.to_csv(args.out / f"{tag}_runs.csv", index=False)
    print(f"wrote {args.out}/{tag}_runs.csv  ({len(df)} rows, "
          f"{time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
