"""Real-data application: South African heart disease.

Response: coronary heart disease, binary.  Index: age.  Covariates: the four
risk factors retained by the standard analysis of these data, standardized.
The eight-covariate model is refitted as well, to record how often the local
likelihood has no usable maximizer.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clsir_study as S
from clsir_core import (LocalIndex, NU0Q_EPAN, fit_clsir_os, fit_wglm,
                        sandwich_variance, slice_labels)

DATA = Path("../data/SAheart.data")
OUT = Path("../results/real")
MAIN = ["tobacco", "ldl", "famhist", "typea"]
ALL8 = ["sbp", "tobacco", "ldl", "adiposity", "famhist", "typea", "obesity", "alcohol"]
METHODS = ["CLSIR-QS", "CLSIR-OS", "LMLE", "OSL", "FZ", "FIRTH", "CGA"]
PRIMARY = "CLSIR-QS"   # the estimator whose curve and intervals are reported
HGRID = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80])
BLOWUP = 50.0
GRID = np.linspace(0.05, 0.95, 91)


def load(cols):
    d = pd.read_csv(DATA)
    d["famhist"] = (d["famhist"] == "Present").astype(float)
    y = d["chd"].to_numpy(float)
    age = d["age"].to_numpy(float)
    z = d[cols].to_numpy(float)
    z = (z - z.mean(0)) / z.std(0, ddof=1)
    u = (age - age.min()) / (age.max() - age.min())
    return u, np.column_stack((np.ones(len(y)), z)), y, age


def build(u, x, y):
    return LocalIndex(u, x, y, slice_labels(y, "binomial", 5))


def folds(n, k=5, seed=7):
    return np.random.default_rng(seed).permutation(np.arange(n) % k)


def cross_validate(u, x, y, k=5):
    """Five-fold predictive deviance per observation, on one common fold split."""
    f = folds(len(y), k)
    tab = {m: np.zeros(HGRID.size) for m in METHODS}
    for j in range(k):
        tr, te = f != j, f == j
        idx = build(u[tr], x[tr], y[tr])
        m_size = S.group_size(int(tr.sum()), x.shape[1])
        for i, h in enumerate(HGRID):
            for m in METHODS:
                e = S.fit_method(m, idx, u[te], h, "binomial", m_size).estimate
                tab[m][i] += S.deviance(y[te], np.einsum("ij,ij->i", x[te], e),
                                        "binomial") * te.sum()
    base = 0.0
    for j in range(k):
        tr, te = f != j, f == j
        b, _, _ = fit_wglm(x[tr], y[tr], np.ones(int(tr.sum())), "binomial", max_iter=100)
        base += S.deviance(y[te], x[te] @ b, "binomial") * te.sum()
    return {m: v / len(y) for m, v in tab.items()}, base / len(y)


def divergence(idx, n, q):
    """Evaluation points at which an estimator returns no usable coefficient vector."""
    out = {m: np.zeros(HGRID.size, int) for m in METHODS}
    for i, h in enumerate(HGRID):
        for m in METHODS:
            e = S.fit_method(m, idx, GRID, h, "binomial", S.group_size(n, q)).estimate
            bad = ~np.all(np.isfinite(e), axis=1)
            big = np.nanmax(np.abs(np.where(np.isfinite(e), e, 0.0)), axis=1) > BLOWUP
            out[m][i] = int((bad | big).sum())
    return out


def variation(est):
    """Integrated squared deviation of each fitted curve from its own mean."""
    return np.mean((est - est.mean(axis=0)) ** 2, axis=0)


def constancy_test(u, x, y, h, beta, reps=999, seed=101):
    """Parametric bootstrap test of a constant coefficient against a varying one.

    Under the null the response is generated from the fitted constant coefficient
    logistic model with the observed index and covariates held fixed, and the
    statistic is the integrated squared deviation of each fitted curve from its
    own mean.
    """
    obs = variation(fit_clsir_os(build(u, x, y), GRID, h, "binomial", quadratic=True).estimate)
    prob = 1.0 / (1.0 + np.exp(-(x @ beta)))
    rng = np.random.default_rng(seed)
    null = np.zeros((reps, obs.size))
    for r in range(reps):
        ys = (rng.random(len(y)) < prob).astype(float)
        if ys.sum() < 5 or ys.sum() > len(ys) - 5:
            null[r] = np.nan
            continue
        null[r] = variation(fit_clsir_os(build(u, x, ys), GRID, h, "binomial", quadratic=True).estimate)
    ok = np.isfinite(null[:, 0])
    p = (1.0 + (null[ok] >= obs).sum(axis=0)) / (1.0 + ok.sum())
    return obs, p, int(ok.sum())


def show(title, tab, fmt="{:10.4f}"):
    print("\n" + title)
    print("     h  " + "".join(f"{m:>10s}" for m in METHODS))
    for i, h in enumerate(HGRID):
        print(f" {h:5.2f}  " + "".join(fmt.format(tab[m][i]) for m in METHODS))


def main() -> None:
    u, x, y, age = load(MAIN)
    n, q = x.shape
    idx = build(u, x, y)
    names = ["intercept"] + MAIN
    print(f"n = {n}, events = {int(y.sum())} ({100 * y.mean():.1f} per cent), "
          f"age {age.min():.0f} to {age.max():.0f}, main model p = {q - 1}")

    cv, base = cross_validate(u, x, y)
    show(f"cross-validated deviance per observation "
         f"(constant coefficients {base:.4f})", cv)
    sel = {m: float(HGRID[int(np.argmin(cv[m]))]) for m in METHODS}
    print("  selected h: " + "  ".join(f"{m} {sel[m]:.2f} ({cv[m].min():.4f})"
                                       for m in METHODS))

    h = sel[PRIMARY]
    est, times = {}, {}
    for m in METHODS:
        t0 = time.perf_counter()
        est[m] = S.fit_method(m, idx, GRID, h, "binomial", S.group_size(n, q)).estimate
        times[m] = time.perf_counter() - t0
    print("\nwall clock at h = %.2f, milliseconds: " % h
          + "  ".join(f"{m} {1000 * times[m]:.0f}" for m in METHODS))

    fit = est[PRIMARY]
    se = np.zeros_like(fit)
    for j, g in enumerate(GRID):
        v = sandwich_variance(idx, g, h, "binomial", fit[j], np.zeros(q),
                              nu0=NU0Q_EPAN)
        se[j] = np.sqrt(np.maximum(np.diag(v), 0.0))
    beta, _, _ = fit_wglm(x, y, np.ones(n), "binomial", max_iter=100)
    ages = age.min() + GRID * (age.max() - age.min())
    print("\nfitted coefficient curves, main model")
    for k, nm in enumerate(names):
        print(f"  {nm:10s} global {beta[k]:+.3f} | curve {fit[:, k].min():+.3f} to "
              f"{fit[:, k].max():+.3f}, mean se {se[:, k].mean():.3f}, "
              f"at 20/40/60 years {np.interp([20, 40, 60], ages, fit[:, k]).round(3)}")

    obs, pval, used = constancy_test(u, x, y, h, beta)
    print(f"\nparametric bootstrap test of a constant coefficient, {used} replications")
    for k, nm in enumerate(names):
        print(f"  {nm:10s} statistic {obs[k]:.4f}   p = {pval[k]:.3f}")

    # calibration by five year age band: observed against fitted proportion
    a_at_u = np.column_stack([np.interp(u, GRID, fit[:, k]) for k in range(q)])
    p_vary = 1.0 / (1.0 + np.exp(-np.einsum("ij,ij->i", x, a_at_u)))
    p_const = 1.0 / (1.0 + np.exp(-(x @ beta)))
    edges = np.arange(15, 70, 5.0)
    mid, prop, cnt, pv, pc = [], [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (age >= lo) & (age < hi)
        if s.sum() >= 10:
            mid.append(0.5 * (lo + hi))
            prop.append(float(y[s].mean()))
            cnt.append(int(s.sum()))
            pv.append(float(p_vary[s].mean()))
            pc.append(float(p_const[s].mean()))
    print("\nage band: observed / varying / constant fitted proportion")
    for m, o, a, b2 in zip(mid, prop, pv, pc):
        print(f"  {m:4.0f}   {o:.3f}   {a:.3f}   {b2:.3f}")
    print("  mean absolute calibration error: varying "
          f"{np.mean(np.abs(np.array(prop) - np.array(pv))):.4f}, constant "
          f"{np.mean(np.abs(np.array(prop) - np.array(pc))):.4f}")

    u8, x8, y8, _ = load(ALL8)
    idx8 = build(u8, x8, y8)
    div8 = divergence(idx8, len(y8), x8.shape[1])
    show("evaluation points out of 91 without a usable coefficient vector, p = 8",
         div8, fmt="{:10d}")

    # the same two effects after adjusting for all eight risk factors
    cv8 = np.array([np.nan] * HGRID.size)
    f = folds(len(y8))
    for i, hh in enumerate(HGRID):
        tot = 0.0
        for j in range(5):
            tr, te = f != j, f == j
            e = fit_clsir_os(build(u8[tr], x8[tr], y8[tr]), u8[te], hh,
                             "binomial", quadratic=True).estimate
            tot += S.deviance(y8[te], np.einsum("ij,ij->i", x8[te], e),
                              "binomial") * te.sum()
        cv8[i] = tot / len(y8)
    h8 = float(HGRID[int(np.nanargmin(cv8))])
    fit8 = fit_clsir_os(idx8, GRID, h8, "binomial", quadratic=True).estimate
    se8 = np.zeros_like(fit8)
    for j, g in enumerate(GRID):
        v = sandwich_variance(idx8, g, h8, "binomial", fit8[j],
                              np.zeros(x8.shape[1]), nu0=NU0Q_EPAN)
        se8[j] = np.sqrt(np.maximum(np.diag(v), 0.0))
    names8 = ["intercept"] + ALL8
    print(f"\neight covariate model, cross-validated h = {h8:.2f} "
          f"(deviance {np.nanmin(cv8):.4f})")
    for k, nm in enumerate(names8):
        print(f"  {nm:10s} curve {fit8[:, k].min():+.3f} to {fit8[:, k].max():+.3f}, "
              f"at 20/40/60 years {np.interp([20, 40, 60], ages, fit8[:, k]).round(3)}")

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "saheart.npz", grid=GRID, ages=ages, h=h, hgrid=HGRID, se=se,
             beta=beta, base=base, names=np.array(names), methods=np.array(METHODS),
             sel=np.array([sel[m] for m in METHODS]),
             times=np.array([times[m] for m in METHODS]),
             band_mid=np.array(mid), band_prop=np.array(prop), band_n=np.array(cnt),
             band_vary=np.array(pv), band_const=np.array(pc),
             test_stat=obs, test_p=pval, h8=h8, cv8=cv8, fit8=fit8, se8=se8,
             names8=np.array(names8),
             **{f"cv_{m}": cv[m] for m in METHODS},
             **{f"div8_{m}": div8[m] for m in METHODS},
             **{f"est_{m}": est[m] for m in METHODS})
    print(f"\nwrote {OUT / 'saheart.npz'}")


if __name__ == "__main__":
    main()
