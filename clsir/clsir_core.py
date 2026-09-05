"""Core routines for the calibrated local SIR (CLSIR) one-step study.

Estimators implemented
----------------------
CLSIR-OS  proposed: coarse-grid local SIR + two-parameter likelihood
          calibration + local-quadratic smoothing + one Newton step.
LC-OS     ablation: identical pipeline with the full q-dimensional
          local-constant likelihood pilot in place of the SIR pilot.
LMLE      fully iterated local-linear maximum likelihood (Cai, Fan and Li, 2000).
OSLMLE    one-step local MLE with sequential grid propagation and periodic
          refreshing (Cai, Fan and Li, 2000, Section 3.5).
GLA       grouped local averaging, the generalized analogue of the local
          average estimator of Peng, Xie and Zhao (2021).
BCGLA     bias-corrected grouped local averaging.

Only NumPy is required.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

EPS = 1e-12

# Set to False to switch off the ridge of Cai, Fan and Li (2000, eq. 4.4).
# Read from the environment so that the flag survives into worker processes,
# which re-import this module rather than inheriting the parent's globals.
RIDGE_ON = os.environ.get("CLSIR_NO_RIDGE", "") != "1"

# --------------------------------------------------------------------------
# kernel and exponential family primitives
# --------------------------------------------------------------------------


def epanechnikov(t: np.ndarray) -> np.ndarray:
    return 0.75 * np.maximum(1.0 - t * t, 0.0)


MU2_EPAN = 0.2          # \int t^2 K(t) dt
NU0_EPAN = 0.6          # \int K(t)^2 dt, local linear
# \int K*(t)^2 dt for the equivalent kernel K*(t) = {(mu4 - mu2 t^2)/(mu4 - mu2^2)} K(t)
# of the local quadratic fit; see Theorem 4.
NU0Q_EPAN = 1.25

# Default pilot ratio c1 = h1 / h, by response family.  Theorem 2 asks only that
# c1 be fixed and positive, so the default may depend on the family, which is
# known to the analyst.  A Bernoulli observation carries at most a quarter of a
# unit of Fisher information about the linear predictor, and much less where the
# success probability is near zero or one, whereas a count carries its mean; the
# pilot window must therefore be wider for a binary response to reach the same
# precision.  See tuning_wide.py and Section S3 of the supplement.
PILOT_RATIO = {"binomial": 3.0}
PILOT_RATIO_DEFAULT = 1.5


def default_pilot_ratio(family: str) -> float:
    return PILOT_RATIO.get(family, PILOT_RATIO_DEFAULT)


def expit(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-np.minimum(x[pos], 40.0)))
    e = np.exp(np.maximum(x[~pos], -40.0))
    out[~pos] = e / (1.0 + e)
    return out


def mean_and_weight(eta: np.ndarray, family: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the conditional mean and the canonical information weight."""
    if family == "gaussian":
        return eta, np.ones_like(eta)
    if family == "binomial":
        mu = np.clip(expit(eta), 1e-10, 1.0 - 1e-10)
        return mu, mu * (1.0 - mu)
    if family == "poisson":
        mu = np.exp(np.clip(eta, -20.0, 20.0))
        return mu, np.maximum(mu, 1e-10)
    raise ValueError(family)


def loglik(y: np.ndarray, eta: np.ndarray, w: np.ndarray, family: str) -> float:
    if family == "gaussian":
        v = -0.5 * (y - eta) ** 2
    elif family == "binomial":
        v = y * eta - np.logaddexp(0.0, eta)
    elif family == "poisson":
        v = y * eta - np.exp(np.clip(eta, -20.0, 20.0))
    else:
        raise ValueError(family)
    return float(np.dot(w, v))


def zeta_prime(mu: np.ndarray, family: str) -> np.ndarray:
    """Curvature coefficient c(mu) = d zeta / d mu with zeta = d mu / d eta."""
    if family == "gaussian":
        return np.zeros_like(mu)
    if family == "binomial":
        return 1.0 - 2.0 * mu
    if family == "poisson":
        return np.ones_like(mu)
    raise ValueError(family)


def ridge_solve(
    info: np.ndarray, score: np.ndarray, rel: float = 1e-8, n_eff: float = 0.0
) -> np.ndarray:
    """Solve info x = score with the ridge of Cai, Fan and Li (2000, eq. 4.4).

    Their ridge parameter equals the approximate diagonal element of the local
    information divided by the effective local sample size N = n h f_U(u); the
    sample analogue used here adds diag(info)/N to the diagonal.  ``n_eff <= 0``
    disables it, leaving only a numerical jitter.
    """
    info = 0.5 * (info + info.T)
    d = info.shape[0]
    scale = max(float(np.trace(info)) / max(d, 1), 1e-10)
    reg = info + rel * scale * np.eye(d)
    if n_eff > 0.0 and RIDGE_ON:
        reg = reg + np.diag(np.maximum(np.diag(info), 0.0)) / n_eff
    try:
        return np.linalg.solve(reg, score)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(reg, score, rcond=None)[0]


# --------------------------------------------------------------------------
# weighted generalized linear model
# --------------------------------------------------------------------------


def start_value(x: np.ndarray, y: np.ndarray, w: np.ndarray, family: str) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    tot = float(np.sum(w))
    ybar = float(np.dot(w, y) / tot) if tot > EPS else 0.0
    if family == "gaussian":
        beta[0] = ybar
    elif family == "binomial":
        pr = min(max(ybar, 1e-3), 1.0 - 1e-3)
        beta[0] = math.log(pr / (1.0 - pr))
    elif family == "poisson":
        beta[0] = math.log(max(ybar, 1e-3))
    return beta


def fit_wglm(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    family: str,
    beta0: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-9,
    rel_ridge: float = 1e-8,
    n_eff: float = 0.0,
) -> tuple[np.ndarray, bool, int]:
    """Weighted GLM by IRLS with step halving.  Returns (beta, converged, iters)."""
    d = x.shape[1]
    if float(np.sum(w > 0)) <= d:
        return np.zeros(d), False, 0
    if family == "gaussian":
        info = x.T @ (x * w[:, None])
        score = x.T @ (w * y)
        return ridge_solve(info, score, rel_ridge, n_eff), True, 1
    beta = start_value(x, y, w, family) if beta0 is None else np.asarray(beta0, float).copy()
    cur = loglik(y, x @ beta, w, family)
    it = 0
    for it in range(1, max_iter + 1):
        eta = x @ beta
        mu, wt = mean_and_weight(eta, family)
        score = x.T @ (w * (y - mu))
        info = x.T @ (x * (w * wt)[:, None])
        step = ridge_solve(info, score, rel_ridge, n_eff)
        nrm = float(np.linalg.norm(step))
        if not np.isfinite(nrm):
            return beta, False, it
        if nrm > 15.0:
            step *= 15.0 / nrm
        ok = False
        frac = 1.0
        for _ in range(20):
            cand = beta + frac * step
            val = loglik(y, x @ cand, w, family)
            if np.isfinite(val) and val >= cur - 1e-10:
                beta, cur, ok = cand, val, True
                break
            frac *= 0.5
        if not ok:
            return beta, False, it
        if frac * nrm <= tol * (1.0 + float(np.linalg.norm(beta))):
            return beta, True, it
    return beta, False, it


def newton_step(
    d_mat: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    family: str,
    theta0: np.ndarray,
    rel_ridge: float = 1e-8,
    cap: float = 50.0,
    n_eff: float = 0.0,
    scale: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """One Newton step, with a trust region in the scaling of the theory.

    ``scale`` is the diagonal of the matrix Lambda that the asymptotics use,
    namely (1, h) for the local linear parametrization and (1, h, h^2) for the
    local quadratic one, each repeated over the p+1 coefficients.  The step is
    capped on ``norm(scale * step)`` rather than on ``norm(step)``: the
    derivative block moves on the scale h^{-1} and the curvature block on
    h^{-2}, so an unscaled cap truncates a perfectly ordinary quadratic step
    almost everywhere and leaves the level barely moved.
    """
    eta = d_mat @ theta0
    mu, wt = mean_and_weight(eta, family)
    score = d_mat.T @ (w * (y - mu))
    info = d_mat.T @ (d_mat * (w * wt)[:, None])
    step = ridge_solve(info, score, rel_ridge, n_eff)
    ok = bool(np.all(np.isfinite(step)))
    if not ok:
        return theta0.copy(), False
    nrm = float(np.linalg.norm(step if scale is None else scale * step))
    if nrm > cap:
        step *= cap / nrm
        ok = False
    return theta0 + step, ok


# --------------------------------------------------------------------------
# local window bookkeeping (compact kernel support)
# --------------------------------------------------------------------------


class LocalIndex:
    """Pre-sorted index supporting O(log n) extraction of {i : |U_i - u| <= h}."""

    def __init__(
        self, u: np.ndarray, x: np.ndarray, y: np.ndarray, lab: np.ndarray | None = None
    ) -> None:
        order = np.argsort(u, kind="mergesort")
        self.u = np.ascontiguousarray(u[order])
        self.x = np.ascontiguousarray(x[order])
        self.y = np.ascontiguousarray(y[order])
        self.lab = np.ascontiguousarray(lab[order]) if lab is not None else None
        self.n = u.size

    def window(self, target: float, h: float):
        """Return (du, w, x, y, lab) restricted to the kernel support, or None."""
        lo = int(np.searchsorted(self.u, target - h, side="left"))
        hi = int(np.searchsorted(self.u, target + h, side="right"))
        if hi <= lo:
            return None
        du = self.u[lo:hi] - target
        w = epanechnikov(du / h)
        keep = w > 0.0
        if not np.any(keep):
            return None
        lab = self.lab[lo:hi][keep] if self.lab is not None else None
        return du[keep], w[keep], self.x[lo:hi][keep], self.y[lo:hi][keep], lab


# --------------------------------------------------------------------------
# local polynomial smoother of a pilot curve
# --------------------------------------------------------------------------


def local_poly_smooth(
    centres: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    b: float,
    degree: int = 2,
    return_curv: bool = False,
):
    """Smooth a pilot curve; return (level, first derivative) at the targets.

    With ``return_curv`` the second derivative implied by the quadratic term is
    returned as a third element.  It costs nothing, the quadratic coefficient
    already being computed, and it is what the bias correction of Section 2.4
    needs.
    """
    g = values.shape[1]
    level = np.zeros((targets.size, g))
    deriv = np.zeros((targets.size, g))
    curv = np.zeros((targets.size, g))
    for j, t in enumerate(targets):
        dt = centres - t
        w = epanechnikov(dt / b)
        eff = int(np.sum(w > 0))
        deg = degree
        while deg >= 1 and eff < deg + 2:
            deg -= 1
        if eff < 3:
            near = np.argsort(np.abs(dt))[: min(5, centres.size)]
            w = np.zeros_like(dt)
            w[near] = 1.0
            deg = min(1, degree)
        cols = [np.ones_like(dt)] + [dt**k for k in range(1, deg + 1)]
        design = np.column_stack(cols)
        sw = np.sqrt(w)[:, None]
        coef = np.linalg.lstsq(design * sw, values * sw, rcond=None)[0]
        level[j] = coef[0]
        deriv[j] = coef[1] if deg >= 1 else 0.0
        if deg >= 2:
            curv[j] = 2.0 * coef[2]
    return (level, deriv, curv) if return_curv else (level, deriv)


# --------------------------------------------------------------------------
# response slicing for SIR
# --------------------------------------------------------------------------


def slice_labels(y: np.ndarray, family: str, n_slices: int) -> np.ndarray:
    if family == "binomial":
        return (y > 0.5).astype(int)
    uniq = np.unique(y)
    if uniq.size <= n_slices:
        table = {v: k for k, v in enumerate(uniq)}
        return np.array([table[v] for v in y], dtype=int)
    order = np.argsort(y, kind="mergesort")
    lab = np.empty(y.size, dtype=int)
    lab[order] = np.minimum((np.arange(y.size) * n_slices) // y.size, n_slices - 1)
    return lab


# --------------------------------------------------------------------------
# stage 1: kernel-weighted SIR direction
# --------------------------------------------------------------------------


def local_sir_direction(
    du: np.ndarray,
    w: np.ndarray,
    z: np.ndarray,
    lab: np.ndarray,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, float, bool]:
    """Leading generalized eigenvector of the kernel-weighted slice matrix."""
    p = z.shape[1]
    tot = float(np.sum(w))
    if tot <= 0.0 or w.size <= p + 2:
        return np.zeros(p), 0.0, False
    m = (w @ z) / tot
    zc = z - m
    sigma = (zc * w[:, None]).T @ zc / tot
    scale = max(float(np.trace(sigma)) / p, 1e-10)
    sigma = sigma + ridge * scale * np.eye(p)

    mat = np.zeros((p, p))
    used = 0
    for s in np.unique(lab):
        mask = lab == s
        ws = float(np.sum(w[mask]))
        if ws <= 1e-8 * tot:
            continue
        ms = (w[mask] @ z[mask]) / ws
        d = ms - m
        mat += (ws / tot) * np.outer(d, d)
        used += 1
    if used < 2:
        return np.zeros(p), 0.0, False
    try:
        evals, evecs = np.linalg.eigh(sigma)
        floor = max(1e-10 * float(evals[-1]), 1e-12)
        isqrt = evecs @ np.diag(1.0 / np.sqrt(np.maximum(evals, floor))) @ evecs.T
        std = isqrt @ mat @ isqrt
        lam, vec = np.linalg.eigh(0.5 * (std + std.T))
        direction = isqrt @ vec[:, -1]
        nrm = float(np.linalg.norm(direction))
        if nrm < 1e-10 or lam[-1] <= 1e-12:
            return np.zeros(p), 0.0, False
        gap = float(lam[-1] - (lam[-2] if p > 1 else 0.0))
        return direction / nrm, gap, True
    except np.linalg.LinAlgError:
        return np.zeros(p), 0.0, False


# --------------------------------------------------------------------------
# stage 2: pilot curves
# --------------------------------------------------------------------------


def clsir_pilot(
    idx: LocalIndex,
    centres: np.ndarray,
    h1: float,
    family: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    n_eff = idx.n * h1
    """Calibrated local SIR pre-estimates at the pilot centres."""
    q = idx.x.shape[1]
    out = np.zeros((centres.size, q))
    fails = 0
    gaps = np.zeros(centres.size)
    prev = None
    for j, t in enumerate(centres):
        win = idx.window(t, h1)
        if win is None:
            fails += 1
            continue
        du, w, xw, yw, lab_w = win
        z = xw[:, 1:]
        d, gap, ok = local_sir_direction(du, w, z, lab_w)
        gaps[j] = gap
        if not ok:
            fails += 1
            d = np.zeros(z.shape[1])
            d[0] = 1.0
        if prev is not None and float(np.dot(d, prev)) < 0.0:
            d = -d
        prev = d
        design = np.column_stack((np.ones(w.size), z @ d))
        gamma, conv, _ = fit_wglm(design, yw, w, family, max_iter=40, n_eff=n_eff)
        if not conv:
            fails += 1
        out[j, 0] = gamma[0]
        out[j, 1:] = gamma[1] * d
    return out, fails, gaps


def lc_pilot(
    idx: LocalIndex,
    centres: np.ndarray,
    h1: float,
    family: str,
) -> tuple[np.ndarray, int]:
    """Full q-dimensional local-constant likelihood pre-estimates (ablation)."""
    n_eff = idx.n * h1
    q = idx.x.shape[1]
    out = np.zeros((centres.size, q))
    fails = 0
    for j, t in enumerate(centres):
        win = idx.window(t, h1)
        if win is None:
            fails += 1
            continue
        _, w, xw, yw, _ = win
        beta, conv, _ = fit_wglm(xw, yw, w, family, max_iter=40, n_eff=n_eff)
        if not conv:
            fails += 1
        out[j] = beta
    return out, fails


def pilot_centres(h1: float, spacing_factor: float = 0.5) -> np.ndarray:
    step = max(spacing_factor * h1, 1e-3)
    m = max(int(math.ceil(1.0 / step)) + 1, 7)
    return np.linspace(0.0, 1.0, m)


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------


@dataclass
class FitResult:
    estimate: np.ndarray
    runtime: float
    pilot_failures: int
    update_failures: int
    iterations: int


def _one_step_from_pilot(
    idx: LocalIndex,
    grid: np.ndarray,
    h: float,
    family: str,
    level: np.ndarray,
    deriv: np.ndarray,
    steps: int = 1,
    curv: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """One Newton step of the local likelihood at every evaluation point.

    With ``curv`` supplied the step is taken in the local *quadratic*
    parametrization, the smoothing pass of Step 3 having already produced an
    initial value for the curvature block at no extra cost.
    """
    q = idx.x.shape[1]
    est = np.zeros((grid.size, q))
    bad = 0
    for j, t in enumerate(grid):
        win = idx.window(t, h)
        if win is None:
            est[j] = level[j]
            bad += 1
            continue
        du, w, xw, yw, _ = win
        if curv is None:
            dmat = np.column_stack((xw, xw * du[:, None]))
            theta = np.concatenate((level[j], deriv[j]))
            scale = np.repeat([1.0, h], q)
        else:
            dmat = np.column_stack((xw, xw * du[:, None],
                                    xw * (0.5 * du * du)[:, None]))
            theta = np.concatenate((level[j], deriv[j], curv[j]))
            scale = np.repeat([1.0, h, h * h], q)
        ok = True
        for _ in range(steps):
            theta, ok_s = newton_step(dmat, yw, w, family, theta,
                                      n_eff=idx.n * h, scale=scale)
            ok = ok and ok_s
        est[j] = theta[:q]
        bad += int(not ok)
    return est, bad


def fit_clsir_os(
    idx: LocalIndex,
    grid: np.ndarray,
    h: float,
    family: str,
    pilot_ratio: float | None = None,
    smooth_ratio: float = 1.2,
    steps: int = 1,
    bias_correct: bool = False,
    quadratic: bool = False,
    curv_ratio: float = 1.0,
) -> FitResult:
    """CLSIR-OS (``quadratic=False``) and CLSIR-QS (``quadratic=True``).

    Both share the pilot, the calibration and the single local-quadratic
    smoothing pass.  That pass returns the level, the derivative and the
    curvature; CLSIR-OS uses the first two and CLSIR-QS all three.  A curvature
    bandwidth other than ``b`` is available through ``curv_ratio`` but is not
    needed: the curvature block of the initial value has only to be O_p(1)
    accurate, and the estimate is numerically indistinguishable over
    ``curv_ratio`` in [1, 3].
    """
    t0 = time.perf_counter()
    if pilot_ratio is None:
        pilot_ratio = default_pilot_ratio(family)
    h1 = pilot_ratio * h
    centres = pilot_centres(h1)
    pilot, fails, _ = clsir_pilot(idx, centres, h1, family)
    b = smooth_ratio * h1
    level, deriv, curv = local_poly_smooth(centres, pilot, grid, b, degree=2,
                                           return_curv=True)
    if curv_ratio != 1.0:
        _, _, curv = local_poly_smooth(centres, pilot, grid, curv_ratio * b,
                                       degree=2, return_curv=True)
    est, bad = _one_step_from_pilot(idx, grid, h, family, level, deriv, steps,
                                    curv=curv if quadratic else None)
    if bias_correct and not quadratic:
        est = est - 0.5 * MU2_EPAN * h * h * curv
    return FitResult(est, time.perf_counter() - t0, fails, bad, steps)


def fit_lc_os(
    idx: LocalIndex,
    grid: np.ndarray,
    h: float,
    family: str,
    pilot_ratio: float | None = None,
    smooth_ratio: float = 1.2,
    steps: int = 1,
) -> FitResult:
    t0 = time.perf_counter()
    if pilot_ratio is None:                 # the ablation gets the same rule
        pilot_ratio = default_pilot_ratio(family)
    h1 = pilot_ratio * h
    centres = pilot_centres(h1)
    pilot, fails = lc_pilot(idx, centres, h1, family)
    b = smooth_ratio * h1
    level, deriv = local_poly_smooth(centres, pilot, grid, b, degree=2)
    est, bad = _one_step_from_pilot(idx, grid, h, family, level, deriv, steps)
    return FitResult(est, time.perf_counter() - t0, fails, bad, steps)


def fit_local_mle(
    idx: LocalIndex,
    grid: np.ndarray,
    h: float,
    family: str,
    warm_start: bool = False,
) -> FitResult:
    """Fully iterated local-linear maximum likelihood estimator.

    ``warm_start`` starts each local fit from the local-linear translation of the
    previous grid point's solution, which is the fastest sensible implementation
    and does not change the estimate, the fit being iterated to convergence.
    """
    t0 = time.perf_counter()
    q = idx.x.shape[1]
    beta_glob, _, _ = fit_wglm(idx.x, idx.y, np.ones(idx.n), family, max_iter=60)
    theta_glob = np.concatenate((beta_glob, np.zeros(q)))
    est = np.zeros((grid.size, q))
    fails = 0
    iters = 0
    prev = None
    for j, t in enumerate(grid):
        win = idx.window(t, h)
        if win is None:
            est[j] = beta_glob
            fails += 1
            prev = None
            continue
        du, w, xw, yw, _ = win
        dmat = np.column_stack((xw, xw * du[:, None]))
        start = theta_glob
        if warm_start and prev is not None:
            start = prev.copy()
            start[:q] = start[:q] + (t - grid[j - 1]) * start[q:]
        theta, conv, it = fit_wglm(dmat, yw, w, family, beta0=start, max_iter=60,
                                   n_eff=idx.n * h)
        if warm_start and not conv:
            theta, conv, it2 = fit_wglm(dmat, yw, w, family, beta0=theta_glob,
                                        max_iter=60, n_eff=idx.n * h)
            it += it2
        est[j] = theta[:q]
        prev = theta if conv else None
        fails += int(not conv)
        iters += it
    return FitResult(est, time.perf_counter() - t0, fails, 0, iters)


def fit_local_firth(
    idx: LocalIndex,
    grid: np.ndarray,
    h: float,
    family: str,
    max_iter: int = 60,
    tol: float = 1e-9,
) -> FitResult:
    """Local-linear likelihood with the Jeffreys-prior (Firth) penalty.

    For a binary response the modified score of Firth (1993) replaces the
    residual y_i - mu_i by y_i - mu_i + lev_i (1/2 - mu_i), with lev_i the
    weighted leverage.  The resulting estimate is finite even under complete
    separation (Kosmidis and Firth, 2021), which makes it the natural penalized
    competitor for the existence comparison of Section 5.  For a non-binary
    family the penalty is not defined in this form and the fit reduces to the
    unpenalized local MLE.
    """
    if family != "binomial":
        return fit_local_mle(idx, grid, h, family)
    t0 = time.perf_counter()
    q = idx.x.shape[1]
    est = np.zeros((grid.size, q))
    fails = 0
    iters = 0
    for j, t in enumerate(grid):
        win = idx.window(t, h)
        if win is None:
            fails += 1
            continue
        du, w, xw, yw, _ = win
        dmat = np.column_stack((xw, xw * du[:, None]))
        theta = np.zeros(2 * q)
        conv = False
        for it in range(1, max_iter + 1):
            eta = dmat @ theta
            mu, wt = mean_and_weight(eta, family)
            info = dmat.T @ (dmat * (w * wt)[:, None])
            try:
                inv = np.linalg.inv(
                    info + 1e-10 * max(np.trace(info) / info.shape[0], 1.0)
                    * np.eye(info.shape[0]))
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(info)
            lev = (w * wt) * np.einsum("ij,jk,ik->i", dmat, inv, dmat)
            score = dmat.T @ (w * (yw - mu) + w * lev * (0.5 - mu))
            step = ridge_solve(info, score, 1e-8, idx.n * h)
            nrm = float(np.linalg.norm(step))
            if not np.isfinite(nrm):
                break
            if nrm > 15.0:
                step *= 15.0 / nrm
            theta = theta + step
            iters += 1
            if nrm <= tol * (1.0 + float(np.linalg.norm(theta))):
                conv = True
                break
        fails += int(not conv)
        est[j] = theta[:q]
    return FitResult(est, time.perf_counter() - t0, fails, 0, iters)


def fit_os_local_mle(
    idx: LocalIndex,
    grid: np.ndarray,
    h: float,
    family: str,
    n_refresh: int = 5,
    steps: int = 1,
) -> FitResult:
    """Cai, Fan and Li (2000), Section 3.5: sequential propagation with refreshing.

    The grid is split into ``n_refresh`` consecutive blocks.  A fully iterated
    local MLE is computed at the centre of each block and propagated outwards
    in both directions by single Newton steps, the initial value at each new
    point being the local-linear translation of the previous solution.
    """
    t0 = time.perf_counter()
    q = idx.x.shape[1]
    g = grid.size
    theta = np.zeros((g, 2 * q))
    beta_glob, _, _ = fit_wglm(idx.x, idx.y, np.ones(idx.n), family, max_iter=60)
    theta_glob = np.concatenate((beta_glob, np.zeros(q)))

    edges = np.linspace(0, g, min(n_refresh, g) + 1).astype(int)
    blocks = [(edges[k], edges[k + 1]) for k in range(len(edges) - 1) if edges[k + 1] > edges[k]]

    fails = 0
    iters = 0
    bad = 0

    def one_step_at(j: int, init: np.ndarray) -> int:
        win = idx.window(grid[j], h)
        if win is None:
            theta[j] = init
            return 1
        du, w, xw, yw, _ = win
        dmat = np.column_stack((xw, xw * du[:, None]))
        cur, ok = init, True
        for _ in range(steps):
            cur, ok_s = newton_step(dmat, yw, w, family, cur, n_eff=idx.n * h,
                                    scale=np.repeat([1.0, h], q))
            ok = ok and ok_s
        theta[j] = cur
        return int(not ok)

    for lo, hi in blocks:
        anchor = (lo + hi) // 2
        win = idx.window(grid[anchor], h)
        if win is None:
            theta[anchor] = theta_glob
            fails += 1
        else:
            du, w, xw, yw, _ = win
            dmat = np.column_stack((xw, xw * du[:, None]))
            th, conv, it = fit_wglm(dmat, yw, w, family, beta0=theta_glob, max_iter=60,
                                    n_eff=idx.n * h)
            theta[anchor] = th
            fails += int(not conv)
            iters += it
        for j in range(anchor + 1, hi):
            init = theta[j - 1].copy()
            init[:q] = init[:q] + (grid[j] - grid[j - 1]) * init[q:]
            bad += one_step_at(j, init)
        for j in range(anchor - 1, lo - 1, -1):
            init = theta[j + 1].copy()
            init[:q] = init[:q] + (grid[j] - grid[j + 1]) * init[q:]
            bad += one_step_at(j, init)
    return FitResult(theta[:, :q], time.perf_counter() - t0, fails, bad, iters)


def fit_fan_zhang(
    idx: LocalIndex,
    grid: np.ndarray,
    h: float,
    family: str,
    under_ratio: float = 0.5,
    degree: int = 3,
) -> FitResult:
    """Two-step estimator of Fan and Zhang (1999), generalized-response version.

    Step one computes the fully iterated local-linear likelihood estimate on a fine
    grid at the undersmoothing bandwidth ``under_ratio * h``; step two re-smooths
    that curve by a local polynomial of the given degree at bandwidth ``h``.
    """
    t0 = time.perf_counter()
    q = idx.x.shape[1]
    h1 = under_ratio * h
    m = max(int(math.ceil(2.0 / max(h1, 1e-3))) + 1, 25)
    centres = np.linspace(0.0, 1.0, m)
    beta_glob, _, _ = fit_wglm(idx.x, idx.y, np.ones(idx.n), family, max_iter=60)
    theta_glob = np.concatenate((beta_glob, np.zeros(q)))
    pilot = np.zeros((centres.size, q))
    fails = 0
    iters = 0
    prev = None
    for j, t in enumerate(centres):
        win = idx.window(t, h1)
        if win is None:
            pilot[j] = beta_glob if prev is None else prev[:q]
            fails += 1
            continue
        du, w, xw, yw, _ = win
        dmat = np.column_stack((xw, xw * du[:, None]))
        start = theta_glob if prev is None else prev
        th, conv, it = fit_wglm(dmat, yw, w, family, beta0=start, max_iter=60,
                                n_eff=idx.n * h1)
        pilot[j] = th[:q]
        prev = th if conv else None
        fails += int(not conv)
        iters += it
    level, _ = local_poly_smooth(centres, pilot, grid, h, degree=degree)
    return FitResult(level, time.perf_counter() - t0, fails, 0, iters)


def grouped_fits(
    idx: LocalIndex,
    m: int,
    family: str,
    bias_correct: bool,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Grouped GLM pre-estimates, optionally Cox-Snell bias corrected."""
    n, q = idx.n, idx.x.shape[1]
    k = n // m
    centres = np.zeros(k)
    values = np.zeros((k, q))
    fails = 0
    ones = np.ones(m)
    for g in range(k):
        sl = slice(g * m, (g + 1) * m)
        xg, yg, ug = idx.x[sl], idx.y[sl], idx.u[sl]
        centres[g] = float(np.mean(ug))
        beta, conv, _ = fit_wglm(xg, yg, ones, family, max_iter=50)
        if not conv:
            fails += 1
        if bias_correct and conv and family != "gaussian":
            eta = xg @ beta
            mu, wt = mean_and_weight(eta, family)
            info = xg.T @ (xg * wt[:, None])
            try:
                inv = np.linalg.inv(info + 1e-10 * np.eye(q) * max(np.trace(info) / q, 1.0))
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(info)
            lev = wt * np.einsum("ij,jk,ik->i", xg, inv, xg)
            xi = zeta_prime(mu, family) * lev
            beta = beta + 0.5 * inv @ (xg.T @ xi)
        values[g] = beta
    return centres, values, fails


def fit_gla(
    idx: LocalIndex,
    grid: np.ndarray,
    h: float,
    family: str,
    m: int,
    bias_correct: bool,
) -> FitResult:
    t0 = time.perf_counter()
    centres, values, fails = grouped_fits(idx, m, family, bias_correct)
    level, _ = local_poly_smooth(centres, values, grid, h, degree=1)
    return FitResult(level, time.perf_counter() - t0, fails, 0, 0)


# --------------------------------------------------------------------------
# asymptotic variance (Cai, Fan and Li, 2000, Theorem 1)
# --------------------------------------------------------------------------


def sandwich_variance(
    idx: LocalIndex,
    target: float,
    h: float,
    family: str,
    a_hat: np.ndarray,
    a_deriv: np.ndarray,
    nu0: float = NU0_EPAN,
    dispersion: float = 1.0,
) -> np.ndarray:
    """Plug-in estimate of sigma^2 nu0 Gamma(u)^{-1} / (n h f_U(u)) for a_hat.

    ``nu0`` is NU0_EPAN for the local linear estimator and NU0Q_EPAN for the
    local quadratic one, and ``dispersion`` is the Pearson estimate of sigma^2,
    which is one for the Bernoulli and Poisson families.
    """
    win = idx.window(target, h)
    if win is None:
        return np.full((a_hat.size, a_hat.size), np.nan)
    du, w, xw, _, _ = win
    eta = xw @ a_hat + (xw @ a_deriv) * du
    _, wt = mean_and_weight(eta, family)
    gamma_hat = (xw * (w * wt)[:, None]).T @ xw / (idx.n * h)
    q = a_hat.size
    scale = max(float(np.trace(gamma_hat)) / q, 1e-12)
    gamma_hat = gamma_hat + 1e-10 * scale * np.eye(q)
    return dispersion * nu0 * np.linalg.inv(gamma_hat) / (idx.n * h)
