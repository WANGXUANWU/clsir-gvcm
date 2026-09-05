"""Compute and cache the mean fitted coefficient functions for every example."""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clsir_study as S
from clsir_core import LocalIndex, slice_labels

MS = ["CLSIR-OS", "CLSIR-QS", "LMLE", "OSL", "FZ", "FIRTH", "CGA"]
# the paper's four examples first, then the two the supplement adds
DESIGNS = ["bin2", "bin10", "poi10", "gau2", "poi2", "binskew"]
OUT = Path("../results/v5/mean_curves.npz")


def _one(job):
    design, n, c, r = job
    d = S.DESIGNS[design]
    grid = np.linspace(0.10, 0.90, 101)
    h = c * S.base_bandwidth(n, d)
    msize = S.group_size(n, d.p + 1)
    rng = np.random.default_rng(444_000 + r)
    u, x, y = S.simulate(rng, n, d)
    idx = LocalIndex(u, x, y, slice_labels(y, d.family, 5))
    out = {}
    for k in MS:
        e = S.fit_method(k, idx, grid, h, d.family, msize).estimate
        ok = bool(np.all(np.isfinite(e)) and np.max(np.abs(e)) <= 50)
        out[k] = (e if ok else None)
    return out


def mean_curves(design, n, c, reps):
    """Mean fitted curve, and the pointwise mean squared error behind it.

    The mean curve alone shows bias and hides variance, which is the larger
    part of the error for a binary response; ``mse`` is the quantity the
    integrated squared error of Table 1 actually integrates, so it is stored
    alongside and the two can be plotted against each other.
    """
    d = S.DESIGNS[design]
    grid = np.linspace(0.10, 0.90, 101)
    truth = S.coefficients(grid, d)
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = list(pool.map(_one, [(design, n, c, r) for r in range(reps)]))
    mean, frac, mse = {}, {}, {}
    for k in MS:
        good = [o[k] for o in res if o[k] is not None]
        frac[k] = len(good) / max(len(res), 1)
        if good:
            mean[k] = np.mean(good, axis=0)
            mse[k] = np.mean([(g - truth) ** 2 for g in good], axis=0)
        else:
            mean[k] = np.full((grid.size, d.p + 1), np.nan)
            mse[k] = np.full((grid.size, d.p + 1), np.nan)
    return grid, mean, frac, truth, mse


def main(n: int = 500, c: float = 0.80, reps: int = 200,
         out: Path | None = None) -> None:
    out = OUT if out is None else Path(out)
    store = {"c": c, "reps": reps, "n": n, "designs": np.array(DESIGNS)}
    for dz in DESIGNS:
        grid, mean, frac, truth, mse = mean_curves(dz, n, c, reps)
        store[f"{dz}_grid"] = grid
        store[f"{dz}_truth"] = truth
        for m in MS:
            store[f"{dz}_mean_{m}"] = mean[m]
            store[f"{dz}_frac_{m}"] = frac[m]
            store[f"{dz}_mse_{m}"] = mse[m]
        print(dz + ": " + "  ".join(
            f"{m} bias2 {np.mean((mean[m] - truth) ** 2):.4f} "
            f"mse {np.nanmean(mse[m]):.4f} ({100 * frac[m]:.0f}%)"
            for m in MS), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **store)
    print(f"wrote {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--c", type=float, default=0.80)
    a = ap.parse_args()
    main(n=a.n, c=a.c, reps=a.reps, out=a.out)
