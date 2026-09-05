"""Wall clock of the update step against the number of worker processes.

Theorem 2 says the initial value at u uses no quantity computed at any other
evaluation point, so Step 4 of Algorithm 1 is embarrassingly parallel.  The
paper claims that as a computational advantage over the propagated one-step
estimator, whose recursion is sequential, and this script measures it rather
than asserting it.

Only Step 4 is distributed: the pilot and the smoothing pass are computed once
in the parent and their output is broadcast to the workers at start-up, which
is how a real implementation would do it.  Reported is the wall clock of the
update relative to the same work on one worker.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Each worker must be single threaded, or the linear algebra inside one worker
# spawns its own thread pool and w workers oversubscribe the machine w-fold.
# The children are spawned and re-import numpy, so setting this before the pool
# is created is enough; it is set before numpy is imported here as well so that
# the parent's own timing is on the same footing.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clsir_core import (  # noqa: E402
    LocalIndex, clsir_pilot, local_poly_smooth, pilot_centres,
    _one_step_from_pilot, slice_labels,
)
import clsir_study as S  # noqa: E402

_STATE: dict = {}


def _init(u, x, y, lab, h, family, level, deriv, curv, grid):
    _STATE["idx"] = LocalIndex(u, x, y, lab)
    _STATE.update(h=h, family=family, level=level, deriv=deriv, curv=curv,
                  grid=grid)


def _chunk(args):
    lo, hi, quadratic = args
    s = _STATE
    est, _ = _one_step_from_pilot(
        s["idx"], s["grid"][lo:hi], s["h"], s["family"],
        s["level"][lo:hi], s["deriv"][lo:hi], 1,
        curv=s["curv"][lo:hi] if quadratic else None)
    return est


def run(design: str, n: int, n_grid: int, workers: int, quadratic: bool,
        seed: int, repeats: int) -> float:
    d = S.DESIGNS[design]
    rng = np.random.default_rng(seed)
    u, x, y = S.simulate(rng, n, d)
    lab = slice_labels(y, d.family, 5)
    idx = LocalIndex(u, x, y, lab)
    grid = np.linspace(0.10, 0.90, n_grid)
    h = S.base_bandwidth(n, d)
    h1 = 1.5 * h
    centres = pilot_centres(h1)
    pilot, _, _ = clsir_pilot(idx, centres, h1, d.family)
    level, deriv, curv = local_poly_smooth(centres, pilot, grid, 1.2 * h1,
                                           degree=2, return_curv=True)
    init = (u, x, y, lab, h, d.family, level, deriv, curv, grid)

    edges = np.linspace(0, n_grid, workers + 1).astype(int)
    jobs = [(int(edges[k]), int(edges[k + 1]), quadratic)
            for k in range(workers) if edges[k + 1] > edges[k]]

    best = np.inf
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=init) as pool:
        list(pool.map(_chunk, jobs[:1]))          # warm the pool
        for _ in range(repeats):
            t0 = time.perf_counter()
            list(pool.map(_chunk, jobs))
            best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", default=["bin10", "poi10"])
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--grid-size", type=int, default=1600)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", type=Path, default=Path("../results/v3"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for dz in a.designs:
        for w in a.workers:
            for quad in (False, True):
                t = run(dz, a.n, a.grid_size, w, quad, a.seed, a.repeats)
                rows.append({"design": dz, "n": a.n, "grid_size": a.grid_size,
                             "workers": w, "quadratic": int(quad), "runtime": t})
                print(f"{dz:8s} workers={w} quad={int(quad)} {t:7.3f}s",
                      flush=True)
    df = pd.DataFrame(rows)
    # the headline figure uses the local linear update
    df[df.quadratic == 0].to_csv(a.out / "parallel_runs.csv", index=False)
    df.to_csv(a.out / "parallel_runs_full.csv", index=False)
    print(f"wrote {a.out}/parallel_runs.csv")


if __name__ == "__main__":
    main()
