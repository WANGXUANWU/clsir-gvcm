"""Figures for the paper and the supplement.

Every figure is drawn at the width it is printed at, so that no scaling takes
place and the type size on the page is the type size set here.

Layout follows one convention throughout: a rectangular grid of small panels
spanning the full text width, a bold lower-case letter marking each row, axis
labels only on the outer panels, and one shared legend below the grid.

main text
  fig_acc.png          accuracy, four examples x two sample sizes   (8 panels)
  fig_curves.png       two slope functions, each above its error  (16 panels)
  fig_time.png         cost, four examples x two sample sizes       (8 panels)
  fig_real.png         the application, two models x five effects   (10 panels)

supplement
  fig_acc_all.png      accuracy over all six examples
  fig_curves_2000.png  the paired figure at the larger sample size
  fig_curves_bias.png  the bias on its own, which the RMSE rows absorb
  fig_curves_a*.png    the fitted curves themselves, every example
  fig_cross.png        where the linear and quadratic updates cross
  fig_time_all.png     cost over all six examples
  fig_parallel.png     the parallel scaling and the cross-validated speed-up
  fig_real_diag.png    calibration, deviance, existence and cost
"""
from __future__ import annotations

import argparse
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

W = 6.5          # text width of the article class with one inch margins
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.6,
    "axes.titlesize": 7.6,
    "axes.labelsize": 7.6,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 6.6,
    "ytick.labelsize": 6.6,
    "axes.linewidth": 0.55,
    "axes.edgecolor": "#444444",
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.2,
    "ytick.major.size": 2.2,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "lines.solid_capstyle": "round",
})

# One colour per estimator, held fixed across every figure.  The two proposed
# estimators share the blue family so that they read as one construction.
STYLE = {
    "CLSIR-QS": dict(color="#1a6fb0", ls="-",  lw=1.7, marker="o", ms=3.3, zorder=6),
    "CLSIR-OS": dict(color="#0d3b5c", ls="-",  lw=1.3, marker="s", ms=2.9, zorder=5),
    "LMLE":     dict(color="#d1651a", ls="--", lw=1.1, marker="v", ms=2.9, zorder=3),
    "OSL":      dict(color="#c0392b", ls="-.", lw=1.1, marker="^", ms=2.9, zorder=3),
    "FZ":       dict(color="#7a5195", ls=":",  lw=1.3, marker="D", ms=2.5, zorder=3),
    "FIRTH":    dict(color="#2e8b57", ls="--", lw=1.1, marker="P", ms=3.0, zorder=3),
    "CGA":      dict(color="#8a8a8a", ls="-",  lw=0.9, marker="x", ms=3.0, zorder=2),
}
LABEL = {"CLSIR-QS": "CLSIR-QS", "CLSIR-OS": "CLSIR-OS", "LMLE": "LMLE",
         "OSL": "OSL", "FZ": "FZ", "FIRTH": "Firth", "CGA": "CGA"}
MS = ["CLSIR-QS", "CLSIR-OS", "LMLE", "OSL", "FZ", "FIRTH", "CGA"]
# the paired subset: the estimators that all solve the same local likelihood
PAIR = ["CLSIR-QS", "CLSIR-OS", "LMLE", "OSL", "FZ"]

# The four examples the paper carries are numbered first, the two the
# supplement adds follow; this order drives every figure and every table.
DESIGNS = ["bin2", "bin10", "poi10", "gau2", "poi2", "binskew"]
MAIN_DESIGNS = ["bin2", "bin10", "poi10", "gau2"]
TITLE = {"bin2": "Example 1", "bin10": "Example 2", "poi10": "Example 3",
         "gau2": "Example 4", "poi2": "Example 5", "binskew": "Example 6"}
SHORT = {"bin2": "Ex. 1", "bin10": "Ex. 2", "poi10": "Ex. 3",
         "gau2": "Ex. 4", "poi2": "Ex. 5", "binskew": "Ex. 6"}
EXCOL = {"bin2": "#1a6fb0", "bin10": "#0d3b5c", "poi10": "#d1651a",
         "gau2": "#7a5195", "poi2": "#2e8b57", "binskew": "#c0392b"}
EXMK = {"bin2": "o", "bin10": "s", "poi10": "^", "gau2": "v",
        "poi2": "D", "binskew": "P"}


# --------------------------------------------------------------------------
# layout helpers
# --------------------------------------------------------------------------


def panel_grid(nrow, ncol, height, wspace=0.34, hspace=0.42,
               left=0.075, right=0.988, top=0.945, bottom=0.135):
    fig, ax = plt.subplots(nrow, ncol, figsize=(W, height),
                           gridspec_kw=dict(wspace=wspace, hspace=hspace))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    return fig, np.atleast_2d(ax)


def row_letters(fig, ax, x=0.008):
    """Bold lower-case letter at the top left of each row of panels."""
    for r in range(ax.shape[0]):
        box = ax[r, 0].get_position()
        fig.text(x, box.y1 + 0.004, string.ascii_lowercase[r],
                 fontsize=8.4, fontweight="bold", va="bottom", ha="left")


def shared_legend(fig, methods, y=0.028, ncol=None):
    handles = [plt.Line2D([], [], label=LABEL[m], **STYLE[m]) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=ncol or len(methods),
               frameon=False, handlelength=2.1, columnspacing=1.4,
               handletextpad=0.5, bbox_to_anchor=(0.5, y - 0.02))


def tidy(ax, xlabel=None, ylabel=None, title=None):
    ax.grid(alpha=0.20, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=3.5)


def logy(ax):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 3.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=320)
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------
# accuracy against the bandwidth
# --------------------------------------------------------------------------


def keep_of(cell):
    """Replications in which every local likelihood estimator succeeded."""
    piv = cell.pivot_table(index="seed", columns="method", values="blown", aggfunc="max")
    have = [m for m in PAIR if m in piv.columns]
    return set(piv.index[piv[have].max(axis=1) == 0])


def cell_means(g, design, n, methods):
    sub = g[(g.design == design) & (g.n == n)]
    cs = sorted(sub.c.unique())
    out = {m: [] for m in methods}
    for c in cs:
        cell = sub[sub.c == c]
        k = keep_of(cell)
        for m in methods:
            v = (cell[cell.method == m]["ise"] if m not in PAIR
                 else cell[(cell.method == m) & (cell.seed.isin(k))]["ise"])
            v = v[np.isfinite(v)]
            out[m].append(float(v.mean()) if len(v) else np.nan)
    return np.array(cs), out


def ise_panel(ax, g, design, n, methods, show_x, show_y):
    cs, v = cell_means(g, design, n, methods)
    for m in methods:
        if m in v and np.any(np.isfinite(v[m])):
            ax.plot(cs, v[m], label=LABEL[m], **STYLE[m])
    logy(ax)
    # the frame follows the estimators that solve the same local likelihood, so
    # that a grouped estimator orders of magnitude worse leaves it
    ref = np.concatenate([v[m] for m in methods if m in PAIR])
    ref = ref[np.isfinite(ref) & (ref > 0)]
    if ref.size:
        ax.set_ylim(ref.min() / 1.8, ref.max() * 2.4)
    # label a subset of the bandwidth constants: six labels do not fit across a
    # panel this narrow, and the markers show where the six values are
    show = [c for c in cs if c in (0.5, 0.8, 1.3, 1.7)] or list(cs)
    ax.set_xticks(show)
    ax.set_xticklabels([f"{c:g}" for c in show])
    tidy(ax,
         xlabel="bandwidth constant $c$" if show_x else None,
         ylabel="mean ISE" if show_y else None,
         title=f"{TITLE[design]}, $n={n}$")


def fig_acc(g, out: Path, designs=None, methods=None):
    """Two panels per example: sample sizes across, examples in rows of two."""
    designs = designs or DESIGNS
    methods = methods or [m for m in MS if m in set(g.method)]
    ns = sorted(g.n.unique())
    nrow = len(designs) // 2
    fig, ax = panel_grid(nrow, 4, 6.3, hspace=0.34, bottom=0.135, top=0.955)
    for r in range(nrow):
        for k, dz in enumerate(designs[2 * r:2 * r + 2]):
            for j, n in enumerate(ns):
                col = 2 * k + j
                ise_panel(ax[r, col], g, dz, n, methods,
                          show_x=(r == nrow - 1), show_y=(col == 0))
    row_letters(fig, ax)
    shared_legend(fig, methods, y=0.040, ncol=len(methods))
    save(fig, out)


# --------------------------------------------------------------------------
# fitted coefficient functions
# --------------------------------------------------------------------------


def load_curves(path: Path):
    z = np.load(path, allow_pickle=True)
    have = [m for m in MS if f"{list(TITLE)[0]}_mean_{m}" in z]
    out = {}
    for dz in DESIGNS:
        if f"{dz}_grid" not in z:
            continue
        mse = ({m: z[f"{dz}_mse_{m}"] for m in have}
               if f"{dz}_mse_{have[0]}" in z else None)
        out[dz] = (z[f"{dz}_grid"],
                   {m: z[f"{dz}_mean_{m}"] for m in have},
                   {m: float(z[f"{dz}_frac_{m}"]) for m in have},
                   z[f"{dz}_truth"],
                   mse)
    return out, float(z["c"]), int(z["n"]), have


def curve_panel(ax, cur, j, methods, show_x, show_y, title=None, bias=False,
                mode=None):
    """The mean fitted curve, its deviation from the truth, or its pointwise RMSE.

    ``mode="rmse"`` plots the square root of the pointwise mean squared error,
    which is what the integrated squared error of Table 1 integrates.  The
    other two modes show the mean fitted curve only, so they display bias and
    hide variance -- which for a binary response is the larger part of the
    error, and the part the pilot is designed to control.  Plotting the
    deviation rather than the curve itself is what makes the count and Gaussian
    designs readable: there every estimator sits on top of the truth at the
    scale of the curve.
    """
    grid, mean, frac, truth = cur[0], cur[1], cur[2], cur[3]
    mse = cur[4] if len(cur) > 4 else None
    shown = [m for m in methods if m in mean and frac.get(m, 0.0) >= 0.05]

    if mode == "rmse":
        if mse is None:
            raise ValueError("mode='rmse' needs a run that stored the mse block")
        for m in shown:
            st = {k: v for k, v in STYLE[m].items() if k not in ("marker", "ms")}
            ax.plot(grid, np.sqrt(np.maximum(mse[m][:, j], 0.0)),
                    label=LABEL[m], **st)
        ax.set_yscale("log")
        # a decade of range often leaves only two labelled ticks, so the minor
        # decades are labelled too
        ax.yaxis.set_major_locator(
            matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        # the frame follows the estimators that solve the same local likelihood,
        # as in Figure 1, so the grouped estimator leaves it rather than
        # flattening everything else onto a single line
        ref = np.concatenate([np.sqrt(np.maximum(mse[m][:, j], 0.0))
                              for m in shown if m in PAIR or m == "FIRTH"])
        ref = ref[np.isfinite(ref) & (ref > 0)]
        if ref.size:
            ax.set_ylim(ref.min() / 1.45, ref.max() * 1.7)
        ax.set_xticks([0.2, 0.5, 0.8])
        tidy(ax, xlabel="$u$" if show_x else None,
             ylabel=f"RMSE, $a_{j}$" if show_y else None, title=title)
        return

    if bias:
        ax.axhline(0.0, color="#b9b9b9", lw=2.2, zorder=1,
                   solid_capstyle="round")
    else:
        ax.plot(grid, truth[:, j], color="#b9b9b9", lw=2.8, zorder=1,
                label="truth", solid_capstyle="round")
    for m in shown:
        st = {k: v for k, v in STYLE[m].items() if k not in ("marker", "ms")}
        y = mean[m][:, j] - truth[:, j] if bias else mean[m][:, j]
        ax.plot(grid, y, label=LABEL[m], **st)
    # the frame follows the truth and the proposed estimates, so that a wildly
    # biased competitor leaves it rather than flattening everything else
    ours = [m for m in ("CLSIR-QS", "CLSIR-OS") if m in mean]
    if bias:
        ref = np.concatenate([mean[m][:, j] - truth[:, j] for m in ours]
                             + [np.zeros(1)]) if ours else np.zeros(2)
        hi = float(np.nanmax(np.abs(ref)))
        hi = max(hi, 1e-9) * 2.6
        ax.set_ylim(-hi, hi)
    else:
        ref = np.concatenate([truth[:, j]]
                             + [mean[m][:, j] for m in ours])
        lo, hi = float(np.nanmin(ref)), float(np.nanmax(ref))
        span = max(hi - lo, 1e-6)
        ax.set_ylim(lo - 0.20 * span, hi + 0.24 * span)
    ax.set_xticks([0.2, 0.5, 0.8])
    lab = f"$\\widehat a_{j}(u)-a_{j}(u)$" if bias else f"$a_{j}(u)$"
    tidy(ax, xlabel="$u$" if show_x else None,
         ylabel=lab if show_y else None, title=title)


def _curve_legend(fig, methods, bias):
    """Estimators only.

    The grey reference line is described in the caption rather than given a
    legend entry of its own, so that this figure and fig_acc carry the same
    key and can be read against each other.
    """
    handles = [plt.Line2D([], [], label=LABEL[m],
                          **{k: v for k, v in STYLE[m].items()
                             if k not in ("marker", "ms")}) for m in methods]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, handlelength=2.1, columnspacing=1.3,
               handletextpad=0.5, bbox_to_anchor=(0.5, 0.012))


def fig_curves(cur, out: Path, designs, methods, ncol=None, columns=None,
               bias=False, mode=None):
    """Rows are the three coefficient functions, columns the examples."""
    if columns is None:
        designs = [d for d in designs if d in cur]
        columns = [(cur, d, TITLE[d]) for d in designs]
    ncol = ncol or len(columns)
    fig, ax = panel_grid(3, ncol, 5.3, hspace=0.38, bottom=0.125, top=0.95)
    for r, j in enumerate((0, 1, 2)):
        for c, (cset, dz, title) in enumerate(columns):
            curve_panel(ax[r, c], cset[dz], j, methods,
                        show_x=(r == 2), show_y=(c == 0),
                        title=title if r == 0 else None, bias=bias, mode=mode)
    row_letters(fig, ax)
    _curve_legend(fig, methods, bias)
    save(fig, out)


def fig_curves_pair(cur, out: Path, designs, methods, coefs=(1, 2)):
    """Two coefficient functions, each as a fitted curve above its own error.

    The curve rows say whether an estimator recovers the shape at all, which is
    where the estimators that need a finite local maximizer come apart in the
    Bernoulli designs; the error rows say how large the error is, which is what
    Table 1 integrates and is the only thing that separates the estimators once
    every one of them recovers the shape.
    """
    designs = [d for d in designs if d in cur]
    nrow = 2 * len(coefs)
    fig, ax = panel_grid(nrow, len(designs), 7.0, hspace=0.36, wspace=0.32,
                         bottom=0.092, top=0.963)
    for k, j in enumerate(coefs):
        for c, dz in enumerate(designs):
            curve_panel(ax[2 * k, c], cur[dz], j, methods,
                        show_x=False, show_y=(c == 0),
                        title=TITLE[dz] if k == 0 else None)
            curve_panel(ax[2 * k + 1, c], cur[dz], j, methods,
                        show_x=(k == len(coefs) - 1), show_y=(c == 0),
                        mode="rmse")
    row_letters(fig, ax)
    _curve_legend(fig, methods, False)
    save(fig, out)


def fig_curves_one_coef(cur, j, out: Path, designs, methods):
    """One coefficient function, every example: three rows of two panels."""
    designs = [d for d in designs if d in cur]
    nrow = (len(designs) + 1) // 2
    fig, ax = panel_grid(nrow, 2, 1.75 * nrow + 1.05, hspace=0.40,
                         bottom=0.115, top=0.955)
    for k, dz in enumerate(designs):
        r, c = k // 2, k % 2
        curve_panel(ax[r, c], cur[dz], j, methods,
                    show_x=(r == nrow - 1), show_y=(c == 0), title=TITLE[dz])
    for k in range(len(designs), nrow * 2):
        ax[k // 2, k % 2].axis("off")
    _curve_legend(fig, methods, False)
    save(fig, out)


# --------------------------------------------------------------------------
# where the linear and the quadratic update cross
# --------------------------------------------------------------------------


def fig_cross(g, out: Path):
    """Ratio of the two proposed estimators against the bandwidth."""
    designs = [d for d in DESIGNS if d in set(g.design)]
    ns = sorted(g.n.unique())
    fig, ax = panel_grid(1, 2, 2.7, bottom=0.235, top=0.90, wspace=0.26)
    for j, n in enumerate(ns):
        a = ax[0, j]
        for dz in designs:
            cs, v = cell_means(g, dz, n, ["CLSIR-OS", "CLSIR-QS"])
            a.plot(cs, np.array(v["CLSIR-QS"]) / np.array(v["CLSIR-OS"]),
                   color=EXCOL[dz], marker=EXMK[dz], ms=3.2, lw=1.2,
                   label=SHORT[dz])
        a.axhline(1.0, color="#444444", lw=0.8, ls="--")
        a.set_xticks(cs)
        a.set_xticklabels([f"{c:g}" for c in cs])
        a.set_yscale("log")
        a.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(
            [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4]))
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}"))
        a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        tidy(a, xlabel="bandwidth constant $c$",
             ylabel="ISE of CLSIR-QS / CLSIR-OS" if j == 0 else None,
             title=f"$n={n}$")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False,
               handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(0.5, 0.005))
    save(fig, out)


# --------------------------------------------------------------------------
# computing time
# --------------------------------------------------------------------------


def cost_series(cost, design, methods, n=None):
    sub = cost[cost.design == design]
    if n is not None and "n" in sub.columns:
        sub = sub[sub.n == n]
    sizes = sorted(sub.grid_size.unique())
    alias = {"LMLE": "LMLE-warm"}
    out = {}
    for m in methods:
        key = alias.get(m, m)
        if key not in set(sub.method):
            key = m
        if key not in set(sub.method):
            continue
        out[m] = np.array([1000 * float(
            sub[(sub.method == key) & (sub.grid_size == s)]["runtime"].mean())
            for s in sizes])
    return np.array(sizes), out


def cost_panel(ax, cost, design, methods, show_y, n=None, show_x=True,
               title=None, ylabel=None):
    sizes, v = cost_series(cost, design, methods, n)
    for m in methods:
        if m in v:
            ax.plot(sizes, v[m], label=LABEL[m], **STYLE[m])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 3.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    tidy(ax, xlabel="evaluation points" if show_x else None,
         ylabel=(ylabel if ylabel is not None else "milliseconds for one fit")
         if show_y else None,
         title=title)


def ratio_panel(ax, cost):
    ref = "CLSIR-OS"
    for dz in DESIGNS:
        if dz not in set(cost.design):
            continue
        sizes, v = cost_series(cost, dz, ["LMLE", ref])
        if "LMLE" not in v or ref not in v:
            continue
        ax.plot(sizes, v["LMLE"] / v[ref], ls="-", marker=EXMK[dz], ms=3.2,
                lw=1.2, color=EXCOL[dz], label=SHORT[dz])
    ax.axhline(1.0, color="#444444", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    tidy(ax, xlabel="evaluation points", ylabel="time relative to CLSIR-OS",
         title="Fully iterated estimator")
    ax.legend(frameon=False, handlelength=1.5, fontsize=6.4, ncol=2,
              columnspacing=0.8, loc="upper left")


def parallel_panel(ax, par):
    """Wall clock of Step 4 against the number of worker processes."""
    for dz in sorted(set(par.design), key=lambda d: DESIGNS.index(d)):
        s = par[par.design == dz].groupby("workers")["runtime"].mean()
        w = np.array(s.index, float)
        ax.plot(w, float(s.iloc[0]) / s.values, marker=EXMK[dz], ms=3.2, lw=1.2,
                color=EXCOL[dz], label=SHORT[dz])
    w = np.array(sorted(set(par.workers)), float)
    ax.plot(w, w, color="#444444", lw=0.8, ls="--", label="linear")
    ax.set_xticks(w)
    ax.set_xticklabels([f"{int(x)}" for x in w])
    tidy(ax, xlabel="worker processes", ylabel="speed-up over one worker",
         title="Updates in parallel")
    ax.legend(frameon=False, handlelength=1.5, fontsize=6.4, ncol=2,
              columnspacing=0.8, loc="upper left")


def analysis_panel(ax, cv):
    """Speed-up of a complete cross-validated analysis, every example."""
    designs = [d for d in DESIGNS if d in set(cv.design)]
    ns = sorted(cv.n.unique())
    width = 0.8 / len(ns)
    xs = np.arange(len(designs))
    for j, n in enumerate(ns):
        vals = []
        for dz in designs:
            s = cv[(cv.design == dz) & (cv.n == n)]
            ours = float(s[s.method == "CLSIR-OS"]["runtime"].mean())
            cand = [float(s[s.method == m]["runtime"].mean())
                    for m in ("LMLE", "LMLE-warm") if m in set(s.method)]
            vals.append(min(cand) / ours if cand and ours > 0 else np.nan)
        ax.bar(xs + (j - (len(ns) - 1) / 2) * width, vals, width * 0.88,
               label=f"$n={n}$", color=("#1a6fb0", "#0d3b5c")[j],
               edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="#c0392b", lw=1.0)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[d] for d in designs], fontsize=6.4)
    lo, hi = ax.get_ylim()
    ax.set_ylim(min(lo, 0.45), hi * 1.45)
    ax.set_yticks([0.5, 1, 2, 5, 10])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    tidy(ax, ylabel="speed-up of a complete analysis",
         title="Cross-validated analysis")
    ax.legend(frameon=False, ncol=2, fontsize=6.6, columnspacing=0.9,
              loc="upper right")


def fig_time(cost, out: Path, designs=None, methods=None):
    """Cost against the resolution of the fitted curve.

    Rows are the two sample sizes and columns the examples, so the figure
    carries the same grid as Figures 1 and 2 and the same seven-estimator key.
    ``LMLE`` is drawn as the warm started implementation, which is the faster
    of its two and therefore the conservative comparison.
    """
    designs = designs or MAIN_DESIGNS
    designs = [d for d in designs if d in set(cost.design)]
    methods = methods or [m for m in MS if m in set(cost.method)
                          or (m == "LMLE" and "LMLE-warm" in set(cost.method))]
    ns = sorted(cost.n.unique()) if "n" in cost.columns else [None]
    fig, ax = panel_grid(len(ns), len(designs), 4.7, hspace=0.34, wspace=0.30,
                         bottom=0.15, top=0.945)
    for r, n in enumerate(ns):
        for c, dz in enumerate(designs):
            cost_panel(ax[r, c], cost, dz, methods, show_y=(c == 0), n=n,
                       show_x=(r == len(ns) - 1),
                       title=TITLE[dz] if r == 0 else None,
                       ylabel=("milliseconds for one fit"
                               + (f", $n={n}$" if n is not None else "")))
    row_letters(fig, ax)
    shared_legend(fig, methods, y=0.045, ncol=len(methods))
    save(fig, out)


def fig_parallel(cv, par, cost, out: Path):
    """The two computational claims the cost curves do not carry.

    Left, the update distributed over worker processes; right, the speed-up of
    a complete cross-validated analysis over the fully iterated estimator.
    """
    fig, ax = panel_grid(1, 2, 2.5, wspace=0.28, bottom=0.20, top=0.90)
    if par is not None and len(par):
        parallel_panel(ax[0, 0], par)
    else:
        ratio_panel(ax[0, 0], cost)
    analysis_panel(ax[0, 1], cv)
    save(fig, out)


# --------------------------------------------------------------------------
# the application
# --------------------------------------------------------------------------


def coef_panel(ax, ages, est, se, ref, title, show_y):
    ax.fill_between(ages, est - 1.96 * se, est + 1.96 * se, color="#cfe0ee", lw=0)
    ax.plot(ages, est, color="#1a6fb0", lw=1.7, zorder=4)
    if ref is not None:
        ax.axhline(ref, color="#d1651a", lw=1.1, ls="--", zorder=3)
    ax.axhline(0.0, color="#999999", lw=0.7)
    tidy(ax, xlabel="age (years)", ylabel="coefficient" if show_y else None,
         title=title)


LAB_REAL = {"intercept": "Intercept", "tobacco": "Cumulative tobacco",
            "famhist": "Family history", "ldl": "LDL cholesterol",
            "typea": "Type A behaviour"}
ORDER_REAL = ["intercept", "tobacco", "ldl", "famhist", "typea"]


def calibration_panel(ax, d, show_y=True):
    """Observed and fitted proportion of cases in five year age bands.

    This is what ties the fitted coefficients back to the data, and it is the
    panel that says the varying coefficient model is worth having: the constant
    coefficient fit is badly wrong at both ends of the age range.
    """
    ax.plot(d["band_mid"], d["band_prop"], color="#444444", lw=1.2, marker="o",
            ms=3.2, label="observed")
    ax.plot(d["band_mid"], d["band_vary"], color="#1a6fb0", lw=1.6, marker="s",
            ms=3.0, label="varying")
    ax.plot(d["band_mid"], d["band_const"], color="#d1651a", lw=1.1, ls="--",
            marker="v", ms=3.0, label="constant")
    tidy(ax, xlabel="age (years)",
         ylabel="proportion of cases" if show_y else None)
    ax.set_title("observed and fitted\ncases", fontsize=7.2, pad=3.0,
                 linespacing=1.25)
    ax.legend(frameon=False, fontsize=6.4, handlelength=1.4,
              borderpad=0.2, labelspacing=0.25)


def fig_real(d, out: Path):
    """The same five effects under two models, one above the other.

    Row (a) is the four covariate model and row (b) the same effects after
    adjusting for all eight recorded risk factors, with the covariates in the
    same columns so that each can be read down.  Row (b) is the model in which
    the competing estimators return nothing at a substantial fraction of the
    ages, so it is the existence claim of Proposition 1 on real data.

    Nothing else belongs here.  The calibration of the fit has age on its
    horizontal axis but a proportion on its vertical one, and the estimator
    comparisons have neither, so neither shares a frame with a coefficient
    function; they are Table 2 and a figure in the supplement.
    """
    names = [str(s) for s in d["names"]]
    names8 = [str(s) for s in d["names8"]] if "names8" in d else []
    ages = d["ages"]
    methods = [str(s) for s in d["methods"]]
    primary = "CLSIR-QS" if "CLSIR-QS" in methods else "CLSIR-OS"
    fit, se = d[f"est_{primary}"], d["se"]
    shown = [nm for nm in ORDER_REAL if nm in names]
    nrow = 2 if names8 else 1
    fig, ax = panel_grid(nrow, len(shown), 2.15 * nrow + 0.55, wspace=0.40,
                         hspace=0.60, left=0.068, right=0.995,
                         bottom=0.115 if nrow == 2 else 0.225,
                         top=0.885 if nrow == 2 else 0.795)
    for k, nm in enumerate(shown):
        j = names.index(nm)
        # name and p-value on two lines: on a panel this narrow a single line
        # of either runs into its neighbour
        coef_panel(ax[0, k], ages, fit[:, j], se[:, j], float(d["beta"][j]),
                   None, show_y=(k == 0))
        ax[0, k].set_title(f"{LAB_REAL[nm]}\n$p={float(d['test_p'][j]):.3f}$",
                           fontsize=7.2, pad=3.0, linespacing=1.25)
        if nrow == 2:
            j8 = names8.index(nm)
            coef_panel(ax[1, k], ages, d["fit8"][:, j8], d["se8"][:, j8], None,
                       None, show_y=(k == 0))
    if nrow == 2:
        row_letters(fig, ax)
    save(fig, out)


def fig_real_diag(d, out: Path):
    """The four diagnostics of the application, with room to be read."""
    methods = [str(s) for s in d["methods"]]
    hg = d["hgrid"]
    fig, ax = panel_grid(1, 4, 2.75, wspace=0.34, bottom=0.315, top=0.905)

    # observed and fitted proportion of cases in five year age bands
    calibration_panel(ax[0, 0], d, show_y=True)

    # five fold predictive deviance against the bandwidth
    a = ax[0, 1]
    for m in [m for m in MS if f"cv_{m}" in d]:
        a.plot(hg, d[f"cv_{m}"], label=LABEL[m], **STYLE[m])
    a.axhline(float(d["base"]), color="#999999", lw=0.9, ls=":")
    lo = min(float(np.nanmin(d[f"cv_{m}"])) for m in MS if f"cv_{m}" in d)
    a.set_ylim(lo - 0.004, float(d["base"]) + 0.035)
    tidy(a, xlabel="bandwidth $h$", ylabel="predictive deviance",
         title="Five fold deviance")

    # evaluation points at which each estimator returns nothing usable
    a = ax[0, 2]
    shown = [m for m in MS if f"div8_{m}" in d]
    for k, m in enumerate(shown):
        a.plot(hg + (k - (len(shown) - 1) / 2) * 0.006, d[f"div8_{m}"],
               label=LABEL[m], **STYLE[m])
    tidy(a, xlabel="bandwidth $h$", ylabel="ages with no estimate",
         title="Existence, eight covariates")

    # wall clock for one fit at the selected bandwidth
    a = ax[0, 3]
    vals = 1000.0 * np.asarray(d["times"], float)
    a.bar(np.arange(len(methods)), vals, 0.64,
          color=[STYLE[m]["color"] for m in methods], edgecolor="white",
          linewidth=0.4)
    a.set_xticks(np.arange(len(methods)))
    a.set_xticklabels([LABEL[m] for m in methods], rotation=42, ha="right",
                      fontsize=6.0)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    tidy(a, ylabel="milliseconds", title="One fit")
    # the two middle panels carry one line per estimator and need the key that
    # the first panel's own legend does not supply
    shared_legend(fig, [m for m in MS if m in methods], y=0.075,
                  ncol=len(methods))
    save(fig, out)


def fig_real8(d, out: Path):
    """The same four effects, refitted with all eight recorded risk factors.

    This is the model in which the competing estimators return nothing at a
    substantial fraction of the evaluation points, so it belongs beside the
    four covariate fit rather than in the supplement: it is the existence claim
    of Proposition 1 on real data.
    """
    names8 = [str(s) for s in d["names8"]]
    ages = d["ages"]
    lab8 = {"tobacco": "cumulative tobacco", "ldl": "LDL cholesterol",
            "famhist": "family history", "typea": "type A behaviour"}
    shown = [nm for nm in ("tobacco", "ldl", "famhist", "typea")
             if nm in names8]
    fig, ax = panel_grid(1, len(shown), 2.15, bottom=0.225, top=0.855,
                         wspace=0.34)
    for k, nm in enumerate(shown):
        j = names8.index(nm)
        coef_panel(ax[0, k], ages, d["fit8"][:, j], d["se8"][:, j], None,
                   lab8[nm], show_y=(k == 0))
    save(fig, out)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("../results/v4"))
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="further result directories to merge, --results winning")
    ap.add_argument("--figures", type=Path, default=Path("../draft/figures-main"))
    ap.add_argument("--figures-supp", type=Path, dest="figures_supp",
                    default=Path("../draft/figures-supp"))
    ap.add_argument("--real", type=Path, default=Path("../results/real/saheart.npz"))
    a = ap.parse_args()
    # the paper's four figures and the supplement's ten live in separate
    # directories, which is how they are laid out in the Overleaf project
    R, F, S = a.results, a.figures, a.figures_supp

    gp = R / "grid_runs.csv"
    if gp.exists():
        frames = [pd.read_csv(gp)]
        for e in list(a.extra) + [R.parent / (R.name + "_firth")]:
            q = Path(e) / "grid_runs.csv"
            if q.exists():
                frames.append(pd.read_csv(q))
        g = pd.concat(frames, ignore_index=True)
        # --results is read first and wins, so a re-run of a subset of the
        # estimators supersedes the same rows in an earlier directory
        g = g.drop_duplicates(subset=["design", "n", "seed", "c", "method"],
                              keep="first")
        have = [m for m in MS if m in set(g.method)]
        main_d = [d for d in MAIN_DESIGNS if d in set(g.design)]
        fig_acc(g, F / "fig_acc.png", designs=main_d, methods=have)
        # the supplement repeats the comparison over all six examples
        fig_acc(g, S / "fig_acc_all.png",
                designs=[d for d in DESIGNS if d in set(g.design)], methods=have)
        fig_cross(g, S / "fig_cross.png")

    # The paper's Figure 2 plots the pointwise root mean squared error, which
    # is what the integrated squared error of Table 1 integrates.  The mean
    # fitted curve shows bias and hides variance, and for a binary response
    # variance is the larger part of the error, so the deviation plots are
    # kept but moved to the supplement.
    for tag, cp in (("", R / "mean_curves.npz"),
                    ("_2000", R / "mean_curves_2000.npz")):
        if not cp.exists():
            continue
        cur, c, n, have = load_curves(cp)
        main_designs = [d for d in MAIN_DESIGNS if d in cur]
        fig_curves_pair(cur, (F if tag == "" else S) / f"fig_curves{tag}.png",
                        main_designs, have, coefs=(1, 2))
        fig_curves(cur, S / f"fig_curves_bias{tag}.png", main_designs, have,
                   bias=True)
        if tag == "":
            for j in (0, 1, 2):
                fig_curves_one_coef(cur, j, S / f"fig_curves_a{j}.png",
                                    DESIGNS, have)

    costp, cvp = R / "cost_runs.csv", R / "cv_runs.csv"
    parp = R / "parallel_runs.csv"
    if costp.exists():
        cost = pd.read_csv(costp)
        fig_time(cost, F / "fig_time.png", designs=MAIN_DESIGNS)
        fig_time(cost, S / "fig_time_all.png", designs=DESIGNS)
        if cvp.exists():
            par = pd.read_csv(parp) if parp.exists() else None
            fig_parallel(pd.read_csv(cvp), par, cost, S / "fig_parallel.png")

    if a.real.exists():
        d = dict(np.load(a.real, allow_pickle=True))
        fig_real(d, F / "fig_real.png")
        fig_real_diag(d, S / "fig_real_diag.png")



if __name__ == "__main__":
    main()
