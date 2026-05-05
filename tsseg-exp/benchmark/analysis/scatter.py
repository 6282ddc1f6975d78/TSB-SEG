"""FIGURE_SCATTER_CPD_SD: monochrome scatter + empirical envelope.

Pools every (algorithm, dataset, trial) outcome from the four sources
(default/grid x non-guided/guided), keeps SD-capable algorithms, and
plots SMS vs BiCovering as a single colour cloud, highlighting points
that fall inside the empirical envelope around y=x.
"""
from __future__ import annotations

# Allow running this file directly: ``python analysis/scatter.py``
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

from analysis.data import load_data
from analysis.helpers import (
    ALGO_COLORS,
    ALGO_TASK,
    COL_W,
    CPD_CLUSTERING_CSV,
    METRIC_CPD,
    METRIC_SD,
    savefig,
)


def make_figure(d=None):
    if d is None:
        d = load_data()

    frames = []
    for src, tag in [
        (d.df_default_ng, "default_ng"),
        (d.df_grid_ng,    "grid_ng"),
        (d.df_default_g,  "default_g"),
        (d.df_grid_g,     "grid_g"),
    ]:
        if src is None or len(src) == 0:
            continue
        if not {"algorithm", "dataset", METRIC_CPD, METRIC_SD}.issubset(src.columns):
            continue
        f = src[["algorithm", "dataset", METRIC_CPD, METRIC_SD]].copy()
        f["source"] = tag
        frames.append(f)

    df_scatter = pd.concat(frames, ignore_index=True)
    sd_algo_set = {a for a, t in ALGO_TASK.items() if t == "SD"}
    df_scatter = df_scatter[df_scatter["algorithm"].isin(sd_algo_set)]
    df_scatter = df_scatter[df_scatter["dataset"].str.lower() != "pamap2"]
    df_scatter = df_scatter.dropna(subset=[METRIC_CPD, METRIC_SD])
    df_scatter = df_scatter[
        (df_scatter[METRIC_CPD].between(0, 1)) & (df_scatter[METRIC_SD].between(-0.3, 1))
    ]

    if CPD_CLUSTERING_CSV is not None and CPD_CLUSTERING_CSV.exists():
        extra = pd.read_csv(CPD_CLUSTERING_CSV)
        cols_ok = {"algorithm", "dataset", METRIC_CPD, METRIC_SD}.issubset(extra.columns)
        if cols_ok:
            extra = extra[["algorithm", "dataset", METRIC_CPD, METRIC_SD]].copy()
            extra["source"] = "cpd_clustering"
            extra = extra.dropna(subset=[METRIC_CPD, METRIC_SD])
            extra = extra[
                extra[METRIC_CPD].between(0, 1) & extra[METRIC_SD].between(-0.3, 1)
            ]
            df_scatter = pd.concat([df_scatter, extra], ignore_index=True)

    x = df_scatter[METRIC_CPD].to_numpy(dtype=float)
    y = df_scatter[METRIC_SD].to_numpy(dtype=float)
    n = x.size

    # Symmetric two-sided ~95% containment band around y=x.
    delta_up = float(np.quantile(y - x, 0.975))
    delta_lo = float(np.quantile(x - y, 0.975))
    inside = (y <= x + delta_up) & (y >= x - delta_lo)
    in_band = float(np.mean(inside))

    print(f"[scatter] N = {n:,}")
    print(f"[scatter] delta_up = {delta_up:.3f}   delta_lo = {delta_lo:.3f}   inside = {in_band*100:.1f}%")

    # Monochrome palette: in-band saturated blue, out-of-band pale blue.
    C_IN  = "#1f4e8c"
    C_UNDER = "#dda0dd"
    C_OVER  = "#ffb6c1"

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.95))

    # Plot out-of-band first (behind), then in-band on top.
    under_mask = y < x - delta_lo
    ax.scatter(
        x[under_mask], y[under_mask],
        c=C_UNDER, s=2.5, alpha=0.3,
        linewidths=0, rasterized=True,
    )
    over_mask = y > x + delta_up
    ax.scatter(
        x[over_mask], y[over_mask],
        c=C_OVER, s=2.5, alpha=0.3,
        linewidths=0, rasterized=True,
    )
    ax.scatter(
        x[inside], y[inside],
        c=C_IN, s=2.5, alpha=0.3,
        linewidths=0, rasterized=True,
    )

    xs = np.linspace(0, 1, 200)
    ax.plot(xs, xs, color="black", lw=1.1, zorder=4.5)
    ax.plot(xs, np.clip(xs + delta_up, -0.3, 1), color="black", lw=0.9, ls=(0, (4, 2)), zorder=4)
    ax.plot(xs, np.clip(xs - delta_lo, -0.3, 1), color="black", lw=0.9, ls=(0, (4, 2)), zorder=4)
    
    ax.fill_between(
        xs, 
        np.clip(xs - delta_lo, -0.3, 1), 
        np.clip(xs + delta_up, -0.3, 1), 
        color="#a9c4e6", alpha=0.2, zorder=3
    )

    ax.set_xlim(0, 1); ax.set_ylim(-0.3, 1)
    ax.set_xlabel("Bi-Covering Score", fontsize=12)
    ax.set_ylabel("State Matching Score", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # In-axes label for the y=x reference line, placed near the top-right
    # of the plot, slightly above the diagonal.
    # ax.text(
    #     0.93, 0.97, r"$y = x$",
    #     fontsize=8, ha="right", va="top", color="black",
    #     zorder=6,
    # )
    plt.rcParams["text.usetex"] = True

    # 2) Legend for the in-band vs out-of-band points
    pct = int(round(in_band * 100))
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_IN, markersize=8, label=rf"$\underline{{\mathrm{{BiC}}}} \leq \mathrm{{SMS}} \leq \overline{{\mathrm{{BiC}}}}$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_OVER, markersize=8, label=r"$\mathrm{SMS} > \overline{\mathrm{BiC}}$"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=C_UNDER, markersize=8, label=r"$\mathrm{SMS} < \underline{\mathrm{BiC}}$"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=False, handletextpad=0.4)

    fig.tight_layout()
    savefig(fig, "fig_scatter_cpd_sd")
    plt.show()

    # Also produce the per-algorithm coloured variant from the same data.
    make_figure_by_algo(df_scatter)


def make_figure_by_algo(df_scatter):
    """Variant of the scatter where points are coloured by algorithm.

    Uses the canonical ``ALGO_COLORS`` palette so colours stay consistent
    with the strip / spider / CD figures in the paper.
    """
    x = df_scatter[METRIC_CPD].to_numpy(dtype=float)
    y = df_scatter[METRIC_SD].to_numpy(dtype=float)
    algos = df_scatter["algorithm"].to_numpy()
    n = x.size

    delta_up = float(np.quantile(y - x, 0.975))
    delta_lo = float(np.quantile(x - y, 0.975))

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.95))

    # Draw algorithms in deterministic order so the legend matches the plot.
    algo_list = sorted(np.unique(algos).tolist())
    for algo in algo_list:
        m = algos == algo
        if not m.any():
            continue
        ax.scatter(
            x[m], y[m],
            c=ALGO_COLORS.get(algo, "#888888"),
            s=2.5, alpha=0.10,
            linewidths=0, rasterized=True,
            label=algo,
        )

    xs = np.linspace(0, 1, 200)
    ax.plot(xs, xs, color="black", lw=1.1, zorder=4.5)
    ax.plot(xs, np.clip(xs + delta_up, -0.3, 1), color="black", lw=0.9, ls=(0, (4, 2)), zorder=4)
    ax.plot(xs, np.clip(xs - delta_lo, -0.3, 1), color="black", lw=0.9, ls=(0, (4, 2)), zorder=4)
    
    ax.fill_between(
        xs, 
        np.clip(xs - delta_lo, -0.3, 1), 
        np.clip(xs + delta_up, -0.3, 1), 
        color="#a9c4e6", alpha=0.2, zorder=3
    )

    ax.set_xlim(0, 1); ax.set_ylim(-0.3, 1)
    ax.set_xlabel("Bi-Covering Score", fontsize=12)
    ax.set_ylabel("State Matching Score", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(
        0.93, 0.97, r"$y = x$",
        fontsize=8, ha="right", va="top", color="black",
        zorder=6,
    )

    # Legend with one larger marker per algorithm, placed below the axes.
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=ALGO_COLORS.get(a, "#888888"),
                   markeredgecolor="none", markersize=4, label=a)
        for a in algo_list
    ]
    ncol = min(4, max(1, len(handles)))
    ax.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=ncol, fontsize=10, frameon=False,
        handletextpad=0.3, columnspacing=0.8,
    )

    fig.tight_layout()
    savefig(fig, "fig_scatter_cpd_sd_by_algo")
    plt.show()


if __name__ == "__main__":
    make_figure()
