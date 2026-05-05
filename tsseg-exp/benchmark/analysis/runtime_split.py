"""FIGURE_RUNTIME_SPLIT: per-algorithm runtime boxplots, NG vs G.

Produces two independent figures (no title), one for CPD and one for SD:

* ``fig_runtime_split_cpd.pdf``
* ``fig_runtime_split_sd.pdf``

For each algorithm we show the per-series runtime of the *best-grid*
configuration (selected per dataset on the corresponding score),
aggregated across all non-PAMAP2 datasets. The lighter left half is the
Non-Guided regime and the darker right half the Guided regime;
algorithms that only support one regime are drawn as a single full-width
box.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis.data import load_data
from analysis.helpers import (
    ALGO_TASK,
    TEXT_W,
    algo_color,
    display_name,
    savefig,
    select_best_grid_per_dataset,
)

METRIC_CPD = "bidirectional_covering_score"
METRIC_SD = "state_matching_score"
RUNTIME_COL = "execution_time_seconds"

# Algorithms excluded from the main paper for systematic timeouts/OOM.
EXCLUDED_ALGOS = {"hidalgo", "kcpd", "dynp", "tglad"}

# Font sizes — bumped uniformly for camera-ready legibility.
FS_TICK_X = 11
FS_TICK_Y = 11
FS_LABEL = 13
FS_LEGEND = 11


def _xtick_label(algo: str) -> str:
    # CLaP has its own box on this figure, so we keep just "CLaSP" for
    # the clasp algorithm (the merged "CLaSP/CLaP" name is reserved for
    # the runtime-accuracy figure where they share a single point).
    if algo == "clasp":
        return "CLaSP"
    return display_name(algo)


def _best_grid_runtimes(
    df_grid: pd.DataFrame, metric: str, algos: set
) -> tuple[dict[str, np.ndarray], dict[str, float], int]:
    df = df_grid[df_grid["algorithm"].isin(algos)].copy()
    df = df[df["dataset"] != "pamap2"]
    df = df.dropna(subset=[metric, RUNTIME_COL, "trial_index", "parent_run_id"])
    if df.empty:
        return {}, {}, 0

    total = int(df.groupby("dataset")["trial_index"].nunique().sum())

    ranked = select_best_grid_per_dataset(df, metric, higher_is_better=True)
    if ranked.empty:
        return {}, {}, total

    best = ranked[ranked["config_rank"] == 1]
    runtimes: dict[str, np.ndarray] = {}
    coverage: dict[str, float] = {}
    for algo, sub in best.groupby("algorithm"):
        rt = sub[RUNTIME_COL].to_numpy(dtype=float)
        rt = rt[np.isfinite(rt) & (rt > 0)]
        if rt.size:
            runtimes[str(algo)] = rt
            n_covered = int(sub.groupby("dataset")["trial_index"].nunique().sum())
            coverage[str(algo)] = n_covered / total if total else 0.0
    return runtimes, coverage, total


def _draw_split_box(ax, x, data, color, side, width=0.7):
    """Draw a half-width box at ``x``. side: 'left', 'right' or 'full'."""
    half = width / 2.0
    if side == "left":
        positions = [x - half / 2.0]
        widths = [half * 0.95]
        face_alpha = 0.35
    elif side == "right":
        positions = [x + half / 2.0]
        widths = [half * 0.95]
        face_alpha = 0.85
    else:
        positions = [x]
        widths = [width * 0.95]
        face_alpha = 0.85

    bp = ax.boxplot(
        [data],
        positions=positions,
        widths=widths,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(color=color, linewidth=0.8),
        capprops=dict(color=color, linewidth=0.8),
        boxprops=dict(linewidth=0.8),
    )
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(face_alpha)
        box.set_edgecolor(color)


def _build_panel(ax, rt_ng: dict, rt_g: dict):
    algos = sorted(set(rt_ng) | set(rt_g))
    if not algos:
        ax.set_visible(False)
        return

    def _key(a):
        if a in rt_g:
            return float(np.median(rt_g[a]))
        return float(np.median(rt_ng[a]))

    algos = sorted(algos, key=_key)
    xs = np.arange(len(algos))

    for x, algo in zip(xs, algos):
        color = algo_color(algo)
        has_ng = algo in rt_ng
        has_g = algo in rt_g
        if has_ng and has_g:
            _draw_split_box(ax, x, rt_ng[algo], color, side="left")
            _draw_split_box(ax, x, rt_g[algo], color, side="right")
        elif has_g:
            _draw_split_box(ax, x, rt_g[algo], color, side="full")
        elif has_ng:
            _draw_split_box(ax, x, rt_ng[algo], color, side="full")

    ax.set_xticks(xs)
    ax.set_xticklabels(
        [_xtick_label(a) for a in algos],
        rotation=45, ha="right", fontsize=FS_TICK_X,
    )
    ax.set_yscale("log")
    ax.set_ylabel("Runtime per series (s)", fontsize=FS_LABEL)
    ax.grid(True, axis="y", which="both", alpha=0.25, linewidth=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=FS_TICK_Y)
    ax.set_xlim(-0.6, len(algos) - 0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_one(rt_ng: dict, rt_g: dict, name: str):
    n = max(len(set(rt_ng) | set(rt_g)), 1)
    width = max(min(TEXT_W, 0.45 * n + 1.2), 4.0)
    fig, ax = plt.subplots(figsize=(width, 3.2))
    _build_panel(ax, rt_ng, rt_g)

    legend_handles = [
        Patch(facecolor="0.55", edgecolor="0.25", alpha=0.35,
              label="Non-Guided (left half)"),
        Patch(facecolor="0.55", edgecolor="0.25", alpha=0.85,
              label="Guided (right half)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        fontsize=FS_LEGEND,
        handlelength=1.6,
    )

    fig.subplots_adjust(left=0.10, right=0.99, top=0.98, bottom=0.28)
    savefig(fig, name)


def make_figure(d=None):
    if d is None:
        d = load_data()

    cpd_algos = {a for a in d.cpd_algos if a != "random" and a not in EXCLUDED_ALGOS}
    sd_algos = {a for a in d.sd_algos if a != "random" and a not in EXCLUDED_ALGOS}

    rt_cpd_ng, _, _ = _best_grid_runtimes(d.df_grid_ng, METRIC_CPD, cpd_algos)
    rt_cpd_g,  _, _ = _best_grid_runtimes(d.df_grid_g,  METRIC_CPD, cpd_algos)
    rt_sd_ng,  _, _ = _best_grid_runtimes(d.df_grid_ng, METRIC_SD,  sd_algos)
    rt_sd_g,   _, _ = _best_grid_runtimes(d.df_grid_g,  METRIC_SD,  sd_algos)

    _save_one(rt_cpd_ng, rt_cpd_g, "fig_runtime_split_cpd")
    _save_one(rt_sd_ng,  rt_sd_g,  "fig_runtime_split_sd")


if __name__ == "__main__":
    make_figure()
