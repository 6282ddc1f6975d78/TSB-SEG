from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis.data import load_data
from analysis.helpers import (
    TEXT_W,
    algo_color,
    display_name,
    savefig,
    select_best_grid_per_dataset,
)

METRIC_CPD = "bidirectional_covering_score"
METRIC_SD = "state_matching_score"
RUNTIME_COL = "execution_time_seconds"

# Exact 14 algorithms that have "scales" status (>= 1 complete config) from PAMAP2 / general check
SCALES_ALGOS = {
    "binseg", "bottomup", "changefinder", "prophet", "window", "tire", "ggs",
    "fluss", "amoc", "pelt", "ticc", "time2state", "e2usd", "autoplait"
}

FS_TICK_X = 11
FS_TICK_Y = 11
FS_LABEL = 13
FS_LEGEND = 11


def _xtick_label(algo: str) -> str:
    if algo == "clasp":
        return "CLaSP"
    return display_name(algo)


def _best_grid_stats(
    df_grid: pd.DataFrame, metric: str, algos: set
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    # Extract both scores and runtimes for best grid
    df = df_grid[df_grid["algorithm"].isin(algos)].copy()
    df = df[df["dataset"] == "pamap2"]
    df = df.dropna(subset=[metric, RUNTIME_COL, "trial_index", "parent_run_id"])
    if df.empty:
        return {}, {}
        
    ranked = select_best_grid_per_dataset(df, metric, higher_is_better=True)
    if ranked.empty:
        return {}, {}

    best = ranked[ranked["config_rank"] == 1]
    scores: dict[str, np.ndarray] = {}
    runtimes: dict[str, np.ndarray] = {}
    
    for algo, sub in best.groupby("algorithm"):
        s = sub[metric].to_numpy(dtype=float)
        rt = sub[RUNTIME_COL].to_numpy(dtype=float)
        
        valid = np.isfinite(s) & np.isfinite(rt) & (rt > 0)
        s = s[valid]
        rt = rt[valid]
        
        if len(s) > 0:
            scores[str(algo)] = s
            runtimes[str(algo)] = rt
            
    return scores, runtimes

def _draw_split_box(ax, x, data, color, side, width=0.7):
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

def _draw_split_bar(ax, x, data, color, side, width=0.7):
    half = width / 2.0
    mean_val = np.mean(data)
    
    if side == "left":
        pos = x - half / 2.0
        w = half * 0.95
        face_alpha = 0.35
    elif side == "right":
        pos = x + half / 2.0
        w = half * 0.95
        face_alpha = 0.85
    else:
        pos = x
        w = width * 0.95
        face_alpha = 0.85

    ax.bar(pos, mean_val, width=w, color=color, alpha=face_alpha, edgecolor=color)

def _save_one(scores_ng: dict, scores_g: dict, rt_ng: dict, rt_g: dict, name: str, ylabel_score: str):
    algos = sorted(set(scores_ng) | set(scores_g))
    if not algos:
        print(f"No algos for {name}")
        return

    def _key(a):
        if a in scores_ng:
            return float(np.median(scores_ng[a]))
        if a in scores_g:
            return float(np.median(scores_g[a]))
        return 0.0

    algos = sorted(algos, key=_key)
    xs = np.arange(len(algos))

    n = max(len(algos), 1)
    width = max(min(TEXT_W, 0.45 * n + 1.2), 6.0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, 5.5), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

    for x, algo in zip(xs, algos):
        color = algo_color(algo)
        has_ng = algo in scores_ng
        has_g = algo in scores_g
        
        # Score boxplot
        if has_ng and has_g:
            _draw_split_box(ax1, x, scores_ng[algo], color, side="left")
            _draw_split_box(ax1, x, scores_g[algo], color, side="right")
        elif has_g:
            _draw_split_box(ax1, x, scores_g[algo], color, side="full")
        elif has_ng:
            _draw_split_box(ax1, x, scores_ng[algo], color, side="full")

        # Runtime barplot
        if has_ng and has_g:
            _draw_split_bar(ax2, x, rt_ng[algo], color, side="left")
            _draw_split_bar(ax2, x, rt_g[algo], color, side="right")
        elif has_g:
            _draw_split_bar(ax2, x, rt_g[algo], color, side="full")
        elif has_ng:
            _draw_split_bar(ax2, x, rt_ng[algo], color, side="full")

    # ax1 formatting (Score)
    ax1.set_ylabel(ylabel_score, fontsize=FS_LABEL)
    ax1.grid(True, axis="y", which="both", alpha=0.25, linewidth=0.3)
    ax1.set_axisbelow(True)
    ax1.tick_params(axis="y", labelsize=FS_TICK_Y)
    ax1.set_xlim(-0.6, len(algos) - 0.4)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim(-0.05, 1.05)

    # ax2 formatting (Runtime)
    ax2.set_yscale("log")
    ax2.set_ylabel("Runtime mean (s)", fontsize=FS_LABEL)
    ax2.grid(True, axis="y", which="both", alpha=0.25, linewidth=0.3)
    ax2.set_axisbelow(True)
    ax2.tick_params(axis="y", labelsize=FS_TICK_Y)
    ax2.set_xlim(-0.6, len(algos) - 0.4)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    ax2.set_xticks(xs)
    ax2.set_xticklabels([_xtick_label(a) for a in algos], rotation=45, ha="right", fontsize=FS_TICK_X)

    legend_handles = [
        Patch(facecolor="0.55", edgecolor="0.25", alpha=0.35, label="Non-Guided (left half)"),
        Patch(facecolor="0.55", edgecolor="0.25", alpha=0.85, label="Guided (right half)"),
    ]
    ax1.legend(handles=legend_handles, loc="upper left", ncol=1, frameon=False, fontsize=FS_LEGEND, handlelength=1.6)

    fig.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.2, hspace=0.15)
    savefig(fig, name)
    print(f"Saved {name}, algos: {len(algos)}")

def make_figure(d=None):
    if d is None:
        d = load_data()

    cpd_algos = {a for a in d.cpd_algos if a in SCALES_ALGOS}
    sd_algos = {a for a in d.sd_algos if a in SCALES_ALGOS}

    scores_cpd_ng, rt_cpd_ng = _best_grid_stats(d.df_grid_ng, METRIC_CPD, cpd_algos)
    scores_cpd_g, rt_cpd_g = _best_grid_stats(d.df_grid_g, METRIC_CPD, cpd_algos)
    
    _save_one(scores_cpd_ng, scores_cpd_g, rt_cpd_ng, rt_cpd_g, "fig_score_runtime_split_cpd", "BiCovering Score")

    scores_sd_ng, rt_sd_ng = _best_grid_stats(d.df_grid_ng, METRIC_SD, sd_algos)
    scores_sd_g, rt_sd_g = _best_grid_stats(d.df_grid_g, METRIC_SD, sd_algos)
    
    _save_one(scores_sd_ng, scores_sd_g, rt_sd_ng, rt_sd_g, "fig_score_runtime_split_sd", "SMS Score")

if __name__ == "__main__":
    make_figure()
