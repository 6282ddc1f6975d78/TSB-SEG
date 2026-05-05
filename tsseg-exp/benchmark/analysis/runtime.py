"""FIGURE_RUNTIME_ACCURACY: runtime vs best-grid score (CPD & SD).

Produces two independent figures (no title), one per task:

* ``fig_runtime_accuracy_cpd.pdf``
* ``fig_runtime_accuracy_sd.pdf``

For every algorithm we plot up to two points -- non-guided (hollow) and
guided (filled) -- connected by a thin line. Labels are placed in black
near the guided point (or NG if guided is missing) and de-overlapped
with adjustText.

* Score   : mean of ``score_best`` per algorithm (df_{cpd,sd}{,_g}).
* Runtime : median ``execution_time_seconds`` of the same best-grid
            config (re-derived from df_grid_{ng,g}).
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.data import load_data
from analysis.helpers import (
    COL_W,
    algo_color,
    display_name,
    savefig,
    select_best_grid_per_dataset,
)

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - adjustText is optional
    adjust_text = None

METRIC_CPD = "bidirectional_covering_score"
METRIC_SD = "state_matching_score"
RUNTIME_COL = "execution_time_seconds"

EXCLUDED_ALGOS = {"hidalgo", "kcpd", "dynp", "hdp-hsmm", "hdphsmm"}

# Font sizes — bumped uniformly for camera-ready legibility.
FS_TICK = 12
FS_LABEL = 13
FS_LEGEND = 11
FS_TEXT = 9


def _label(algo: str) -> str:
    al = algo.lower()
    if al == "clasp":
        return "CLaSP/CLaP"
    if al == "clap":
        return "CLaP"
    if "ggs" in al:
        return "GGS"
    if al == "changefinder":
        return "CF"
    if al == "time2state":
        return "T2S"
    if al == "bottomup":
        return "Bot.Up"
    return display_name(algo)


def _runtime_table(
    df_score: pd.DataFrame,
    df_grid: pd.DataFrame,
    score_metric: str,
    algo_filter: set,
) -> pd.DataFrame:
    """Per-algorithm (mean best-grid score, median best-grid runtime)."""
    s = df_score[df_score["algorithm"].isin(algo_filter)]
    s = s[s["dataset"] != "pamap2"]
    score = s.groupby("algorithm")["score_best"].mean()

    g = df_grid[df_grid["algorithm"].isin(algo_filter)].copy()
    g = g[g["dataset"] != "pamap2"]
    g = g.dropna(subset=[score_metric, RUNTIME_COL, "trial_index", "parent_run_id"])
    if g.empty:
        runtime = pd.Series(dtype=float)
    else:
        ranked = select_best_grid_per_dataset(g, score_metric, higher_is_better=True)
        best = ranked[ranked["config_rank"] == 1]
        runtime = best.groupby("algorithm")[RUNTIME_COL].median()

    out = pd.concat([score.rename("score"), runtime.rename("runtime")], axis=1).dropna()
    out = out[out["runtime"] > 0]
    return out


def _ylim_from_data(*tables, pad=0.05) -> tuple[float, float]:
    vals = np.concatenate([t["score"].to_numpy() for t in tables if not t.empty])
    if vals.size == 0:
        return 0.0, 1.0
    lo = max(0.0, float(vals.min()) - pad)
    hi = min(1.0, float(vals.max()) + pad)
    # round to nearest 0.05 for readable ticks
    lo = np.floor(lo * 20) / 20
    hi = np.ceil(hi * 20) / 20
    if hi - lo < 0.2:
        hi = min(1.0, lo + 0.2)
    return lo, hi


def _plot(ax, tbl: pd.DataFrame, ylabel: str,
          ylim: tuple[float, float] | None = None,
          title: str = ""):
    from matplotlib.patches import FancyBboxPatch

    # ── Markers. ──
    for algo, row in tbl.iterrows():
        ax.scatter(
            row["runtime"], row["score"],
            marker="o", s=55, color=algo_color(algo),
            edgecolors="white", linewidths=0.5, zorder=4,
        )

    # ── Text labels ──
    texts = []
    anchors: dict[str, tuple[float, float]] = {}
    for algo in tbl.index:
        ax_, ay_ = tbl.loc[algo, "runtime"], tbl.loc[algo, "score"]
        anchors[algo] = (ax_, ay_)
        t = ax.text(
            ax_, ay_, _label(algo),
            fontsize=FS_TEXT, color="black", zorder=7,
            ha="center", va="center",
        )
        t._algo = algo  # type: ignore[attr-defined]
        texts.append(t)

    ax.set_xscale("log")
    ax.set_xlabel("Median Runtime (s)", fontsize=FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL)
    if title:
        ax.set_title(title, fontsize=FS_LABEL, pad=10)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    lo, hi = _ylim_from_data(tbl)
    if ylim is not None:
        lo, hi = ylim
    # Add a touch of headroom on both sides.
    span = hi - lo
    if ylim is None:
        lo = max(0.0, lo - 0.06 * span)
        hi = min(1.0, hi + 0.10 * span)
    ax.set_ylim(lo, hi)
    
    # Slight horizontal padding in log-space.
    xs = tbl["runtime"].to_numpy()
    if xs.size:
        ax.set_xlim(xs.min() / 1.15, xs.max() * 1.15)

    if title == "Non-Guided":
        ax.annotate(
            "Faster & More Accurate",
            xy=(0.02, 0.96), xycoords="axes fraction",
            xytext=(0.15, 0.96), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
            fontsize=FS_TEXT, color="black", fontweight="bold",
            ha="left", va="center"
        )

    if adjust_text is not None and texts:
        adjust_text(
            texts, ax=ax,
            expand=(1.1, 1.1),
            expand_axes=True,
            arrowprops=dict(
                arrowstyle="-", color="0.4", lw=0.6, alpha=0.7,
                shrinkA=1, shrinkB=1,
            ),
            force_text=(0.05, 0.05),
            force_static=(0.05, 0.05),
            iter_lim=500,
        )

def make_figure(d=None):
    if d is None:
        d = load_data()

    cpd_algos = {a for a in d.cpd_algos if a != "random" and a not in EXCLUDED_ALGOS}
    sd_algos = {a for a in d.sd_algos if a != "random" and a not in EXCLUDED_ALGOS}

    ng_cpd = _runtime_table(d.df_cpd,   d.df_grid_ng, METRIC_CPD, cpd_algos)
    g_cpd  = _runtime_table(d.df_cpd_g, d.df_grid_g,  METRIC_CPD, cpd_algos)
    ng_sd  = _runtime_table(d.df_sd,    d.df_grid_ng, METRIC_SD,  sd_algos)
    g_sd   = _runtime_table(d.df_sd_g,  d.df_grid_g,  METRIC_SD,  sd_algos)

    # CPD NG
    cpd_ylim = _ylim_from_data(ng_cpd, g_cpd, pad=0.02)
    fig, ax = plt.subplots(figsize=(COL_W * 1.1, COL_W * 1.1))
    _plot(ax, ng_cpd, ylabel="Mean Bi-Covering Score", ylim=cpd_ylim, title="Non-Guided")
    savefig(fig, "fig_runtime_accuracy_cpd_ng")

    # CPD G
    fig, ax = plt.subplots(figsize=(COL_W * 1.1, COL_W * 1.1))
    _plot(ax, g_cpd, ylabel="Mean Bi-Covering Score", ylim=cpd_ylim, title="Guided")
    savefig(fig, "fig_runtime_accuracy_cpd_g")

    # SD NG
    sd_ylim = _ylim_from_data(ng_sd, g_sd, pad=0.02)
    fig, ax = plt.subplots(figsize=(COL_W * 1.1, COL_W * 1.1))
    _plot(ax, ng_sd, ylabel="Mean Best-Grid SMS", ylim=sd_ylim, title="Non-Guided")
    savefig(fig, "fig_runtime_accuracy_sd_ng")

    # SD G
    fig, ax = plt.subplots(figsize=(COL_W * 1.1, COL_W * 1.1))
    _plot(ax, g_sd, ylabel="Mean Best-Grid SMS", ylim=sd_ylim, title="Guided")
    savefig(fig, "fig_runtime_accuracy_sd_g")

    print(f"CPD: {len(ng_cpd)} NG, {len(g_cpd)} G")
    print(f"SD : {len(ng_sd)} NG, {len(g_sd)} G")


if __name__ == "__main__":
    make_figure()
