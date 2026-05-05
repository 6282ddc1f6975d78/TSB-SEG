"""Performance figures: strip + CD + spider composites.

Produces FIGURE_CPD_MAIN, FIGURE_SD_MAIN, FIGURE_CPD_GUIDED and
FIGURE_SD_GUIDED. All DataFrames are loaded through ``data.load_data()``
which uses the on-disk pickle cache.
"""
from __future__ import annotations

# Allow running this file directly: ``python analysis/performance.py``
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from io import BytesIO
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from PIL import Image

from analysis.data import load_data
from analysis.helpers import (
    ALGO_COLORS,
    ALGO_TASK,
    DOMAIN_ORDER,
    TEXT_W,
    display_name,
    savefig,
)
# ── Hatching for SD algorithms in the CPD strip plot ─────────────────
_SD_HATCH = "///"
# Linewidth used on every bar edge so that hatches render in PDF
# (PDF backend ignores hatches when edge linewidth is 0).
_BAR_EDGE_LW = 0.6

# Font/style overrides for stand-alone panels (used in the main-text layout
# where each panel is embedded directly rather than packed inside a 2x2
# composite). We bump font sizes so panels remain legible after LaTeX
# scaling, and disable global tight-bbox so the saved file size matches the
# requested figsize (otherwise legends can balloon the bounding box).
_PANEL_RC = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "hatch.linewidth": 1.8,
    "hatch.color": "black",
    "savefig.bbox": "standard",
    "savefig.pad_inches": 0.05,
}


@contextmanager
def _panel_rc():
    with mpl.rc_context(_PANEL_RC):
        yield

# Maximally distinct 5-colour palette for the spider plot, indexed by
# top-rank position (rank 1 → first colour, …). Overrides the per-algo
# palette to guarantee that the 5 best algorithms always look distinct.
_SPIDER_TOP5_COLORS = [
    "#1F3A93",  # deep blue
    "#E63946",  # vivid red
    "#2A9D8F",  # teal-green
    "#F4A261",  # warm orange
    "#6A4C93",  # purple
]


def _is_sd_algo(algo: str) -> bool:
    return ALGO_TASK.get(algo, "") == "SD"


def _cd_algo_order(df, score_col, maximize=True):
    """Return algorithm order consistent with the CD diagram (mean rank)."""
    df_clean = df.copy()
    df_clean["series_id"] = df_clean["dataset"] + "_" + df_clean["trial_index"].astype(str)
    pivot = df_clean.pivot_table(
        index="series_id", columns="algorithm", values=score_col, aggfunc="first"
    )
    if "random" in pivot.columns:
        pivot = pivot.drop(columns=["random"])
    if pivot.empty or pivot.shape[1] < 2:
        return df.groupby("algorithm")[score_col].mean().sort_values(ascending=False).index.tolist()
    pivot = pivot.fillna(0.0)
    ranks = pivot.rank(axis=1, ascending=not maximize, method="average")
    return ranks.mean().sort_values().index.tolist()


def plot_strip(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_col: str = "score_best",
    ylabel: str = "Score",
    algo_order: list | None = None,
    show_hatch: bool = True,
    show_legend: bool = True,
    legend_ncol: int = 2,
    ylim: tuple | None = None,
):
    """Strip plot: 3 summary statistics per algorithm (mean across series).

    ``show_hatch=False`` disables the SD hatch overlay (useful when all
    algorithms are state detectors and the hatch is therefore meaningless).
    """
    if algo_order is None:
        # Worst on the left, best on the right.
        algo_order = (
            df.groupby("algorithm")[score_col]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )

    _random_val = None
    if "random" in set(df["algorithm"]):
        _random_val = df.loc[df["algorithm"] == "random", score_col].mean()
    algo_order = [a for a in algo_order if a != "random"]

    x_pos = {a: i for i, a in enumerate(algo_order)}
    n = len(algo_order)

    if _random_val is not None:
        ax.axhline(_random_val, ls="--", lw=0.9, color="#888888", alpha=0.55, zorder=0)
        # Place the "Random" label slightly to the right of the leftmost
        # bar (≈1 bar in) and just below the dashed line, so it does not
        # collide with the y-axis tick labels nor with bar tops.
        ax.text(
            0.5, _random_val - 0.01, "Random",
            ha="left", va="top", fontsize=7, color="#888888", fontstyle="italic",
            zorder=0,
        )

    stats_spec = [
        ("score_best",    "^", 40, 0.90, "Best grid"),
        ("score_default", "D", 30, 0.85, "Default"),
        ("score_worst",   "v", 25, 0.70, "Worst grid"),
    ]

    bar_width = 0.32
    bar_alpha = 0.55

    def _bar(xc, h, bottom, width, color, hatch=None):
        # The PDF backend often drops hatches when alpha < 1 is applied to a
        # rectangle's facecolor (or via ``alpha=`` on bar()). We work around
        # this by drawing the bar in two fully-opaque layers:
        #   1. translucent fill (RGBA, no hatch)
        #   2. transparent fill with opaque edge + hatch on top
        ax.bar(
            xc, h, bottom=bottom, width=width,
            facecolor=to_rgba(color, bar_alpha),
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
        ax.bar(
            xc, h, bottom=bottom, width=width,
            facecolor="none",
            edgecolor=color,
            linewidth=_BAR_EDGE_LW,
            hatch=hatch,
            zorder=2.1,
        )

    for idx, algo in enumerate(algo_order):
        sub = df[df["algorithm"] == algo]
        xc = x_pos[algo]
        c = ALGO_COLORS.get(algo, "#888888")

        best_val = sub["score_best"].mean() if "score_best" in sub.columns and sub["score_best"].notna().any() else None
        worst_val = sub["score_worst"].mean() if "score_worst" in sub.columns and sub["score_worst"].notna().any() else None
        median_val = sub["score_median"].mean() if "score_median" in sub.columns and sub["score_median"].notna().any() else None
        bar_h = (best_val - worst_val) if (best_val is not None and worst_val is not None) else None

        if bar_h is not None:
            _bar(xc, bar_h, worst_val, bar_width * 2, c, hatch=None)

        if median_val is not None:
            ax.hlines(
                median_val, xc - bar_width, xc + bar_width,
                colors=c, linewidths=1.2, alpha=0.55, zorder=3,
            )

        for col, marker, ms, alpha, _label in stats_spec:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna()
            if vals.empty:
                continue
            ax.scatter(
                xc, vals.mean(),
                marker=marker, s=ms, color=c,
                edgecolors="none", alpha=alpha, zorder=4,
            )

    ax.set_xticks(range(n))
    ax.set_xticklabels([display_name(a) for a in algo_order], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, n - 0.5)
    if ylim is not None:
        ax.set_ylim(*ylim)

    legend_handles = [
        Line2D([], [], marker="^", color="gray", markersize=5, linestyle="None",
               markeredgecolor="none", alpha=0.9, label="Best grid"),
        Line2D([], [], marker="v", color="gray", markersize=4, linestyle="None",
               markeredgecolor="none", alpha=0.7, label="Worst grid"),
        Line2D([], [], marker="D", color="gray", markersize=5, linestyle="None",
               markeredgecolor="none", alpha=0.85, label="Default"),
        Line2D([], [], color="gray", linewidth=1.2, alpha=0.55, label="Median"),
    ]
    if show_legend:
        ax.legend(handles=legend_handles, loc="upper left", framealpha=0.85,
                  ncol=legend_ncol, fontsize=6)

    return algo_order


def plot_spider(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_col: str = "score_best",
    top_k: int = 5,
    legend_outside: bool = False,
    group_by: str = "domain",
):
    """Radar / spider plot — performance by domain or dataset for top-K algorithms."""
    algo_means = df.groupby("algorithm")[score_col].mean().sort_values(ascending=False)
    top_algos = [a for a in algo_means.index if a != "random"][:top_k]

    group_means = (
        df[df["algorithm"].isin(top_algos)]
        .groupby(["algorithm", group_by])[score_col]
        .mean()
        .unstack(fill_value=0)
    )

    if group_by == "domain":
        active_groups = [d for d in DOMAIN_ORDER if d in group_means.columns]
    else:
        active_groups = sorted([d for d in group_means.columns if d != "pamap2"])

    group_means = group_means[active_groups]

    N = len(active_groups)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2 + np.pi / N)
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(0)

    for i, algo in enumerate(top_algos):
        values = group_means.loc[algo].tolist()
        values += values[:1]
        # Use a maximally-distinct palette indexed by rank position so
        # the top-K curves are always visually well separated, even if
        # the global per-algo palette would give similar hues.
        c = _SPIDER_TOP5_COLORS[i % len(_SPIDER_TOP5_COLORS)]
        ax.plot(angles, values, linewidth=1.5, label=display_name(algo), color=c)
        ax.fill(angles, values, alpha=0.10, color=c)

    ax.set_xticks(angles[:-1])
    label_size = 14 if group_by == "domain" else 10
    ax.set_xticklabels(active_groups, size=label_size, color="black", fontweight="semibold")
    ax.tick_params(axis="y", labelsize=12)

    ax.set_ylim(0, 1)
    if legend_outside:
        # Place legend to the right, fully outside the polar axes so it
        # does not overlap any spoke or label.
        ax.legend(loc="center left", bbox_to_anchor=(1.25, 0.5),
                  fontsize=12, ncol=1, framealpha=0.8, borderaxespad=0)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(-0.85, 1.12), fontsize=12,
                  ncol=1, framealpha=0.8, borderaxespad=0)


def plot_cd_diagram(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_col: str = "score_best",
    maximize: bool = True,
):
    """Critical difference diagram via aeon — rendered into the target axes."""
    try:
        from aeon.visualisation import plot_critical_difference
    except ImportError:
        ax.text(0.5, 0.5, "aeon not installed", transform=ax.transAxes, ha="center")
        return

    # ... earlier code handles missing data
    df_clean = df.copy()
    df_clean["series_id"] = df_clean["dataset"] + "_" + df_clean["trial_index"].astype(str)

    pivot = df_clean.pivot_table(
        index="series_id", columns="algorithm", values=score_col, aggfunc="first"
    )

    if "random" in pivot.columns:
        pivot = pivot.drop(columns=["random"])

    if pivot.empty or pivot.shape[1] < 3:
        ax.text(0.5, 0.5, "Insufficient data for CD (<3 algos)", transform=ax.transAxes, ha="center")
        return

    pivot = pivot.fillna(0.0)
    pivot.columns = [display_name(c) for c in pivot.columns]

    # ``reverse=True`` puts the best (lowest mean rank) algorithm on
    # the RIGHT side of the diagram (aeon's default), matching the
    # worst→best left-to-right ordering of the strip plot.
    _cd_font_rc = {
        "font.size": 24,
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        "legend.fontsize": 22,
    }
    with mpl.rc_context(_cd_font_rc):
        cd_fig, _cd_ax = plot_critical_difference(
            pivot.values,
            list(pivot.columns),
            lower_better=not maximize,
            reverse=True,
        )

    # Post-process the aeon plot to hide alternating rank ticks and prevent overlap
    # aeon draws the rank axis numbers as text objects with y-position < 0.2.
    for t in _cd_ax.texts:
        text_str = t.get_text()
        y_pos = t.get_position()[1]
        if text_str.isdigit() and y_pos < 0.2:
            val = int(text_str)
            # Keep ONLY odd numbers (1, 3, 5...) to double the spacing
            if val % 2 == 0:
                t.set_visible(False)

    # Render aeon's CD into a high-DPI PNG and embed at the natural
    # aspect ratio (``aspect='equal'``) so the diagram is not stretched.
    # Note: aeon's CD is built with matplotlib in pixel-space (uses
    # ``imshow``-incompatible coordinates), so SVG/PDF re-embedding is
    # not straightforward. We rasterise at high DPI for crisp output.
    buf = BytesIO()
    cd_fig.savefig(buf, format="png", dpi=1200, bbox_inches="tight", pad_inches=0.02)
    plt.close(cd_fig)
    buf.seek(0)
    cd_img = np.array(Image.open(buf))

    # Render the CD into a high-DPI PNG, then embed at its natural
    # aspect ratio (anchored to the top of the slot) so it is not
    # stretched vertically.
    ax.imshow(cd_img, interpolation="none", aspect="equal")
    ax.set_anchor("N")
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────


def _save_cd_directly(df: pd.DataFrame, score_col: str, maximize: bool,
                      name: str, width: float = 5.0):
    """Save the aeon CD figure directly with tight bbox (no whitespace)."""
    try:
        from aeon.visualisation import plot_critical_difference
    except ImportError:
        return

    df_clean = df.copy()
    df_clean["series_id"] = df_clean["dataset"] + "_" + df_clean["trial_index"].astype(str)
    pivot = df_clean.pivot_table(
        index="series_id", columns="algorithm", values=score_col, aggfunc="first"
    )
    if "random" in pivot.columns:
        pivot = pivot.drop(columns=["random"])
    if pivot.empty or pivot.shape[1] < 3:
        return
    pivot = pivot.fillna(0.0)
    pivot.columns = [display_name(c) for c in pivot.columns]

    _cd_font_rc = {
        "font.size": 28,
        "axes.titlesize": 28,
        "axes.labelsize": 28,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "legend.fontsize": 26,
    }
    with mpl.rc_context(_cd_font_rc):
        cd_fig, _cd_ax = plot_critical_difference(
            pivot.values,
            list(pivot.columns),
            lower_better=not maximize,
            reverse=True,
            width=width,
        )

    # Apply the same post-processing trick for individual panels
    for t in _cd_ax.texts:
        text_str = t.get_text()
        if text_str.isdigit() and t.get_position()[1] < 0.2:
            if int(text_str) % 2 == 0:
                t.set_visible(False)

    savefig(cd_fig, name, tight=True)
    plt.close(cd_fig)


def _composite(df: pd.DataFrame, title: str, ylabel: str, name: str,
               fig_w=TEXT_W, show_hatch: bool = True,
               panel_legend: bool = True, maximize: bool = True):
    # Square figure (NeurIPS text-width × text-width).
    # Layout: row 0 = full-width strip, row 1 = CD (left) + spider (right).
    fig = plt.figure(figsize=(fig_w, fig_w), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 2.5], width_ratios=[4, 2])

    ax_strip = fig.add_subplot(gs[0, :])
    ax_cd = fig.add_subplot(gs[1, 0])
    ax_spider = fig.add_subplot(gs[1, 1], polar=True)

    plot_strip(ax_strip, df, score_col="score_best", ylabel=ylabel,
               show_hatch=show_hatch)
    ax_strip.set_title(f"(a) {title}", loc="left", fontweight="bold")

    plot_cd_diagram(ax_cd, df, score_col="score_best", maximize=maximize)
    ax_cd.set_title("(b) Critical Difference", loc="left", fontweight="bold")

    plot_spider(ax_spider, df, score_col="score_best", top_k=5)
    ax_spider.set_title("(c) Domain", loc="left", fontweight="bold", y=1.08)

    savefig(fig, name)

    # Also export each panel as a stand-alone PDF/PNG. SD panels (no hatch)
    # are exported as a square strip so they can fit in a 4-column main-text
    # row alongside their CD diagram and their guided counterparts.
    _export_panels(df, name, ylabel, show_hatch, square=not show_hatch,
                   show_legend=panel_legend, maximize=maximize)


def _export_panels(df: pd.DataFrame, name_base: str, ylabel: str,
                   show_hatch: bool, square: bool = False,
                   show_legend: bool = True, maximize: bool = True):
    """Save the strip / CD / spider panels of a composite as individual figures.

    ``square=True`` produces a square strip plot (used for SD with few algos
    so the panel can be placed alongside others in a 4-column row).
    ``show_legend=False`` hides the strip legend (used for the panels that
    appear next to a panel that already shows it).
    """
    with _panel_rc():
        # Strip — fixed figsize, no bbox-tight expansion (otherwise the
        # legend's bbox can balloon the saved file beyond the requested
        # dimensions). For SD (``square=True``) we use a narrow figure
        # whose bar density (figure_width / n_algos) roughly matches
        # the CPD strip plot, so that bars look the same physical width
        # once both panels are embedded in LaTeX.
        n_algos = max(1, df["algorithm"].nunique() - (1 if "random" in set(df["algorithm"]) else 0))
        if square:
            # CPD ref: ~15 algos in 0.55*TEXT_W → ~0.037*TEXT_W per algo.
            sd_w = max(TEXT_W * 0.22, n_algos * TEXT_W * 0.045)
            fig, ax = plt.subplots(figsize=(sd_w, TEXT_W * 0.25))
        else:
            fig, ax = plt.subplots(figsize=(TEXT_W * 0.55, TEXT_W * 0.25))
        plot_strip(ax, df, score_col="score_best", ylabel=ylabel,
                   show_hatch=show_hatch, show_legend=show_legend,
                   legend_ncol=1 if square else 2,
                   ylim=(0.0, 0.8) if square else None)
        # Add headroom on top so the legend (when shown) does not overlap
        # bars. Skip this for ``square`` (SD) panels, otherwise NG and G
        # would no longer share the same y-axis range (NG has the legend,
        # G doesn't).
        if show_legend and not square:
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax + 0.00 * (ymax - ymin))
        # Use tight bounding box so rotated x-tick labels are never
        # clipped (the in-axes legend stays inside the axes via the
        # 20% y-headroom above, so tight bbox is safe).
        savefig(fig, f"{name_base}_strip", tight=True)

        # Critical difference: save the aeon figure directly (tightly
        # cropped) to avoid the whitespace introduced by the imshow
        # re-embedding pipeline used inside ``plot_cd_diagram``. Use a
        # narrower CD when there are few algorithms (SD), wider when many
        # (CPD), so that the per-algorithm spacing stays readable.
        cd_width = 5.0 if square else 5.0
        _save_cd_directly(df, score_col="score_best", maximize=maximize,
                          name=f"{name_base}_cd", width=cd_width)

        # Spider / radar (kept for the appendix). Larger figsize to
        # accommodate the bigger fonts set in ``plot_spider``. Use tight
        # bbox to ensure the right-side legend and the leftmost
        # axis-label ("Other") are not clipped.
        fig = plt.figure(figsize=(TEXT_W * 1.10, TEXT_W * 0.5))
        ax = fig.add_subplot(111, polar=True)
        plot_spider(ax, df, score_col="score_best", top_k=5, legend_outside=True, group_by="domain")
        savefig(fig, f"{name_base}_spider", tight=True)
        plt.close(fig)

        # Spider / radar (by dataset for the appendix).
        fig = plt.figure(figsize=(TEXT_W * 1.10, TEXT_W * 0.70))
        ax = fig.add_subplot(111, polar=True)
        plot_spider(ax, df, score_col="score_best", top_k=5, legend_outside=True, group_by="dataset")
        savefig(fig, f"{name_base}_spider_dataset", tight=True)
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# Split strip (Non-Guided left half / Guided right half)
# ──────────────────────────────────────────────────────────────────

_NG_ALPHA = 0.35
_G_ALPHA = 0.85


def _algo_stats(df: pd.DataFrame, algo: str) -> dict | None:
    sub = df[df["algorithm"] == algo]
    if sub.empty:
        return None
    def _m(col):
        if col not in sub.columns:
            return None
        v = sub[col].dropna()
        return float(v.mean()) if not v.empty else None
    out = dict(best=_m("score_best"), worst=_m("score_worst"),
               default=_m("score_default"), median=_m("score_median"))
    if out["best"] is None and out["worst"] is None:
        return None
    return out


def _draw_strip_half(ax, x, side, color, stats, width=0.8):
    """Draw NG (left), G (right), or full bar for one algorithm."""
    half = width / 2.0
    if side == "left":
        xc = x - half / 2.0
        bw = half * 0.95
        face_alpha = _NG_ALPHA
    elif side == "right":
        xc = x + half / 2.0
        bw = half * 0.95
        face_alpha = _G_ALPHA
    else:  # full
        xc = x
        bw = width * 0.95
        face_alpha = _G_ALPHA

    best = stats.get("best")
    worst = stats.get("worst")
    default = stats.get("default")
    median = stats.get("median")

    if best is not None and worst is not None:
        h = best - worst
        ax.bar(xc, h, bottom=worst, width=bw,
               facecolor=to_rgba(color, face_alpha),
               edgecolor=color, linewidth=_BAR_EDGE_LW, zorder=2)
    if median is not None:
        ax.hlines(median, xc - bw / 2, xc + bw / 2,
                  colors=color, linewidth=1.0, alpha=0.7, zorder=3)
    for val, marker, ms, alpha in (
        (best, "^", 28, 0.95),
        (default, "D", 22, 0.9),
        (worst, "v", 20, 0.75),
    ):
        if val is None:
            continue
        ax.scatter(xc, val, marker=marker, s=ms, color=color,
                   edgecolors="none", alpha=alpha, zorder=4)


def _split_strip(df_ng: pd.DataFrame, df_g: pd.DataFrame, ylabel: str,
                 name: str, title: str | None = None):
    """Build a split-strip figure: NG (left half) vs G (right half) per algo.

    Algorithms with no NG run (guided-only) are drawn as a full-width bar
    and pushed to the right of the x-axis. Ordering is by best-grid score
    in the Non-Guided regime (descending).
    """
    algos = {a for a in set(df_ng["algorithm"]) | set(df_g["algorithm"]) if a != "random"}
    stats_ng = {a: _algo_stats(df_ng, a) for a in algos}
    stats_g = {a: _algo_stats(df_g, a) for a in algos}
    stats_ng = {a: s for a, s in stats_ng.items() if s is not None}
    stats_g = {a: s for a, s in stats_g.items() if s is not None}

    # Order worst → best (left to right) by NG best score.
    has_ng = [a for a in stats_ng if stats_ng[a].get("best") is not None]
    has_ng.sort(key=lambda a: stats_ng[a]["best"])
    # Guided-only algos: also worst → best, placed on the FAR LEFT so
    # that the right end of the axis still corresponds to the best NG.
    g_only = [a for a in stats_g if a not in stats_ng]
    g_only.sort(key=lambda a: (stats_g[a].get("best") or -np.inf))
    ordered = g_only + has_ng

    if not ordered:
        return

    fig, ax = plt.subplots(figsize=(TEXT_W, 3.4))

    # Random baseline (NG mean if available, else G).
    rand_val = None
    if "random" in set(df_ng["algorithm"]):
        rand_val = df_ng.loc[df_ng["algorithm"] == "random", "score_best"].mean()
    elif "random" in set(df_g["algorithm"]):
        rand_val = df_g.loc[df_g["algorithm"] == "random", "score_best"].mean()
    if rand_val is not None and np.isfinite(rand_val):
        ax.axhline(rand_val, ls="--", lw=0.9, color="#888888", alpha=0.55, zorder=0)
        ax.text(-0.5, rand_val + 0.01, "Random",
                ha="left", va="bottom", fontsize=6,
                color="#888888", fontstyle="italic", zorder=0)

    for i, a in enumerate(ordered):
        c = ALGO_COLORS.get(a, "#888888")
        sng = stats_ng.get(a)
        sg = stats_g.get(a)
        if sng is not None and sg is not None:
            _draw_strip_half(ax, i, "left", c, sng)
            _draw_strip_half(ax, i, "right", c, sg)
        elif sg is not None:
            _draw_strip_half(ax, i, "full", c, sg)
        elif sng is not None:
            _draw_strip_half(ax, i, "full", c, sng)

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([display_name(a) for a in ordered],
                       rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(ordered) - 0.4)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.3)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=8.5)

    legend_handles = [
        Line2D([], [], marker="^", color="gray", markersize=5, linestyle="None",
               markeredgecolor="none", alpha=0.95, label="Best grid"),
        Line2D([], [], marker="v", color="gray", markersize=4, linestyle="None",
               markeredgecolor="none", alpha=0.75, label="Worst grid"),
        Line2D([], [], marker="D", color="gray", markersize=5, linestyle="None",
               markeredgecolor="none", alpha=0.9, label="Default"),
        mpl.patches.Patch(facecolor=to_rgba("gray", _NG_ALPHA), edgecolor="gray",
                          linewidth=_BAR_EDGE_LW, label="Non-Guided (left)"),
        mpl.patches.Patch(facecolor=to_rgba("gray", _G_ALPHA), edgecolor="gray",
                          linewidth=_BAR_EDGE_LW, label="Guided (right)"),
        Line2D([], [], color="gray", linewidth=1.0, alpha=0.7, label="Median"),
        Line2D([], [], ls="--", lw=0.9, color="#888888", alpha=0.55,
               label="Random baseline"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=6,
              ncol=2, framealpha=0.85)

    fig.tight_layout()
    savefig(fig, name)


def make_all_figures(d=None):
    """Generate composites + per-panel exports + split (NG/G) strip figures."""
    if d is None:
        d = load_data()
    # CPD plots mix CPD + SD algos → hatch is meaningful.
    # SD plots only show SD algos → hatch is redundant, disable it.
    # Legend is shown only on the Non-Guided strips; Guided strips reuse it
    # via the LaTeX caption to save space in the main-text 3-row layout.
    _composite(d.df_cpd,   "Change Point Detection Performance — Non-Guided",
               "Bi-Covering Score",     "fig_cpd_main",   fig_w=TEXT_W,
               show_hatch=True,  panel_legend=True)
    _composite(d.df_sd,    "State Detection Performance — Non-Guided",
               "State Matching Score",  "fig_sd_main",    fig_w=TEXT_W,
               show_hatch=False, panel_legend=True)
    _composite(d.df_cpd_g, "Change Point Detection Performance — Guided",
               "Bi-Covering Score",     "fig_cpd_guided", fig_w=TEXT_W,
               show_hatch=True,  panel_legend=False)
    _composite(d.df_sd_g,  "State Detection Performance — Guided",
               "State Matching Score",  "fig_sd_guided",  fig_w=TEXT_W,
               show_hatch=False, panel_legend=False)

    # Split strip figures: NG (left half) + G (right half).
    _split_strip(d.df_cpd, d.df_cpd_g,
                 ylabel="Bi-Covering Score",
                 name="fig_cpd_split",
                 title="Change-Point Detection (CPD)")
    _split_strip(d.df_sd, d.df_sd_g,
                 ylabel="State Matching Score",
                 name="fig_sd_split",
                 title="State Detection (SD)")

    print("\n--- Generating plots for all other metrics ---")
    _generate_metrics_figures(d)


def _build_metric_df(df_grid: pd.DataFrame, df_default: pd.DataFrame, algos: set, metric: str) -> pd.DataFrame:
    from analysis.helpers import (
        ALGO_CATEGORY, ALGO_DISPLAY, EXCLUDE_INCOMPLETE_ALGOS, get_domain,
        select_best_grid_per_dataset, aggregate_grid_summaries
    )
    higher_is_better = metric != "execution_time_seconds"
    ranked = select_best_grid_per_dataset(df_grid, metric, higher_is_better=higher_is_better)
    if not ranked.empty:
        grid_summary = aggregate_grid_summaries(ranked, metric)
    else:
        grid_summary = pd.DataFrame()

    if metric in df_default.columns:
        default_summary = df_default[["algorithm", "dataset", "trial_index", metric]].copy()
        default_summary = default_summary.rename(columns={metric: "score_default"})
    else:
        default_summary = pd.DataFrame(columns=["algorithm", "dataset", "trial_index", "score_default"])

    if not grid_summary.empty:
        df_metric = grid_summary.merge(
            default_summary, on=["algorithm", "dataset", "trial_index"], how="outer"
        )
    else:
        df_metric = default_summary.copy()
        df_metric["score_best"] = np.nan
        df_metric["score_worst"] = np.nan
        df_metric["score_median"] = np.nan

    df_metric["domain"] = df_metric.apply(lambda r: get_domain(r["dataset"], r["trial_index"]), axis=1)
    df_metric["category"] = df_metric["algorithm"].map(ALGO_CATEGORY).fillna("?")
    df_metric["display"] = df_metric["algorithm"].map(ALGO_DISPLAY).fillna(df_metric["algorithm"])

    df_metric = df_metric[df_metric["algorithm"].isin(algos)].copy()
    df_metric = df_metric[df_metric["dataset"] != "pamap2"].copy()

    n_datasets = df_metric["dataset"].nunique()
    algo_ds_count = df_metric.groupby("algorithm")["dataset"].nunique()
    complete_algos = set(algo_ds_count[algo_ds_count == n_datasets].index)
    df_metric = df_metric[df_metric["algorithm"].isin(complete_algos)].copy()

    from analysis.helpers import CPD_METRICS
    if metric in CPD_METRICS:
        df_metric = df_metric[df_metric["algorithm"] != "clap"].copy()

    if EXCLUDE_INCOMPLETE_ALGOS:
        nan_per_algo = df_metric.groupby("algorithm")["score_best"].apply(lambda s: s.isna().sum())
        algos_with_gaps = set(nan_per_algo[nan_per_algo > 0].index)
        if algos_with_gaps:
            gap_detail = {a: int(nan_per_algo[a]) for a in sorted(algos_with_gaps)}
            print(f"  Excluding algos with NaN score_best for {metric}: {gap_detail}")
            df_metric = df_metric[~df_metric["algorithm"].isin(algos_with_gaps)].copy()
            
    return df_metric


def _generate_metrics_figures(d):
    from analysis.helpers import CPD_METRICS, SD_METRICS, OUTPUT_DIR, TEXT_W
    from analysis.data import _build_algo_sets

    cpd_algos, sd_algos = _build_algo_sets()
    
    out_dir = OUTPUT_DIR / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for metric in sorted(set(CPD_METRICS + SD_METRICS)):
        print(f"Generating figures for metric: {metric}")
        # Generate for CPD
        if metric in CPD_METRICS:
            df = _build_metric_df(d.df_grid_ng, d.df_default_ng, cpd_algos, metric)
            if not df.empty:
                max_better = metric != "execution_time_seconds"
                _composite(df, f"CPD — {metric}", metric, str(out_dir / f"fig_cpd_{metric}_main"),
                           fig_w=TEXT_W, show_hatch=True, panel_legend=True, maximize=max_better)
            df_g = _build_metric_df(d.df_grid_g, d.df_default_g, cpd_algos, metric)
            if not df_g.empty:
                max_better = metric != "execution_time_seconds"
                _composite(df_g, f"CPD Guided — {metric}", metric, str(out_dir / f"fig_cpd_{metric}_guided"),
                           fig_w=TEXT_W, show_hatch=True, panel_legend=False, maximize=max_better)

        # Generate for SD
        if metric in SD_METRICS:
            df = _build_metric_df(d.df_grid_ng, d.df_default_ng, sd_algos, metric)
            if not df.empty:
                max_better = metric != "execution_time_seconds"
                _composite(df, f"SD — {metric}", metric, str(out_dir / f"fig_sd_{metric}_main"),
                           fig_w=TEXT_W, show_hatch=False, panel_legend=True, maximize=max_better)
            df_g = _build_metric_df(d.df_grid_g, d.df_default_g, sd_algos, metric)
            if not df_g.empty:
                max_better = metric != "execution_time_seconds"
                _composite(df_g, f"SD Guided — {metric}", metric, str(out_dir / f"fig_sd_{metric}_guided"),
                           fig_w=TEXT_W, show_hatch=False, panel_legend=False, maximize=max_better)

print("Plot helpers defined: plot_strip, plot_spider, plot_cd_diagram, make_all_figures")


if __name__ == "__main__":
    make_all_figures()

