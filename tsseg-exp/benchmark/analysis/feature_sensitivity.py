"""FIGURE_FEATURE_SENSITIVITY: correlation of best-grid scores with
input time-series features.

For every algorithm and every supervision mode (NG = non-guided,
G = guided), we compute the Spearman rank correlation between its
``score_best`` (best grid configuration on the target metric) and four
per-series features:

    * length            (number of samples)
    * dimensions        (number of channels)
    * n_change_points   (ground-truth segmentation count)
    * n_states          (ground-truth state count)

Two heatmaps are produced side by side:
  - left  : CPD score, all algorithms (CLaP merged into CLaSP upstream).
  - right : SD score, SD-capable algorithms only.

Each algorithm spans two rows (NG / G). Cells where the algorithm does
not support that supervision mode are shown in grey with a cross.
The ``random`` baseline is dropped.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as _scstats

from analysis.data import load_data
from analysis.helpers import (
    ALGO_DISPLAY,
    ALGO_TASK,
    BENCH_DIR,
    TEXT_W,
    savefig,
)

FEATURES_CSV = BENCH_DIR / "datasets" / "dataset_features.csv"

FEATURE_COLS = ["length", "dimensions", "n_change_points", "n_states"]
FEATURE_LABELS = {
    "length":          r"$n$",
    "dimensions":      r"$d$",
    "n_change_points": r"$n_{CP}$",
    "n_states":        r"$n_s$",
}

CMAP = plt.get_cmap("PRGn")
_NA_COLOR = "#EEEEEE"


def _load_features() -> pd.DataFrame:
    f = pd.read_csv(FEATURES_CSV)
    f = f.rename(columns={"series_index": "trial_index"})
    f["trial_index"] = f["trial_index"].astype("Int64")
    return f[["dataset", "trial_index", *FEATURE_COLS]]


def _merge_features(df: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["trial_index"] = pd.to_numeric(df["trial_index"], errors="coerce").astype("Int64")
    return df.merge(feats, on=["dataset", "trial_index"], how="left")


def _spearman_table(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for algo, sub in df.groupby("algorithm"):
        sub = sub.dropna(subset=[score_col, *FEATURE_COLS])
        if len(sub) < 5:
            continue
        rho_row: dict[str, float] = {}
        for feat in FEATURE_COLS:
            x = sub[feat].to_numpy(dtype=float)
            y = sub[score_col].to_numpy(dtype=float)
            if np.unique(x).size < 2 or np.unique(y).size < 2:
                rho_row[feat] = np.nan
            else:
                rho, _ = _scstats.spearmanr(x, y)
                rho_row[feat] = float(rho)
        rows[algo] = rho_row
    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS)
    return pd.DataFrame.from_dict(rows, orient="index")[FEATURE_COLS]


def _build_task_table(
    df_ng: pd.DataFrame,
    df_g: pd.DataFrame,
    feats: pd.DataFrame,
    algo_filter: set | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    df_ng = _merge_features(df_ng, feats)
    df_g = _merge_features(df_g, feats)
    if algo_filter is not None:
        df_ng = df_ng[df_ng["algorithm"].isin(algo_filter)]
        df_g = df_g[df_g["algorithm"].isin(algo_filter)]
    df_ng = df_ng[df_ng["algorithm"] != "random"]
    df_g = df_g[df_g["algorithm"] != "random"]

    corr_ng = _spearman_table(df_ng, "score_best")
    corr_g = _spearman_table(df_g, "score_best")

    algos = sorted(set(corr_ng.index) | set(corr_g.index))

    score_means: dict[str, float] = {}
    for a in algos:
        vals = []
        for src in (df_ng, df_g):
            v = src.loc[src["algorithm"] == a, "score_best"].dropna().to_numpy()
            if v.size:
                vals.append(float(v.mean()))
        score_means[a] = float(np.mean(vals)) if vals else -np.inf
    algos = sorted(algos, key=lambda a: score_means[a], reverse=True)

    rows: list[list[float]] = []
    support: list[list[bool]] = []
    index: list[tuple[str, str]] = []
    for a in algos:
        for mode, table in (("NG", corr_ng), ("G", corr_g)):
            if a not in table.index:
                continue
            index.append((a, mode))
            rows.append(table.loc[a].to_numpy(dtype=float).tolist())
            support.append([True] * len(FEATURE_COLS))
    corr = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=("algorithm", "mode")),
        columns=FEATURE_COLS,
    )
    return corr, np.asarray(support, dtype=bool)


def _draw_heatmap(ax, corr, support, title):
    # Transpose: features become rows, (algo, mode) pairs become columns.
    data = corr.to_numpy(dtype=float).T
    sup = support.T
    n_rows, n_cols = data.shape  # n_rows = #features, n_cols = 2 * #algos

    bg = np.where(sup, np.nan, 0.0)
    ax.imshow(
        bg,
        cmap=LinearSegmentedColormap.from_list("na", [_NA_COLOR, _NA_COLOR]),
        vmin=0, vmax=1, aspect="auto",
    )
    masked = np.ma.array(data, mask=~sup)
    im = ax.imshow(masked, cmap=CMAP, vmin=-1.0, vmax=1.0, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            if not sup[i, j]:
                continue
            v = data[i, j]
            if np.isnan(v):
                ax.text(j, i, ".", ha="center", va="center",
                        fontsize=6, color="0.4")
                continue
            color = "white" if abs(v) > 0.55 else "black"
            # Compact label: drop leading 0 ("0.27" → ".27") so values fit in narrow cells.
            s = f"{abs(v):.2f}"
            if s.startswith("0"):
                s = s[1:]
            txt = ("\u2212" + s) if v < 0 else s
            ax.text(j, i, txt,
                    ha="center", va="center",
                    fontsize=4.0, color=color)

    # X axis: algo names + NG/G sub-labels.
    # corr.index is MultiIndex of (algo, mode) — pairs come consecutively.
    pairs = list(corr.index)
    modes = [m for (_a, m) in pairs]
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(modes, fontsize=6)
    ax.tick_params(axis="x", which="major", length=0, pad=1)

    # Algo group labels: place at the center of each consecutive algo run.
    groups: list[tuple[str, int, int]] = []  # (algo, start_col, end_col_exclusive)
    for j, (a, _m) in enumerate(pairs):
        if groups and groups[-1][0] == a:
            algo, s, _e = groups[-1]
            groups[-1] = (algo, s, j + 1)
        else:
            groups.append((a, j, j + 1))
    import matplotlib.transforms as mtransforms
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for (a, s, e) in groups:
        x_center = (s + e - 1) / 2 + 0.5
        disp = ALGO_DISPLAY.get(a, a)
        ax.annotate(
            disp,
            xy=(x_center, 0), xycoords=trans,
            xytext=(0, -12), textcoords="offset points",
            ha="right", va="top",
            fontsize=6.0, rotation=45, rotation_mode="anchor",
        )

    # Y axis: feature labels.
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([FEATURE_LABELS[c] for c in corr.columns], fontsize=7)

    # Cell separators + heavier separator between consecutive algos.
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(False, which="minor")
    
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    for (_a, _s, e) in groups[:-1]:
        ax.axvline(e - 0.5, color="0.35", linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_title(title, fontsize=8, pad=4)
    return im


def make_figure(d=None):
    if d is None:
        d = load_data()

    feats = _load_features()
    corr_cpd, sup_cpd = _build_task_table(d.df_cpd, d.df_cpd_g, feats)
    sd_algo_set = {a for a, t in ALGO_TASK.items() if t == "SD"}
    corr_sd, sup_sd = _build_task_table(
        d.df_sd, d.df_sd_g, feats, algo_filter=sd_algo_set,
    )

    n_cpd = corr_cpd.shape[0]  # rows = (algo, mode) pairs → columns after transpose
    n_sd = corr_sd.shape[0]
    n_feat = len(FEATURE_COLS)

    # Stack CPD on top, SD below; both transposed (features as rows, algos as cols).
    fig = plt.figure(figsize=(TEXT_W, 4.2))
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[1.0, 1.0, 0.05],
        hspace=1.0,
    )
    ax_cpd = fig.add_subplot(gs[0, 0])
    ax_sd = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[2, 0])

    _draw_heatmap(ax_cpd, corr_cpd, sup_cpd, "CPD score (BiCovering)")
    im = _draw_heatmap(ax_sd, corr_sd, sup_sd, "SD score (SMS)")

    cbar = plt.colorbar(im, cax=ax_cb, orientation="horizontal")
    cbar.set_label(r"Spearman $\rho$", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    fig.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.12)

    savefig(fig, "fig_feature_sensitivity")
    plt.show()


if __name__ == "__main__":
    make_figure()
