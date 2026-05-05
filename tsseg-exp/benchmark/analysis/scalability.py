"""FIGURE_SCALABILITY: per-algorithm runtime vs series length.

Shows, in log-log space, how each algorithm's per-series runtime scales
with the input length ``n``. Combines all benchmark datasets (including
PAMAP2) so that the long PAMAP2 series serve as a natural extrapolation
point for the runtime law of each method.

For each algorithm we fit ``log(runtime) = alpha * log(n) + b`` and
report the empirical exponent ``alpha`` in the legend; this directly
quantifies whether a method behaves as advertised by its theoretical
complexity (e.g. ClaSP's quadratic dependence).

Data: per-series runtime from the *default* runs (NG + G merged); for
each (algo, dataset, series) we take the median runtime across both
regimes (when available). Algorithms that systematically fail on PAMAP2
are still shown, but use only their non-PAMAP2 points.

Outputs:

* ``figures/paper/fig_scalability.pdf``
* ``figures/paper/fig_scalability.png``
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.helpers import (
    ALGO_TASK,
    ALL_METRICS,
    BENCH_DIR,
    TEXT_W,
    _coalesce_sms,
    algo_color,
    analyzer,
    display_name,
    savefig,
)

RUNTIME_COL = "execution_time_seconds"
DATASET_FEATURES_CSV = BENCH_DIR / "datasets" / "dataset_features.csv"

# Algorithms excluded from the figure (systematic timeouts or never
# produced any meaningful runtime data — they would only add clutter).
EXCLUDED_ALGOS = {"random"}

# Minimum number of distinct lengths to attempt a regression.
MIN_POINTS_FIT = 8

# Visual style
FS_TICK = 9
FS_LABEL = 10
FS_LEGEND = 7

# Reference timeout band (drawn as a horizontal stripe).
TIMEOUT_S = 4 * 3600  # 4h job-level cap used in the cluster runs


def _fetch_default(experiment_key: str) -> pd.DataFrame:
    """Return all finished default children with runtime, across all datasets."""
    df_parents = analyzer.fetch_parent_stats([experiment_key])
    if df_parents.empty:
        return pd.DataFrame()
    df_valid = analyzer.validate_completeness(df_parents, strategy="relaxed")
    df_valid = (
        df_valid.sort_values(["children_finished", "start_time"], ascending=[False, False])
        .drop_duplicates(subset=["algorithm", "dataset"], keep="first")
    )
    df = analyzer.fetch_metrics(df_valid, metric_keys=ALL_METRICS, deduplicate_series=True)
    return _coalesce_sms(df)


def _load_runtime_table() -> pd.DataFrame:
    """Per (algo, dataset, series_index): median runtime over NG+G."""
    df_ng = _fetch_default("non_guided")
    df_g = _fetch_default("guided")
    if df_ng.empty and df_g.empty:
        return pd.DataFrame()

    parts = []
    for df, regime in ((df_ng, "ng"), (df_g, "g")):
        if df.empty:
            continue
        keep = ["algorithm", "dataset", "trial_index", RUNTIME_COL]
        sub = df[[c for c in keep if c in df.columns]].copy()
        sub[RUNTIME_COL] = pd.to_numeric(sub[RUNTIME_COL], errors="coerce")
        sub = sub.dropna(subset=[RUNTIME_COL])
        sub = sub[sub[RUNTIME_COL] > 0]
        sub["regime"] = regime
        parts.append(sub)

    if not parts:
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    # Median runtime across regimes (per series).
    df = (
        df.groupby(["algorithm", "dataset", "trial_index"])[RUNTIME_COL]
        .median()
        .reset_index()
    )
    df["trial_index"] = pd.to_numeric(df["trial_index"], errors="coerce").astype("Int64")
    return df


def _load_lengths() -> pd.DataFrame:
    """(dataset, series_index) → length, dimensions."""
    df = pd.read_csv(DATASET_FEATURES_CSV)
    df = df.rename(columns={"series_index": "trial_index"})
    df["dataset"] = df["dataset"].str.lower()
    df["trial_index"] = df["trial_index"].astype("Int64")
    return df[["dataset", "trial_index", "length", "dimensions"]]


def _fit_powerlaw(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) of a log-log linear fit."""
    lx, ly = np.log10(x), np.log10(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    yhat = slope * lx + intercept
    ss_res = np.sum((ly - yhat) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2)


def _build_panel(ax, df: pd.DataFrame, task: str) -> None:
    sub = df[df["task"] == task].copy()
    if sub.empty:
        ax.set_visible(False)
        return

    algos = sorted(sub["algorithm"].unique(),
                   key=lambda a: -sub.loc[sub["algorithm"] == a, "length"].nunique())

    handles_legend: list = []
    labels_legend: list[str] = []

    for algo in algos:
        s = sub[sub["algorithm"] == algo]
        x = s["length"].to_numpy(dtype=float)
        y = s[RUNTIME_COL].to_numpy(dtype=float)
        m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            continue

        col = algo_color(algo)
        ax.scatter(x, y, s=14, color=col, alpha=0.55,
                   edgecolors="white", linewidths=0.3, zorder=3)

        # Power-law fit on per-length medians (avoids overweighting datasets
        # with many series at the same scale).
        df_alg = pd.DataFrame({"x": x, "y": y})
        med = df_alg.groupby("x")["y"].median().reset_index()
        if len(med) >= MIN_POINTS_FIT:
            slope, intercept, r2 = _fit_powerlaw(med["x"].to_numpy(),
                                                  med["y"].to_numpy())
            xs = np.array([med["x"].min(), med["x"].max()])
            ys = 10 ** (slope * np.log10(xs) + intercept)
            ax.plot(xs, ys, color=col, lw=1.0, alpha=0.85, zorder=2)
            label = f"{display_name(algo)}  $\\alpha\\!\\approx\\!{slope:.2f}$"
        else:
            label = f"{display_name(algo)}  (n={len(med)})"

        handles_legend.append(plt.Line2D([], [], marker="o", color=col,
                                         linestyle="-", markersize=5,
                                         markeredgecolor="white",
                                         markeredgewidth=0.4))
        labels_legend.append(label)

    # Timeout reference band
    ax.axhspan(TIMEOUT_S, TIMEOUT_S * 100, color="0.85", alpha=0.5, zorder=0)
    ax.axhline(TIMEOUT_S, color="0.55", lw=0.6, ls="--", zorder=1)
    ax.text(ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 1e6, TIMEOUT_S * 1.15,
            "4h job cap",
            fontsize=FS_TICK - 1, color="0.40",
            ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Series length $n$ (samples, log scale)", fontsize=FS_LABEL)
    ax.set_ylabel("Runtime per series (s, log scale)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, which="both", alpha=0.20, lw=0.3)
    ax.set_title(f"{task}", fontsize=FS_LABEL + 1)

    if handles_legend:
        ax.legend(handles_legend, labels_legend, fontsize=FS_LEGEND,
                  loc="lower right", frameon=True, ncol=2,
                  handlelength=1.4, columnspacing=0.8, labelspacing=0.25)


def make_figure() -> None:
    print("Fetching default runtime data ...")
    df_rt = _load_runtime_table()
    if df_rt.empty:
        print("⚠ No runtime data fetched — aborting.")
        return

    df_len = _load_lengths()
    df = df_rt.merge(df_len, on=["dataset", "trial_index"], how="inner")
    df = df[~df["algorithm"].isin(EXCLUDED_ALGOS)].copy()
    df["task"] = df["algorithm"].map(ALGO_TASK).fillna("?")

    print(f"  merged: {len(df)} (algo, series) rows, "
          f"{df['algorithm'].nunique()} algos, "
          f"{df['dataset'].nunique()} datasets")
    print(f"  pamap2 points: {(df['dataset'] == 'pamap2').sum()}")

    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, TEXT_W * 0.42), sharey=True)
    _build_panel(axes[0], df, "CPD")
    _build_panel(axes[1], df, "SD")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.14, wspace=0.06)
    savefig(fig, "fig_scalability")


if __name__ == "__main__":
    make_figure()
