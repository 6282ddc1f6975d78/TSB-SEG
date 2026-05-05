"""
Analyze the influence of hyper-parameters on the target score
(bi-covering for CPD, SMS for SD). Creates figures for Non-Guided and Guided
CPD and SD tasks.

Design notes:
* Algorithms whose grids are intrinsically uninteresting (no tunable param,
  or trivial-only) are excluded: ``amoc``, ``autoplait``, ``prophet`` (and
  ``clap`` is dropped from the CPD plots because it is a SD-first method).
  Additionally ``kcpd``, ``tglad``, ``dynp`` and ``hidalgo`` are dropped at
  the paper level (results not reported in the main pool).
* The most influential hyper-parameter (largest std of marginal score
  means) is used as the primary x-axis. Up to ``MAX_HPS`` parameters are
  shown per panel.
* Param names and categorical values are abbreviated through ``SHORT_NAME``
  / ``SHORT_VALUE`` so legends fit in-axes.
* Colors come from ``seaborn``'s colour-blind palette.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from analysis.data import load_data
from analysis.helpers import METRIC_CPD, METRIC_SD, analyzer, display_name

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

EXCLUDED_ALGOS = {
    "amoc", "autoplait", "prophet",
    # Removed at user request (paper-level decision):
    "kcpd", "tglad", "dynp", "hidalgo",
}
EXCLUDED_PER_TASK = {"CPD": {"clap"}, "SD": set()}

COMPLETION_THRESHOLD_DEFAULT = 0.65
COMPLETION_THRESHOLD_BY_ALGO: dict[str, float] = {"hdp-hsmm": 0.60}

MAX_HPS = 3

OUT_DIR = Path("figures/paper/hyperparams")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHORT_NAME = {
    "window_size": "win",
    "k_neighbours": "k_n",
    "excl_radius": "excl",
    "exclusion_factor": "excl_f",
    "n_estimators": "n_est",
    "max_epochs": "epochs",
    "multivariate_strategy": "mv",
    "smooth_window": "sm_win",
    "discount": "disc",
    "penalty": "pen",
    "model": "mdl",
    "classifier": "clf",
    "distance": "dist",
    "domain": "dom",
    "lambda_parameter": "lambda",
    "n_max_states": "n_states",
    "dur_alpha": "dur_a",
    "chain_len": "chain",
    "min_size": "min",
    "kernel": "kern",
    "threshold": "thr",
}

SHORT_VALUE = {
    "znormed_euclidean_distance": "znorm_eucl",
    "cinvariant_euclidean_distance": "cinv_eucl",
    "euclidean_distance": "eucl",
    "ensembling": "ens",
    "mrhydra": "mrhy",
    "rocket": "rkt",
}


def _short_name(key: str) -> str:
    return SHORT_NAME.get(key, key)


def _short_val(val) -> str:
    s = str(val)
    return SHORT_VALUE.get(s, s)


# ─────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────


def get_grid_params(run_ids):
    df_p = analyzer.manager.fetch_run_params(list(run_ids))
    if df_p.empty:
        return pd.DataFrame()
    return df_p.pivot(index="run_id", columns="key", values="value").reset_index()


def _per_algo_max_runs(df_grid: pd.DataFrame) -> dict[str, int]:
    """Per-algorithm theoretical run budget.

    For each (algorithm, dataset) pair we take the maximum number of distinct
    series ever completed by a single hyper-parameter combination, then sum
    across the datasets the algorithm actually exercises. This rewards
    algorithms that only run on a subset of datasets (e.g.\ multivariate-only
    methods such as ``ticc``) by not penalising them for the datasets they
    cannot process.
    """
    df_valid = df_grid[df_grid["dataset"] != "pamap2"]
    counts = (
        df_valid.groupby(["algorithm", "parent_run_id", "dataset"])["run_id"]
        .nunique()
        .reset_index()
    )
    per_algo_dataset = (
        counts.groupby(["algorithm", "dataset"])["run_id"].max().reset_index()
    )
    return per_algo_dataset.groupby("algorithm")["run_id"].sum().to_dict()


# ─────────────────────────────────────────────────────────────────────
# Main per-task processing
# ─────────────────────────────────────────────────────────────────────


def process_task(task_name, target_metric, algos, df_grid, filename):
    print(f"\nProcessing {task_name} for {filename}...")
    excluded = EXCLUDED_ALGOS | EXCLUDED_PER_TASK.get(task_name, set())
    fig_data: list[dict] = []
    algo_max_runs = _per_algo_max_runs(df_grid)

    for algo in tqdm(algos):
        if algo in excluded:
            continue
        max_runs = algo_max_runs.get(algo, 0)
        if max_runs <= 0:
            continue

        df_algo = df_grid[
            (df_grid["algorithm"] == algo) & (df_grid["dataset"] != "pamap2")
        ].copy()
        if df_algo.empty:
            continue

        parent_ids = set(df_algo["parent_run_id"].dropna().unique())
        if not parent_ids:
            continue

        df_params = get_grid_params(parent_ids)
        if df_params.empty:
            continue

        # Detect varying grid_* params, drop the bookkeeping ones.
        hyperparams: list[str] = []
        for c in list(df_params.columns):
            if not c.startswith("grid_") or c == "grid_execution_role":
                continue
            clean = c[5:]
            df_params[clean] = df_params[c].astype(str)
            if df_params[clean].nunique() > 1:
                hyperparams.append(clean)

        if not hyperparams:
            continue

        merged_all = df_algo.merge(
            df_params, left_on="parent_run_id", right_on="run_id"
        )

        # Aggregate per (hyperparam combination):
        #   sum of scores over the series that ran;
        #   count of distinct (parent, dataset, series) runs.
        # Padded score = sum / global_max_runs (treats missing as 0).
        grouped = merged_all.groupby(hyperparams)
        sums = grouped[target_metric].sum().rename("sum_score")
        counts = grouped["run_id_x"].nunique().rename("n_runs")
        scores = pd.concat([sums, counts], axis=1).reset_index()
        scores["completion"] = scores["n_runs"] / max_runs
        scores[target_metric] = scores["sum_score"] / max_runs

        threshold = COMPLETION_THRESHOLD_BY_ALGO.get(
            algo, COMPLETION_THRESHOLD_DEFAULT
        )
        scores = scores[scores["completion"] >= threshold].copy()
        if scores.empty:
            continue

        # Drop params that collapsed to a single value after the
        # completion filter — they would clutter the legend with a
        # constant label that carries no information.
        hyperparams = [hp for hp in hyperparams if scores[hp].nunique() > 1]
        if not hyperparams:
            continue

        # Influence ranking: std of marginal means per parameter.
        # The most influential param is used as primary x-axis.
        influence = []
        for hp in hyperparams:
            infl = scores.groupby(hp)[target_metric].mean().std()
            influence.append((0.0 if pd.isna(infl) else infl, hp))
        influence.sort(reverse=True)
        sorted_hps = [hp for _, hp in influence]

        # ChangeFinder: split by multivariate_strategy as separate panels.
        if algo == "changefinder" and "multivariate_strategy" in sorted_hps:
            other_hps = [h for h in sorted_hps if h != "multivariate_strategy"][
                :MAX_HPS
            ]
            for strat in ["l2", "ensembling"]:
                sub = scores[scores["multivariate_strategy"] == strat].copy()
                if sub.empty:
                    continue
                fig_data.append(
                    {
                        "algo": f"changefinder ({strat})",
                        "data": sub,
                        "hps": other_hps,
                        "target": target_metric,
                    }
                )
        else:
            fig_data.append(
                {
                    "algo": algo,
                    "data": scores,
                    "hps": sorted_hps[:MAX_HPS],
                    "target": target_metric,
                }
            )

    if not fig_data:
        print("No valid algorithms found for plotting.")
        return

    _render_figure(fig_data, filename)


# ─────────────────────────────────────────────────────────────────────
# Plot rendering
# ─────────────────────────────────────────────────────────────────────


def _render_figure(fig_data: list[dict], filename: str) -> None:
    n_plots = len(fig_data)
    cols = 2 if n_plots <= 4 else 3
    rows = (n_plots + cols - 1) // cols

    # Slightly wider, much shorter cells: legend goes BELOW each axis,
    # so we reserve a thin band of space rather than stretching height.
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.6))
    axes = np.atleast_1d(axes).flatten()

    palette = sns.color_palette("colorblind", 10)
    shapes = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]

    for i, item in enumerate(fig_data):
        ax = axes[i]
        merged = item["data"]
        hps = item["hps"]
        target = item["target"]
        algo_name = item["algo"]

        if not hps:
            continue

        main_hp = hps[0]
        try:
            x_vals = sorted(merged[main_hp].unique(), key=lambda v: float(v))
        except (TypeError, ValueError):
            x_vals = sorted(merged[main_hp].unique())
        x_map = {v: xi for xi, v in enumerate(x_vals)}

        second_hp = hps[1] if len(hps) >= 2 else None
        third_hp = hps[2] if len(hps) >= 3 else None

        s_vals = []
        if second_hp:
            try:
                s_vals = sorted(merged[second_hp].unique(), key=lambda v: float(v))
            except (TypeError, ValueError):
                s_vals = sorted(merged[second_hp].unique())

        t_vals = []
        if third_hp:
            try:
                t_vals = sorted(merged[third_hp].unique(), key=lambda v: float(v))
            except (TypeError, ValueError):
                t_vals = sorted(merged[third_hp].unique())

        for _, row in merged.iterrows():
            base_x = x_map[row[main_hp]]
            offset = 0.0
            color = palette[0]
            marker = "o"

            if second_hp:
                s_idx = s_vals.index(row[second_hp])
                color = palette[s_idx % len(palette)]
                if third_hp and len(s_vals) > 1:
                    offset = np.linspace(-0.22, 0.22, len(s_vals))[s_idx]
                if not third_hp:
                    marker = shapes[s_idx % len(shapes)]

            if third_hp:
                t_idx = t_vals.index(row[third_hp])
                marker = shapes[t_idx % len(shapes)]

            ax.scatter(
                base_x + offset,
                row[target],
                marker=marker,
                color=color,
                alpha=0.9,
                s=130,
                edgecolors="black",
                linewidths=0.5,
            )

        for xi in range(len(x_vals) - 1):
            ax.axvline(xi + 0.5, color="gray", alpha=0.18, linestyle="--", lw=0.7)

        # Compact in-axes legend (color = second_hp; marker = third_hp).
        legend_handles: list = []
        if second_hp:
            for s_idx, s_val in enumerate(s_vals):
                color = palette[s_idx % len(palette)]
                marker = "o" if third_hp else shapes[s_idx % len(shapes)]
                legend_handles.append(
                    mlines.Line2D(
                        [], [],
                        color=color, marker=marker, linestyle="None",
                        markersize=10, markeredgecolor="black", markeredgewidth=0.5,
                        label=f"{_short_name(second_hp)}={_short_val(s_val)}",
                    )
                )
        if third_hp:
            for t_idx, t_val in enumerate(t_vals):
                legend_handles.append(
                    mlines.Line2D(
                        [], [],
                        color="gray", marker=shapes[t_idx % len(shapes)],
                        linestyle="None", markersize=10,
                        markeredgecolor="black", markeredgewidth=0.5,
                        label=f"{_short_name(third_hp)}={_short_val(t_val)}",
                    )
                )

        if legend_handles:
            ncol = min(len(legend_handles), 4)
            ax.legend(
                handles=legend_handles,
                loc="upper center", bbox_to_anchor=(0.5, -0.32),
                fontsize=10, frameon=False, ncol=ncol,
                handletextpad=0.4, columnspacing=1.0,
                borderpad=0.2, labelspacing=0.3,
            )

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([_short_val(v) for v in x_vals], rotation=0, fontsize=12)
        ax.tick_params(axis="y", labelsize=12)

        title_extra = "" if "(" not in algo_name else " " + algo_name[algo_name.find("("):]
        ax.set_title(
            f"{display_name(algo_name.split(' ')[0])}{title_extra}", fontsize=15
        )
        ax.set_xlabel(_short_name(main_hp), fontweight="bold", fontsize=13)
        ax.set_ylabel("")
        ax.grid(alpha=0.3, axis="y")

    for j in range(len(fig_data), len(axes)):
        fig.delaxes(axes[j])

    # Common metric label, top-left of the figure.
    metric_label = "BiC" if any(
        item["target"] == METRIC_CPD for item in fig_data
    ) else "SMS"
    fig.suptitle(f"score: {metric_label}", x=0.01, y=0.995,
                 ha="left", fontsize=12, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=4.0, w_pad=1.5)
    out_pdf = OUT_DIR / f"{filename}.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_pdf}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("Loading data...")
    d = load_data()

    cpd_algos = [a for a in d.cpd_algos if a != "random"]
    sd_algos = [a for a in d.sd_algos if a != "random"]

    process_task("CPD", METRIC_CPD, cpd_algos, d.df_grid_ng, "fig_hyperparams_cpd_ng")
    process_task("SD",  METRIC_SD,  sd_algos,  d.df_grid_ng, "fig_hyperparams_sd_ng")
    process_task("CPD", METRIC_CPD, cpd_algos, d.df_grid_g,  "fig_hyperparams_cpd_g")
    process_task("SD",  METRIC_SD,  sd_algos,  d.df_grid_g,  "fig_hyperparams_sd_g")
