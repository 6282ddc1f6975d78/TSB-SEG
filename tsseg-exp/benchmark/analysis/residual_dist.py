"""FIGURE_RESIDUALS: histogram of SMS - BiC residuals coloured by zone.

Pools every (algorithm, dataset, trial) outcome from the four data
sources, keeps SD-capable algorithms, and plots the empirical
distribution of the residual r = SMS - BiCovering, split into three
zones around y = x using the same envelope as scatter.py.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from analysis.data import load_data
from analysis.helpers import ALGO_TASK, COL_W, METRIC_CPD, METRIC_SD, savefig

plt.rcParams["text.usetex"] = True

L_UNDER = r"$\scriptstyle \mathrm{SMS} < \underline{\mathrm{BiC}}$"
L_NEAR  = r"$\scriptstyle \mathrm{{SMS}} \in [ \underline{{\mathrm{{BiC}}}}, \overline{{\mathrm{{BiC}}}} ]$"
L_OVER  = r"$\scriptstyle \mathrm{SMS} > \overline{\mathrm{BiC}}$"

ZONE_COLORS = {
    L_UNDER: "#dda0dd",
    L_NEAR:  "#1f4e8c",
    L_OVER:  "#ffb6c1",
}
ZONE_ORDER = [L_UNDER, L_NEAR, L_OVER]


def pool_runs(d) -> pd.DataFrame:
    frames = []
    for src in (d.df_default_ng, d.df_grid_ng, d.df_default_g, d.df_grid_g):
        if src is None or len(src) == 0:
            continue
        cols = {"run_id", "algorithm", "dataset", METRIC_CPD, METRIC_SD}
        if not cols.issubset(src.columns):
            continue
        frames.append(src[list(cols)].copy())
    df = pd.concat(frames, ignore_index=True)
    sd_algos = {a for a, t in ALGO_TASK.items() if t == "SD"}
    df = df[df["algorithm"].isin(sd_algos)]
    df = df[df["dataset"].str.lower() != "pamap2"]
    df = df.dropna(subset=[METRIC_CPD, METRIC_SD, "run_id"])
    df = df[df[METRIC_CPD].between(0, 1) & df[METRIC_SD].between(0, 1)]
    return df.drop_duplicates(subset="run_id")


def envelope(df: pd.DataFrame) -> tuple[float, float]:
    delta_up = float(np.quantile(df[METRIC_SD] - df[METRIC_CPD], 0.975))
    delta_lo = float(np.quantile(df[METRIC_CPD] - df[METRIC_SD], 0.975))
    return delta_lo, delta_up


def assign_zone(df: pd.DataFrame, delta_lo: float, delta_up: float) -> pd.DataFrame:
    def _z(r: float) -> str:
        if r < -delta_lo:
            return L_UNDER
        if r > delta_up:
            return L_OVER
        return L_NEAR
    df = df.copy()
    df["residual"] = df[METRIC_SD] - df[METRIC_CPD]
    df["zone"] = df["residual"].map(_z)
    return df


def make_figure(d=None):
    if d is None:
        d = load_data()
    df = pool_runs(d)
    delta_lo, delta_up = envelope(df)
    df = assign_zone(df, delta_lo, delta_up)

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.75))

    bins = np.linspace(-0.6, 0.6, 61)
    r = df["residual"].clip(-0.6, 0.6).to_numpy(dtype=float)
    total_runs = len(df)
    # Normalize by the total number of runs so the y-axis shows proportions.
    weights = np.full(r.shape, 1.0 / total_runs, dtype=float)
    for zone in ZONE_ORDER:
        mask = df["zone"].values == zone
        ax.hist(r[mask], bins=bins,
                weights=weights[mask],
                color=ZONE_COLORS[zone], alpha=0.85,
                edgecolor="white", linewidth=0.2, label=zone)

    ax.axvline(-delta_lo, color="black", lw=0.8, ls=(0, (3, 2)))
    ax.axvline(+delta_up, color="black", lw=0.8, ls=(0, (3, 2)), ymin=0, ymax=0.55)
    ax.axvline(0.0, color="black", lw=0.8)

    y_top = ax.get_ylim()[1]
    ax.text(-delta_lo - 0.01, y_top * 0.35, rf"$-{delta_lo:.2f}$",
            fontsize=9, ha="right", va="center", rotation=90)
    ax.text(+delta_up + 0.01, y_top * 0.35, rf"$+{delta_up:.2f}$",
            fontsize=9, ha="left", va="center", rotation=90)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(-0.55, 0.72)
    ax.set_xlabel(r"Residual $r = $ SMS $-$ Bi-Covering", fontsize=11)
    ax.set_ylabel("Proportion of runs", fontsize=11)
    ax.set_yticks([0.05, 0.10])
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.tick_params(labelsize=10)
    ax.legend(loc="upper right", fontsize=11, frameon=False,
              handlelength=1.2, handletextpad=0.4, borderaxespad=0.3)

    fig.tight_layout()
    savefig(fig, "fig_residual_dist")
    plt.show()


if __name__ == "__main__":
    make_figure()
