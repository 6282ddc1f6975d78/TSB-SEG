"""FIGURE_METRIC_CORRELATIONS: pairwise correlation of evaluation metrics.

Aggregates **all grid (non-guided) runs** across every algorithm/dataset and
computes pairwise Pearson correlations between the CPD and SD metrics. The
output is a single combined figure with two masked-triangular heatmaps
(CPD on the left, SD on the right):

* ``fig_metric_correlations.pdf``

This supersedes the legacy notebook-based ``heatmap_scores_{cpd,sms}.pdf``
pair and is intended for the appendix of the paper.
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis.data import load_data
from analysis.helpers import COL_W, savefig

CPD_METRICS = {
    "f1_score": "$F_1$",
    "covering_score": "Covering",
    "gaussian_f1_score": "Gaussian $F_1$",
    "bidirectional_covering_score": "Bi-Covering",
}
SD_METRICS = {
    "adjusted_rand_index_score": "ARI",
    "adjusted_mutual_info_score": "AMI",
    "weighted_adjusted_rand_index_score": "WARI",
    "state_matching_score": "SMS",
}

FS_TICK = 12
FS_ANNOT = 12
FS_TITLE = 13


def _heatmap(ax, df, metrics: dict, title: str):
    cols = list(metrics.keys())
    sub = df[cols].dropna(how="any")
    corr = sub.corr().rename(index=metrics, columns=metrics)
    # Drop the first row and the last column: with the diagonal masked, those
    # are entirely empty (only contained the masked diagonal cell).
    corr_disp = corr.iloc[1:, :-1]
    # Mask the upper triangle including the diagonal of the displayed matrix.
    n_rows, n_cols = corr_disp.shape
    mask = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        for j in range(n_cols):
            # Cell (i, j) of corr_disp corresponds to (i+1, j) of corr.
            if (i + 1) <= j:  # strictly upper triangle or diagonal of full corr
                mask[i, j] = True
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    sns.heatmap(
        corr_disp, mask=mask, cmap=cmap, vmin=0, vmax=1,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.7, "pad": 0.02, "ticks": [0, 0.25, 0.5, 0.75, 1.0]},
        annot=True, fmt=".2f", annot_kws={"size": FS_ANNOT},
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right",
                       fontsize=FS_TICK)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=FS_TICK)
    ax.set_title(title, fontsize=FS_TITLE)
    off = corr.values[np.tril_indices_from(corr.values, k=-1)]
    return float(np.nanmin(off)), float(np.nanmax(off)), len(sub)


def make_figure(d=None):
    if d is None:
        d = load_data()

    df = pd.concat([d.df_grid_ng, d.df_grid_g], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(COL_W * 2.6, COL_W * 1.4))
    cpd_min, cpd_max, n_cpd = _heatmap(axes[0], df, CPD_METRICS, "CPD metrics")
    sd_min,  sd_max,  n_sd  = _heatmap(axes[1], df, SD_METRICS,  "SD metrics")
    fig.tight_layout()
    savefig(fig, "fig_metric_correlations")

    print(f"CPD: {n_cpd} rows, off-diag corr in [{cpd_min:.2f}, {cpd_max:.2f}]")
    print(f"SD : {n_sd} rows, off-diag corr in [{sd_min:.2f}, {sd_max:.2f}]")


if __name__ == "__main__":
    make_figure()
