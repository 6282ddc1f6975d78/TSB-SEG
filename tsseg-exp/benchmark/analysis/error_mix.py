"""FIGURE_ERROR_MIX: SMS error-type composition per residual zone.

Joins the pooled SD-capable runs with sms_error_breakdown.csv on
run_id, computes per-run penalty proportions (Delay / Transition /
Isolation / Missing), and shows the average proportion per zone as a
stacked horizontal bar.  Runs with zero total penalty (perfect SMS=1)
are dropped so each zone's bar sums to exactly 100%.
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
from analysis.helpers import COL_W, SMS_ERRORS_CSV, savefig
from analysis.residual_dist import (
    ZONE_COLORS,
    ZONE_ORDER,
    assign_zone,
    envelope,
    pool_runs,
)

PENALTY_COLS = ["penalty_delay", "penalty_transition",
                "penalty_isolation", "penalty_missing"]
PENALTY_LABELS = ["Delay", "Transition", "Isolation", "Missing"]
PENALTY_COLORS = {
    "penalty_delay":      "#F8961E",
    "penalty_transition": "#FFD166",
    "penalty_isolation":  "#4CC9F0",
    "penalty_missing":    "#325586",
}


def make_figure(d=None):
    if d is None:
        d = load_data()
    df = pool_runs(d)
    delta_lo, delta_up = envelope(df)
    df = assign_zone(df, delta_lo, delta_up)

    sms = pd.read_csv(SMS_ERRORS_CSV).rename(columns={"run_uuid": "run_id"})
    sms = sms.drop(columns=["algorithm", "dataset", "trial_index", "supervision"],
                   errors="ignore")
    sms = sms[["run_id"] + PENALTY_COLS].copy()

    merged = df.merge(sms, on="run_id", how="inner")
    totals = merged[PENALTY_COLS].sum(axis=1)
    # Drop perfect-SMS runs (no error to decompose).
    merged = merged[totals > 0].copy()
    totals = merged[PENALTY_COLS].sum(axis=1)
    merged[PENALTY_COLS] = merged[PENALTY_COLS].div(totals, axis=0)

    zone_means = merged.groupby("zone")[PENALTY_COLS].mean().reindex(ZONE_ORDER)
    counts = merged["zone"].value_counts().reindex(ZONE_ORDER).fillna(0).astype(int)

    print("[error_mix] N with errors =", len(merged))
    print(zone_means.round(3))
    print("row sums (should all be 1.0):", zone_means.sum(axis=1).round(3).to_dict())

    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.80))

    y_pos = np.arange(len(ZONE_ORDER))
    left = np.zeros(len(ZONE_ORDER))
    for col in PENALTY_COLS:
        vals = zone_means[col].to_numpy()
        ax.barh(y_pos, vals, left=left, height=0.75,
                color=PENALTY_COLORS[col],
                edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v >= 0.05:
                ax.text(left[i] + v / 2, y_pos[i],
                        f"{int(round(v * 100))}\%",
                        ha="center", va="center", fontsize=9.0,
                        color="white" if col in ("penalty_delay",
                                                 "penalty_missing")
                        else "black")
        left += vals

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(y_pos)
    ylabels = ax.set_yticklabels(ZONE_ORDER, fontsize=15)
    for label, z in zip(ylabels, ZONE_ORDER):
        label.set_bbox(dict(facecolor=ZONE_COLORS[z], alpha=0.3, edgecolor='none', boxstyle='round,pad=0.1'))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Average penalty share", fontsize=11)
    ax.tick_params(axis='x', labelsize=10)

    handles = [Patch(facecolor=PENALTY_COLORS[c], label=l)
               for c, l in zip(PENALTY_COLS, PENALTY_LABELS)]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.3, -0.3),
              fontsize=10, frameon=False, ncols=4,
              handlelength=1.2, handletextpad=0.4,
              columnspacing=1.0, borderaxespad=0.0)

    fig.tight_layout()
    savefig(fig, "fig_error_mix")
    plt.show()


if __name__ == "__main__":
    make_figure()
