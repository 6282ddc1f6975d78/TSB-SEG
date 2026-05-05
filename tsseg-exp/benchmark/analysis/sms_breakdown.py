"""FIGURE_SMS_ERRORS: stacked vertical bars per algo, both supervisions."""
from __future__ import annotations

# Allow running this file directly: ``python analysis/sms_breakdown.py``
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis.helpers import ALGO_DISPLAY, COL_W, SMS_ERRORS_CSV, savefig


def make_figure():
    if not SMS_ERRORS_CSV.exists():
        raise FileNotFoundError(f"SMS errors CSV not found: {SMS_ERRORS_CSV}")

    df_sms = pd.read_csv(SMS_ERRORS_CSV)
    df_sms = df_sms[df_sms["dataset"] != "pamap2"].copy()
    df_sms = df_sms[df_sms["algorithm"] != "random"].copy()
    # GGS is a CPD-only method (no state recurrence); exclude from SD/SMS plot.
    df_sms = df_sms[df_sms["algorithm"] != "ggs"].copy()

    PENALTY_COLS = ["penalty_delay", "penalty_transition", "penalty_isolation", "penalty_missing"]
    PENALTY_LABELS = ["Delay", "Transition", "Isolation", "Missing"]

    row_totals = df_sms[PENALTY_COLS].sum(axis=1)
    row_props = df_sms[PENALTY_COLS].div(row_totals.replace(0, np.nan), axis=0)
    df_sms_props = pd.concat([df_sms[["algorithm", "supervision"]], row_props], axis=1).dropna()

    props = df_sms_props.groupby(["algorithm", "supervision"])[PENALTY_COLS].mean()

    _GUIDED_ONLY = {"ticc", "autoplait", "hidalgo", "dynp", "espresso", "prophet"}
    _all_algos = sorted(props.index.get_level_values(0).unique())
    ordered_algos = [a for a in _all_algos if a not in _GUIDED_ONLY] + \
                    [a for a in _all_algos if a in _GUIDED_ONLY]

    _ERROR_COLORS = {
        "penalty_transition": "#FFD166",
        "penalty_delay":      "#F8961E",
        "penalty_isolation":  "#4CC9F0",
        "penalty_missing":    "#325586",
    }

    SUPS = [("unsupervised", "NG", -0.20), ("semi_supervised", "G", +0.20)]
    BAR_W = 0.36
    TAG_FS = 9
    LABEL_FS = 12
    TICK_FS = 11
    LEGEND_FS = 10
    TITLE_FS = 13

    fig, ax = plt.subplots(figsize=(COL_W * 2, COL_W * 1.5))
    x_centres = np.arange(len(ordered_algos))

    for i, algo in enumerate(ordered_algos):
        present = [sup for sup, _, _ in SUPS if (algo, sup) in props.index]
        guided_only = present == ["semi_supervised"]
        for sup, tag, offset in SUPS:
            if (algo, sup) not in props.index:
                continue
            # Centre the bar if the algorithm only has the guided regime.
            x = i if guided_only else i + offset
            bottom = 0.0
            for col in PENALTY_COLS:
                h = props.loc[(algo, sup), col]
                ax.bar(x, h, bottom=bottom, width=BAR_W,
                       color=_ERROR_COLORS[col], edgecolor="white", linewidth=0.4)
                bottom += h
            ax.text(x, -0.04, tag, ha="center", va="top",
                    fontsize=TAG_FS, color="#555", family="monospace")

    ax.set_xticks(x_centres)
    ax.set_xticklabels([ALGO_DISPLAY.get(a, a) for a in ordered_algos],
                       rotation=30, ha="right", fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)

    legend_handles = [Patch(facecolor=_ERROR_COLORS[c], label=l)
                      for c, l in zip(PENALTY_COLS, PENALTY_LABELS)]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=LEGEND_FS, framealpha=0.85, ncol=4,
              bbox_to_anchor=(1.0, -0.22))

    ax.set_ylabel("Proportion of Errors", fontsize=LABEL_FS)
    ax.set_ylim(-0.07, 1.02)
    ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlim(-0.6, len(ordered_algos) - 0.4)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)

    savefig(fig, "fig_sms_errors")


if __name__ == "__main__":
    make_figure()
