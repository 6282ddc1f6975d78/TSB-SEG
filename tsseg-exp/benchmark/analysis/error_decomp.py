"""FIGURE_ERROR_DECOMP: residual distribution + SMS error mix per zone.

For every SD-capable run pooled across the four data sources
(default/grid x non-guided/guided), we compute the residual
``r = SMS - BiCovering``.  The same empirical envelope used in
``scatter.py`` defines three residual zones around y = x:

    Under-predicted : r < -delta_lo   (SMS << BiC, segmentation good
                                       but state assignment fails)
    Near fit        : -delta_lo <= r <= +delta_up
    Over-predicted  : r > +delta_up   (SMS >> BiC, lucky state
                                       grouping despite poor CPs)

The figure has two panels:

  (a) histogram of residuals coloured by zone, with vertical envelope
      thresholds.
  (b) stacked horizontal bar of the average proportion of each SMS
      penalty type (Delay / Transition / Isolation / Missing) per zone,
      computed by joining with ``sms_error_breakdown.csv`` on run_id.
"""
from __future__ import annotations

# Allow running this file directly: ``python analysis/error_decomp.py``
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from analysis.data import load_data
from analysis.helpers import (
    ALGO_TASK,
    COL_W,
    METRIC_CPD,
    METRIC_SD,
    SMS_ERRORS_CSV,
    savefig,
)


PENALTY_COLS = ["penalty_delay", "penalty_transition", "penalty_isolation", "penalty_missing"]
PENALTY_LABELS = ["Delay", "Transition", "Isolation", "Missing"]
PENALTY_COLORS = {
    "penalty_delay":      "#F8961E",
    "penalty_transition": "#FFD166",
    "penalty_isolation":  "#4CC9F0",
    "penalty_missing":    "#325586",
}

# Zone colours (match the scatter palette: in-band = saturated blue).
ZONE_COLORS = {
    r"$\mathrm{SMS} < \underline{\mathrm{BiC}}$": "#dda0dd",   # pale purple
    r"$\underline{\mathrm{BiC}} \leq \mathrm{SMS} \leq \overline{\mathrm{BiC}}$": "#1f4e8c",   # saturated blue (in-band)
    r"$\mathrm{SMS} > \overline{\mathrm{BiC}}$": "#ffb6c1",   # pale pink
}
ZONE_ORDER = [
    r"$\mathrm{SMS} < \underline{\mathrm{BiC}}$", 
    r"$\underline{\mathrm{BiC}} \leq \mathrm{SMS} \leq \overline{\mathrm{BiC}}$", 
    r"$\mathrm{SMS} > \overline{\mathrm{BiC}}$"
]


def _pool_runs(d) -> pd.DataFrame:
    """Pool per-run (run_id, algo, dataset, BiC, SMS) across the 4 sources."""
    frames = []
    for src in (d.df_default_ng, d.df_grid_ng, d.df_default_g, d.df_grid_g):
        if src is None or len(src) == 0:
            continue
        cols = {"run_id", "algorithm", "dataset", METRIC_CPD, METRIC_SD}
        if not cols.issubset(src.columns):
            continue
        f = src[list(cols)].copy()
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)

    sd_algo_set = {a for a, t in ALGO_TASK.items() if t == "SD"}
    df = df[df["algorithm"].isin(sd_algo_set)]
    df = df[df["dataset"].str.lower() != "pamap2"]
    df = df.dropna(subset=[METRIC_CPD, METRIC_SD, "run_id"])
    df = df[df[METRIC_CPD].between(0, 1) & df[METRIC_SD].between(0, 1)]
    return df.drop_duplicates(subset="run_id")


def make_figure(d=None):
    if d is None:
        d = load_data()

    df = _pool_runs(d)
    df["residual"] = df[METRIC_SD] - df[METRIC_CPD]

    # Same empirical envelope as scatter.py (two-sided ~95% containment).
    delta_up = float(np.quantile(df[METRIC_SD] - df[METRIC_CPD], 0.975))
    delta_lo = float(np.quantile(df[METRIC_CPD] - df[METRIC_SD], 0.975))

    def _zone(r: float) -> str:
        if r < -delta_lo:
            return r"$\mathrm{SMS} < \underline{\mathrm{BiC}}$"
        if r > delta_up:
            return r"$\mathrm{SMS} > \overline{\mathrm{BiC}}$"
        return r"$\underline{\mathrm{BiC}} \leq \mathrm{SMS} \leq \overline{\mathrm{BiC}}$"

    df["zone"] = df["residual"].apply(_zone)

    # ── Join SMS penalty breakdown on run_id ───────────────────────
    if not SMS_ERRORS_CSV.exists():
        raise FileNotFoundError(f"SMS errors CSV not found: {SMS_ERRORS_CSV}")
    sms = pd.read_csv(SMS_ERRORS_CSV)
    sms = sms.rename(columns={"run_uuid": "run_id"})
    sms = sms[["run_id"] + PENALTY_COLS].copy()

    merged = df.merge(sms, on="run_id", how="inner")
    # Normalise penalties to proportions per run (rows summing to 1).
    totals = merged[PENALTY_COLS].sum(axis=1)
    props = merged[PENALTY_COLS].div(totals.replace(0, np.nan), axis=0)
    props = props.fillna(0.0)
    merged[PENALTY_COLS] = props

    print(f"[error_decomp] N pooled = {len(df):,}   "
          f"N with SMS breakdown = {len(merged):,}")
    counts = df["zone"].value_counts().reindex(ZONE_ORDER)
    matched = merged["zone"].value_counts().reindex(ZONE_ORDER)
    print(f"[error_decomp] delta_lo = {delta_lo:.3f}   delta_up = {delta_up:.3f}")
    for z in ZONE_ORDER:
        print(f"  {z:<16s}  all={int(counts.get(z, 0)):6d}  "
              f"matched={int(matched.get(z, 0)):6d}")

    zone_means = merged.groupby("zone")[PENALTY_COLS].mean().reindex(ZONE_ORDER)

    # ── Figure layout ──────────────────────────────────────────────
    fig, (ax_hist, ax_bar) = plt.subplots(
        1, 2,
        figsize=(COL_W * 2, COL_W * 0.9),
        gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.55},
    )

    # ── Panel (a): residual histogram, coloured by zone ────────────
    r = df["residual"].to_numpy(dtype=float)
    bins = np.linspace(-1.0, 1.0, 81)

    for zone in ZONE_ORDER:
        mask = df["zone"].values == zone
        ax_hist.hist(
            r[mask], bins=bins,
            color=ZONE_COLORS[zone], alpha=0.85,
            edgecolor="white", linewidth=0.2,
            label=f"{zone} (n={int(mask.sum()):,})",
        )

    ax_hist.axvline(-delta_lo, color="black", lw=0.8, ls=(0, (3, 2)))
    ax_hist.axvline(+delta_up, color="black", lw=0.8, ls=(0, (3, 2)))
    ax_hist.axvline(0.0, color="black", lw=0.8)

    y_top = ax_hist.get_ylim()[1]
    ax_hist.text(-delta_lo - 0.01, y_top * 0.55, rf"$-{delta_lo:.2f}$",
                 fontsize=6, ha="right", va="center", rotation=90,
                 bbox=dict(boxstyle="square,pad=0.15", fc="white",
                           ec="none", alpha=0.65))
    ax_hist.text(+delta_up + 0.01, y_top * 0.55, rf"$+{delta_up:.2f}$",
                 fontsize=6, ha="left", va="center", rotation=90,
                 bbox=dict(boxstyle="square,pad=0.15", fc="white",
                           ec="none", alpha=0.65))

    ax_hist.set_xlim(-1.0, 1.0)
    ax_hist.set_xlabel(r"Residual $r = $ SMS $-$ Bi-Covering", fontsize=10)
    ax_hist.set_ylabel("Number of runs", fontsize=10)
    ax_hist.tick_params(labelsize=9)
    ax_hist.legend(loc="upper right", fontsize=5.5, frameon=False,
                   handlelength=1.2, handletextpad=0.4, borderaxespad=0.3)
    ax_hist.set_title("(a) Residual distribution",
                      fontsize=12, loc="left", pad=4)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)

    # ── Panel (b): stacked horizontal bar of penalty mix per zone ──
    y_pos = np.arange(len(ZONE_ORDER))
    left = np.zeros(len(ZONE_ORDER))
    for col in PENALTY_COLS:
        vals = zone_means[col].to_numpy()
        ax_bar.barh(y_pos, vals, left=left, height=0.62,
                    color=PENALTY_COLORS[col],
                    edgecolor="white", linewidth=0.5)
        # Annotate slices wider than 6%.
        for i, v in enumerate(vals):
            if v >= 0.06:
                pct_val = int(round(v * 100))
                ax_bar.text(left[i] + v / 2, y_pos[i], f"{pct_val}\%",
                            ha="center", va="center", fontsize=8,
                            color="white" if col in ("penalty_delay",
                                                     "penalty_missing")
                            else "black")
        left += vals

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(
        [f"{z}\n(n={int(matched.get(z, 0)):,})" for z in ZONE_ORDER],
        fontsize=10,
    )
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Average penalty share", fontsize=10)
    ax_bar.tick_params(labelsize=9)
    ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax_bar.set_title("(b) SMS error mix per residual zone",
                     fontsize=7, loc="left", pad=4)

    legend_handles = [Patch(facecolor=PENALTY_COLORS[c], label=l)
                      for c, l in zip(PENALTY_COLS, PENALTY_LABELS)]
    ax_bar.legend(handles=legend_handles,
                  loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  fontsize=6, frameon=False, ncols=4,
                  handlelength=1.2, handletextpad=0.4,
                  columnspacing=1.0, borderaxespad=0.0)

    fig.tight_layout()
    savefig(fig, "fig_error_decomp")
    plt.show()


if __name__ == "__main__":
    make_figure()
