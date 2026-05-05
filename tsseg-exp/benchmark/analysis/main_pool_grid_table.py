"""Per-(algorithm, dataset) score table for the 20 main-pool algorithms.

Generates two paper-ready tables (CPD and SD), each one row per
(algorithm, dataset) and 8 score columns spanning the two supervision
modes (Non-Guided and Guided) and the four configurations of interest:
``Default``, ``Worst-grid``, ``Median-grid`` and ``Best-grid``.

The grid configurations are selected per (algorithm, dataset) on the
mean primary metric across series (BiC for CPD, SMS for SD), as in the
main pipeline (:func:`select_best_grid_per_dataset` /
:func:`aggregate_grid_summaries`). ``Default`` is the mean default-mode
score across the dataset's series. Cells with no eligible configuration
(e.g.\\ algorithms that are intrinsically unguided) display ``--``.

Outputs in ``figures/paper/``:

  - ``table_main_pool_cpd.tex`` / ``.pdf``
  - ``table_main_pool_sd.tex``  / ``.pdf``
  - ``table_main_pool.csv``      (long-form raw data)

Run::

    python benchmark/analysis/main_pool_grid_table.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from analysis.data import load_data
from analysis.helpers import (
    ALGO_DISPLAY,
    LIMITED_COVERAGE_ALGOS,
    METRIC_CPD,
    METRIC_SD,
    OUTPUT_DIR,
    TYPOLOGY,
    aggregate_grid_summaries,
    select_best_grid_per_dataset,
)

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

MAIN_DATASETS = ["actrectut", "has", "mocap", "pump", "skab", "tssb", "usc-had", "utsa"]
DATASET_DISPLAY = {
    "actrectut": "ActRecTut", "has": "HAS", "mocap": "MoCap", "pump": "PUMP",
    "skab": "SKAB", "tssb": "TSSB", "usc-had": "USC-HAD", "utsa": "UTSA",
}

METRIC_BY_TASK = {"CPD": METRIC_CPD, "SD": METRIC_SD}
METRIC_DISPLAY = {"CPD": "BiC", "SD": "SMS"}

# Secondary metrics emitted as separate (annex-of-annex) tables.
# (task, metric_key, file_suffix, display name, table label, caption-extra).
SECONDARY_METRICS: list[tuple[str, str, str, str]] = [
    ("CPD", "gaussian_f1_score",                 "gf1",  r"Gaussian $F_1$"),
    ("SD",  "adjusted_rand_index_score",         "ari",  r"ARI"),
    ("SD",  "weighted_adjusted_rand_index_score", "wari", r"WARI"),
]


def _algo_task(algo: str) -> str:
    return TYPOLOGY["algorithms"][algo]["task"]


# Main pool = every algo of the typology not flagged as
# limited-coverage. Typology order is preserved.
MAIN_POOL_ALGOS = [
    a for a in TYPOLOGY["algorithms"].keys() if a not in LIMITED_COVERAGE_ALGOS
]
CPD_MAIN = [a for a in MAIN_POOL_ALGOS if _algo_task(a) == "CPD"]
SD_MAIN = [a for a in MAIN_POOL_ALGOS if _algo_task(a) == "SD"]


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────

def _per_cell_default(df_default: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean default-mode score per (algorithm, dataset)."""
    if df_default is None or df_default.empty or metric not in df_default.columns:
        return pd.DataFrame(columns=["algorithm", "dataset", "score_default"])
    return (
        df_default.dropna(subset=[metric])
        .groupby(["algorithm", "dataset"], as_index=False)[metric]
        .mean()
        .rename(columns={metric: "score_default"})
    )


def _per_cell_grid(df_grid: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean best/worst/median grid scores + median best-grid runtime."""
    cols = ["algorithm", "dataset", "score_best", "score_worst", "score_median",
            "time_best_median"]
    if df_grid is None or df_grid.empty or metric not in df_grid.columns:
        return pd.DataFrame(columns=cols)

    ranked = select_best_grid_per_dataset(df_grid, metric, higher_is_better=True)
    if ranked.empty:
        return pd.DataFrame(columns=cols)

    summary = aggregate_grid_summaries(ranked, metric)
    if summary.empty:
        return pd.DataFrame(columns=cols)

    cell = (
        summary.groupby(["algorithm", "dataset"], as_index=False)[
            ["score_best", "score_worst", "score_median"]
        ]
        .mean()
    )

    if "execution_time_seconds" in df_grid.columns:
        best_rows = ranked[ranked["config_rank"] == 1]
        times = (
            best_rows.groupby(["algorithm", "dataset"], as_index=False)[
                "execution_time_seconds"
            ]
            .median()
            .rename(columns={"execution_time_seconds": "time_best_median"})
        )
        cell = cell.merge(times, on=["algorithm", "dataset"], how="left")
    else:
        cell["time_best_median"] = np.nan

    return cell


def _build_task_frame(d, task: str, metric: str | None = None) -> pd.DataFrame:
    """Long-form per-(algo, dataset, mode) frame for one task and metric."""
    if metric is None:
        metric = METRIC_BY_TASK[task]
    pool = CPD_MAIN if task == "CPD" else SD_MAIN

    rows = []
    for mode_key, df_default, df_grid in [
        ("NG", d.df_default_ng, d.df_grid_ng),
        ("G",  d.df_default_g,  d.df_grid_g),
    ]:
        defaults = _per_cell_default(df_default, metric)
        grids    = _per_cell_grid(df_grid, metric)
        cell = grids.merge(defaults, on=["algorithm", "dataset"], how="outer")
        cell = cell[cell["algorithm"].isin(pool)].copy()
        cell = cell[cell["dataset"].isin(MAIN_DATASETS)].copy()
        cell["mode"] = mode_key
        rows.append(cell)

    df = pd.concat(rows, ignore_index=True)
    if df.empty:
        return df
    # Pivot to wide: one row per (algo, dataset), columns per mode/config.
    long = df.melt(
        id_vars=["algorithm", "dataset", "mode"],
        value_vars=["score_default", "score_worst", "score_median",
                    "score_best", "time_best_median"],
        var_name="kind",
        value_name="value",
    )
    # Strip BOTH possible prefixes so the time column ends up named
    # "<MODE>_time" rather than "<MODE>_time_best_median" (and matches the
    # lookup keys NG_default/NG_worst/NG_median/NG_best/NG_time below).
    long["short_kind"] = (
        long["kind"]
        .str.replace("score_", "", regex=False)
        .str.replace("time_best_median", "time", regex=False)
    )
    long["col"] = long["mode"] + "_" + long["short_kind"]
    wide = long.pivot_table(
        index=["algorithm", "dataset"],
        columns="col",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # Order rows by algo display name then dataset canonical order.
    wide["_algo_display"] = wide["algorithm"].map(lambda a: ALGO_DISPLAY.get(a, a))
    wide["_ds_order"] = wide["dataset"].map(lambda d: MAIN_DATASETS.index(d))
    wide = wide.sort_values(["_algo_display", "_ds_order"]).reset_index(drop=True)
    wide = wide.drop(columns=["_algo_display", "_ds_order"])
    return wide


# ─────────────────────────────────────────────────────────────────────
# LaTeX rendering
# ─────────────────────────────────────────────────────────────────────

def _fmt_score(v) -> str:
    if v is None or pd.isna(v):
        return "--"
    return f"{v:.2f}"


def _fmt_time(v) -> str:
    if v is None or pd.isna(v):
        return "--"
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    if v >= 100:
        return f"{v:.0f}"
    return f"{v:.1f}"


def _render_table(
    df_task: pd.DataFrame, task: str, metric_display: str,
    label: str, caption: str,
) -> str:
    """longtable: per (algo, dataset) row, 4 grid configs * 2 modes per row.

    Emits a self-contained ``longtable`` so it can break across pages.
    The caption and ``\\label`` are baked into the table; the consumer
    just needs to ``\\input`` the file inside the document body.
    """
    if df_task.empty:
        return "% (empty)"

    header_groups = (
        r"\multicolumn{2}{c}{} & "
        r"\multicolumn{5}{c}{\textbf{Non-Guided}} & "
        r"\multicolumn{5}{c}{\textbf{Guided}} \\"
        r" \cmidrule(lr){3-7} \cmidrule(lr){8-12}"
    )
    header_cols = (
        r" \textbf{Algo} & \textbf{Dataset} "
        r" & Def. & Wrst & Med. & Best & $\tilde{t}$ [s] "
        r" & Def. & Wrst & Med. & Best & $\tilde{t}$ [s] \\"
    )

    lines = [
        "% Auto-generated by analysis/main_pool_grid_table.py",
        f"% 20 main-pool algorithms - primary metric = {metric_display}.",
        "% Per (algorithm, dataset) cell: mean Default / Worst-grid / Median-grid /",
        "% Best-grid score in both supervision modes, plus median wall-clock of the",
        "% Best-grid configuration ($\\tilde{t}$, seconds). '--' = not evaluated.",
        r"\begingroup",
        r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{0.95}\small",
        r"\begin{longtable}{l l c c c c c c c c c c}",
        rf"\caption{{{caption}}} \label{{{label}}} \\",
        r"\toprule",
        header_groups,
        header_cols,
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{12}{l}{\emph{(continued from previous page)}} \\",
        r"\toprule",
        header_groups,
        header_cols,
        r"\midrule",
        r"\endhead",
        r"\midrule \multicolumn{12}{r}{\emph{(continued on next page)}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    algos_order = list(dict.fromkeys(df_task["algorithm"].tolist()))
    for algo in algos_order:
        sub = df_task[df_task["algorithm"] == algo]
        n_rows = len(sub)
        for i, (_, r) in enumerate(sub.iterrows()):
            algo_cell = (
                rf"\multirow{{{n_rows}}}{{*}}{{\textbf{{{ALGO_DISPLAY.get(algo, algo)}}}}}"
                if i == 0 else ""
            )
            ds_cell = DATASET_DISPLAY.get(r["dataset"], r["dataset"])
            ng_def  = _fmt_score(r.get("NG_default"))
            ng_w    = _fmt_score(r.get("NG_worst"))
            ng_m    = _fmt_score(r.get("NG_median"))
            ng_b    = _fmt_score(r.get("NG_best"))
            ng_t    = _fmt_time(r.get("NG_time"))
            g_def   = _fmt_score(r.get("G_default"))
            g_w     = _fmt_score(r.get("G_worst"))
            g_m     = _fmt_score(r.get("G_median"))
            g_b     = _fmt_score(r.get("G_best"))
            g_t     = _fmt_time(r.get("G_time"))
            lines.append(
                f"{algo_cell} & {ds_cell} & "
                f"{ng_def} & {ng_w} & {ng_m} & {ng_b} & {ng_t} & "
                f"{g_def} & {g_w} & {g_m} & {g_b} & {g_t} \\\\"
            )
        if algo != algos_order[-1]:
            lines.append(r"\midrule")

    lines.extend([r"\end{longtable}", r"\endgroup"])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# Standalone PDF compilation (sanity-check preview)
# ─────────────────────────────────────────────────────────────────────

def _compile_pdf(tex_body: str, out_pdf: Path) -> None:
    doc = (
        r"\documentclass[a4paper,11pt]{article}" "\n"
        r"\usepackage[a4paper,margin=1cm,landscape]{geometry}" "\n"
        r"\usepackage{booktabs}" "\n"
        r"\usepackage{multirow}" "\n"
        r"\usepackage{longtable}" "\n"
        r"\usepackage[table]{xcolor}" "\n"
        r"\pagestyle{empty}" "\n"
        r"\begin{document}" "\n"
        + tex_body + "\n"
        r"\end{document}" "\n"
    )
    tmp = out_pdf.with_suffix(".tex.tmp")
    tmp.write_text(doc)
    try:
        subprocess.run(
            [
                "pdflatex", "-interaction=nonstopmode",
                f"-output-directory={out_pdf.parent}", tmp.name,
            ],
            cwd=str(tmp.parent),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        produced = out_pdf.parent / (tmp.stem + ".pdf")
        if produced.exists():
            produced.rename(out_pdf)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  [warn] pdflatex failed for {out_pdf.name}: {exc}")
    finally:
        for ext in (".tex.tmp", ".aux", ".log"):
            f = out_pdf.parent / (tmp.stem + ext)
            if f.exists():
                f.unlink()


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading benchmark frames…")
    d = load_data()

    print("Building CPD frame…")
    df_cpd = _build_task_frame(d, "CPD")
    print("Building SD frame…")
    df_sd  = _build_task_frame(d, "SD")

    long_rows = []
    for task, df_task in [("CPD", df_cpd), ("SD", df_sd)]:
        df_long = df_task.copy()
        df_long.insert(0, "task", task)
        long_rows.append(df_long)
    csv_path = out_dir / "table_main_pool.csv"
    pd.concat(long_rows, ignore_index=True).to_csv(csv_path, index=False)
    print(f"  CSV written: {csv_path}")

    captions = {
        "CPD": (
            r"\textbf{Main-pool CPD algorithms: per-(algorithm, dataset) "
            r"Default / Worst-grid / Median-grid / Best-grid Bidirectional "
            r"Covering, in both Non-Guided (NG) and Guided (G) modes.} "
            r"$\tilde{t}$ is the median wall-clock of the Best-grid "
            r"configuration in seconds. ``\texttt{--}'' = (mode, dataset) "
            r"pair not evaluated for that algorithm (e.g.\ intrinsically "
            r"Guided methods have no NG entries)."
        ),
        "SD": (
            r"\textbf{Main-pool SD algorithms: per-(algorithm, dataset) "
            r"Default / Worst-grid / Median-grid / Best-grid State Matching "
            r"Score, in both Non-Guided (NG) and Guided (G) modes.} "
            r"Conventions identical to Table~\ref{tab:main_pool_cpd}."
        ),
    }
    labels = {"CPD": "tab:main_pool_cpd", "SD": "tab:main_pool_sd"}

    for task, df_task in [("CPD", df_cpd), ("SD", df_sd)]:
        tex = _render_table(
            df_task, task, METRIC_DISPLAY[task],
            label=labels[task], caption=captions[task],
        )
        tex_path = out_dir / f"table_main_pool_{task.lower()}.tex"
        pdf_path = out_dir / f"table_main_pool_{task.lower()}.pdf"
        tex_path.write_text(tex)
        _compile_pdf(tex, pdf_path)
        print(f"  Wrote {tex_path.name} (+ {pdf_path.name} preview)")

    # ---- Secondary metrics (Gaussian F1 for CPD, ARI / WARI for SD) ----
    for task, metric_key, suffix, display in SECONDARY_METRICS:
        print(f"Building {task} frame for secondary metric {display}…")
        df_task = _build_task_frame(d, task, metric=metric_key)
        label = f"tab:main_pool_{task.lower()}_{suffix}"
        primary_label = labels[task]
        primary_display = METRIC_DISPLAY[task]
        caption = (
            rf"\textbf{{Main-pool {task} algorithms: per-(algorithm, dataset) "
            rf"Default / Worst-grid / Median-grid / Best-grid {display}, "
            rf"in both Non-Guided (NG) and Guided (G) modes.}} "
            rf"Same conventions as Table~\ref{{{primary_label}}}, with "
            rf"{primary_display} replaced by {display}. The Best-grid "
            rf"configuration in this table is selected on {display} itself "
            rf"(not on {primary_display}), so each table is self-contained."
        )
        tex = _render_table(
            df_task, task, display,
            label=label, caption=caption,
        )
        tex_path = out_dir / f"table_main_pool_{task.lower()}_{suffix}.tex"
        pdf_path = out_dir / f"table_main_pool_{task.lower()}_{suffix}.pdf"
        tex_path.write_text(tex)
        _compile_pdf(tex, pdf_path)
        print(f"  Wrote {tex_path.name} (+ {pdf_path.name} preview)")


if __name__ == "__main__":
    main()
