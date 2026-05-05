"""TABLE_PAMAP2 + scalability inputs for RQ3.

Builds a per-algorithm summary of how each method behaves on the PAMAP2
dataset (9 long, multivariate HAR series). PAMAP2 is excluded from the
main analyses (filtered upstream in ``analysis/data.py``) because most
algorithms cannot complete it in the allotted budget; here we
explicitly report which algorithms scale and which do not.

For each algorithm we report, separately for the non-guided (NG) and
guided (G) regimes:

* ``default_finished``   number of finished children out of 9 (default config).
* ``score_default``      mean task score across the finished children.
* ``runtime_default``    median per-series runtime (seconds).
* ``grid_complete``      number of grid configs that completed all 9 series.
* ``grid_total``         number of grid configs attempted.
* ``score_best``         mean task score of the best-grid config (per dataset).
* ``runtime_best``       median per-series runtime of that best-grid config.

Outputs:

* ``temp/table_pamap2.csv``                machine-readable summary.
* ``figures/paper/table_pamap2.tex``       booktabs LaTeX table fragment.
* ``figures/paper/table_pamap2.pdf``       standalone-compiled PDF preview.

Run as::

    python analysis/pamap2.py
"""
from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.helpers import (
    ALGO_CATEGORY,
    ALGO_DISPLAY,
    ALGO_TASK,
    ALL_METRICS,
    BENCH_DIR,
    METRIC_CPD,
    METRIC_SD,
    OUTPUT_DIR,
    _coalesce_sms,
    analyzer,
    select_best_grid_per_dataset,
)

DATASET = "pamap2"
EXPECTED = 9  # series count on pamap2
RUNTIME_COL = "execution_time_seconds"

CSV_PATH = BENCH_DIR.parent / "temp" / "table_pamap2.csv"
TEX_PATH = OUTPUT_DIR / "table_pamap2.tex"
PDF_PATH = OUTPUT_DIR / "table_pamap2.pdf"


# ─────────────────────────────────────────────────────────────────────
# Raw fetches restricted to PAMAP2
# ─────────────────────────────────────────────────────────────────────

def _fetch_default_pamap2(experiment_key: str) -> pd.DataFrame:
    """Return finished default children on PAMAP2 (relaxed, no completeness filter)."""
    df_parents = analyzer.fetch_parent_stats([experiment_key])
    if df_parents.empty:
        return pd.DataFrame()
    df_parents = df_parents[df_parents["dataset"] == DATASET].copy()
    if df_parents.empty:
        return pd.DataFrame()
    # Keep all parents (relaxed): we want partial runs too.
    df_valid = analyzer.validate_completeness(df_parents, strategy="relaxed")
    # Among reruns of the same (algo, dataset), keep the one with the most
    # finished children; tiebreak by latest start.
    df_valid = (
        df_valid.sort_values(["children_finished", "start_time"], ascending=[False, False])
        .drop_duplicates(subset=["algorithm", "dataset"], keep="first")
    )
    df = analyzer.fetch_metrics(df_valid, metric_keys=ALL_METRICS, deduplicate_series=True)
    return _coalesce_sms(df)


def _fetch_grid_pamap2(base_mode: str) -> pd.DataFrame:
    df = analyzer.fetch_grid_metrics(base_mode, metric_keys=ALL_METRICS)
    if df.empty:
        return df
    df = df[df["dataset"] == DATASET].copy()
    return _coalesce_sms(df)


# ─────────────────────────────────────────────────────────────────────
# Per-algo summary builder
# ─────────────────────────────────────────────────────────────────────

def _algo_metric(algo: str) -> str:
    """Pick the right task metric for the algorithm."""
    return METRIC_SD if ALGO_TASK.get(algo) == "SD" else METRIC_CPD


def _default_summary(df_default: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Per-algo (n_finished, score_default, runtime_default) on pamap2."""
    if df_default.empty:
        return pd.DataFrame(
            columns=["algorithm", f"default_finished_{suffix}",
                     f"score_default_{suffix}", f"runtime_default_{suffix}"]
        )

    rows = []
    for algo, sub in df_default.groupby("algorithm"):
        metric = _algo_metric(algo)
        score_series = sub[metric] if metric in sub.columns else pd.Series(dtype=float)
        score_series = pd.to_numeric(score_series, errors="coerce").dropna()
        rt_series = pd.to_numeric(sub.get(RUNTIME_COL, pd.Series(dtype=float)),
                                   errors="coerce").dropna()
        rows.append({
            "algorithm": algo,
            f"default_finished_{suffix}": int(sub["trial_index"].nunique()),
            f"score_default_{suffix}":
                float(score_series.mean()) if not score_series.empty else np.nan,
            f"runtime_default_{suffix}":
                float(rt_series.median()) if not rt_series.empty else np.nan,
        })
    return pd.DataFrame(rows)


def _grid_summary(df_grid: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Per-algo grid stats on pamap2.

    Returns: ``algorithm, grid_total_<s>, grid_complete_<s>,
    score_best_<s>, runtime_best_<s>``.
    """
    cols = [
        "algorithm",
        f"grid_total_{suffix}", f"grid_complete_{suffix}",
        f"score_best_{suffix}", f"runtime_best_{suffix}",
    ]
    if df_grid.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for algo, sub in df_grid.groupby("algorithm"):
        metric = _algo_metric(algo)
        if metric not in sub.columns:
            rows.append({
                "algorithm": algo,
                f"grid_total_{suffix}": int(sub["parent_run_id"].nunique()),
                f"grid_complete_{suffix}": 0,
                f"score_best_{suffix}": np.nan,
                f"runtime_best_{suffix}": np.nan,
            })
            continue

        sub = sub.dropna(subset=[metric, "trial_index", "parent_run_id"]).copy()
        # Per-config series count
        per_cfg = sub.groupby("parent_run_id")["trial_index"].nunique()
        n_total = int(per_cfg.shape[0])
        n_complete = int((per_cfg >= EXPECTED).sum())

        ranked = select_best_grid_per_dataset(sub, metric, higher_is_better=True)
        if ranked.empty:
            rows.append({
                "algorithm": algo,
                f"grid_total_{suffix}": n_total,
                f"grid_complete_{suffix}": n_complete,
                f"score_best_{suffix}": np.nan,
                f"runtime_best_{suffix}": np.nan,
            })
            continue

        best = ranked[ranked["config_rank"] == 1]
        score_best = float(best[metric].mean()) if not best.empty else np.nan
        rt = pd.to_numeric(best.get(RUNTIME_COL, pd.Series(dtype=float)),
                           errors="coerce").dropna()
        runtime_best = float(rt.median()) if not rt.empty else np.nan

        rows.append({
            "algorithm": algo,
            f"grid_total_{suffix}": n_total,
            f"grid_complete_{suffix}": n_complete,
            f"score_best_{suffix}": score_best,
            f"runtime_best_{suffix}": runtime_best,
        })
    return pd.DataFrame(rows)


def _status(row: pd.Series) -> str:
    """Final status flag from default-completion in either regime."""
    n_ng = int(row.get("default_finished_ng", 0) or 0)
    n_g = int(row.get("default_finished_g", 0) or 0)
    n_max = max(n_ng, n_g)
    if n_max >= EXPECTED:
        return "scales"
    if n_max > 0:
        return "partial"
    return "fails"


def build_dataframe() -> pd.DataFrame:
    print("Fetching PAMAP2 default (NG, G) ...")
    df_def_ng = _fetch_default_pamap2("non_guided")
    df_def_g = _fetch_default_pamap2("guided")
    print(f"  default NG: {len(df_def_ng)} children, "
          f"{df_def_ng['algorithm'].nunique() if not df_def_ng.empty else 0} algos")
    print(f"  default  G: {len(df_def_g)} children, "
          f"{df_def_g['algorithm'].nunique() if not df_def_g.empty else 0} algos")

    print("Fetching PAMAP2 grid (NG, G) ...")
    # NB: ``fetch_grid_metrics`` builds the group key as ``grid_<base_mode>`` and
    # ``benchmark_config_v3.yaml`` declares ``grid_non-guided`` (with hyphen).
    df_grid_ng = _fetch_grid_pamap2("non-guided")
    df_grid_g = _fetch_grid_pamap2("guided")
    print(f"  grid NG: {len(df_grid_ng)} children, "
          f"{df_grid_ng['algorithm'].nunique() if not df_grid_ng.empty else 0} algos")
    print(f"  grid  G: {len(df_grid_g)} children, "
          f"{df_grid_g['algorithm'].nunique() if not df_grid_g.empty else 0} algos")

    parts = [
        _default_summary(df_def_ng, "ng"),
        _default_summary(df_def_g, "g"),
        _grid_summary(df_grid_ng, "ng"),
        _grid_summary(df_grid_g, "g"),
    ]

    # Universe of algorithms = anything that was actually attempted on
    # PAMAP2 (i.e. has at least one parent run, finished or not, in any
    # of the 4 experiments) plus anything observed in the metric fetches.
    # This keeps the table grounded in our experiments and avoids listing
    # typology entries we never tried on PAMAP2.
    print("Listing PAMAP2 parents (all statuses) ...")
    pamap_algos: set[str] = set()
    for key in ("non_guided", "guided", "grid_non_guided", "grid_guided"):
        dfp = analyzer.fetch_parent_stats([key])
        if dfp.empty:
            continue
        pamap_algos.update(dfp.loc[dfp["dataset"] == DATASET, "algorithm"].dropna().tolist())

    all_algos: set[str] = set(pamap_algos)
    for p in parts:
        if not p.empty:
            all_algos.update(p["algorithm"].tolist())
    print(f"  PAMAP2 universe: {len(all_algos)} algorithms")

    df = pd.DataFrame({"algorithm": sorted(all_algos)})
    for p in parts:
        if not p.empty:
            df = df.merge(p, on="algorithm", how="left")

    # Defaults for missing integer columns
    for c in ["default_finished_ng", "default_finished_g",
              "grid_total_ng", "grid_complete_ng",
              "grid_total_g", "grid_complete_g"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0).astype(int)

    # Defaults for missing float columns (e.g. when one regime is fully empty)
    for c in ["score_default_ng", "score_default_g",
              "runtime_default_ng", "runtime_default_g",
              "score_best_ng", "score_best_g",
              "runtime_best_ng", "runtime_best_g"]:
        if c not in df.columns:
            df[c] = np.nan

    df["task"] = df["algorithm"].map(ALGO_TASK).fillna("?")
    df["category"] = df["algorithm"].map(ALGO_CATEGORY).fillna("?")
    df["display"] = df["algorithm"].map(ALGO_DISPLAY).fillna(df["algorithm"])
    df["metric"] = df["algorithm"].map(_algo_metric)
    df["status"] = df.apply(_status, axis=1)

    # Order: scales first then partial then fails; within each, by best-score desc
    df["best_score_overall"] = df[["score_best_ng", "score_best_g"]].max(axis=1)
    status_rank = {"scales": 0, "partial": 1, "fails": 2}
    df["_sr"] = df["status"].map(status_rank)
    df = df.sort_values(
        ["_sr", "task", "best_score_overall"],
        ascending=[True, True, False],
        na_position="last",
    ).drop(columns=["_sr"])

    return df


# ─────────────────────────────────────────────────────────────────────
# LaTeX rendering
# ─────────────────────────────────────────────────────────────────────

_STATUS_GLYPH = {
    "scales":  r"\textcolor{green!55!black}{\checkmark}",
    "partial": r"\textcolor{orange!80!black}{$\sim$}",
    "fails":   r"\textcolor{red!75!black}{$\times$}",
}

_TABLE_PREAMBLE = r"""% Auto-generated by analysis/pamap2.py
% Reports per-algorithm behaviour on the PAMAP2 dataset (9 long HAR series,
% n in [8.5k, 447k], d=[18,27]). NG = non-guided, G = guided.
% Score: Bidirectional Covering (CPD) or SMS (SD), depending on the algo's task.
"""


def _fmt_score(x) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.2f}"


def _fmt_runtime(x) -> str:
    if pd.isna(x) or x is None:
        return "--"
    if x < 1:
        return f"{x:.2f}\\,s"
    if x < 60:
        return f"{x:.1f}\\,s"
    if x < 3600:
        return f"{x / 60:.1f}\\,m"
    return f"{x / 3600:.1f}\\,h"


def _fmt_count(n: int, total: int) -> str:
    if total == 0:
        return "--"
    return f"{int(n)}/{int(total)}"


def render_latex(df: pd.DataFrame) -> str:
    rows = []
    for _, r in df.iterrows():
        rows.append(" & ".join([
            r["display"],
            r["task"],
            _STATUS_GLYPH.get(r["status"], r["status"]),
            _fmt_count(r["default_finished_ng"], EXPECTED),
            _fmt_score(r["score_default_ng"]),
            _fmt_runtime(r["runtime_default_ng"]),
            _fmt_count(r["default_finished_g"], EXPECTED),
            _fmt_score(r["score_default_g"]),
            _fmt_runtime(r["runtime_default_g"]),
            _fmt_count(r["grid_complete_ng"], r["grid_total_ng"]),
            _fmt_score(r["score_best_ng"]),
            _fmt_runtime(r["runtime_best_ng"]),
            _fmt_count(r["grid_complete_g"], r["grid_total_g"]),
            _fmt_score(r["score_best_g"]),
            _fmt_runtime(r["runtime_best_g"]),
        ]) + r" \\")

    body = "\n".join(rows)
    header = (
        r"\begin{tabular}{l c c | c c c | c c c | c c c | c c c}" "\n"
        r"\toprule" "\n"
        r" & & & \multicolumn{6}{c|}{\textbf{Default config}} & "
        r"\multicolumn{6}{c}{\textbf{Best-grid config}} \\" "\n"
        r"\cmidrule(lr){4-9} \cmidrule(lr){10-15}" "\n"
        r" & & & \multicolumn{3}{c|}{Non-guided} & \multicolumn{3}{c|}{Guided}"
        r" & \multicolumn{3}{c|}{Non-guided} & \multicolumn{3}{c}{Guided} \\" "\n"
        r"\textbf{Algorithm} & \textbf{Task} & \textbf{Status}"
        r" & \#fin. & score & runtime"
        r" & \#fin. & score & runtime"
        r" & \#cfg. & score & runtime"
        r" & \#cfg. & score & runtime \\" "\n"
        r"\midrule" "\n"
    )
    footer = "\n" + r"\bottomrule" "\n" + r"\end{tabular}"
    return _TABLE_PREAMBLE + header + body + footer


_STANDALONE_PREAMBLE = r"""\documentclass[border=4pt]{standalone}
\usepackage{booktabs}
\usepackage{xcolor}
\usepackage{amssymb}
\begin{document}
"""
_STANDALONE_END = "\n" + r"\end{document}" + "\n"


def compile_pdf(tex_body: str, out_pdf: Path) -> bool:
    """Compile the LaTeX table to a standalone PDF if ``pdflatex`` is available."""
    if not shutil.which("pdflatex"):
        print("  pdflatex not found — skipping PDF compile")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "table.tex"
        src.write_text(_STANDALONE_PREAMBLE + tex_body + _STANDALONE_END)
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "table.tex"],
                cwd=tmp_dir, check=True, capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print("  ⚠ pdflatex failed:")
            print(e.stdout.decode(errors="ignore")[-1500:])
            return False
        produced = tmp_dir / "table.pdf"
        if not produced.exists():
            print("  ⚠ pdflatex produced no PDF")
            return False
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, out_pdf)
        return True


# ─────────────────────────────────────────────────────────────────────
# Main entry
# ─────────────────────────────────────────────────────────────────────

def main():
    df = build_dataframe()

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"\nCSV → {CSV_PATH} ({len(df)} algos)")

    tex = render_latex(df)
    TEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEX_PATH.write_text(tex)
    print(f"TEX → {TEX_PATH}")

    if compile_pdf(tex, PDF_PATH):
        print(f"PDF → {PDF_PATH}")

    # Console preview
    print("\nSummary by status:")
    print(df.groupby("status").size().to_string())


if __name__ == "__main__":
    main()
