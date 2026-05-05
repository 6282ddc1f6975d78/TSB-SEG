import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import subprocess
import numpy as np
import pandas as pd
from analysis.data import load_data, _build_algo_sets
from analysis.helpers import (
    ALGO_DISPLAY, CPD_METRICS, SD_METRICS, OUTPUT_DIR, EXCLUDE_INCOMPLETE_ALGOS,
    select_best_grid_per_dataset, aggregate_grid_summaries
)

MAIN_DATASETS = ["actrectut", "has", "mocap", "pump", "skab", "tssb", "usc-had", "utsa"]

def _compile_pdf(tex_body: str, out_pdf: Path):
    doc = r"""\documentclass[preview,varwidth=50cm,margin=2mm]{standalone}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage[table]{xcolor}
\begin{document}
""" + tex_body + r"""
\end{document}
"""
    tmp_tex = out_pdf.with_suffix(".tex.tmp")
    tmp_tex.write_text(doc)
    try:
        subprocess.run([
            "pdflatex", "-interaction=nonstopmode",
            f"-output-directory={out_pdf.parent}",
            tmp_tex.name
        ], cwd=str(tmp_tex.parent), check=True, stdout=subprocess.DEVNULL)
        
        pdf_out = out_pdf.parent / (tmp_tex.stem + ".pdf")
        if pdf_out.exists():
            pdf_out.rename(out_pdf)
    finally:
        for ext in [".tex.tmp", ".aux", ".log", ".pdf.tmp", ".pdf"]:
            f = out_pdf.parent / (tmp_tex.stem + ext)
            if f.exists():
                f.unlink()

def _build_latex_all_metrics(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{l l l c c c c c}",
        r"\toprule",
        r"\textbf{Algorithm} & \textbf{Dataset} & \textbf{Metric} & \textbf{Best} & \textbf{Median} & \textbf{Worst} & \textbf{Default} & \textbf{Time (s)} \\",
        r"\midrule"
    ]
    algos_in_df = list(df["algorithm"].unique())
    
    for algo, group_algo in df.groupby("algorithm", sort=False):
        n_rows_algo = len(group_algo)
        first_algo = True
        
        # Group by dataset within algorithm
        ds_order = [ds for ds in MAIN_DATASETS if ds in group_algo["dataset"].unique()]
        for ds in ds_order:
            group_ds = group_algo[group_algo["dataset"] == ds]
            n_rows_ds = len(group_ds)
            first_ds = True
            
            for i, r in group_ds.iterrows():
                algo_cell = rf"\multirow{{{n_rows_algo}}}{{*}}{{{ALGO_DISPLAY.get(algo, algo)}}}" if first_algo else ""
                ds_cell = rf"\multirow{{{n_rows_ds}}}{{*}}{{{ds}}}" if first_ds else ""
                
                metric_name = r["metric"].replace("_", "\\_")
                
                def f(val):
                    return f"{val:.3f}" if pd.notna(val) else "--"
                def ft(val):
                    return f"{val:.1f}" if pd.notna(val) else "--"
                    
                best = f(r["score_best"])
                med = f(r["score_median"])
                worst = f(r["score_worst"])
                dftl = f(r["score_default"])
                time_val = ft(r["time_median"])
                
                lines.append(f"{algo_cell} & {ds_cell} & {metric_name} & {best} & {med} & {worst} & {dftl} & {time_val} \\\\")
                
                first_algo = False
                first_ds = False
                
        if algo != algos_in_df[-1]:
            lines.append(r"\midrule")
            
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}"
    ])
    return "\n".join(lines)


def run():
    d = load_data()
    cpd_algos, sd_algos = _build_algo_sets()
    
    out_dir = OUTPUT_DIR / "performance" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cpd_valid = [m for m in set(CPD_METRICS) if m != "execution_time_seconds"]
    sd_valid = [m for m in set(SD_METRICS) if m != "execution_time_seconds"]
    
    runs = [
        ("CPD", "NG", cpd_algos, cpd_valid, d.df_grid_ng, d.df_default_ng),
        ("CPD", "G", cpd_algos, cpd_valid, d.df_grid_g, d.df_default_g),
        ("SD", "NG", sd_algos, sd_valid, d.df_grid_ng, d.df_default_ng),
        ("SD", "G", sd_algos, sd_valid, d.df_grid_g, d.df_default_g),
    ]
    
    for task, mode, algos, metrics_to_run, df_grid, df_default in runs:
        print(f"Building aggregated table for {task} ({mode})...")
        
        all_metrics_dfs = []
        for metric in sorted(metrics_to_run):
            higher_is_better = True
            
            ranked = select_best_grid_per_dataset(df_grid, metric, higher_is_better=higher_is_better)
            if ranked.empty:
                continue
            grid_summary = aggregate_grid_summaries(ranked, metric)
            
            if "execution_time_seconds" in df_grid.columns:
                best_configs = ranked[ranked["config_rank"] == 1]
                times = best_configs.groupby(["algorithm", "dataset"])["execution_time_seconds"].median().reset_index()
                times = times.rename(columns={"execution_time_seconds": "time_median"})
                grid_summary = grid_summary.merge(times, on=["algorithm", "dataset"], how="left")
            else:
                grid_summary["time_median"] = np.nan
                
            if metric in df_default.columns:
                default_summary = df_default[["algorithm", "dataset", "trial_index", metric]].copy()
                default_summary = default_summary.rename(columns={metric: "score_default"})
            else:
                default_summary = pd.DataFrame(columns=["algorithm", "dataset", "trial_index", "score_default"])
                
            if not grid_summary.empty:
                gs = grid_summary.groupby(["algorithm", "dataset"], as_index=False)[
                    ["score_best", "score_median", "score_worst", "time_median"]
                ].mean()
            else:
                gs = pd.DataFrame(columns=["algorithm", "dataset", "score_best", "score_median", "score_worst", "time_median"])
                
            if not default_summary.empty:
                ds = default_summary.groupby(["algorithm", "dataset"], as_index=False)["score_default"].mean()
            else:
                ds = pd.DataFrame(columns=["algorithm", "dataset", "score_default"])
                
            df_metric = gs.merge(ds, on=["algorithm", "dataset"], how="outer")
            df_metric["metric"] = metric
            all_metrics_dfs.append(df_metric)
            
        if not all_metrics_dfs:
            continue
            
        df_all = pd.concat(all_metrics_dfs, ignore_index=True)
        
        df_all = df_all[df_all["algorithm"].isin(algos)].copy()
        df_all = df_all[df_all["dataset"] != "pamap2"].copy()
        
        if "random" in df_all["algorithm"].values or "Random" in df_all["algorithm"].values:
            df_all = df_all[~df_all["algorithm"].isin(["random", "Random"])].copy()

        n_datasets = len([x for x in MAIN_DATASETS if x != "pamap2"])
        algo_ds_count = df_all.groupby("algorithm")["dataset"].nunique()
        complete_algos = set(algo_ds_count[algo_ds_count == len(MAIN_DATASETS)].index)
        df_all = df_all[df_all["algorithm"].isin(complete_algos)].copy()
        
        if task == "CPD":
            df_all = df_all[df_all["algorithm"] != "clap"].copy()
            
        if EXCLUDE_INCOMPLETE_ALGOS:
            nan_per_algo = df_all.groupby("algorithm")["score_best"].apply(lambda s: s.isna().sum())
            algos_with_gaps = set(nan_per_algo[nan_per_algo > 0].index)
            if algos_with_gaps:
                df_all = df_all[~df_all["algorithm"].isin(algos_with_gaps)].copy()
                
        if df_all.empty:
            continue
            
        algo_order = df_all.groupby("algorithm")["score_best"].mean().sort_values(ascending=False).index.tolist()
        df_all["algorithm"] = pd.Categorical(df_all["algorithm"], categories=algo_order, ordered=True)
        df_all = df_all.sort_values(["algorithm", "dataset", "metric"])
        
        tex_str = _build_latex_all_metrics(df_all)
        
        out_name = f"table_{task.lower()}_all_metrics_{mode.lower()}"
        tex_path = out_dir / f"{out_name}.tex"
        pdf_path = out_dir / f"{out_name}.pdf"
        
        tex_path.write_text(tex_str)
        _compile_pdf(tex_str, pdf_path)
            
if __name__ == "__main__":
    run()
