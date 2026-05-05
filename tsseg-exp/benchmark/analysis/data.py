"""Data pipeline — fetches metrics from MLflow and builds unified DataFrames.

The expensive ``analyzer.fetch_*`` calls are cached as pickles under
``analysis/cache/`` so figure modules can be re-run quickly without
re-querying the SQLite tracking database.

Usage:
    from analysis.data import load_data
    d = load_data()                   # uses cache if available
    d = load_data(refresh=True)       # force re-fetch from MLflow
    d.df_cpd, d.df_sd, d.df_cpd_g, d.df_sd_g, d.cpd_algos, d.sd_algos
"""
from __future__ import annotations

# Allow running this file directly: ``python analysis/data.py``
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis.helpers import (
    ALGO_CATEGORY,
    ALGO_DISPLAY,
    ALL_METRICS,
    CACHE_DIR,
    EXCLUDE_INCOMPLETE_ALGOS,
    GUIDED_TIMEOUT_ALGOS,
    LIMITED_COVERAGE_ALGOS,
    METRIC_CPD,
    METRIC_SD,
    TYPOLOGY,
    _coalesce_sms,
    aggregate_grid_summaries,
    analyzer,
    apply_gap_tolerance,
    get_domain,
    select_best_grid_per_dataset,
)

# Cached intermediate DataFrames ----------------------------------------
_RAW_CACHE_NAMES = ("df_default_ng", "df_default_g", "df_grid_ng", "df_grid_g")
_BUILT_CACHE_NAMES = ("df_cpd", "df_sd", "df_cpd_g", "df_sd_g")
ALL_CACHE_NAMES = _RAW_CACHE_NAMES + _BUILT_CACHE_NAMES


def _cache_path(name: str):
    return CACHE_DIR / f"{name}.pkl"


def clear_cache(names=None) -> None:
    """Delete pickled DataFrames from the cache directory."""
    targets = names or ALL_CACHE_NAMES
    for n in targets:
        p = _cache_path(n)
        if p.exists():
            p.unlink()
            print(f"  removed {p.name}")


def _load_pickle(name: str):
    p = _cache_path(name)
    return pd.read_pickle(p) if p.exists() else None


def _save_pickle(name: str, df: pd.DataFrame) -> None:
    df.to_pickle(_cache_path(name))


# ─────────────────────────────────────────────────────────────────────
# Raw fetches (default + grid, non-guided + guided)
# ─────────────────────────────────────────────────────────────────────

def _fetch_default(experiments_key: str, label: str) -> pd.DataFrame:
    print(f"Fetching {label} default parents...")
    df_parents = analyzer.fetch_parent_stats([experiments_key])
    df_valid = analyzer.validate_completeness(df_parents, strategy="strict")
    print(
        f"  Valid parents: {len(df_valid)} "
        f"({df_valid['algorithm'].nunique()} algos, {df_valid['dataset'].nunique()} datasets)"
    )
    print(f"Fetching {label} default metrics...")
    df = analyzer.fetch_metrics(df_valid, metric_keys=ALL_METRICS, deduplicate_series=True)
    df = _coalesce_sms(df)
    print(f"  Children: {len(df)}")
    return df


def _fetch_grid(mode: str, label: str) -> pd.DataFrame:
    print(f"Fetching grid {label} metrics...")
    df = analyzer.fetch_grid_metrics(mode, metric_keys=ALL_METRICS)
    df = _coalesce_sms(df)
    print(f"  Grid children: {len(df)} ({df['algorithm'].nunique()} algos)")
    return df


# ─────────────────────────────────────────────────────────────────────
# Manual patches applied to raw fetches
# ─────────────────────────────────────────────────────────────────────

def _patch_remove_invalid_grid_combos(df_grid: pd.DataFrame, label: str) -> pd.DataFrame:
    """Explicitly remove meaningless hyperparameter combinations.
    e.g. e2usd and time2state with window_size=64 and step=100.
    """
    mask = df_grid["algorithm"].isin(["e2usd", "time2state"])
    if not mask.any():
        return df_grid
        
    df_suspect = df_grid[mask]
    parent_ids = list(df_suspect["parent_run_id"].dropna().unique())
    if not parent_ids:
        return df_grid
        
    df_p = analyzer.manager.fetch_run_params(parent_ids)
    if df_p.empty:
        return df_grid
        
    pivoted = df_p.pivot(index="run_id", columns="key", values="value").reset_index()
    if "grid_window_size" not in pivoted.columns or "grid_step" not in pivoted.columns:
        return df_grid
        
    bad_parents = pivoted[
        (pivoted["grid_window_size"] == "64") & (pivoted["grid_step"] == "100")
    ]["run_id"].tolist()
    
    if bad_parents:
        print(
            f"  [patch] {label}: removing {len(bad_parents)} bad parent configs "
            "(window_size=64, step=100) for e2usd/time2state"
        )
        return df_grid[~df_grid["parent_run_id"].isin(bad_parents)].copy()
        
    return df_grid


def _patch_remove_tglad_test_epochs(df_grid: pd.DataFrame, label: str) -> pd.DataFrame:
    """Remove tglad runs with grid_epochs=10 (just test runs)."""
    mask = df_grid["algorithm"] == "tglad"
    if not mask.any():
        return df_grid
        
    df_suspect = df_grid[mask]
    parent_ids = list(df_suspect["parent_run_id"].dropna().unique())
    if not parent_ids:
        return df_grid
        
    df_p = analyzer.manager.fetch_run_params(parent_ids)
    if df_p.empty:
        return df_grid
        
    pivoted = df_p.pivot(index="run_id", columns="key", values="value").reset_index()
    if "grid_epochs" not in pivoted.columns:
        return df_grid
        
    bad_parents = pivoted[pivoted["grid_epochs"] == "10"]["run_id"].tolist()
    
    if bad_parents:
        print(
            f"  [patch] {label}: removing {len(bad_parents)} bad parent configs "
            "(epochs=10) for tglad"
        )
        return df_grid[~df_grid["parent_run_id"].isin(bad_parents)].copy()
        
    return df_grid


def _patch_grid_g_prophet_pump(
    df_grid_g: pd.DataFrame, df_default_g: pd.DataFrame
) -> pd.DataFrame:
    """Inject Prophet/pump default-guided runs into ``df_grid_g``.

    The grid-guided pipeline never dispatched the ``prophet × pump`` cell;
    however Prophet is essentially parameter-free in our wrapper (only the
    multivariate aggregation method varies), so the default-guided run is
    a faithful stand-in for a one-config grid. We append those rows under
    a single synthetic ``parent_run_id`` so that
    ``select_best_grid_per_dataset`` treats them as one valid config.
    """
    mask_existing = (df_grid_g["algorithm"] == "prophet") & (
        df_grid_g["dataset"] == "pump"
    )
    if mask_existing.any():
        return df_grid_g  # already present, nothing to patch

    src = df_default_g[
        (df_default_g["algorithm"] == "prophet")
        & (df_default_g["dataset"] == "pump")
    ].copy()
    if src.empty:
        return df_grid_g

    src = src[[c for c in df_grid_g.columns if c in src.columns]].copy()
    # Single synthetic parent_run_id → one "config" for the grid selector.
    src["parent_run_id"] = "synthetic_prophet_pump_default_as_grid"
    src["mode"] = "grid_guided"
    print(
        f"  [patch] grid_g: injecting {len(src)} prophet/pump rows "
        "from default_g (parameter-free method)"
    )
    return pd.concat([df_grid_g, src], ignore_index=True)


# ─────────────────────────────────────────────────────────────────────
# Build unified per-task DataFrames
# ─────────────────────────────────────────────────────────────────────

def _build_cpd(df_grid: pd.DataFrame, df_default: pd.DataFrame, cpd_algos: set) -> pd.DataFrame:
    print("Ranking grid configs for CPD (non-guided)...")
    ranked = select_best_grid_per_dataset(df_grid, METRIC_CPD, higher_is_better=True)
    if not ranked.empty:
        grid_summary = aggregate_grid_summaries(ranked, METRIC_CPD)
        print(
            f"  Grid CPD summaries: {len(grid_summary)} rows "
            f"({grid_summary['algorithm'].nunique()} algos)"
        )
    else:
        grid_summary = pd.DataFrame()
        print("  ⚠ No grid CPD data")

    default_cpd = df_default[["algorithm", "dataset", "trial_index", METRIC_CPD]].copy()
    default_cpd = default_cpd.rename(columns={METRIC_CPD: "score_default"})

    if not grid_summary.empty:
        df_cpd = grid_summary.merge(
            default_cpd, on=["algorithm", "dataset", "trial_index"], how="outer"
        )
    else:
        df_cpd = default_cpd.copy()
        df_cpd["score_best"] = np.nan
        df_cpd["score_worst"] = np.nan
        df_cpd["score_median"] = np.nan

    df_cpd["domain"] = df_cpd.apply(lambda r: get_domain(r["dataset"], r["trial_index"]), axis=1)
    df_cpd["category"] = df_cpd["algorithm"].map(ALGO_CATEGORY)
    df_cpd["display"] = df_cpd["algorithm"].map(ALGO_DISPLAY)

    df_cpd = df_cpd[df_cpd["algorithm"].isin(cpd_algos)].copy()
    df_cpd = df_cpd[df_cpd["dataset"] != "pamap2"].copy()
    df_cpd = df_cpd[~df_cpd["algorithm"].isin(LIMITED_COVERAGE_ALGOS)].copy()

    n_datasets = df_cpd["dataset"].nunique()
    algo_ds_count = df_cpd.groupby("algorithm")["dataset"].nunique()
    complete_algos = set(algo_ds_count[algo_ds_count == n_datasets].index)
    incomplete_algos = set(algo_ds_count[algo_ds_count < n_datasets].index)
    if incomplete_algos:
        print(f"  Dropping algos missing entire datasets: {sorted(incomplete_algos)}")
    df_cpd = df_cpd[df_cpd["algorithm"].isin(complete_algos)].copy()

    df_cpd = df_cpd[df_cpd["algorithm"] != "clap"].copy()

    if EXCLUDE_INCOMPLETE_ALGOS:
        df_cpd, _, _ = apply_gap_tolerance(df_cpd, "CPD", force_keep={"hdp-hsmm"})

    print(
        f"\ndf_cpd: {len(df_cpd)} rows, {df_cpd['algorithm'].nunique()} algos, "
        f"{df_cpd['dataset'].nunique()} datasets"
    )
    print(f"  Domains: {df_cpd['domain'].value_counts().to_dict()}")
    return df_cpd


def _build_sd(df_grid: pd.DataFrame, df_default: pd.DataFrame, sd_algos: set) -> pd.DataFrame:
    print("Ranking grid configs for SD (non-guided)...")
    ranked = select_best_grid_per_dataset(df_grid, METRIC_SD, higher_is_better=True)
    if not ranked.empty:
        grid_summary = aggregate_grid_summaries(ranked, METRIC_SD)
        print(
            f"  Grid SD summaries: {len(grid_summary)} rows "
            f"({grid_summary['algorithm'].nunique()} algos)"
        )
    else:
        grid_summary = pd.DataFrame()
        print("  ⚠ No grid SD data")

    default_sd = df_default[["algorithm", "dataset", "trial_index", METRIC_SD]].copy()
    default_sd = default_sd.rename(columns={METRIC_SD: "score_default"})

    if not grid_summary.empty:
        df_sd = grid_summary.merge(
            default_sd, on=["algorithm", "dataset", "trial_index"], how="outer"
        )
    else:
        df_sd = default_sd.copy()
        df_sd["score_best"] = np.nan
        df_sd["score_worst"] = np.nan
        df_sd["score_median"] = np.nan

    df_sd["domain"] = df_sd.apply(lambda r: get_domain(r["dataset"], r["trial_index"]), axis=1)
    df_sd["category"] = df_sd["algorithm"].map(ALGO_CATEGORY)
    df_sd["display"] = df_sd["algorithm"].map(ALGO_DISPLAY)

    df_sd = df_sd[df_sd["algorithm"].isin(sd_algos)].copy()
    df_sd = df_sd[df_sd["dataset"] != "pamap2"].copy()
    df_sd = df_sd[~df_sd["algorithm"].isin(LIMITED_COVERAGE_ALGOS)].copy()

    if EXCLUDE_INCOMPLETE_ALGOS:
        df_sd, _, _ = apply_gap_tolerance(df_sd, "SD", force_keep={"hdp-hsmm"})

    print(
        f"\ndf_sd: {len(df_sd)} rows, {df_sd['algorithm'].nunique()} algos, "
        f"{df_sd['dataset'].nunique()} datasets"
    )
    print(f"  Domains: {df_sd['domain'].value_counts().to_dict()}")
    return df_sd


def _build_cpd_guided(
    df_grid: pd.DataFrame, df_default: pd.DataFrame, cpd_algos: set
) -> pd.DataFrame:
    print("Ranking grid configs for CPD (guided)...")
    ranked = select_best_grid_per_dataset(df_grid, METRIC_CPD, higher_is_better=True)
    if not ranked.empty:
        grid_summary = aggregate_grid_summaries(ranked, METRIC_CPD)
        print(
            f"  Grid CPD-g summaries: {len(grid_summary)} rows "
            f"({grid_summary['algorithm'].nunique()} algos)"
        )
    else:
        grid_summary = pd.DataFrame()
        print("  ⚠ No grid CPD-g data")

    default_cpd = df_default[["algorithm", "dataset", "trial_index", METRIC_CPD]].copy()
    default_cpd = default_cpd.rename(columns={METRIC_CPD: "score_default"})

    if not grid_summary.empty:
        df_cpd_g = grid_summary.merge(
            default_cpd, on=["algorithm", "dataset", "trial_index"], how="outer"
        )
    else:
        df_cpd_g = default_cpd.copy()
        df_cpd_g[["score_best", "score_worst", "score_median"]] = np.nan

    df_cpd_g["domain"] = df_cpd_g.apply(
        lambda r: get_domain(r["dataset"], r["trial_index"]), axis=1
    )
    df_cpd_g["category"] = df_cpd_g["algorithm"].map(ALGO_CATEGORY)
    df_cpd_g["display"] = df_cpd_g["algorithm"].map(ALGO_DISPLAY)

    df_cpd_g = df_cpd_g[df_cpd_g["algorithm"].isin(cpd_algos)].copy()
    df_cpd_g = df_cpd_g[df_cpd_g["dataset"] != "pamap2"].copy()
    df_cpd_g = df_cpd_g[~df_cpd_g["algorithm"].isin(LIMITED_COVERAGE_ALGOS)].copy()

    pre = df_cpd_g["algorithm"].nunique()
    df_cpd_g = df_cpd_g[~df_cpd_g["algorithm"].isin(GUIDED_TIMEOUT_ALGOS)].copy()
    print(
        f"  Excluded guided-timeout: {GUIDED_TIMEOUT_ALGOS} "
        f"({pre}→{df_cpd_g['algorithm'].nunique()} algos)"
    )

    n_datasets_g = df_cpd_g["dataset"].nunique()
    algo_ds_count_g = df_cpd_g.groupby("algorithm")["dataset"].nunique()
    complete_algos_g = set(algo_ds_count_g[algo_ds_count_g == n_datasets_g].index)
    incomplete_algos_g = set(algo_ds_count_g[algo_ds_count_g < n_datasets_g].index)
    if incomplete_algos_g:
        print(f"  Dropping algos missing entire datasets: {sorted(incomplete_algos_g)}")
    df_cpd_g = df_cpd_g[df_cpd_g["algorithm"].isin(complete_algos_g)].copy()

    df_cpd_g = df_cpd_g[df_cpd_g["algorithm"] != "clap"].copy()

    if EXCLUDE_INCOMPLETE_ALGOS:
        df_cpd_g, _, _ = apply_gap_tolerance(df_cpd_g, "CPD-g")

    print(
        f"\ndf_cpd_g: {len(df_cpd_g)} rows, {df_cpd_g['algorithm'].nunique()} algos, "
        f"{df_cpd_g['dataset'].nunique()} datasets"
    )
    return df_cpd_g


def _build_sd_guided(
    df_grid: pd.DataFrame, df_default: pd.DataFrame, sd_algos: set
) -> pd.DataFrame:
    print("\nRanking grid configs for SD (guided)...")
    ranked = select_best_grid_per_dataset(df_grid, METRIC_SD, higher_is_better=True)
    if not ranked.empty:
        grid_summary = aggregate_grid_summaries(ranked, METRIC_SD)
        print(
            f"  Grid SD-g summaries: {len(grid_summary)} rows "
            f"({grid_summary['algorithm'].nunique()} algos)"
        )
    else:
        grid_summary = pd.DataFrame()
        print("  ⚠ No grid SD-g data")

    default_sd = df_default[["algorithm", "dataset", "trial_index", METRIC_SD]].copy()
    default_sd = default_sd.rename(columns={METRIC_SD: "score_default"})

    if not grid_summary.empty:
        df_sd_g = grid_summary.merge(
            default_sd, on=["algorithm", "dataset", "trial_index"], how="outer"
        )
    else:
        df_sd_g = default_sd.copy()
        df_sd_g[["score_best", "score_worst", "score_median"]] = np.nan

    df_sd_g["domain"] = df_sd_g.apply(lambda r: get_domain(r["dataset"], r["trial_index"]), axis=1)
    df_sd_g["category"] = df_sd_g["algorithm"].map(ALGO_CATEGORY)
    df_sd_g["display"] = df_sd_g["algorithm"].map(ALGO_DISPLAY)

    df_sd_g = df_sd_g[df_sd_g["algorithm"].isin(sd_algos)].copy()
    df_sd_g = df_sd_g[df_sd_g["dataset"] != "pamap2"].copy()
    df_sd_g = df_sd_g[~df_sd_g["algorithm"].isin(LIMITED_COVERAGE_ALGOS)].copy()

    if EXCLUDE_INCOMPLETE_ALGOS:
        df_sd_g, _, _ = apply_gap_tolerance(df_sd_g, "SD-g")

    print(
        f"\ndf_sd_g: {len(df_sd_g)} rows, {df_sd_g['algorithm'].nunique()} algos, "
        f"{df_sd_g['dataset'].nunique()} datasets"
    )
    return df_sd_g


def _build_algo_sets():
    """CPD + SD algorithm sets used to filter df_cpd / df_sd."""
    cpd_algos: set = set()
    for cat_info in TYPOLOGY["categories"].values():
        cpd_algos.update(cat_info["algorithms"])  # both CPD and SD
    cpd_algos.add("changefinder")
    cpd_algos.add("random")

    sd_algos: set = set()
    for cat_info in TYPOLOGY["categories"].values():
        if cat_info["task"] == "SD":
            sd_algos.update(cat_info["algorithms"])
    sd_algos.add("random")

    return cpd_algos, sd_algos


def load_data(refresh: bool = False, refresh_built: bool = False) -> SimpleNamespace:
    """Return a namespace with all DataFrames + algo sets.

    Parameters
    ----------
    refresh
        If True, re-fetch the raw default/grid metrics from MLflow
        (slowest step; otherwise loaded from the on-disk pickle cache).
    refresh_built
        If True, rebuild df_cpd / df_sd / df_cpd_g / df_sd_g from raw
        even if their cached versions exist.
    """
    cpd_algos, sd_algos = _build_algo_sets()

    cache: dict = {n: None for n in ALL_CACHE_NAMES}
    if not refresh:
        for n in _RAW_CACHE_NAMES:
            cache[n] = _load_pickle(n)
            if cache[n] is not None:
                print(f"  [cache] loaded {n}.pkl ({len(cache[n])} rows)")

    if cache["df_default_ng"] is None:
        cache["df_default_ng"] = _fetch_default("non_guided", "non-guided")
        _save_pickle("df_default_ng", cache["df_default_ng"])
    if cache["df_default_g"] is None:
        cache["df_default_g"] = _fetch_default("guided", "guided")
        _save_pickle("df_default_g", cache["df_default_g"])
    if cache["df_grid_ng"] is None:
        cache["df_grid_ng"] = _fetch_grid("non-guided", "non-guided (exp 8)")
        _save_pickle("df_grid_ng", cache["df_grid_ng"])
    if cache["df_grid_g"] is None:
        cache["df_grid_g"] = _fetch_grid("guided", "guided (exp 9)")
        _save_pickle("df_grid_g", cache["df_grid_g"])

    # Apply manual patches on every load (kept out of the pickle so the
    # raw cache stays a faithful mirror of MLflow).
    cache["df_grid_ng"] = _patch_remove_invalid_grid_combos(
        cache["df_grid_ng"], "grid_ng"
    )
    cache["df_grid_g"] = _patch_remove_invalid_grid_combos(
        cache["df_grid_g"], "grid_g"
    )
    cache["df_grid_ng"] = _patch_remove_tglad_test_epochs(
        cache["df_grid_ng"], "grid_ng"
    )
    cache["df_grid_g"] = _patch_remove_tglad_test_epochs(
        cache["df_grid_g"], "grid_g"
    )
    cache["df_grid_g"] = _patch_grid_g_prophet_pump(
        cache["df_grid_g"], cache["df_default_g"]
    )

    if not (refresh or refresh_built):
        for n in _BUILT_CACHE_NAMES:
            cache[n] = _load_pickle(n)
            if cache[n] is not None:
                print(f"  [cache] loaded {n}.pkl ({len(cache[n])} rows)")

    if cache["df_cpd"] is None:
        cache["df_cpd"] = _build_cpd(cache["df_grid_ng"], cache["df_default_ng"], cpd_algos)
        _save_pickle("df_cpd", cache["df_cpd"])
    if cache["df_sd"] is None:
        cache["df_sd"] = _build_sd(cache["df_grid_ng"], cache["df_default_ng"], sd_algos)
        _save_pickle("df_sd", cache["df_sd"])
    if cache["df_cpd_g"] is None:
        cache["df_cpd_g"] = _build_cpd_guided(
            cache["df_grid_g"], cache["df_default_g"], cpd_algos
        )
        _save_pickle("df_cpd_g", cache["df_cpd_g"])
    if cache["df_sd_g"] is None:
        cache["df_sd_g"] = _build_sd_guided(cache["df_grid_g"], cache["df_default_g"], sd_algos)
        _save_pickle("df_sd_g", cache["df_sd_g"])

    return SimpleNamespace(cpd_algos=cpd_algos, sd_algos=sd_algos, **cache)


if __name__ == "__main__":
    d = load_data()
