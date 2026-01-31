import pandas as pd
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Literal
from mlflow_manager import MLflowBenchmarkManager

# Expected Series Counts (Hardcoded from original notebook)
EXPECTED_SERIES = {
    'actrectut': 2, 
    'has': 250, 
    'mocap': 9, 
    'pamap2': 9, 
    'skab': 34, 
    'tssb': 75, 
    'usc-had': 70, 
    'utsa': 32
}

# Whitelists/Blacklists
WHITELISTED_ALGOS = {'hdp-hsmm', 'ticc', 'hdp_hsmm'}
BLACKLISTED_ALGOS = {'vsax'}

class BenchmarkAnalyzer:
    """
    Logic layer for validating and aggregating benchmark results.
    Separates data fetching (MLflowBenchmarkManager) from analysis (this class).
    """
    
    def __init__(self, manager: MLflowBenchmarkManager):
        self.manager = manager
        
    def fetch_parent_runs_stats(self, experiment_keys: List[str] = None) -> pd.DataFrame:
        """
        Fetches parent runs and computes child stats (success/failure counts).
        Uses SQL backend for performance.
        """
        # Determine keys to fetch
        if experiment_keys is None:
            # Flatten all keys in all groups
            groups = self.manager.config.get('groups', {})
            all_keys = []
            for k_list in groups.values():
                all_keys.extend(k_list)
            experiment_keys = list(set(all_keys))

        # OPTIMIZATION: Use SQL backend if available
        print(f"Fetching parent stats (SQL accelerated) for keys: {experiment_keys}...")
        df_parents = self.manager.fetch_parent_stats_sql(experiment_keys)
        
        if df_parents.empty:
            print("No data found via SQL fetch.")
            return pd.DataFrame()
            
        # Post-process SQL result
        # 1. Assign Modes
        df_parents = self.assign_modes(df_parents)
        
        # 2. Fix Datetime types (SQLite returns ms ints)
        if 'start_time' in df_parents.columns:
            # auto-detect if int
            if pd.api.types.is_numeric_dtype(df_parents['start_time']):
                df_parents['start_time'] = pd.to_datetime(df_parents['start_time'], unit='ms')

        # 4. Standardize Dataset names
        if 'dataset' in df_parents.columns:
            df_parents['dataset'] = df_parents['dataset'].str.lower()
            
        return df_parents

    def assign_modes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigns a 'mode' column based on the groups defined in config.
        """
        if df.empty or 'experiment_id' not in df.columns:
            return df
            
        groups = self.manager.config.get('groups', {})
        df['mode'] = 'other'
        
        # Safe comparison by converting both sides to string
        # (SQLite returns ints, MLflow returns strings)
        exp_id_str = df['experiment_id'].astype(str)
        
        for group_name, keys in groups.items():
            # Get exp IDs for these keys
            exp_ids = self.manager.get_experiment_ids(keys)
            if exp_ids:
                # Ensure target list is also strings
                exp_ids_str = [str(x) for x in exp_ids]
                mask = exp_id_str.isin(exp_ids_str)
                df.loc[mask, 'mode'] = group_name
                
        return df

    def validate_completeness(self, df_parents: pd.DataFrame, strategy: Literal['strict', 'merge'] = 'strict') -> pd.DataFrame:
        """
        Applies completeness logic.
        strategy='strict':
            - Complete = (children_finished == EXPECTED_SERIES[dataset])
            - Deduplicates parents (keeps latest start_time).
            - Returns only Complete or Whitelisted parents.
        strategy='merge':
            - Returns ALL parents for valid algorithms (Whitelisted or normally valid).
            - Does NOT deduplicate parents.
            - Used when we want to merge series from multiple runs (patching).
        """
        if df_parents.empty:
            return df_parents

        # 1. Basic Algo Filtering (Blacklist)
        algo_col = 'algorithm'
        valid_mask = ~df_parents[algo_col].isin(BLACKLISTED_ALGOS)
        filtered = df_parents[valid_mask].copy()

        if strategy == 'merge':
            # In merge mode, we keep all runs that might contribute.
            # We don't filter by 'is_complete' because we might merge partial runs.
            # We assume downstream/imputation handles missing data.
            return filtered.sort_values('start_time', ascending=False)

        # 2. Strict Mode Logic
        def is_complete(row):
            ds = row.get('dataset')
            if ds not in EXPECTED_SERIES:
                return False
            return row['children_finished'] == EXPECTED_SERIES[ds]
            
        filtered['is_complete'] = filtered.apply(is_complete, axis=1)
        
        whitelist_mask = filtered[algo_col].isin(WHITELISTED_ALGOS)
        
        # Keep if Complete OR Whitelisted
        valid_mask = filtered['is_complete'] | whitelist_mask
        valid_parents = filtered[valid_mask].copy()
        
        # Deduplication: Keep latest start_time per (Experiment, Dataset, Algorithm)
        valid_parents = valid_parents.sort_values('start_time', ascending=False)
        valid_parents = valid_parents.drop_duplicates(
            subset=['experiment_id', 'dataset', 'algorithm'], 
            keep='first'
        )
        
        return valid_parents

    def fetch_metrics_for_parents(self, valid_parents: pd.DataFrame, deduplicate_series: bool = False) -> pd.DataFrame:
        """
        Fetches metrics for all children.
        deduplicate_series: If True, keeps only the latest child run per (algorithm, dataset, trial_index).
        """
        if valid_parents.empty:
            return pd.DataFrame()
            
        parent_ids = list(valid_parents['run_id'].unique())
        print(f"Fetching metrics for children of {len(parent_ids)} parents...")
        
        # 1. Fetch
        df_long = self.manager.fetch_metrics_sql(parent_ids)
        
        if df_long.empty:
            print("No metrics found via SQL.")
            return pd.DataFrame()

        # Pivot
        # We need to preserve start_time and trial_index meta-data for deduplication
        print(f" pivoting {len(df_long)} metric records...")
        
        # Columns to keep fixed per run_id
        index_cols = ['run_id', 'parent_run_id']
        if 'start_time' in df_long.columns: index_cols.append('start_time')
        if 'trial_index' in df_long.columns: index_cols.append('trial_index')
        
        df_children = df_long.pivot_table(
            index=index_cols,
            columns='key',
            values='value'
        ).reset_index()
        
        # Merge with Parent Metadata
        meta_cols = ['run_id', 'experiment_id', 'algorithm', 'dataset']
        if 'mode' in valid_parents.columns:
            meta_cols.append('mode')
            
        parent_meta = valid_parents[meta_cols].rename(
            columns={'run_id': 'parent_run_id'}
        )
        
        df_merged = df_children.merge(parent_meta, on='parent_run_id', suffixes=('_child', ''))
        
        if deduplicate_series:
            # Drop duplicates by (mode, algorithm, dataset, trial_index) keeping latest start_time
            # We MUST include 'mode' so we don't mix Unsupervised vs Guided runs.
            # We assume valid_parents has 'mode'.
            
            subset_cols = ['algorithm', 'dataset', 'trial_index']
            if 'mode' in df_merged.columns:
                subset_cols.insert(0, 'mode')
            
            if 'trial_index' in df_merged.columns and 'start_time' in df_merged.columns:
                print(f" Deduplicating series (latest run) on subset: {subset_cols}...")
                
                # Split Valid/NaN trial_index
                mask_valid = df_merged['trial_index'].notna()
                df_valid = df_merged[mask_valid].copy()
                df_nan = df_merged[~mask_valid].copy()
                
                if not df_nan.empty:
                    print(f"  Warning: {len(df_nan)} runs have missing 'dataset_trial_index' and cannot be deduplicated properly. Keeping them.")
                
                # Sort by start_time (ascending -> last is latest)
                df_valid = df_valid.sort_values('start_time', ascending=True)
                
                before_dedupe = len(df_valid)
                df_valid = df_valid.drop_duplicates(subset=subset_cols, keep='last')
                print(f"  Deduplication: Reduced valid runs from {before_dedupe} to {len(df_valid)}.")
                
                # Recombine
                df_merged = pd.concat([df_valid, df_nan], ignore_index=True)
                
            else:
                print("Warning: Missing trial_index or start_time columns. Skipping deduplication.")

        return df_merged

    def fetch_grid_runs_raw(self, base_mode: str) -> pd.DataFrame:
        """
        Fetches all grid search runs for the corresponding base mode ('default' or 'guided').
        Maps 'default' -> 'grid_default', etc.
        Returns raw dataframe (one row per run, no deduplication).
        """
        grid_mode = f"grid_{base_mode}"
        groups = self.manager.config.get('groups', {})
        if grid_mode not in groups:
            print(f"Grid mode {grid_mode} not found in config.")
            return pd.DataFrame()
            
        keys = groups[grid_mode]
        
        # Fetch Parents
        # We can pass keys directly
        df_parents = self.fetch_parent_runs_stats(keys)
        
        # Validate (Relaxed for Grid)
        # Use strategy='merge' to keep all valid parents regardless of completion count
        df_valid_parents = self.validate_completeness(df_parents, strategy='merge')
        
        if df_valid_parents.empty:
            print("No valid grid parents found.")
            return pd.DataFrame()
            
        # Fetch Metrics (Full, no dedupe)
        df_metrics = self.fetch_metrics_for_parents(df_valid_parents, deduplicate_series=False)
        
        return df_metrics

    def get_best_grid_runs(self, df_grid_raw: pd.DataFrame, metric: str) -> pd.DataFrame:
        """
        Given raw grid runs, selects the best run (max metric) per (algorithm, dataset, trial_index).
        """
        if df_grid_raw.empty or metric not in df_grid_raw.columns:
            return pd.DataFrame()
            
        # Group by Series
        # We assume trial_index identifies the series within a dataset
        # Also need invalid trial_index handling? Assuming grid has trial_index.
        group_cols = ['algorithm', 'dataset', 'trial_index']
        
        # Check columns exist
        if not all(col in df_grid_raw.columns for col in group_cols):
             print(f"Missing grouping columns: {group_cols}")
             return pd.DataFrame()

        # Filter for valid metric and trial_index
        df = df_grid_raw.dropna(subset=[metric, 'trial_index']).copy()
        
        # Get indices of max metric
        # Use simple sort and drop_duplicates for robustness if idxmax fails on multi-index
        df = df.sort_values(metric, ascending=False)
        best_runs = df.drop_duplicates(subset=group_cols, keep='first')
        
        return best_runs

def plot_heatmap(valid_parents: pd.DataFrame, mode_name: str):
    """Plots Series count heatmap."""
    if valid_parents.empty:
        print(f"No data for {mode_name}")
        return
        
    pivot = valid_parents.pivot_table(
        index='dataset', 
        columns='algorithm', 
        values='children_finished',
        aggfunc='max'
    )
    
    # Reindex
    all_datasets = sorted(EXPECTED_SERIES.keys())
    # Filter datasets that actually exist in the data to avoid empty rows if user data is partial
    present_datasets = [d for d in all_datasets if d in valid_parents['dataset'].unique()]
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='Greens', cbar=False, linewidths=.5)
    plt.title(f"Validated Complete Series Count - {mode_name}")
    plt.tight_layout()
    plt.show()

def plot_metric_boxplot(df_merged: pd.DataFrame, metric_col: str, title: str):
    """Plots boxplot for a given metric key (e.g. 'metrics.f1_score')."""
    if metric_col not in df_merged.columns:
        print(f"Metric {metric_col} not found.")
        return
        
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df_merged, x='dataset', y=metric_col, hue='algorithm')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_cd_diagram(df_merged: pd.DataFrame, metric_col: str, mode_name: str = ""):
    """
    Plots Critical Difference diagram using aeon.
    Requires 'aeon' package.
    """
    try:
        from aeon.visualisation import plot_critical_difference
    except ImportError:
        print("aeon not installed. Skipping CD diagram.")
        return

    # Filter for metric
    if metric_col not in df_merged.columns:
        print(f"Metric {metric_col} not found.")
        return
        
    df = df_merged.dropna(subset=[metric_col]).copy()
    
    # Align by Series Index
    df = df.sort_values(['dataset', 'algorithm'])
    df['series_idx'] = df.groupby(['dataset', 'algorithm']).cumcount()
    df['instance_id'] = df['dataset'] + "_" + df['series_idx'].astype(str)
    
    # Pivot
    pivot = df.pivot(index='instance_id', columns='algorithm', values=metric_col)
    
    # Drop columns with all NaNs
    pivot = pivot.dropna(axis=1, how='all')
    
    # Impute missing (simple zero fill for visualization, ideally mean per dataset)
    pivot = pivot.fillna(0) 
    
    if pivot.empty:
        print("No valid data for CD Diagram.")
        return

    # Create figure
    # plot_critical_difference creates its own figure usually, but let's try wrapping
    try:
        fig, ax = plot_critical_difference(pivot.values, pivot.columns.tolist())
        plt.title(f"Critical Difference - {mode_name} - {metric_col}")
        plt.show()
    except Exception as e:
        print(f"Error plotting CD Diagram: {e}")

def plot_pareto(df_merged: pd.DataFrame, quality_col: str, time_col: str, title: str):
    """
    Plots Pareto frontier (Quality vs Time).
    """
    if quality_col not in df_merged.columns or time_col not in df_merged.columns:
        print(f"Metrics {quality_col} or {time_col} not found.")
        return
        
    # Aggregate by algorithm (mean across all datasets)
    agg = df_merged.groupby('algorithm')[[quality_col, time_col]].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=agg, x=time_col, y=quality_col, hue='algorithm', s=100, style='algorithm')
    
    # Annotate
    for i, row in agg.iterrows():
        plt.text(row[time_col], row[quality_col]+0.01, row['algorithm'], fontsize=9, ha='center')
        
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel(f"Time ({time_col}) - Log Scale")
    plt.ylabel(quality_col)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
