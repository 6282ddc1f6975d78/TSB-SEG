import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors

# =============================================================================
# 1. Environment & MLflow Setup
# =============================================================================

# Ensure we can import local modules
current_dir = Path(os.getcwd())
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from mlflow_manager import MLflowBenchmarkManager
from benchmark_analysis import BenchmarkAnalyzer

# Setup Paths & URI
PROJECT_ROOT = Path("..").resolve()
MLFLOW_DB_PATH = PROJECT_ROOT / "results/mlflow_snapshot.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
CONFIG_PATH = PROJECT_ROOT / "configs/benchmark_config.yaml"

# Output Directory Setup
OUTPUT_DIR = "../figures/param_tuning"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Figures will be saved to: {os.path.abspath(OUTPUT_DIR)}")

# Initialize Manager
print(f"Tracking URI: {TRACKING_URI}")
manager = MLflowBenchmarkManager(CONFIG_PATH, tracking_uri=TRACKING_URI)
analyzer = BenchmarkAnalyzer(manager)
print("Environment Setup Complete.")

# =============================================================================
# 2. Global Style Registry & Config
# =============================================================================

# Expanded Color Pool (~90 distinct colors)
EXPANDED_COLOR_POOL = (
    list(plt.get_cmap('tab20').colors) +
    list(plt.get_cmap('tab20b').colors) +
    list(plt.get_cmap('tab20c').colors) +
    list(plt.get_cmap('Dark2').colors) +
    list(plt.get_cmap('Set1').colors) +
    list(plt.get_cmap('Paired').colors)
)

# Marker Pool (Filled markers only)
MARKER_POOL = ['o', 's', 'D', '^', 'v', 'X', 'P', '*', 'h', 'H', 'p', '<', '>', 'd', '8']

ALGO_REGISTRY = {}

def init_global_registry(analyzer, modes=['default', 'guided']):
    """Scans database to register all algorithms with consistent colors/markers."""
    global ALGO_REGISTRY
    all_algos = set()
    print("Initializing Global Style Registry... Scanning database...")
    
    for mode in modes:
        # Scan Baseline
        grp_keys = analyzer.manager.config.get('groups', {}).get(mode, [])
        if grp_keys:
            df_p = analyzer.fetch_parent_runs_stats(grp_keys)
            if not df_p.empty and 'algorithm' in df_p.columns:
                all_algos.update(df_p['algorithm'].dropna().unique())
        
        # Scan Grid
        df_grid = analyzer.fetch_grid_runs_raw(mode)
        if not df_grid.empty and 'algorithm' in df_grid.columns:
            all_algos.update(df_grid['algorithm'].dropna().unique())
            
    sorted_algos = sorted(list(all_algos))
    ALGO_REGISTRY = {} 
    for idx, name in enumerate(sorted_algos):
        ALGO_REGISTRY[name] = {
            'color': EXPANDED_COLOR_POOL[idx % len(EXPANDED_COLOR_POOL)],
            'marker': MARKER_POOL[idx % len(MARKER_POOL)]
        }
    print("Global Registry Ready.")

def get_style_dicts(algo_names=None):
    """Returns palette and markers dicts."""
    global ALGO_REGISTRY
    if not ALGO_REGISTRY:
        print("Warning: Registry empty. Run init_global_registry first.")
    
    if algo_names is None:
        target_names = sorted(ALGO_REGISTRY.keys())
    else:
        target_names = sorted(list(set(algo_names)))
        for name in target_names:
            if name not in ALGO_REGISTRY:
                idx = len(ALGO_REGISTRY)
                ALGO_REGISTRY[name] = {
                    'color': EXPANDED_COLOR_POOL[idx % len(EXPANDED_COLOR_POOL)],
                    'marker': MARKER_POOL[idx % len(MARKER_POOL)]
                }
    
    palette = {name: ALGO_REGISTRY[name]['color'] for name in target_names}
    markers = {name: ALGO_REGISTRY[name]['marker'] for name in target_names}
    return palette, markers

# Initialize Registry Once
init_global_registry(analyzer)

# =============================================================================
# 3. Data Helper Functions
# =============================================================================

def get_plot_data(mode, metric, target_algos=None, exclude_algos=None, strict_fairness=True):
    """Fetches data for Scatter Plots."""
    grp_keys = analyzer.manager.config.get('groups', {}).get(mode, [])
    if not grp_keys: return pd.DataFrame(), None, None
        
    df_p = analyzer.fetch_parent_runs_stats(grp_keys)
    df_valid = analyzer.validate_completeness(df_p)
    df_base = analyzer.fetch_metrics_for_parents(df_valid, deduplicate_series=True)
    df_grid_raw = analyzer.fetch_grid_runs_raw(mode)
    df_grid_best = analyzer.get_best_grid_runs(df_grid_raw, metric)
    
    merge_cols = ['algorithm', 'dataset', 'trial_index']
    df_merged = pd.merge(df_base, df_grid_best, on=merge_cols, suffixes=('_base', '_grid'), how='inner')
    df_merged = df_merged[df_merged['dataset'] != 'pamap2']
    
    if strict_fairness:
        all_datasets = df_merged['dataset'].unique()
        n_required = len(all_datasets)
        coverage = df_merged.groupby('algorithm')['dataset'].nunique()
        valid_algos = coverage[coverage == n_required].index.tolist()
        df_merged = df_merged[df_merged['algorithm'].isin(valid_algos)]
    
    if target_algos:
        df_merged = df_merged[df_merged['algorithm'].isin(target_algos)]
    if exclude_algos:
        df_merged = df_merged[~df_merged['algorithm'].isin(exclude_algos)]
        
    col_base = f"{metric}_base"
    col_grid = f"{metric}_grid"
    col_opt = f"{metric}_optimized"
    df_merged[col_opt] = df_merged[[col_base, col_grid]].max(axis=1)
    
    df_plot = df_merged.groupby(['algorithm'])[[col_base, col_opt]].mean().reset_index()
    return df_plot, col_base, col_opt


def get_bar_data_sorted(mode, metric, target_algos=None, exclude_algos=None, strict_fairness=True):
    """
    Fetches data for Bar Plots, sorted by improvement.
    Includes Strict Fairness Check to ensure algorithms cover ALL datasets.
    """
    # 1. Fetch Data
    grp_keys = analyzer.manager.config.get('groups', {}).get(mode, [])
    if not grp_keys: return pd.DataFrame()
        
    df_p = analyzer.fetch_parent_runs_stats(grp_keys)
    df_base = analyzer.fetch_metrics_for_parents(analyzer.validate_completeness(df_p), deduplicate_series=True)
    df_grid_raw = analyzer.fetch_grid_runs_raw(mode)
    df_grid_best = analyzer.get_best_grid_runs(df_grid_raw, metric)
    
    # 2. Merge
    merge_cols = ['algorithm', 'dataset', 'trial_index']
    df_merged = pd.merge(df_base, df_grid_best, on=merge_cols, suffixes=('_base', '_grid'), how='inner')
    
    # 3. Filter Dataset
    df_merged = df_merged[df_merged['dataset'] != 'pamap2']
    
    # =========================================================
    # 4. Strict Fairness Check (Added)
    # =========================================================
    if strict_fairness:
        all_datasets = df_merged['dataset'].unique()
        n_required = len(all_datasets)
        
        coverage = df_merged.groupby('algorithm')['dataset'].nunique()
        valid_algos = coverage[coverage == n_required].index.tolist()
        
        dropped = set(coverage.index) - set(valid_algos)
        if dropped:
            print(f"[{mode} - BarPlot] Dropped (Incomplete Runs): {dropped}")
            
        df_merged = df_merged[df_merged['algorithm'].isin(valid_algos)]
    # =========================================================
    
    # 5. Filter Algorithms (User Specified)
    if target_algos:
        df_merged = df_merged[df_merged['algorithm'].isin(target_algos)]
    if exclude_algos:
        df_merged = df_merged[~df_merged['algorithm'].isin(exclude_algos)]
        
    # 6. Calculate Improvement
    col_base = f"{metric}_base"
    col_grid = f"{metric}_grid"
    col_opt = f"{metric}_optimized"
    df_merged[col_opt] = df_merged[[col_base, col_grid]].max(axis=1)
    
    df_agg = df_merged.groupby('algorithm')[[col_base, col_opt]].mean().reset_index()
    df_agg['improvement'] = df_agg[col_opt] - df_agg[col_base]
    
    # Sort Ascending
    df_agg = df_agg.sort_values(by='improvement', ascending=True)
    return df_agg

# =============================================================================
# 4. Plotting: Scatter - Bidirectional Covering Score
# =============================================================================
print("Generating Scatter: Bidirectional Covering...")

metric_name = 'metrics.bidirectional_covering_score'
target_algos = ['binseg', 'clap', 'clasp', 'e2usd', 'ggs', 'ticc', 'bottomup', 'fluss', 'icid', 'time2state']

df_l, base_col, opt_col = get_plot_data('default', metric_name, target_algos=target_algos)
df_r, _, _ = get_plot_data('guided', metric_name, target_algos=target_algos)

palette, markers = get_style_dicts(None)

fig, axes = plt.subplots(1, 2, figsize=(32, 11))
sns.set_context("notebook", font_scale=1.4)

# Left: Default
ax = axes[0]
sns.scatterplot(
    data=df_l, x=base_col, y=opt_col,
    hue='algorithm', style='algorithm',
    palette=palette, markers=markers,
    s=3000, alpha=0.75, ax=ax, legend='brief'
)

# Right: Guided
ax = axes[1]
sns.scatterplot(
    data=df_r, x=base_col, y=opt_col,
    hue='algorithm', style='algorithm',
    palette=palette, markers=markers,
    s=3000, alpha=0.75, ax=ax, legend='brief'
)

# Common Formatting
lims = [0.15, 0.9] 
for ax in axes:
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=3, label='No Improvement')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Baseline (Default Param)", fontsize=45)
    ax.set_ylabel("Best Achievable (Grid)", fontsize=45)
    ax.tick_params(axis='both', which='major', labelsize=40)

# Legend Logic
h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
combined = dict(zip(l1 + l2, h1 + h2))

axes[0].get_legend().remove()
axes[1].get_legend().remove()

fig.legend(
    combined.values(), combined.keys(),
    loc='lower center', 
    bbox_to_anchor=(0.5, 0.87),
    ncol=6, 
    fontsize=40, 
    title_fontsize=16,
    borderaxespad=0.,
    frameon=True 
)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save
save_path = os.path.join(OUTPUT_DIR, "param_tuning_scatter_bi_covering.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Saved: {save_path}")
plt.show()

# =============================================================================
# 5. Plotting: Scatter - State Matching Score
# =============================================================================
print("Generating Scatter: State Matching...")

# 【关键修正】使用用户指定的正确 Metric Key
metric_name = 'metrics.state_matching_score_score' 
exclude_algos = ['random'] 

df_l, base_col, opt_col = get_plot_data('default', metric_name, exclude_algos=exclude_algos)
df_r, _, _ = get_plot_data('guided', metric_name, exclude_algos=exclude_algos)

palette, markers = get_style_dicts(None)

fig, axes = plt.subplots(1, 2, figsize=(32, 11))
sns.set_context("notebook", font_scale=1.4)

# Left: Default
ax = axes[0]
sns.scatterplot(
    data=df_l, x=base_col, y=opt_col,
    hue='algorithm', style='algorithm',
    palette=palette, markers=markers,
    s=3000, alpha=0.75, ax=ax, legend='brief'
)

# Right: Guided
ax = axes[1]
sns.scatterplot(
    data=df_r, x=base_col, y=opt_col,
    hue='algorithm', style='algorithm',
    palette=palette, markers=markers,
    s=3000, alpha=0.75, ax=ax, legend='brief'
)

# Common Formatting
lims = [0.1, 0.9] 
for ax in axes:
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=3, label='No Improvement')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel("Baseline (Default Param)", fontsize=45)
    ax.set_ylabel("Best Achievable (Grid)", fontsize=45)
    ax.tick_params(axis='both', which='major', labelsize=40)

# Legend Logic (Sorted + No Improvement at end)
h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
combined = dict(zip(l1 + l2, h1 + h2))

special_key = 'No Improvement'
special_handle = None
if special_key in combined:
    special_handle = combined.pop(special_key)

sorted_names = sorted(combined.keys())
sorted_handles = [combined[name] for name in sorted_names]

if special_handle:
    sorted_names.append(special_key)
    sorted_handles.append(special_handle)

axes[0].get_legend().remove()
axes[1].get_legend().remove()

fig.legend(
    sorted_handles, sorted_names,
    loc='lower center', 
    bbox_to_anchor=(0.5, 0.87),
    ncol=6, 
    fontsize=40, 
    title_fontsize=16,
    borderaxespad=0.,
    frameon=True 
)

plt.tight_layout()
plt.subplots_adjust(top=0.85)

# Save
save_path = os.path.join(OUTPUT_DIR, "param_tuning_scatter_sms.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Saved: {save_path}")
plt.show()


# =============================================================================
# 6. Plotting: Bar - Bidirectional Covering Score
# =============================================================================
print("Generating Bar: Bidirectional Covering...")

metric_name = 'metrics.bidirectional_covering_score'
target_algos = ['binseg', 'clap', 'clasp', 'e2usd', 'ggs', 'ticc', 'bottomup', 'fluss', 'icid', 'time2state']
exclude_algos = None

# Bar Settings
FIXED_BAR_WIDTH = 0.6
BAR_EDGE_COLOR = 'black'
BAR_LINE_WIDTH = 2.0
Y_LIMITS = (0, 0.4)
BAR_ALPHA = 0.7

df_l = get_bar_data_sorted('default', metric_name, target_algos=target_algos, exclude_algos=exclude_algos) 
df_r = get_bar_data_sorted('guided', metric_name, target_algos=target_algos, exclude_algos=exclude_algos)
palette, _ = get_style_dicts(None)

# Unified X-Axis Range
max_bars = max(len(df_l), len(df_r))
common_xlim = (-0.6, max_bars - 0.4)

fig, axes = plt.subplots(1, 2, figsize=(32, 11))

def draw_bar_subplot(ax, df, title_mode):
    bar_colors = [palette.get(algo, '#333333') for algo in df['algorithm']]
    bars = ax.bar(
        x=np.arange(len(df)), 
        height=df['improvement'],
        color=bar_colors,
        width=FIXED_BAR_WIDTH,
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_LINE_WIDTH,
        alpha=BAR_ALPHA 
    )
    
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df['algorithm'], rotation=45, ha='right', fontsize=35)
    ax.set_title(f"Mode: {title_mode}", fontsize=45, y=1.22)
    ax.set_ylabel(r"Performance Gain ($\Delta$)", fontsize=40)
    ax.set_ylim(Y_LIMITS)
    ax.set_xlim(common_xlim)
    ax.tick_params(axis='y', which='major', labelsize=35)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.005,
            f'{height:.3f}',
            ha='center', va='bottom', fontsize=28
        )

draw_bar_subplot(axes[0], df_l, "Default")
draw_bar_subplot(axes[1], df_r, "Guided")

plt.tight_layout()
plt.subplots_adjust(top=0.88, wspace=0.15)

# Save
save_path = os.path.join(OUTPUT_DIR, "param_tuning_bar_bi_covering.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Saved: {save_path}")
plt.show()

# =============================================================================
# 7. Plotting: Bar - State Matching Score
# =============================================================================
print("Generating Bar: State Matching...")

# 【关键修正】使用用户指定的正确 Metric Key
metric_name = 'metrics.state_matching_score_score'
target_algos = None
exclude_algos = ['random']

# Bar Settings (Same as above)
FIXED_BAR_WIDTH = 0.6
BAR_EDGE_COLOR = 'black'
BAR_LINE_WIDTH = 2.0
Y_LIMITS = (0, 0.4)
BAR_ALPHA = 0.7

df_l = get_bar_data_sorted('default', metric_name, target_algos=target_algos, exclude_algos=exclude_algos) 
df_r = get_bar_data_sorted('guided', metric_name, target_algos=target_algos, exclude_algos=exclude_algos)
palette, _ = get_style_dicts(None)

max_bars = max(len(df_l), len(df_r))
common_xlim = (-0.6, max_bars - 0.4)

fig, axes = plt.subplots(1, 2, figsize=(32, 11))

# Re-use draw_bar_subplot defined in section 6 or re-define if strictly separated
# Since this is a script, previous definition is available.
draw_bar_subplot(axes[0], df_l, "Default")
draw_bar_subplot(axes[1], df_r, "Guided")

plt.tight_layout()
plt.subplots_adjust(top=0.88, wspace=0.15)

# Save
save_path = os.path.join(OUTPUT_DIR, "param_tuning_bar_sms.png")
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Saved: {save_path}")
plt.show()