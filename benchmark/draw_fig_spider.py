import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import matplotlib.colors as mcolors

# =============================================================================
# 1. Environment & Output Configuration
# =============================================================================

current_dir = Path(os.getcwd())
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from mlflow_manager import MLflowBenchmarkManager
    from benchmark_analysis import BenchmarkAnalyzer
except ImportError:
    print("Warning: Local modules (mlflow_manager, benchmark_analysis) not found.")

OUTPUT_DIR = "../figures/spider_performance"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output Directory: {os.path.abspath(OUTPUT_DIR)}")

PROJECT_ROOT = Path("..").resolve()
MLFLOW_DB_PATH = PROJECT_ROOT / "results/mlflow_snapshot.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH}"
CONFIG_PATH = PROJECT_ROOT / "configs/benchmark_config.yaml"

print(f"Tracking URI: {TRACKING_URI}")
manager = MLflowBenchmarkManager(CONFIG_PATH, tracking_uri=TRACKING_URI)
analyzer = BenchmarkAnalyzer(manager)
print("Environment Setup Complete.")

# =============================================================================
# 2. Data Helper Functions
# =============================================================================

def get_radar_matrix(mode, metric, target_algos=None, exclude_algos=None, strict_fairness=True):
    """
    Fetches data and transforms it into a Pivot DataFrame for Radar Charts.
    """
    grp_keys = analyzer.manager.config.get('groups', {}).get(mode, [])
    if not grp_keys: return pd.DataFrame()
        
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
        
        dropped = set(coverage.index) - set(valid_algos)
        if dropped:
            print(f"[{mode}] Fairness Filter: Dropped algorithms {dropped}")
            
        df_merged = df_merged[df_merged['algorithm'].isin(valid_algos)]

    if target_algos:
        df_merged = df_merged[df_merged['algorithm'].isin(target_algos)]
    if exclude_algos:
        df_merged = df_merged[~df_merged['algorithm'].isin(exclude_algos)]
        
    col_base = f"{metric}_base"
    col_grid = f"{metric}_grid"
    col_opt = f"{metric}_optimized"
    df_merged[col_opt] = df_merged[[col_base, col_grid]].max(axis=1)
    
    df_pivot = df_merged.groupby(['algorithm', 'dataset'])[col_opt].mean().unstack()
    return df_pivot

# =============================================================================
# 3. Style & Color Configuration
# =============================================================================

CUSTOM_PALETTE = {
    'amoc': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765), 
    'autoplait': (0.6823529411764706, 0.7803921568627451, 0.9098039215686274), 
    'binseg': (1.0, 0.4980392156862745, 0.054901960784313725), 
    'bocd': (1.0, 0.7333333333333333, 0.47058823529411764), 
    'bottomup': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), 
    'clap': (0.596078431372549, 0.8745098039215686, 0.5411764705882353), 
    'clasp': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), 
    'dynp': (1.0, 0.596078431372549, 0.5882352941176471), 
    'e2usd': (0.5803921568627451, 0.403921568627451, 0.7411764705882353), 
    'eagglo': (0.7725490196078432, 0.6901960784313725, 0.8352941176470589), 
    'espresso': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), 
    'fluss': (0.7686274509803922, 0.611764705882353, 0.5803921568627451), 
    'ggs': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), 
    'hdp-hsmm': (0.9686274509803922, 0.7137254901960784, 0.8235294117647058), 
    'hdp-hsmm-legacy': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), 
    'hidalgo': (0.7803921568627451, 0.7803921568627451, 0.7803921568627451), 
    'icid': (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), 
    'igts': (0.8588235294117647, 0.8588235294117647, 0.5529411764705883), 
    'kcpd': (0.09019607843137255, 0.7450980392156863, 0.8117647058823529), 
    'patss': (0.6196078431372549, 0.8549019607843137, 0.8980392156862745), 
    'pelt': (0.2235294117647059, 0.23137254901960785, 0.4745098039215686), 
    'prophet': (0.3215686274509804, 0.32941176470588235, 0.6392156862745098), 
    'random': (0.4196078431372549, 0.43137254901960786, 0.8117647058823529), 
    'tglad': (0.611764705882353, 0.6196078431372549, 0.8705882352941177), 
    'ticc': (0.38823529411764707, 0.4745098039215686, 0.2235294117647059), 
    'time2state': (0.5490196078431373, 0.6352941176470588, 0.3215686274509804), 
    'tire': (0.7098039215686275, 0.8117647058823529, 0.4196078431372549), 
    'tscp2': (0.807843137254902, 0.8588235294117647, 0.611764705882353), 
    'vsax': (0.5490196078431373, 0.42745098039215684, 0.19215686274509805), 
    'window': (0.7411764705882353, 0.6196078431372549, 0.2235294117647059)
}

# =============================================================================
# 4. Visualization 1: Bidirectional Covering Score
# =============================================================================

print("\n>>> Generating Radar Chart: Bidirectional Covering Score...")

# Configuration
metric_name = 'metrics.bidirectional_covering_score'
target_algos = ['binseg', 'clap', 'clasp', 'e2usd', 'ggs', 'ticc', 'bottomup', 'fluss', 'icid', 'time2state']
SAVE_FILENAME = "param_tuning_radar_bi_covering.png"

# 1. Fetch Data
df_pivot_l = get_radar_matrix('default', metric_name, target_algos=target_algos, strict_fairness=True)
df_pivot_r = get_radar_matrix('guided', metric_name, target_algos=target_algos, strict_fairness=True)

# 2. Align Axes
all_datasets = sorted(list(set(df_pivot_l.columns) | set(df_pivot_r.columns)))
for d in all_datasets:
    if d not in df_pivot_l.columns: df_pivot_l[d] = 0.0
    if d not in df_pivot_r.columns: df_pivot_r[d] = 0.0

df_pivot_l = df_pivot_l[all_datasets]
df_pivot_r = df_pivot_r[all_datasets]

# 3. Setup Geometry
categories = all_datasets
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] 

# 4. Initialize Plot
fig, axes = plt.subplots(1, 2, figsize=(32, 16), subplot_kw={'projection': 'polar'})

def plot_radar_subplot(ax, df_pivot, title_mode):
    """
    Helper to draw a single radar subplot.
    Updated to use ax-specific methods for fonts to ensure consistency.
    """
    ax.set_theta_offset(pi / 2) 
    ax.set_theta_direction(-1) 
    
    # Draw X-Axis (Dataset Labels)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=43)
    ax.tick_params(axis='x', which='major', pad=60) 
    
    # Draw Y-Axis (Grid Lines) & Labels (The numbers 0.2, 0.4...)
    # [FIX] Use ax.set_yticks instead of plt.yticks to ensure it applies to THIS subplot
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    # [FIX] Set font size explicitly here
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=30)
    ax.set_ylim(0, 1.0)
    
    # Plot Algorithms
    for algo_name, row in df_pivot.iterrows():
        values = row.tolist()
        values += values[:1] 
        color = CUSTOM_PALETTE.get(algo_name, '#333333')
        ax.plot(angles, values, linewidth=4, linestyle='solid', label=algo_name, color=color)
        ax.fill(angles, values, color=color, alpha=0.05) 
    
    ax.set_title(f"Mode: {title_mode}", size=50, y=1.35)

# 5. Draw Subplots
plot_radar_subplot(axes[0], df_pivot_l, "Default")
plot_radar_subplot(axes[1], df_pivot_r, "Guided")

# 6. Global Legend
h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
combined = dict(zip(l1 + l2, h1 + h2))
sorted_names = sorted(combined.keys())
sorted_handles = [combined[name] for name in sorted_names]

fig.legend(
    sorted_handles, sorted_names,
    loc='upper center',       
    bbox_to_anchor=(0.5, 0.98), 
    ncol=6, 
    fontsize=43, 
    frameon=True 
)

# 7. Save
plt.tight_layout()
plt.subplots_adjust(top=0.75, wspace=0.3) 
save_path = os.path.join(OUTPUT_DIR, SAVE_FILENAME)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Saved Figure: {save_path}")
plt.show()


# =============================================================================
# 5. Visualization 2: State Matching Score
# =============================================================================

print("\n>>> Generating Radar Chart: State Matching Score...")

# Configuration
metric_name = 'metrics.state_matching_score_score'
target_algos = None
exclude_algos = ['random']
SAVE_FILENAME = "param_tuning_radar_sms.png"

# 1. Fetch Data
df_pivot_l = get_radar_matrix('default', metric_name, target_algos=target_algos, exclude_algos=exclude_algos, strict_fairness=True)
df_pivot_r = get_radar_matrix('guided', metric_name, target_algos=target_algos, exclude_algos=exclude_algos, strict_fairness=True)

# 2. Align Axes
all_datasets = sorted(list(set(df_pivot_l.columns) | set(df_pivot_r.columns)))
for d in all_datasets:
    if d not in df_pivot_l.columns: df_pivot_l[d] = 0.0
    if d not in df_pivot_r.columns: df_pivot_r[d] = 0.0

df_pivot_l = df_pivot_l[all_datasets]
df_pivot_r = df_pivot_r[all_datasets]

# 3. Setup Geometry
categories = all_datasets
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1] 

# 4. Initialize Plot
fig, axes = plt.subplots(1, 2, figsize=(32, 16), subplot_kw={'projection': 'polar'})

# 5. Draw Subplots (Reuse corrected helper function)
plot_radar_subplot(axes[0], df_pivot_l, "Default")
plot_radar_subplot(axes[1], df_pivot_r, "Guided")

# 6. Global Legend
h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
combined = dict(zip(l1 + l2, h1 + h2))
sorted_names = sorted(combined.keys())
sorted_handles = [combined[name] for name in sorted_names]

fig.legend(
    sorted_handles, sorted_names,
    loc='upper center',       
    bbox_to_anchor=(0.5, 0.98), 
    ncol=6, 
    fontsize=43, 
    frameon=True 
)

# 7. Save
plt.tight_layout()
plt.subplots_adjust(top=0.75, wspace=0.3) 
save_path = os.path.join(OUTPUT_DIR, SAVE_FILENAME)
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Saved Figure: {save_path}")
plt.show()