import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Try to setup paths to find source code
# We assume this script is in tsseg-exp/benchmark/tex/
current_dir = Path(__file__).resolve().parent
tsseg_exp_root = current_dir.parents[2] # Go up: tex -> benchmark -> tsseg-exp
src_path = tsseg_exp_root / "src"
tsseg_lib_path = tsseg_exp_root.parent / "tsseg"

# Add paths if not present
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
if str(tsseg_lib_path) not in sys.path:
    sys.path.append(str(tsseg_lib_path))

try:
    from tsseg_exp.datasets.loaders import load_dataset
except ImportError:
    print(f"Could not import tsseg_exp. Submodule path added: {src_path}")
    sys.exit(1)

def main():
    dataset_name = "mocap"
    data_root = tsseg_exp_root / "data"
    
    print(f"Loading '{dataset_name}' from {data_root}...")
    try:
        X_list, y_list = load_dataset(dataset_name, data_root=data_root)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    if not X_list or len(X_list) == 0:
        print("No data found.")
        return

    # Select the first time series
    X = X_list[0]
    y = y_list[0].astype(int)
    n_samples, n_dims = X.shape
    
    # 16:9 Ratio
    # We can set figsize in inches. e.g. 16x9.
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Plot Dimensions
    for d in range(n_dims):
        ax.plot(X[:, d], label=f'Dim {d}', linewidth=1.5, alpha=0.9)
        
    # Plot Segmentation background
    unique_states = np.unique(y)
    cmap = plt.get_cmap('tab10')
    colors = {s: cmap(i % 10) for i, s in enumerate(unique_states)}
    
    # Identify segment boundaries
    # Segments start where label changes
    changes = np.concatenate(([0], np.where(y[1:] != y[:-1])[0] + 1, [n_samples]))
    
    for i in range(len(changes) - 1):
        start, end = changes[i], changes[i+1]
        state = y[start]
        color = colors[state]
        ax.axvspan(start, end, color=color, alpha=0.2, lw=0)
        
    ax.set_xlim(0, n_samples)
    
    # Styling: Remove axes and spines for a clean graphic (similar to gallery)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout(pad=0)
    
    output_filename = "mocap_16_9.svg"
    output_path = current_dir / output_filename
    
    plt.savefig(output_path, format="svg", bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Successfully saved {output_path}")

if __name__ == "__main__":
    main()
