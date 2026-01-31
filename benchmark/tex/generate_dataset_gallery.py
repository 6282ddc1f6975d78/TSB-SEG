import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

# Ensure we can import local modules from src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Also ensure valid import of 'tsseg' (sibling project)
tsseg_lib_path = project_root.parent / "tsseg"
if str(tsseg_lib_path) not in sys.path:
    sys.path.append(str(tsseg_lib_path))

try:
    from tsseg_exp.datasets.loaders import load_dataset
except ImportError:
    # Fallback
    sys.path.append(str(src_path))
    from tsseg_exp.datasets.loaders import load_dataset

def compute_reoccurring(y):
    """
    Check if any state reoccurs after visiting another state.
    """
    if len(y) == 0:
        return False
    msgs = y[np.concatenate(([True], y[1:] != y[:-1]))]
    unique_states = np.unique(msgs)
    return len(msgs) > len(unique_states)

def plot_series(X, y, dataset_name, save_path):
    """
    Plots the time series and ground truth segmentation.
    Saves to save_path.
    """
    n_samples, n_dims = X.shape
    
    # Plotting config
    fig_height = 0.5
    # Determine dims to plot: Plot all dims
    plot_dims = n_dims
    
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    # Normalize X for cleaner plotting if needed, but raw is usually fine for shape
    # Just plot
    for d in range(plot_dims):
        ax.plot(X[:, d], label=f'Dim {d}', linewidth=0.8, alpha=0.8)
        
    # Plot segmentation (background color)
    # y is the state label
    unique_states = np.unique(y)
    cmap = plt.get_cmap('tab10')
    colors = {s: cmap(i % 10) for i, s in enumerate(unique_states)}
    
    # Identify segments
    changes = np.concatenate(([0], np.where(y[1:] != y[:-1])[0] + 1, [n_samples]))
    
    for i in range(len(changes) - 1):
        start, end = changes[i], changes[i+1]
        state = y[start]
        color = colors[state]
        ax.axvspan(start, end, color=color, alpha=0.2, lw=0)
        
    ax.set_xlim(0, n_samples)
    ax.set_xticks([]) # Remove x ticks for compactness
    ax.set_yticks([]) # Remove y ticks
    
    # Minimal border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    plt.tight_layout(pad=0)
    
    # Save PNG
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Save SVG
    svg_path = Path(save_path).with_suffix('.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close(fig)

def generate_tex_content(data, figures_rel_path="figures"):
    """
    Generates the LaTeX content for the table.
    """
    tex = []
    tex.append(r"\documentclass{article}")
    tex.append(r"\usepackage[landscape, margin=0.5in]{geometry}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{longtable}")
    tex.append(r"\usepackage{array}")
    tex.append(r"\usepackage[table]{xcolor}")
    tex.append(r"\begin{document}")
    
    tex.append(r"\section*{Dataset Gallery}")
    
    # Table Definition
    # Columns: Name, #, Length, Dimensions, States, CPs, Plot
    tex.append(r"\begin{longtable}{l c c c c c m{8cm}}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Dataset} & \textbf{\#} & \textbf{Length} & \textbf{D} & \textbf{K} & \textbf{CPs} & \textbf{Example Series} \\")
    tex.append(r"\midrule")
    tex.append(r"\endfirsthead")
    
    tex.append(r"\toprule")
    tex.append(r"\textbf{Dataset} & \textbf{\#} & \textbf{Length} & \textbf{D} & \textbf{K} & \textbf{CPs} & \textbf{Example Series} \\")
    tex.append(r"\midrule")
    tex.append(r"\endhead")
    
    tex.append(r"\bottomrule")
    tex.append(r"\endfoot")
    
    for row in data:
        # Escape underscores in LaTeX
        name_esc = row['dataset'].replace("_", r"\_")
        
        # Image Path relative to the tex file
        img_path = f"{figures_rel_path}/{row['dataset']}.png"
        
        # Row Content
        
        line = f"{name_esc} & {row['n_series']} & {row['length_str']} & {row['dim_str']} & {row['states_str']} & {row['cps_str']} & "
        line += r"\includegraphics[width=8cm, height=0.7cm, keepaspectratio=false]{" + img_path + r"} \\"
        
        tex.append(line)
        
    tex.append(r"\end{longtable}")
    tex.append(r"\end{document}")
    
    return "\n".join(tex)

def main():
    data_root = project_root / "data"
    config_dir = project_root / "configs" / "dataset"
    output_dir = current_dir
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    results = []

    print(f"Scanning for dataset configs in: {config_dir}")
    print(f"Loading data from: {data_root}")

    # list all yaml files
    dataset_configs = sorted(list(config_dir.glob("*.yaml")))
    
    if not dataset_configs:
        print("No config files found.")
        return

    for config_file in dataset_configs:
        dataset_name = config_file.stem

        # Exclude specific datasets
        if dataset_name in ["knot-tying", "needle-passing", "pump", "suturing"]:
             continue

        print(f"Processing {dataset_name}...")
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    X_list, y_list = load_dataset(dataset_name, data_root=data_root)
                except Exception as e:
                    print(f"  Skipping {dataset_name}: {e}")
                    continue
            
            if not X_list or len(X_list) == 0:
                print(f"  Skipping {dataset_name}: No data returned.")
                continue

            # Pick specific series for plotting if selected
            if dataset_name == "actrectut":
                X_ex = X_list[0]
                y_ex = y_list[0]
            elif dataset_name == "has":
                X_ex = X_list[134]
                y_ex = y_list[134]
            elif dataset_name == "mocap":
                X_ex = X_list[1]
                y_ex = y_list[1]
            elif dataset_name == "pamap2":
                X_ex = X_list[1]
                y_ex = y_list[1]
            elif dataset_name == "skab":
                X_ex = X_list[12]
                y_ex = y_list[12]
            elif dataset_name == "tssb":
                X_ex = X_list[20]
                y_ex = y_list[20]
            elif dataset_name == "usc-had":
                X_ex = X_list[0]
                y_ex = y_list[0]
            elif dataset_name == "utsa":
                X_ex = X_list[2]
                y_ex = y_list[2]
            else:
                X_ex = X_list[0]
                y_ex = y_list[0]

            y_int_ex = y_ex.astype(int)
            
            # Generate Plot
            save_path = figures_dir / f"{dataset_name}.png"
            plot_series(X_ex[::5], y_int_ex[::5], dataset_name, save_path)

            # Generate Truncated Plot (10k)
            save_path_10k = figures_dir / f"{dataset_name}_10k.png"
            plot_series(X_ex[:10000], y_int_ex[:10000], dataset_name, save_path_10k)

            # Generate Normalized Plot
            save_path_norm = figures_dir / f"{dataset_name}_norm.png"
            std = X_ex.std(axis=0)
            std[std == 0] = 1.0
            X_norm = (X_ex - X_ex.mean(axis=0)) / std
            plot_series(X_norm[::5], y_int_ex[::5], dataset_name, save_path_norm)

            # Generate Normalized Plot (10k)
            save_path_norm_10k = figures_dir / f"{dataset_name}_norm_10k.png"
            plot_series(X_norm[:10000], y_int_ex[:10000], dataset_name, save_path_norm_10k)

            # Aggregate Features over the whole dataset
            lengths = []
            dims_list = []
            states_list = []
            cps_list = []

            for X, y in zip(X_list, y_list):
                 if X is None or y is None: continue
                 lengths.append(X.shape[0])
                 dims_list.append(X.shape[1])
                 yt = y.astype(int)
                 states_list.append(len(np.unique(yt)))
                 # Count CPs
                 cps_list.append(len(np.where(yt[1:] != yt[:-1])[0]))
            
            def get_range_str(lst, suffix="", div=1):
                if not lst: return "-"
                mn, mx = min(lst), max(lst)
                mn_fmt = f"{mn/div:.1f}" if div != 1 else f"{mn}"
                mx_fmt = f"{mx/div:.1f}" if div != 1 else f"{mx}"
                if mn == mx: return f"{mn_fmt}{suffix}"
                return f"{mn_fmt}-{mx_fmt}{suffix}"
            
            results.append({
                "dataset": dataset_name,
                "n_series": len(X_list),
                "length_str": get_range_str(lengths, "k", 1000),
                "dim_str": get_range_str(dims_list),
                "states_str": get_range_str(states_list),
                "cps_str": get_range_str(cps_list)
            })
            print(f"  -> Generated figure and stats.")
            
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            # traceback.print_exc()

    # Generate TeX
    if results:
        print("Generating LaTeX document...")
        tex_content = generate_tex_content(results)
        
        tex_file = output_dir / "dataset_gallery.tex"
        with open(tex_file, "w") as f:
            f.write(tex_content)
            
        print(f"Done. Main file: {tex_file}")
        print(f"Figures stored in: {figures_dir}")
    else:
        print("No results to generate.")

if __name__ == "__main__":
    main()
