import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

# Ensure we can import local modules from src
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

try:
    from tsseg_exp.datasets.loaders import load_dataset
except ImportError:
    # Fallback to direct import if package is not installed in editable mode
    sys.path.append(str(src_path))
    from tsseg_exp.datasets.loaders import load_dataset

def compute_reoccurring(y):
    """
    Check if any state reoccurs after visiting another state.
    True if sequence is like A -> B -> A.
    False if sequence is like A -> B -> C.
    """
    if len(y) == 0:
        return False
    # Compress consecutive duplicates to get sequence of segments
    # e.g. [0, 0, 1, 1, 2, 0] -> [0, 1, 2, 0]
    msgs = y[np.concatenate(([True], y[1:] != y[:-1]))]
    
    unique_states = np.unique(msgs)
    # If segment sequence length > unique states, a state must have repeated disjointly
    return len(msgs) > len(unique_states)

def main():
    data_root = project_root / "data"
    config_dir = project_root / "configs" / "dataset"
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
        print(f"\nProcessing {dataset_name}...")
        
        try:
            # Load all series for this dataset
            # We catch exceptions as some datasets might not be downloaded locally
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # load_dataset returns (list[X], list[y]) when no params are provided
                    X_list, y_list = load_dataset(dataset_name, data_root=data_root)
                except FileNotFoundError:
                    print(f"  Skipping {dataset_name}: Data not found.")
                    continue
                except Exception as e:
                    print(f"  Skipping {dataset_name}: Load error ({str(e)})")
                    continue
            
            if not X_list:
                print(f"  Skipping {dataset_name}: No data returned.")
                continue

            # Handle case where single result is returned (though load_all usually returns list)
            if isinstance(X_list, np.ndarray) and isinstance(y_list, np.ndarray):
                # Check if it is a list of arrays or a single array which is one series
                # Usually load_all returns list. If it's a single series disguised as array of arrays...
                # We assume load_all returns list.
                pass 
                
            count = 0
            for i, (X, y) in enumerate(zip(X_list, y_list)):
                if X is None or y is None:
                    continue
                
                # Basic Properties
                n_samples, n_dims = X.shape
                
                # Labels
                # Ensure y is integer for unique counting
                y_int = y.astype(int)
                
                # Change Points (boundaries where label changes)
                # Count transitions
                transitions = np.where(y_int[1:] != y_int[:-1])[0]
                n_cps = len(transitions)
                
                # States
                unique_states = np.unique(y_int)
                n_states = len(unique_states)
                
                # Reoccurring
                is_reoccurring = compute_reoccurring(y_int)
                
                results.append({
                    "dataset": dataset_name,
                    "series_index": i,
                    "length": n_samples,
                    "dimensions": n_dims,
                    "n_change_points": n_cps,
                    "n_states": n_states,
                    "reoccurring": is_reoccurring
                })
                count += 1
            
            print(f"  Extracted features for {count} series.")
                
        except Exception as e:
            print(f"  Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    # Save Results
    if results:
        df = pd.DataFrame(results)
        
        # Summary
        summary = df.groupby("dataset").agg(
            n_series=("length", "count"),
            mean_len=("length", lambda x: f"{x.mean():.1f}"),
            median_len=("length", lambda x: f"{x.median():.1f}"),
            dim=("dimensions", "mean"),
            mean_cps=("n_change_points", lambda x: f"{x.mean():.1f}"),
            mean_states=("n_states", lambda x: f"{x.mean():.1f}"),
            pct_reocc=("reoccurring", lambda x: f"{x.mean()*100:.1f}%")
        )
        
        print("\n=== Dataset Summary ===")
        print(summary)
        
        output_path = current_dir / "dataset_features.csv"
        df.to_csv(output_path, index=False)
        print(f"\nDetailed features saved to: {output_path}")
    else:
        print("No features extracted.")

if __name__ == "__main__":
    main()
