"""Test SNLDSDetector on TSSB benchmark series with increased training steps."""

import sys
import os
import time
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ensure both tsseg and tsseg_exp are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../tsseg-exp/src"))

from tsseg.algorithms.snlds import SNLDSDetector
from tsseg_exp.data_loading.loaders import load_dataset


def compute_ari(y_true, y_pred):
    """Compute Adjusted Rand Index (subsampled for large n)."""
    n = len(y_true)
    if n < 2:
        return 0.0
    # Subsample for speed
    if n > 1000:
        idx = np.linspace(0, n - 1, 1000, dtype=int)
        y_true, y_pred = y_true[idx], y_pred[idx]
        n = 1000

    from itertools import combinations
    tp = fp = fn = tn = 0
    for i, j in combinations(range(n), 2):
        same_true = y_true[i] == y_true[j]
        same_pred = y_pred[i] == y_pred[j]
        if same_true and same_pred:
            tp += 1
        elif not same_true and same_pred:
            fp += 1
        elif same_true and not same_pred:
            fn += 1
        else:
            tn += 1
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    return num / den if den != 0 else 0.0


def compute_nmi(y_true, y_pred):
    """Compute Normalised Mutual Information."""
    n = len(y_true)

    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / n
        return -np.sum(p * np.log(p + 1e-12))

    h_true, h_pred = entropy(y_true), entropy(y_pred)
    if h_true == 0 or h_pred == 0:
        return 0.0

    labels_true, labels_pred = np.unique(y_true), np.unique(y_pred)
    mi = 0.0
    for lt in labels_true:
        for lp in labels_pred:
            n_ij = ((y_true == lt) & (y_pred == lp)).sum()
            if n_ij > 0:
                n_i = (y_true == lt).sum()
                n_j = (y_pred == lp).sum()
                mi += (n_ij / n) * np.log((n * n_ij) / (n_i * n_j + 1e-12) + 1e-12)
    return mi / np.sqrt(h_true * h_pred)


# Selected TSSB series covering diverse characteristics
SERIES_SELECTION = [
    # (name, n_states_override or None to auto-detect)
    ("CBF", None),               # 960pts, 3 states, 2 CPs — short, classic
    ("Coffee", None),            # 1000pts, 2 states, 1 CP — simple
    ("Adiac", None),             # 1408pts, 4 states, 3 CPs
    ("InsectWingbeatSound", None),  # 1280pts, 4 states, 3 CPs
    ("TwoLeadECG", None),       # 471pts, 2 states, 1 CP — very short
]

N_TRAIN_STEPS = 3000  # 6x vs previous 500


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data_root = os.path.join(os.path.dirname(__file__), "../../../../tsseg-exp/data/")
    data_root = os.path.abspath(data_root)

    # Load all TSSB
    print(f"Loading TSSB from {data_root}...")
    X_list, y_list = load_dataset("tssb", data_root=data_root)

    # Get series names
    desc_path = os.path.join(data_root, "tssb", "desc.txt")
    names = []
    with open(desc_path) as f:
        for line in f:
            names.append(line.strip().split(",")[0])

    # Build name->index
    name_to_idx = {name: i for i, name in enumerate(names)}

    results = []
    total_t0 = time.time()

    for series_name, n_states_override in SERIES_SELECTION:
        idx = name_to_idx.get(series_name)
        if idx is None:
            print(f"\n[SKIP] Series '{series_name}' not found in TSSB")
            continue

        X, y = X_list[idx], y_list[idx]
        n_states = n_states_override or len(np.unique(y))
        n_cps_true = int(np.sum(np.diff(y) != 0))

        print(f"\n{'='*60}")
        print(f"TSSB: {series_name}")
        print(f"  Shape: {X.shape}, true states: {n_states}, true CPs: {n_cps_true}")
        print(f"  Training steps: {N_TRAIN_STEPS}")

        det = SNLDSDetector(
            n_states=n_states,
            hidden_dim=8,
            rnn_dim=4,
            n_train_steps=N_TRAIN_STEPS,
            learning_rate=1e-3,
            verbose=1,
            random_state=42,
        )

        t0 = time.time()
        labels = det.fit_predict(X)
        elapsed = time.time() - t0

        ari = compute_ari(y, labels)
        nmi = compute_nmi(y, labels)
        n_cps_pred = int(np.sum(np.diff(labels) != 0))

        print(f"  Predicted labels: {np.unique(labels)}")
        print(f"  Time: {elapsed:.1f}s ({elapsed/N_TRAIN_STEPS*1000:.0f}ms/step)")
        print(f"  ARI: {ari:.4f}")
        print(f"  NMI: {nmi:.4f}")
        print(f"  CPs: {n_cps_pred} predicted vs {n_cps_true} true")

        results.append({
            "name": series_name,
            "T": X.shape[0],
            "n_states": n_states,
            "ari": ari,
            "nmi": nmi,
            "cps_pred": n_cps_pred,
            "cps_true": n_cps_true,
            "time": elapsed,
        })

    # Summary table
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"SUMMARY — {N_TRAIN_STEPS} training steps")
    print(f"{'='*60}")
    print(f"{'Series':<25} {'T':>5} {'K':>3} {'ARI':>7} {'NMI':>7} {'CPs':>8} {'Time':>8}")
    print("-" * 66)
    for r in results:
        cp_str = f"{r['cps_pred']}/{r['cps_true']}"
        print(f"{r['name']:<25} {r['T']:>5} {r['n_states']:>3} {r['ari']:>7.4f} {r['nmi']:>7.4f} {cp_str:>8} {r['time']:>7.1f}s")
    print("-" * 66)
    ari_mean = np.mean([r["ari"] for r in results]) if results else 0
    nmi_mean = np.mean([r["nmi"] for r in results]) if results else 0
    print(f"{'MEAN':<25} {'':>5} {'':>3} {ari_mean:>7.4f} {nmi_mean:>7.4f} {'':>8} {total_elapsed:>7.1f}s")


if __name__ == "__main__":
    main()
