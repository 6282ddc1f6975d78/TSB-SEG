"""Quick smoke test for the SNLDSDetector on MoCap and synthetic switching data."""

import sys
import os
import time
import numpy as np

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Ensure tsseg is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tsseg.algorithms.snlds import SNLDSDetector
from tsseg.data.datasets import load_mocap


def compute_ari(y_true, y_pred):
    """Compute Adjusted Rand Index without sklearn dependency."""
    from itertools import combinations
    n = len(y_true)
    if n < 2:
        return 0.0

    # Build contingency via pair counting
    tp, fp, fn, tn = 0, 0, 0, 0
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

    # ARI = 2(TP*TN - FP*FN) / ((TP+FP)(FP+TN) + (TP+FN)(FN+TN))
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fp) * (fp + tn) + (tp + fn) * (fn + tn)
    if den == 0:
        return 0.0
    return num / den


def compute_nmi(y_true, y_pred):
    """Compute Normalised Mutual Information."""
    n = len(y_true)

    # Compute entropies
    def entropy(labels):
        _, counts = np.unique(labels, return_counts=True)
        p = counts / n
        return -np.sum(p * np.log(p + 1e-12))

    h_true = entropy(y_true)
    h_pred = entropy(y_pred)

    if h_true == 0 or h_pred == 0:
        return 0.0

    # Mutual information via contingency
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    mi = 0.0
    for lt in labels_true:
        for lp in labels_pred:
            mask = (y_true == lt) & (y_pred == lp)
            n_ij = mask.sum()
            if n_ij > 0:
                n_i = (y_true == lt).sum()
                n_j = (y_pred == lp).sum()
                mi += (n_ij / n) * np.log((n * n_ij) / (n_i * n_j + 1e-12) + 1e-12)

    return mi / np.sqrt(h_true * h_pred)


def generate_switching_series(n_timepoints=2000, n_states=3, seed=42):
    """Generate a synthetic switching Gaussian time series."""
    rng = np.random.default_rng(seed)

    means = rng.uniform(-3, 3, size=n_states)
    stds = rng.uniform(0.3, 1.0, size=n_states)

    # Generate segments
    labels = np.zeros(n_timepoints, dtype=int)
    t = 0
    current_state = 0
    while t < n_timepoints:
        seg_len = rng.integers(100, 400)
        end = min(t + seg_len, n_timepoints)
        labels[t:end] = current_state
        t = end
        current_state = (current_state + 1) % n_states

    X = np.zeros((n_timepoints, 1))
    for k in range(n_states):
        mask = labels == k
        X[mask, 0] = rng.normal(means[k], stds[k], size=mask.sum())

    return X, labels


def run_test(name, X, y_true, n_states, n_train_steps):
    """Run SNLDS detector and print metrics."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"  Shape: {X.shape}, n_states={n_states}, n_train_steps={n_train_steps}")
    print(f"  True labels: {np.unique(y_true)}")

    det = SNLDSDetector(
        n_states=n_states,
        hidden_dim=8,
        rnn_dim=4,
        n_train_steps=n_train_steps,
        learning_rate=1e-3,
        verbose=1,
        random_state=42,
    )

    t0 = time.time()
    labels = det.fit_predict(X)
    elapsed = time.time() - t0

    print(f"  Predicted labels: {np.unique(labels)}")
    print(f"  Time: {elapsed:.1f}s")

    # Subsample for ARI (full pairwise is O(n^2))
    n = len(y_true)
    if n > 1000:
        idx = np.linspace(0, n - 1, 1000, dtype=int)
        ari = compute_ari(y_true[idx], labels[idx])
    else:
        ari = compute_ari(y_true, labels)

    nmi = compute_nmi(y_true, labels)
    print(f"  ARI: {ari:.4f}")
    print(f"  NMI: {nmi:.4f}")

    # Show segment structure
    changes = np.where(np.diff(labels) != 0)[0] + 1
    print(f"  Detected {len(changes)} change points")
    if len(changes) <= 20:
        print(f"  CPs: {changes.tolist()}")

    return labels, ari, nmi


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Test 1: Synthetic switching Gaussian
    X_synth, y_synth = generate_switching_series(n_timepoints=300, n_states=3)
    run_test("Synthetic Switching Gaussian (1D, 300pts)", X_synth, y_synth,
             n_states=3, n_train_steps=500)

    # Test 2: MoCap trial 0 (4D, ~5000 pts)
    X_mocap, y_mocap = load_mocap(trial=0)
    # Subsample for speed — take every 10th point
    X_mocap_sub = X_mocap[::10]
    y_mocap_sub = y_mocap[::10]
    run_test("MoCap Trial 0 (4D, subsampled x10)", X_mocap_sub, y_mocap_sub,
             n_states=5, n_train_steps=500)

    # Test 3: MoCap trial 0, less subsampled
    X_mocap_sub2 = X_mocap[::4]
    y_mocap_sub2 = y_mocap[::4]
    run_test("MoCap Trial 0 (4D, subsamp x4)", X_mocap_sub2, y_mocap_sub2,
             n_states=5, n_train_steps=500)

    print("\n" + "="*60)
    print("All tests completed.")
