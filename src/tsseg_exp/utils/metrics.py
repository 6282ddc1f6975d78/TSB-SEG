import numpy as np

def changepoints_to_labels(changepoints, n_samples):
    """
    Converts a list of changepoint indices to a full label array.

    Args:
        changepoints (list or np.ndarray): A list of indices where new segments start.
        n_samples (int): The total number of samples in the time series.

    Returns:
        np.ndarray: An array of length n_samples with integer labels for each segment.
    """
    # Ensure changepoints are sorted and unique, and convert to a list
    changepoints = sorted(list(set(changepoints)))
    
    # Add 0 to the changepoints if it's not there to handle the first segment
    if not changepoints or changepoints[0] != 0:
        changepoints.insert(0, 0)

    # Add the last sample index if it's not there to ensure the last segment is included
    if changepoints[-1] != n_samples:
        changepoints.append(n_samples)

    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(len(changepoints) - 1):
        start_idx = changepoints[i]
        end_idx = changepoints[i+1]
        labels[start_idx:end_idx] = i
    
    return labels
