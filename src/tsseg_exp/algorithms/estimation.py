import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import ruptures as rpt

def _estimate_noise_variance(X):
    """
    Estimate noise variance using Median Absolute Deviation of first differences.
    """
    if X.shape[0] < 2:
        return 1.0
        
    diff = np.diff(X, axis=0)
    # MAD estimator for standard deviation: sigma = MAD / 0.6745
    # MAD = median(|x - median(x)|)
    # Here we assume differences have mean 0, so MAD = median(|diff|)
    # But let's be safe and subtract median
    mad = np.median(np.abs(diff - np.median(diff, axis=0)), axis=0)
    sigma = mad / 0.6745
    
    # If sigma is 0 (constant signal), replace with small epsilon to avoid zero penalty
    sigma[sigma == 0] = 1e-6
    
    # We return the mean variance across dimensions
    return np.mean(sigma**2)

def estimate_n_states_elbow(X, max_k=10, random_state=42):
    """
    Estimate number of states using the Elbow method on K-Means inertia.
    """
    n_samples = X.shape[0]
    if n_samples < max_k:
        max_k = n_samples
    
    inertias = []
    ks = range(1, max_k + 1)
    
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        
    # Find elbow point using the "distance to line" method
    # Line from (1, inertia[0]) to (max_k, inertia[-1])
    p1 = np.array([1, inertias[0]])
    p2 = np.array([max_k, inertias[-1]])
    
    max_dist = -1
    best_k = 1
    
    for i, k in enumerate(ks):
        p = np.array([k, inertias[i]])
        # Distance from point p to line p1-p2
        # If p1 and p2 are the same (e.g. max_k=1), dist is 0
        if np.array_equal(p1, p2):
            dist = 0
        else:
            dist = np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)
            
        if dist > max_dist:
            max_dist = dist
            best_k = k
            
    return best_k

def estimate_n_states_silhouette(X, max_k=10, random_state=42):
    """
    Estimate number of states using the Silhouette Score.
    """
    n_samples = X.shape[0]
    if n_samples < max_k:
        max_k = n_samples
        
    best_score = -1
    best_k = 1 # Default to 1 if no split is good
    
    # Silhouette requires at least 2 clusters
    ks = range(2, max_k + 1)
    
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        # Check if we actually have k clusters (sometimes kmeans drops empty clusters)
        if len(np.unique(labels)) < 2:
            continue
            
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score = score
            best_k = k
            
    return best_k

def estimate_n_states_bic(X, max_k=10, random_state=42):
    """
    Estimate number of states using BIC on Gaussian Mixture Models.
    """
    n_samples = X.shape[0]
    if n_samples < max_k:
        max_k = n_samples
        
    min_bic = np.inf
    best_k = 1
    
    ks = range(1, max_k + 1)
    
    for k in ks:
        try:
            gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=5)
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < min_bic:
                min_bic = bic
                best_k = k
        except Exception:
            # GMM might fail convergence or other issues
            continue
            
    return best_k

def estimate_n_cps_pelt(X, penalty="bic", min_size=3, jump=5):
    """
    Estimate number of change points using PELT.
    """
    model = "l2"  # "l2" is standard for mean shift
    try:
        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(X)
        
        if penalty == "bic":
            n_samples, n_dims = X.shape
            sigma_sq = _estimate_noise_variance(X)
            # BIC penalty: log(T) * dim * sigma^2
            # Note: ruptures documentation suggests penalty for l2 is proportional to sigma^2
            pen = np.log(n_samples) * n_dims * sigma_sq
        else:
            pen = float(penalty)
            
        bkps = algo.predict(pen=pen)
        # bkps includes the end of the signal, so n_cps = len(bkps) - 1
        return len(bkps) - 1
    except Exception:
        return 0

def estimate_n_cps_binseg(X, penalty="bic", min_size=3, jump=5):
    """
    Estimate number of change points using Binary Segmentation.
    """
    model = "l2"
    try:
        algo = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(X)
        
        if penalty == "bic":
            n_samples, n_dims = X.shape
            sigma_sq = _estimate_noise_variance(X)
            pen = np.log(n_samples) * n_dims * sigma_sq
        else:
            pen = float(penalty)
            
        bkps = algo.predict(pen=pen)
        return len(bkps) - 1
    except Exception:
        return 0
