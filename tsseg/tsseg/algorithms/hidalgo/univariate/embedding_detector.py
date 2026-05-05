"""Hidalgo detector with time-delay embedding for univariate series.

STATUS: ARCHIVED — experimental attempt, concluded negative.
See README.md in this directory for full analysis and conclusions.
"""

import numpy as np

from tsseg.algorithms.base import BaseSegmenter
from tsseg.algorithms.hidalgo.detector import HidalgoDetector


class HidalgoEmbeddingDetector(BaseSegmenter):
    """Hidalgo with built-in time-delay (Takens) embedding.

    Lifts a univariate time series to a multivariate representation via
    time-delay embedding, then runs Hidalgo on the embedded space.  Also
    works directly on multivariate data (embedding is skipped).

    Parameters
    ----------
    embed_dim : int, default=10
        Embedding dimension (m).  The univariate signal is lifted to R^m.
    delay : int, default=1
        Time delay (tau) between consecutive coordinates.
    metric : str, default="euclidean"
        Distance metric for nearest neighbour computation.
    K_states : int, default=2
        Number of manifolds / states.
    zeta : float, default=0.8
        Local homogeneity level.
    q : int, default=3
        Local homogeneity range.
    n_iter : int, default=1000
        Number of Gibbs sampling iterations.
    n_replicas : int, default=1
        Number of random restarts.
    burn_in : float, default=0.9
        Fraction of iterations discarded as burn-in.
    sampling_rate : int, default=10
        Keep every sampling_rate-th post-burn-in sample.
    fixed_Z : bool, default=False
        Fix Z during sampling.
    use_Potts : bool, default=True
        Enable Potts-like local homogeneity term.
    estimate_zeta : bool, default=False
        Update zeta during sampling.
    seed : int, default=0
        Random seed.
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "fit_is_empty": False,
        "returns_dense": False,
        "detector_type": "state_detection",
        "capability:unsupervised": False,
        "capability:semi_supervised": True,
    }

    def __init__(
        self,
        embed_dim=10,
        delay=1,
        metric="euclidean",
        K_states=2,
        zeta=0.8,
        q=3,
        n_iter=1000,
        n_replicas=1,
        burn_in=0.9,
        fixed_Z=False,
        use_Potts=True,
        estimate_zeta=False,
        sampling_rate=10,
        seed=0,
    ):
        self.embed_dim = embed_dim
        self.delay = delay
        self.metric = metric
        self.K_states = K_states
        self.zeta = zeta
        self.q = q
        self.n_iter = n_iter
        self.n_replicas = n_replicas
        self.burn_in = burn_in
        self.fixed_Z = fixed_Z
        self.use_Potts = use_Potts
        self.estimate_zeta = estimate_zeta
        self.sampling_rate = sampling_rate
        self.seed = seed

        super().__init__(axis=0)

    @staticmethod
    def _time_delay_embedding(x, m, tau):
        """Build time-delay embedding matrix from a 1-D signal.

        Parameters
        ----------
        x : np.ndarray, shape (T,)
        m : int, embedding dimension
        tau : int, delay

        Returns
        -------
        X_embed : np.ndarray, shape (T - (m-1)*tau, m)
        """
        T = len(x)
        n_rows = T - (m - 1) * tau
        indices = np.arange(m) * tau
        return np.array([x[i + indices] for i in range(n_rows)])

    def _fit(self, X, y=None):
        # X arrives as (n_timepoints, n_channels) from BaseSegmenter
        n_timepoints, n_channels = X.shape

        if n_channels == 1:
            # Univariate: apply time-delay embedding
            signal = X[:, 0]
            X_embed = self._time_delay_embedding(
                signal, self.embed_dim, self.delay
            )
            self._embedded_len = len(X_embed)
            self._original_len = n_timepoints
        else:
            # Multivariate: use directly
            X_embed = X
            self._embedded_len = n_timepoints
            self._original_len = n_timepoints

        self._inner = HidalgoDetector(
            metric=self.metric,
            K_states=self.K_states,
            zeta=self.zeta,
            q=self.q,
            n_iter=self.n_iter,
            n_replicas=self.n_replicas,
            burn_in=self.burn_in,
            fixed_Z=self.fixed_Z,
            use_Potts=self.use_Potts,
            estimate_zeta=self.estimate_zeta,
            sampling_rate=self.sampling_rate,
            seed=self.seed,
        )
        self._inner._fit(X_embed, y=y)

        # Store diagnostics
        self._d = self._inner._d
        self._p = self._inner._p
        self._lik = self._inner._lik

        return self

    def _predict(self, X, y=None):
        Z_embed = self._inner._predict(X, y=y)

        if self._original_len == self._embedded_len:
            return Z_embed

        # Pad labels back to original length by extending the last label
        # to cover the samples lost from the embedding.
        # With delay=tau, embed_dim=m: we lose (m-1)*tau samples.
        # Embedding row i maps to original time index i.
        n_lost = self._original_len - self._embedded_len
        Z_full = np.concatenate([Z_embed, np.full(n_lost, Z_embed[-1])])
        return Z_full

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {
            "embed_dim": 5,
            "delay": 1,
            "metric": "euclidean",
            "K_states": 1,
            "zeta": 0.8,
            "q": 3,
            "n_iter": 10,
            "n_replicas": 1,
            "burn_in": 0.5,
            "fixed_Z": False,
            "use_Potts": True,
            "estimate_zeta": False,
            "sampling_rate": 2,
            "seed": 1,
        }
