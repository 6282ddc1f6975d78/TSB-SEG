"""
Time2Feat state detector for multivariate time series segmentation.

Adapts the Time2Feat clustering pipeline (intra-signal + inter-signal
feature extraction → feature selection → clustering) to segment a single
multivariate time series by treating sliding windows as individual samples
to be clustered.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

from ..base import BaseSegmenter
from .time2feat import (
    ClusterWrapper,
    extract_pair_series_features,
    extract_univariate_features,
    features_scoring_selection,
    get_transformer,
)


class Time2FeatDetector(BaseSegmenter):
    """Time2Feat state detector for multivariate time series segmentation.

    Segments a multivariate time series by extracting intra-signal features
    (via tsfresh) and inter-signal features (pair-wise distance metrics) on
    sliding windows, selecting the most informative features, and then
    clustering the windows into states.

    The algorithm was originally designed for clustering collections of
    multivariate time series.  Here it is adapted for single-series
    segmentation: the input series is first split into overlapping (or
    non-overlapping) windows, each window is treated as a separate sample,
    features are extracted and selected, and finally clustering assigns a
    state to each window.  The per-window labels are then mapped back to
    per-timepoint labels.

    Parameters
    ----------
    n_states : int, default=3
        Number of states (clusters) to discover.
    window_size : int, default=50
        Length of each sliding window in number of time steps.
    stride : int or None, default=None
        Step between consecutive windows.  If ``None``, defaults to
        ``window_size`` (non-overlapping windows).
    model_type : {"KMeans", "Hierarchical", "Spectral"}, default="KMeans"
        Clustering algorithm used to assign states.
    transform_type : {"std", "minmax", "robust"} or None, default="std"
        Scaler applied to the feature matrix before clustering.
        ``None`` means no scaling.
    top_k : int, default=50
        Number of top features to retain during feature selection.
    feature_selection_strategy : str, default="none"
        Feature selection strategy.  ``"none"`` keeps all features (after
        variance thresholding).  ``"sk_base"`` uses SelectKBest scores but
        requires supervised labels and is therefore not usable in the
        default unsupervised windowed-clustering mode.
    batch_size : int, default=-1
        Batch size for tsfresh feature extraction.  ``-1`` extracts all
        windows in a single batch.
    axis : int, default=0
        The time axis of the input array.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_timepoints,)
        Per-timepoint state labels computed during ``fit``.
    features_ : pd.DataFrame
        Feature matrix extracted from the windows (one row per window).
    selected_features_ : list[str]
        Names of features retained after feature selection.

    References
    ----------
    .. [1] A. Bonifati, F. Del Buono, F. Guerra and D. Tiano,
       "Time2Feat: Learning Interpretable Representations for Multivariate
       Time Series Clustering", Proc. VLDB Endow., 16(2), 193–201, 2022.

    Examples
    --------
    >>> import numpy as np
    >>> from tsseg.algorithms.time2feat import Time2FeatDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.concatenate([rng.normal(0, 1, (100, 3)),
    ...                     rng.normal(5, 1, (100, 3))])
    >>> det = Time2FeatDetector(n_states=2, window_size=50)
    >>> labels = det.fit_predict(X)
    """

    _tags = {
        "capability:univariate": True,
        "capability:multivariate": True,
        "detector_type": "state_detection",
        "fit_is_empty": False,
        "returns_dense": False,
        "capability:unsupervised": True,
        "capability:semi_supervised": True,
        "python_dependencies": "tsfresh",
    }

    def __init__(
        self,
        n_states: int = 3,
        window_size: int = 50,
        stride: int | None = None,
        model_type: Literal["KMeans", "Hierarchical", "Spectral"] = "KMeans",
        transform_type: Literal["std", "minmax", "robust"] | None = "std",
        top_k: int = 50,
        feature_selection_strategy: str = "none",
        batch_size: int = -1,
        axis: int = 0,
    ) -> None:
        self.n_states = n_states
        self.window_size = window_size
        self.stride = stride
        self.model_type = model_type
        self.transform_type = transform_type
        self.top_k = top_k
        self.feature_selection_strategy = feature_selection_strategy
        self.batch_size = batch_size
        super().__init__(axis=axis)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    def _make_windows(self, X: np.ndarray) -> list[np.ndarray]:
        """Split *X* (n_timepoints, n_channels) into sliding windows."""
        n_timepoints = X.shape[0]
        stride = self.stride if self.stride is not None else self.window_size
        windows: list[np.ndarray] = []
        start = 0
        while start + self.window_size <= n_timepoints:
            windows.append(X[start : start + self.window_size])
            start += stride
        # Include a trailing partial window only when it would otherwise be
        # completely lost (i.e. the last full window does not touch the end).
        if start < n_timepoints and (not windows or (start - stride + self.window_size) < n_timepoints):
            windows.append(X[n_timepoints - self.window_size : n_timepoints])
        return windows

    @staticmethod
    def _extract_features_for_window(
        window: np.ndarray,
        sensors_name: list[str],
    ) -> dict:
        """Extract intra- and inter-signal features for a single window."""
        features: dict = {}

        # Intra-signal (univariate tsfresh features per channel)
        uni_feats = extract_univariate_features(window, sensors_name)
        features.update({f"single__{k}": v for k, v in uni_feats.items()})

        # Inter-signal (pair-wise distance features)
        if window.shape[1] > 1:
            pair_feats = extract_pair_series_features(window)
            features.update(pair_feats)

        return features

    def _extract_all_features(
        self, windows: list[np.ndarray], sensors_name: list[str]
    ) -> pd.DataFrame:
        """Build the feature matrix for all windows."""
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        rows: list[dict] = []
        for w in windows:
            rows.append(self._extract_features_for_window(w, sensors_name))
        return pd.DataFrame(rows)

    def _select_features(self, df_feats: pd.DataFrame) -> list[str]:
        """Run feature selection and return the chosen column names."""
        if self.feature_selection_strategy == "none":
            return features_scoring_selection(
                df_feats, [], mode="simple", top_k=1, strategy="none"
            )
        return features_scoring_selection(
            df_feats,
            [],
            mode="simple",
            top_k=self.top_k,
            strategy=self.feature_selection_strategy,
        )

    def _cluster_windows(self, df_feats: pd.DataFrame, features: list[str]) -> np.ndarray:
        """Cluster the windows using the selected features."""
        X_cluster = df_feats[features].values

        if self.transform_type is not None:
            transformer = get_transformer(self.transform_type)
            X_cluster = transformer.fit_transform(X_cluster)

        model = ClusterWrapper(
            n_clusters=self.n_states,
            model_type=self.model_type,
        )
        return model.fit_predict(X_cluster)

    def _window_labels_to_timepoint_labels(
        self, window_labels: np.ndarray, n_timepoints: int
    ) -> np.ndarray:
        """Map per-window labels back to per-timepoint labels.

        When windows overlap, the label assigned to each timepoint is the
        one that occurs most frequently among the windows covering it
        (majority vote).
        """
        stride = self.stride if self.stride is not None else self.window_size
        n_windows = len(window_labels)
        # Accumulate votes per timepoint for each state
        votes = np.zeros((n_timepoints, self.n_states), dtype=int)

        for i, label in enumerate(window_labels):
            start = min(i * stride, n_timepoints - self.window_size)
            end = start + self.window_size
            if label < self.n_states:
                votes[start:end, label] += 1

        labels = np.argmax(votes, axis=1)
        return labels

    # ------------------------------------------------------------------ #
    #  BaseSegmenter hooks                                                 #
    # ------------------------------------------------------------------ #

    def _fit(self, X: np.ndarray, y=None):
        # X arrives as (n_timepoints, n_channels) after BaseSegmenter preprocessing
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if self.axis == 1:
            X = X.T

        n_timepoints, n_channels = X.shape
        sensors_name = [str(i) for i in range(n_channels)]

        # 1. Create sliding windows
        windows = self._make_windows(X)
        if len(windows) < self.n_states:
            raise ValueError(
                f"Not enough windows ({len(windows)}) for {self.n_states} states. "
                "Reduce window_size or increase the input length."
            )

        # 2. Extract features
        self.features_ = self._extract_all_features(windows, sensors_name)

        # 3. Feature selection
        self.selected_features_ = self._select_features(self.features_)
        if not self.selected_features_:
            # Fallback: use all non-constant columns
            self.selected_features_ = list(self.features_.columns)

        # 4. Clustering
        window_labels = self._cluster_windows(self.features_, self.selected_features_)

        # 5. Map window labels → timepoint labels
        self.labels_ = self._window_labels_to_timepoint_labels(window_labels, n_timepoints)

        return self

    def _predict(self, X: np.ndarray):
        return self.labels_

    @classmethod
    def _get_test_params(cls, parameter_set: str = "default") -> dict:
        return {
            "n_states": 2,
            "window_size": 20,
            "stride": 10,
            "model_type": "KMeans",
            "transform_type": "std",
            "top_k": 10,
            "feature_selection_strategy": "none",
        }
