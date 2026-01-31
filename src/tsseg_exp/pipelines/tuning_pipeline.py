"""Hyperparameter tuning pipeline for TS segmentation experiments.

This module provides a dataset-aware tuning routine that can be invoked from
Hydra configs or programmatically. The pipeline coordinates three main steps:

1. Load dataset trials using the existing dataset loaders.
2. Split the available trials into train/test partitions when enough series
    exist; otherwise fall back to the algorithm defaults without tuning.
3. Perform a grid-search style hyper-parameter sweep defined by algorithm tags,
    rank candidates with train metrics, and evaluate the best configuration on
    the held-out test partition.

The pipeline relies on algorithm tags to describe searchable hyperparameters
and gracefully falls back to the algorithm's default configuration when the
dataset cannot be split for tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import random

import hydra
import mlflow
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .default_pipeline import TimeoutException
from tsseg_exp.utils.main_helpers import (
    build_supervision_param_overrides,
    configure_mlflow,
    extract_prediction_components,
    get_algorithm_tags,
    infer_trial_modality,
    normalize_task_value,
    resolve_algorithm_tag,
    resolve_capabilities,
    resolve_parent_deadline,
)

@dataclass
class TrialSegment:
    """Container describing a segment of a dataset trial."""

    dataset_name: str
    trial_index: int
    split: str
    modality: str
    features: np.ndarray
    labels: np.ndarray

    @property
    def n_timepoints(self) -> int:
        return int(self.features.shape[0])

    @property
    def n_channels(self) -> int:
        if self.features.ndim == 1:
            return 1
        return int(self.features.shape[1])


class DatasetTooSmallError(RuntimeError):
    """Raised when a dataset cannot be meaningfully split."""


class HyperparameterTuningPipeline:
    """Coordinate dataset splitting, hyper-parameter search, and evaluation."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.dataset_cfg = cfg.dataset
        self.algorithm_cfg = cfg.algorithm
        self.metric_cfg = cfg.metric
        self.tuning_cfg = cfg.get("tuning", {})

        self.random_state = int(self.tuning_cfg.get("random_state", 42))
        self.train_fraction = self._resolve_train_fraction()
        self.execution_mode = self._resolve_execution_mode()
        self.is_worker = self.execution_mode == "worker"

        self.algorithm_tags = self._resolve_algorithm_tags()
        self.algorithm_task = normalize_task_value(
            resolve_algorithm_tag(self.algorithm_cfg.get("task", "auto"), self.algorithm_tags),
            default="state",
        )
        supports_univariate, supports_multivariate = resolve_capabilities(self.algorithm_tags)
        self.supports_univariate = supports_univariate
        self.supports_multivariate = supports_multivariate
        self.returns_dense = bool(self.algorithm_tags.get("returns_dense", False))
        self.semi_supervised_flag = bool(self.algorithm_cfg.instance.get("semi_supervised", False))

        tuning_metric_tag = self.algorithm_tags.get("tuning_metric", {})
        self.tuning_metric_name = str(
            self.tuning_cfg.get(
                "metric_name",
                tuning_metric_tag.get("name", self._pick_first_metric_name()),
            )
        )
        self.tuning_metric_mode = str(
            self.tuning_cfg.get("metric_mode", tuning_metric_tag.get("mode", "max"))
        ).lower()
        if self.tuning_metric_mode not in {"max", "min"}:
            raise ValueError("tuning.metric_mode must be either 'max' or 'min'.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute the tuning loop (train/test only) and return metrics."""

        trials = self._load_trials()
        print(
            f"[tuning] Loaded {len(trials)} compatible trials for dataset '{self.dataset_cfg.name}'."
        )
        requested_test_fraction = max(0.0, 1.0 - self.train_fraction)
        print(
            f"[tuning] Requested train_fraction={self.train_fraction:.2f} "
            f"(test_fraction={requested_test_fraction:.2f})."
        )
        try:
            segments = self._split_trials(trials)
        except DatasetTooSmallError:
            return self._evaluate_default_parameters(trials)

        if self.is_worker:
            print("[tuning] Worker mode active; evaluating current configuration only.")
            return self._evaluate_current_configuration(trials)

        param_grid = self._build_parameter_grid()
        if not param_grid:
            raise ValueError("No candidate hyper-parameter configurations were generated.")

        train_trial_indices = [segment.trial_index for segment in segments["train"]]
        search_history: List[Dict[str, Any]] = []
        best_candidate: Optional[Dict[str, Any]] = None
        best_metric_value: Optional[float] = None

        for candidate in param_grid:
            train_metrics = self._evaluate_segments(candidate, segments["train"], split_name="train")
            metric_value_raw = train_metrics.get(self.tuning_metric_name)
            params_native = {key: self._to_native(val) for key, val in candidate.items()}
            search_history.append(
                {
                    "params": params_native,
                    "train_metrics": {
                        metric_key: self._to_native(metric_val)
                        for metric_key, metric_val in train_metrics.items()
                    },
                    "tuning_metric_value": None
                    if metric_value_raw is None
                    else float(metric_value_raw),
                }
            )
            if metric_value_raw is None:
                continue
            metric_value = float(metric_value_raw)
            if math.isnan(metric_value):
                continue
            if best_candidate is None or self._is_better(metric_value, best_metric_value):
                best_candidate = candidate
                best_metric_value = metric_value

        if best_candidate is None:
            fallback = self._evaluate_default_parameters(trials)
            fallback["search_history"] = search_history
            fallback["train_trial_indices"] = train_trial_indices
            return fallback

        train_metrics = self._evaluate_segments(best_candidate, segments["train"], split_name="train")
        test_metrics = self._evaluate_segments(best_candidate, segments["test"], split_name="test")

        return {
            "best_params": best_candidate,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "split_summary": self._summarise_segments(segments),
            "tuning_performed": True,
            "search_history": search_history,
            "train_trial_indices": train_trial_indices,
        }

    # ------------------------------------------------------------------
    # Internal helpers - configuration
    # ------------------------------------------------------------------
    def _resolve_execution_mode(self) -> str:
        mode_cfg = str(self.tuning_cfg.get("execution_mode", "auto")).lower()
        if mode_cfg not in {"auto", "controller", "worker"}:
            raise ValueError("tuning.execution_mode must be one of {'auto', 'controller', 'worker' }.")

        if mode_cfg == "worker":
            return "worker"
        if mode_cfg == "controller":
            return "controller"

        # auto mode: infer from Hydra runtime
        try:
            hydra_mode = HydraConfig.get().mode
        except Exception:  # pragma: no cover - Hydra may be unavailable in tests
            hydra_mode = None

        if hydra_mode and str(hydra_mode).upper() == "MULTIRUN":
            return "worker"
        return "controller"

    def _resolve_train_fraction(self) -> float:
        train_fraction_cfg = self.tuning_cfg.get("train_fraction")
        if train_fraction_cfg is None:
            if "test_fraction" in self.tuning_cfg:
                train_fraction_cfg = 1.0 - float(self.tuning_cfg.get("test_fraction", 0.0))
            else:
                train_fraction_cfg = 0.2

        train_fraction = float(train_fraction_cfg)
        if not math.isfinite(train_fraction):
            raise ValueError("tuning.train_fraction must be a finite number.")
        train_fraction = max(0.0, min(train_fraction, 1.0))
        return train_fraction

    def _resolve_algorithm_tags(self) -> Dict[str, Any]:
        algorithm_for_tags = hydra.utils.instantiate(self.algorithm_cfg.instance)
        tags = get_algorithm_tags(algorithm_for_tags)
        del algorithm_for_tags

        tag_override = self.algorithm_cfg.get("tags")
        if tag_override:
            tags = {
                **tags,
                **OmegaConf.to_container(tag_override, resolve=True, throw_on_missing=False),
            }

        if "tunable_parameters" not in tags and "tunable_parameters" in self.algorithm_cfg:
            tags["tunable_parameters"] = OmegaConf.to_container(
                self.algorithm_cfg.tunable_parameters,
                resolve=True,
                throw_on_missing=False,
            )

        if "tunable_parameters" in tags:
            tags["tunable_parameters"] = self._normalise_tunable_parameters(tags["tunable_parameters"])

        if "tuning_metric" not in tags and "tuning_metric" in self.algorithm_cfg:
            tags["tuning_metric"] = OmegaConf.to_container(
                self.algorithm_cfg.tuning_metric,
                resolve=True,
                throw_on_missing=False,
            )

        if "capabilities" not in tags and "capabilities" in self.algorithm_cfg:
            tags["capabilities"] = OmegaConf.to_container(
                self.algorithm_cfg.capabilities,
                resolve=True,
                throw_on_missing=False,
            )

        if "tunable_parameters" not in tags:
            raise ValueError(
                "Algorithm configuration must expose 'tunable_parameters' via tags for tuning to proceed."
            )
        return tags

    def _normalise_tunable_parameters(self, value: Any) -> Dict[str, Any]:
        """Convert list-style tunable parameter specs into a name->spec mapping."""

        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True, throw_on_missing=False)

        if isinstance(value, dict):
            return value

        if isinstance(value, list):
            normalized: Dict[str, Any] = {}
            for entry in value:
                if not isinstance(entry, dict) or len(entry) != 1:
                    raise ValueError(
                        "Each 'tunable_parameters' list entry must be a single-key mapping of name to spec."
                    )
                param_name, spec = next(iter(entry.items()))
                normalized[str(param_name)] = spec
            return normalized

        raise ValueError("Unsupported 'tunable_parameters' structure; expected dict or list of mappings.")

    def _pick_first_metric_name(self) -> str:
        for metric_name, metric_conf in self.metric_cfg.items():
            if isinstance(metric_conf, DictConfig) and "_target_" in metric_conf:
                return f"{metric_name}_score"
        raise ValueError("No valid metric definitions found in configuration.")

    def _is_better(self, candidate_value: float, reference_value: float) -> bool:
        if self.tuning_metric_mode == "max":
            return candidate_value > reference_value
        return candidate_value < reference_value

    # ------------------------------------------------------------------
    # Dataset loading and splitting
    # ------------------------------------------------------------------
    def _load_trials(self) -> List[Tuple[int, np.ndarray, np.ndarray, str]]:
        dataset_name = self.dataset_cfg.name
        loader_cfg = self.dataset_cfg.loader
        X_list, y_list = hydra.utils.call(loader_cfg)
        if not isinstance(X_list, list):
            X_list, y_list = [X_list], [y_list]

        trials: List[Tuple[int, np.ndarray, np.ndarray, str]] = []
        for idx, (X, y) in enumerate(zip(X_list, y_list)):
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            y_arr = np.asarray(y)
            modality = infer_trial_modality(X_arr)
            if modality == "univariate" and not self.supports_univariate:
                print(
                    f"[tuning] Skipping trial {idx}: univariate modality not supported by algorithm."
                )
                continue
            if modality == "multivariate" and not self.supports_multivariate:
                print(
                    f"[tuning] Skipping trial {idx}: multivariate modality not supported by algorithm."
                )
                continue
            trials.append((idx, X_arr, y_arr, modality))

        if not trials:
            raise ValueError(
                f"No trials compatible with algorithm '{self.algorithm_cfg.name}' were found for dataset '{dataset_name}'."
            )
        return trials

    def _split_trials(self, trials: List[Tuple[int, np.ndarray, np.ndarray, str]]) -> Dict[str, List[TrialSegment]]:
        dataset_name = self.dataset_cfg.name
        splits: Dict[str, List[TrialSegment]] = {"train": [], "test": []}
        n_trials = len(trials)

        if n_trials < 2:
            raise DatasetTooSmallError(
                "Need at least two trials to create a train/test split for tuning."
            )

        rng = random.Random(self.random_state)
        indices = list(range(n_trials))
        rng.shuffle(indices)
        train_count = max(1, int(round(self.train_fraction * n_trials)))
        train_count = min(train_count, n_trials - 1)
        test_count = n_trials - train_count
        if test_count < 1:
            test_count = 1
            train_count = n_trials - test_count

        train_indices = indices[:train_count]
        test_indices = indices[train_count:]

        for split_name, chosen_indices in (("train", train_indices), ("test", test_indices)):
            for idx in chosen_indices:
                orig_idx, X_arr, y_arr, modality = trials[idx]
                splits[split_name].append(
                    TrialSegment(
                        dataset_name=dataset_name,
                        trial_index=orig_idx,
                        split=split_name,
                        modality=modality,
                        features=X_arr,
                        labels=y_arr,
                    )
                )

        for split_name, segment_list in splits.items():
            if not segment_list:
                raise DatasetTooSmallError(
                    f"Split '{split_name}' is empty. Provide more trials to enable tuning."
                )
            self._log_split_details(split_name, segment_list)

        return splits

    def _evaluate_default_parameters(
        self,
        trials: List[Tuple[int, np.ndarray, np.ndarray, str]],
    ) -> Dict[str, Any]:
        dataset_name = self.dataset_cfg.name
        test_segments = [
            TrialSegment(
                dataset_name=dataset_name,
                trial_index=idx,
                split="test",
                modality=modality,
                features=X_arr,
                labels=y_arr,
            )
            for idx, X_arr, y_arr, modality in trials
        ]

        self._log_split_details("test", test_segments)

        test_metrics = self._evaluate_segments({}, test_segments, split_name="test")
        split_summary = {
            "train": [],
            "test": [
                {
                    "dataset": dataset_name,
                    "trial_index": segment.trial_index,
                    "modality": segment.modality,
                    "n_timepoints": segment.n_timepoints,
                    "n_channels": segment.n_channels,
                }
                for segment in test_segments
            ],
        }

        return {
            "best_params": {},
            "train_metrics": {},
            "test_metrics": test_metrics,
            "split_summary": split_summary,
            "tuning_performed": False,
            "search_history": [],
            "train_trial_indices": [],
        }

    def _evaluate_current_configuration(
        self,
        trials: List[Tuple[int, np.ndarray, np.ndarray, str]],
    ) -> Dict[str, Any]:
        segments = self._split_trials(trials)
        train_segments = segments["train"]
        test_segments = segments["test"]

        current_params = self._current_algorithm_params()
        train_metrics = self._evaluate_segments({}, train_segments, split_name="train")
        test_metrics = self._evaluate_segments({}, test_segments, split_name="test")

        tuning_metric_value = train_metrics.get(self.tuning_metric_name)
        search_history = [
            {
                "params": current_params,
                "train_metrics": {
                    metric_key: self._to_native(metric_val)
                    for metric_key, metric_val in train_metrics.items()
                },
                "tuning_metric_value": None
                if tuning_metric_value is None
                else float(tuning_metric_value),
            }
        ]

        return {
            "best_params": current_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "split_summary": self._summarise_segments(segments),
            "tuning_performed": False,
            "search_history": search_history,
            "train_trial_indices": [segment.trial_index for segment in train_segments],
        }

    # ------------------------------------------------------------------
    # Hyper-parameter search utilities
    # ------------------------------------------------------------------
    def _build_parameter_grid(self) -> List[Dict[str, Any]]:
        tunable_params = self.algorithm_tags.get("tunable_parameters", {})
        grid_axes: List[List[Tuple[str, Any]]] = []

        for param_name, spec in tunable_params.items():
            axis_values = self._expand_parameter_axis(param_name, spec)
            if not axis_values:
                continue
            grid_axes.append([(param_name, value) for value in axis_values])

        if not grid_axes:
            return []

        param_grid: List[Dict[str, Any]] = []
        for combo in self._product(*grid_axes):
            candidate = {name: value for name, value in combo}
            param_grid.append(candidate)
        return param_grid

    def _expand_parameter_axis(self, name: str, spec: Any) -> List[Any]:
        if isinstance(spec, (list, tuple)):
            return list(spec)
        if "values" in spec:
            return list(spec["values"])
        if "grid" in spec:
            grid_spec = spec["grid"]
            start = float(grid_spec.get("start", 0.0))
            stop = float(grid_spec.get("stop", 1.0))
            num = int(grid_spec.get("num", 5))
            base = float(grid_spec.get("base", math.e))
            if num <= 1:
                return [base ** start]
            return list(base ** np.linspace(start, stop, num=num))
        if {"min", "max", "step"}.issubset(spec.keys()):
            current = float(spec["min"])
            max_value = float(spec["max"])
            step = float(spec["step"])
            values = []
            while current <= max_value + 1e-12:
                values.append(current)
                current += step
            return values
        raise ValueError(
            f"Unsupported specification for parameter '{name}'. Provide either 'values', 'grid', or min/max/step."
        )

    def _product(self, *axes: Sequence[Tuple[str, Any]]) -> Iterable[List[Tuple[str, Any]]]:
        if not axes:
            return []
        def recurse(idx: int, acc: List[Tuple[str, Any]]):
            if idx == len(axes):
                yield list(acc)
                return
            for item in axes[idx]:
                acc.append(item)
                yield from recurse(idx + 1, acc)
                acc.pop()
        return recurse(0, [])

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate_segments(
        self,
        candidate_params: Dict[str, Any],
        segments: List[TrialSegment],
        split_name: str,
    ) -> Dict[str, float]:
        if not segments:
            return {}

        print(
            f"[tuning] Evaluating split '{split_name}' with {len(segments)} series; params={candidate_params}."
        )

        metrics = list(self._iter_metrics_for_split(split_name))
        metrics_accumulator: Dict[str, List[float]] = {}

        # Pre-resolve valid config keys to avoid checking every time
        valid_config_keys = set(OmegaConf.to_container(self.algorithm_cfg.instance, resolve=True).keys())

        for segment in segments:
            print(
                f"[tuning] -> Trial {segment.trial_index} ({segment.modality}), length={segment.n_timepoints}, channels={segment.n_channels}."
            )

            # Prepare overrides: start with grid search candidates
            overrides = dict(candidate_params)

            # If semi-supervised, calculate and inject ground-truth derived parameters
            if self.semi_supervised_flag and segment.labels.size > 0:
                supervision_overrides = build_supervision_param_overrides(segment.labels)
                for param_name, param_value in supervision_overrides.items():
                    # Only inject if the algorithm actually has this parameter in its config
                    if param_name in valid_config_keys:
                        overrides[param_name] = int(param_value)

            algorithm = self._instantiate_algorithm(overrides)
            fit_kwargs = {"axis": 0}
            algorithm.fit(segment.features, **fit_kwargs)

            prediction = algorithm.predict(segment.features, axis=0)
            y_pred, _, _ = extract_prediction_components(
                prediction,
                series_length=segment.n_timepoints,
                returns_dense=self.returns_dense,
                algorithm_task=self.algorithm_task,
            )

            if not segment.labels.size:
                continue

            for metric_name, metric_callable in metrics:
                result = metric_callable(segment.labels, y_pred)
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, (int, float, np.floating)):
                            metrics_accumulator.setdefault(f"{metric_name}_{key}", []).append(float(value))
                elif isinstance(result, (int, float, np.floating)):
                    metrics_accumulator.setdefault(metric_name, []).append(float(result))

        aggregated = {
            metric_name: float(np.mean(values))
            for metric_name, values in metrics_accumulator.items()
            if values
        }
        return aggregated

    def _instantiate_algorithm(self, overrides: Dict[str, Any]):
        base_conf = OmegaConf.create(OmegaConf.to_container(self.algorithm_cfg.instance, resolve=False))
        for key, value in overrides.items():
            base_conf[key] = value
        return hydra.utils.instantiate(base_conf)

    def _iter_metrics_for_split(self, split_name: str):
        allowed_tasks = {"train": {"state", "change_point"}, "test": {"state", "change_point"}}
        for metric_name, metric_conf in self.metric_cfg.items():
            if not isinstance(metric_conf, DictConfig) or "_target_" not in metric_conf:
                continue
            metric_task = normalize_task_value(metric_conf.get("task", "state"))
            if metric_task not in allowed_tasks.get(split_name, {"state", "change_point"}):
                continue
            compute_params = OmegaConf.to_container(metric_conf.get("compute_params", {}), resolve=True) or {}
            instantiate_conf = {
                key: value
                for key, value in metric_conf.items()
                if key not in {"compute_params", "task"}
            }
            metric_obj = hydra.utils.instantiate(instantiate_conf, **compute_params)
            yield metric_name, metric_obj.compute

    def _summarise_segments(self, segments: Dict[str, List[TrialSegment]]) -> Dict[str, Any]:
        summary = {}
        for split_name, segment_list in segments.items():
            summary[split_name] = [
                {
                    "dataset": segment.dataset_name,
                    "trial_index": segment.trial_index,
                    "modality": segment.modality,
                    "n_timepoints": segment.n_timepoints,
                    "n_channels": segment.n_channels,
                }
                for segment in segment_list
            ]
        return summary

    def _current_algorithm_params(self) -> Dict[str, Any]:
        instance_cfg = OmegaConf.to_container(
            self.algorithm_cfg.instance,
            resolve=True,
            throw_on_missing=False,
        )
        if not isinstance(instance_cfg, dict):
            return {}
        return {
            key: self._to_native(value)
            for key, value in instance_cfg.items()
            if key != "_target_"
        }

    def _to_native(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _log_split_details(self, split_name: str, segment_list: List[TrialSegment]) -> None:
        """Emit human-friendly information about a split."""

        indices = [segment.trial_index for segment in segment_list]
        modalities = sorted({segment.modality for segment in segment_list})
        total_timepoints = sum(segment.n_timepoints for segment in segment_list)
        print(
            f"[tuning] Split '{split_name}': {len(segment_list)} series (indices={indices}), "
            f"modalities={modalities}, total_timepoints={total_timepoints}."
        )


def run_tuning_pipeline(cfg: DictConfig) -> None:
    """Entry point for the hyper-parameter tuning pipeline."""

    configure_mlflow(cfg)
    mlflow.set_experiment(cfg.experiment.name)
    default_name = f"{cfg.algorithm.name}_{cfg.dataset.name}_tuning"
    run_name = cfg.get("run_name", default_name)

    timeout_seconds, deadline = resolve_parent_deadline(
        timeout_seconds=cfg.experiment.get("timeout_seconds"),
        timeout_hours=cfg.experiment.get("timeout_hours"),
    )

    with mlflow.start_run(run_name=run_name) as run:
        print(f"--- Starting tuning run: {run_name} (MLflow Run ID: {run.info.run_id}) ---")
        print("Config:")
        print(OmegaConf.to_yaml(cfg))

        mlflow.set_tag("pipeline", "tuning")
        mlflow.log_param("pipeline", "tuning")

        algo_params = OmegaConf.to_container(cfg.algorithm.instance, resolve=True, throw_on_missing=True)
        mlflow.log_params({f"algo_{k}": v for k, v in algo_params.items()})

        dataset_params = OmegaConf.to_container(cfg.dataset.loader, resolve=True, throw_on_missing=True)
        mlflow.log_params({f"dataset_{k}": v for k, v in dataset_params.items()})

        if "preprocessing" in cfg and cfg.preprocessing is not None:
            preproc_params = OmegaConf.to_container(cfg.preprocessing, resolve=True, throw_on_missing=True)
            mlflow.log_params({f"preproc_{k}": v for k, v in preproc_params.items()})

        mlflow.log_param("dataset_name", cfg.dataset.name)
        mlflow.log_param("algorithm_name", cfg.algorithm.name)

        pipeline = HyperparameterTuningPipeline(cfg)

        mlflow.log_param("train_fraction", pipeline.train_fraction)
        mlflow.log_param("test_fraction", max(0.0, 1.0 - pipeline.train_fraction))
        mlflow.log_param("tuning_execution_mode", pipeline.execution_mode)
        mlflow.set_tag("tuning_execution_mode", pipeline.execution_mode)
        if timeout_seconds is not None:
            mlflow.log_param("timeout_seconds", timeout_seconds)

        if timeout_seconds is not None:
            print(
                f"Deadline set to {int(timeout_seconds)} seconds (~{timeout_seconds/3600:.2f} h) from start of tuning run."
            )

        try:
            result = pipeline.run()
            best_params = result.get("best_params", {})
            if best_params:
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

            train_metrics = result.get("train_metrics", {})
            test_metrics = result.get("test_metrics", {})
            metrics_payload: Dict[str, float] = {}
            metrics_payload.update({f"train_{k}": v for k, v in train_metrics.items()})
            metrics_payload.update({f"test_{k}": v for k, v in test_metrics.items()})
            if metrics_payload:
                mlflow.log_metrics(metrics_payload)

            mlflow.log_dict(
                {
                    "split_summary": result.get("split_summary", {}),
                    "tuning_performed": result.get("tuning_performed", False),
                },
                "tuning_summary.json",
            )

            mlflow.log_dict(
                {
                    "train_trial_indices": result.get("train_trial_indices", []),
                    "search_history": result.get("search_history", []),
                },
                "training_search_history.json",
            )

            mlflow.set_tag("tuning_performed", str(result.get("tuning_performed", False)))
            mlflow.set_tag("status", "completed")
        except TimeoutException:
            timeout_display = int(timeout_seconds) if timeout_seconds else "configured budget"
            print(f"!!! Tuning run {run_name} timed out after {timeout_display} seconds. !!!")
            mlflow.set_tag("status", "timeout")
        except Exception as exc:  # pragma: no cover - logging for debugging
            print(f"!!! Tuning run {run_name} failed with an exception: {exc} !!!")
            import traceback

            traceback.print_exc()
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error_message", str(exc))
        finally:
            print(f"--- Finished tuning run: {run_name} ---")


__all__ = [
    "HyperparameterTuningPipeline",
    "TrialSegment",
    "DatasetTooSmallError",
    "run_tuning_pipeline",
]
