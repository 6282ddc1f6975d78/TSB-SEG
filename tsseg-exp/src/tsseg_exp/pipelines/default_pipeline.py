"""Default experiment pipeline for TS segmentation experiments."""
from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import hydra
import mlflow
import numpy as np
import pandas as pd
from mlflow.data import from_pandas as mlflow_from_pandas
from omegaconf import DictConfig, OmegaConf, open_dict

from tsseg_exp.utils.main_helpers import (
    build_supervision_param_overrides,
    configure_mlflow,
    count_change_points,
    count_unique_states,
    estimate_window_size_fft,
    extract_prediction_components,
    get_algorithm_tags,
    infer_trial_modality,
    labels_to_change_points,
    resolve_algorithm_tag,
    resolve_capabilities,
    is_deadline_exceeded,
    resolve_parent_deadline,
)


# ── Styled logging helpers ────────────────────────────────────────────

_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"
_CHECK = "✓"
_CROSS = "✗"
_ARROW = "→"
_BULLET = "•"


def _header(title: str, run_id: str = "") -> None:
    """Print a prominent run header."""
    width = 72
    print(f"\n{_BOLD}{'━' * width}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    if run_id:
        print(f"{_DIM}  MLflow Run ID: {run_id}{_RESET}")
    print(f"{_BOLD}{'━' * width}{_RESET}")


def _footer(title: str, duration: float) -> None:
    """Print a run footer with elapsed time."""
    width = 72
    mins, secs = divmod(duration, 60)
    time_str = f"{int(mins)}m {secs:.1f}s" if mins else f"{secs:.1f}s"
    print(f"{_DIM}{'─' * width}{_RESET}")
    print(f"{_DIM}  {_CHECK} {title} completed in {time_str}{_RESET}\n")


def _log_step(msg: str) -> None:
    """Print a pipeline step."""
    print(f"  {_CYAN}{_ARROW}{_RESET} {msg}")


def _log_info(msg: str) -> None:
    """Print an informational note."""
    print(f"  {_DIM}{_BULLET} {msg}{_RESET}")


def _log_warn(msg: str) -> None:
    """Print a warning."""
    print(f"  {_YELLOW}⚠ {msg}{_RESET}")


def _log_error(msg: str) -> None:
    """Print an error."""
    print(f"  {_RED}{_CROSS} {msg}{_RESET}")


def _log_success(msg: str) -> None:
    """Print a success message."""
    print(f"  {_GREEN}{_CHECK} {msg}{_RESET}")


def _format_metrics_table(
    metrics: Dict[str, float],
    *,
    title: str = "Metrics",
    max_rows: int | None = None,
) -> str:
    """Render a dict of metrics as a compact aligned table."""
    if not metrics:
        return f"  {_DIM}(no metrics){_RESET}"

    items = sorted(metrics.items())
    if max_rows is not None and len(items) > max_rows:
        items = items[:max_rows]

    name_width = max(len(k) for k, _ in items)
    lines = [f"  {_BOLD}{title}{_RESET}"]
    for name, value in items:
        bar = _metric_bar(value)
        lines.append(f"    {name:<{name_width}}  {value:>8.4f}  {bar}")
    return "\n".join(lines)


def _metric_bar(value: float, width: int = 16) -> str:
    """Tiny inline spark-bar for values in [0, 1]."""
    clamped = max(0.0, min(1.0, value))
    filled = int(round(clamped * width))
    return f"{_DIM}{'█' * filled}{'░' * (width - filled)}{_RESET}"


class TimeoutException(Exception):
    """Raised when a trial or run exceeds the configured wall time."""


def _resolve_modality_for_trial(dataset_modality_cfg: str, inferred_trials: List[str], idx: int) -> str:
    configured = dataset_modality_cfg
    if configured in {"univariate", "multivariate"}:
        return configured
    if configured in {"mixed", "infer", "auto"}:
        return inferred_trials[idx]
    _log_warn(f"Unknown dataset modality '{configured}'. Falling back to inference.")
    return inferred_trials[idx]


def _log_dataset_overview(dataset_name: str, dataset_overview_rows: List[Dict[str, float]]):
    if not dataset_overview_rows:
        return

    overview_df = pd.DataFrame(dataset_overview_rows)
    numeric_cols = ["trial_index", "n_timepoints", "n_channels", "label_cardinality"]
    overview_df[numeric_cols] = overview_df[numeric_cols].astype(float)

    try:
        dataset_overview = mlflow_from_pandas(overview_df, name=dataset_name)
        mlflow.log_input(dataset_overview, context="dataset")
    except Exception as exc:  # pragma: no cover - best effort logging
        _log_warn(f"Failed to log dataset to MLflow: {exc}")


def run_single_experiment(
    cfg: DictConfig,
    extra_params: Optional[Dict[str, Any]] = None,
    extra_tags: Optional[Dict[str, Any]] = None,
    run_suffix: str = "",
    fallback_window: int = 10,  # fallback window size for FFT estimation
    deadline: Optional[float] = None,
) -> Tuple[Dict[str, float], object]:
    """Execute one dataset/algorithm experiment and log nested MLflow runs.

    Parameters
    ----------
    cfg: DictConfig
        Experiment configuration hydrated by Hydra.
    extra_params: Optional[Dict[str, Any]]
        Additional parameters to log.
    extra_tags: Optional[Dict[str, Any]]
        Additional tags to set.
    run_suffix: str
        Extra suffix appended to the MLflow run name.
    fallback_window: int
        Fallback window size used when FFT estimation fails.
    deadline: Optional[float]
        Monotonic deadline (seconds) for the parent run. Trials starting after
        this point will be skipped and a ``TimeoutException`` raised. When
        ``None`` the execution runs without a time budget.
    """

    _log_step(f"Loading dataset: {_BOLD}{cfg.dataset.name}{_RESET}")
    X_list, y_list = hydra.utils.call(cfg.dataset.loader)

    if not isinstance(X_list, list):
        X_list, y_list = [X_list], [y_list]

    dataset_modality_cfg = str(cfg.dataset.get("modality", "infer")).lower()
    dataset_periodic_cfg = bool(cfg.dataset.get("periodic", False))
    inferred_trials = [infer_trial_modality(trial) for trial in X_list]

    # Resolve experiment config (support both hierarchical and flattened structures)
    if "experiment" in cfg:
        experiment_cfg = cfg.experiment
    else:
        experiment_cfg = cfg

    supervision_mode_raw = experiment_cfg.get("supervision_mode", "unsupervised")
    supervision_mode = str(supervision_mode_raw if supervision_mode_raw is not None else "unsupervised")
    supervision_mode = supervision_mode.strip().lower().replace(" ", "_").replace("-", "_")
    if supervision_mode not in {"semi_supervised", "supervised"}:
        supervision_mode = "unsupervised"

    mlflow.log_param("supervision_mode", supervision_mode)
    mlflow.set_tag("supervision_mode", supervision_mode)
    is_supervised_mode = supervision_mode in {"semi_supervised", "supervised"}

    per_trial_windows: List[Optional[int]] = [None] * len(X_list)
    if dataset_periodic_cfg:
        estimated_windows = estimate_window_size_fft(trials=X_list, fallback=fallback_window) or []
        if not estimated_windows:
            _log_warn(
                f"FFT window estimation failed; falling back to {fallback_window} for all trials."
            )
            per_trial_windows = [fallback_window] * len(X_list)
        else:
            for idx in range(len(X_list)):
                estimate = estimated_windows[idx] if idx < len(estimated_windows) else None
                if isinstance(estimate, int) and estimate > 0:
                    per_trial_windows[idx] = int(estimate)
                else:
                    _log_warn(f"Trial {idx+1}: invalid FFT estimate; using fallback={fallback_window}.")
                    per_trial_windows[idx] = fallback_window

            _log_info("Using per-trial FFT-derived window sizes (periodic dataset).")
    else:
        _log_info("Non-periodic dataset; using configured window_size.")

    dataset_overview_rows: List[Dict[str, float]] = []
    for idx, (trial_X, trial_y, modality) in enumerate(zip(X_list, y_list, inferred_trials), start=1):
        trial_X_arr = np.asarray(trial_X)
        if trial_X_arr.ndim == 1:
            trial_X_arr = trial_X_arr.reshape(-1, 1)
        n_timepoints = int(trial_X_arr.shape[0])
        n_channels = int(trial_X_arr.shape[1])
        trial_y_arr = np.asarray(trial_y)
        label_cardinality = int(np.unique(trial_y_arr).size) if trial_y_arr.size else 0
        dataset_overview_rows.append(
            {
                "dataset": cfg.dataset.name,
                "trial_index": idx,
                "modality_inferred": modality,
                "n_timepoints": n_timepoints,
                "n_channels": n_channels,
                "label_cardinality": label_cardinality,
            }
        )

    trial_modalities = [
        _resolve_modality_for_trial(dataset_modality_cfg, inferred_trials, idx)
        for idx in range(len(X_list))
    ]
    unique_modalities = sorted(set(trial_modalities))
    modality_summary = ",".join(unique_modalities)
    mlflow.log_param("modality", modality_summary)
    mlflow.set_tag("modality", modality_summary)

    _log_dataset_overview(cfg.dataset.name, dataset_overview_rows)

    mlflow.set_tag("dataset_name", cfg.dataset.name)
    mlflow.log_param("dataset_name", cfg.dataset.name)

    algorithm_conf = cfg.algorithm.instance
    algorithm_cfg_for_tags = OmegaConf.create(
        OmegaConf.to_container(algorithm_conf, resolve=True, throw_on_missing=True)
    )

    if is_supervised_mode:
        seed_labels = next(
            (np.asarray(labels) for labels in y_list if np.asarray(labels).size),
            None,
        )
        if seed_labels is not None:
            supervision_seed_overrides = build_supervision_param_overrides(seed_labels)
            if supervision_seed_overrides:
                with open_dict(algorithm_cfg_for_tags):
                    for param_name, param_value in supervision_seed_overrides.items():
                        if param_name in algorithm_cfg_for_tags:
                            current_value = algorithm_cfg_for_tags[param_name]
                            if current_value is None:
                                algorithm_cfg_for_tags[param_name] = int(param_value)

    algorithm_for_tags = hydra.utils.instantiate(algorithm_cfg_for_tags)
    algo_tags = get_algorithm_tags(algorithm_for_tags)
    supports_univariate, supports_multivariate = resolve_capabilities(algo_tags)
    supports_unsupervised = bool(algo_tags.get("capability:unsupervised", False))
    supports_semi_supervised = bool(algo_tags.get("capability:semi_supervised", False))
    returns_dense = bool(algo_tags.get("returns_dense", False))
    del algorithm_for_tags

    configured_semi_supervised = bool(algorithm_conf.get("semi_supervised", False))
    # fit_with_labels logic is now dynamic based on experiment mode and capabilities
    
    mlflow.log_param("algorithm_name", cfg.algorithm.name)
    mlflow.log_param("algorithm_config_semi_supervised", configured_semi_supervised)
    mlflow.log_param("algorithm_supports_unsupervised", supports_unsupervised)
    mlflow.log_param("algorithm_supports_semi_supervised", supports_semi_supervised)

    raw_algorithm_task = cfg.algorithm.get("task", "auto")
    algorithm_task = resolve_algorithm_tag(raw_algorithm_task, algo_tags)

    if algorithm_task == "change_point":
        allowed_metric_tasks = {"change_point"}
    else:
        allowed_metric_tasks = {"state", "change_point"}

    mlflow.set_tag("algorithm_task", algorithm_task)
    mlflow.log_param("dataset_modality_config", dataset_modality_cfg)
    mlflow.log_param("algorithm_supports_univariate", supports_univariate)
    mlflow.log_param("algorithm_supports_multivariate", supports_multivariate)
    mlflow.log_param("algorithm_returns_dense", returns_dense)

    filtered_trials: List[Tuple[int, np.ndarray, np.ndarray, str, Optional[int]]] = []
    for idx, (single_X, single_y_true, modality) in enumerate(zip(X_list, y_list, trial_modalities)):
        if modality == "univariate" and not supports_univariate:
            _log_warn(f"Skipping trial {idx+1}: univariate data not supported.")
            continue
        if modality == "multivariate" and not supports_multivariate:
            _log_warn(f"Skipping trial {idx+1}: multivariate data not supported.")
            continue

        if not is_supervised_mode and not supports_unsupervised:
            _log_warn(f"Skipping trial {idx+1}: algorithm requires supervision.")
            continue
            
        trial_window = per_trial_windows[idx] if dataset_periodic_cfg else None
        filtered_trials.append((idx, single_X, single_y_true, modality, trial_window))

    if not filtered_trials:
        raise ValueError("No compatible trials found after applying algorithm capability filters.")

    all_metrics: List[Dict[str, float]] = []
    last_fitted_algorithm = None

    suffix_fragment = run_suffix or ""
    mode_fragment = f"_{supervision_mode}" if supervision_mode else ""
    base_run_name = f"{cfg.algorithm.name}_{cfg.dataset.name}{mode_fragment}{suffix_fragment}"

    for trial_idx, single_X, single_y_true, trial_modality, trial_window in filtered_trials:
        if is_deadline_exceeded(deadline):
            _log_warn("Deadline reached — skipping remaining trials.")
            raise TimeoutException(
                "Parent run deadline reached before launching the next trial."
            )
        run_name = f"{base_run_name}_{trial_idx+1}"
        with mlflow.start_run(run_name=run_name, nested=True) as trial_run:
            def _ensure_time_budget(stage: str) -> None:
                if is_deadline_exceeded(deadline):
                    _log_warn(f"Trial {trial_idx+1}: deadline reached during {stage}.")
                    mlflow.set_tag("status", "timeout")
                    raise TimeoutException(
                        f"Parent run deadline reached while {stage} (trial {trial_idx+1})."
                    )

            _ensure_time_budget("trial setup")
            trial_label = f"Trial {trial_idx+1}/{len(filtered_trials)}"
            _log_step(f"{_BOLD}{trial_label}{_RESET}  {_DIM}({trial_run.info.run_id[:12]}){_RESET}")
            mlflow.log_param("algorithm_name", cfg.algorithm.name)
            mlflow.set_tag("algorithm_name", cfg.algorithm.name)
            mlflow.set_tag("modality", trial_modality)
            mlflow.log_param("modality", trial_modality)
            mlflow.set_tag("dataset_name", cfg.dataset.name)
            mlflow.set_tag("dataset_trial_index", trial_idx + 1)
            mlflow.set_tag("algorithm_task", algorithm_task)
            mlflow.set_tag("status", "running")
            mlflow.set_tag("supervision_mode", supervision_mode)
            mlflow.log_param("supervision_mode", supervision_mode)
            
            use_labels = is_supervised_mode and supports_semi_supervised
            mlflow.log_param("algorithm_semi_supervised", use_labels)
            mlflow.set_tag("algorithm_semi_supervised", use_labels)

            if extra_params:
                mlflow.log_params({key: str(value) for key, value in extra_params.items()})
            if extra_tags:
                for key, value in extra_tags.items():
                    mlflow.set_tag(key, str(value))

            trial_start_time = time.monotonic()
            try:
                trial_X_arr = np.asarray(single_X)
                if trial_X_arr.ndim == 1:
                    trial_X_arr = trial_X_arr.reshape(-1, 1)
                trial = pd.DataFrame(
                    [
                        {
                            "dataset": cfg.dataset.name,
                            "trial_index": float(trial_idx + 1),
                            "modality": trial_modality,
                            "n_timepoints": float(trial_X_arr.shape[0]),
                            "n_channels": float(trial_X_arr.shape[1]),
                            "label_cardinality": float(np.unique(np.asarray(single_y_true)).size),
                        }
                    ]
                )
                try:
                    trial_dataset = mlflow_from_pandas(
                        trial,
                        name=f"{cfg.dataset.name}_trial_{trial_idx+1}",
                    )
                    mlflow.log_input(trial_dataset, context="trial")
                except Exception as exc:  # pragma: no cover - best effort logging
                    _log_warn(f"Failed to log trial dataset: {exc}")

                if "preprocessing" in cfg and cfg.preprocessing is not None:
                    _log_info("Preprocessing...")
                    preprocessor = hydra.utils.instantiate(cfg.preprocessing)

                    if np.isnan(single_X).any() or np.isinf(single_X).any():
                        _log_warn("NaN/Inf detected — applying linear interpolation.")
                        single_X[np.isinf(single_X)] = np.nan

                        df = pd.DataFrame(single_X)
                        df.interpolate(method="linear", axis=0, inplace=True, limit_direction="both")

                        if df.isnull().values.any():
                            _log_warn("NaNs persist after interpolation; filling with 0.")
                            df.fillna(0, inplace=True)

                        single_X = df.to_numpy()

                    single_X_3d = single_X.T.reshape(1, single_X.shape[1], single_X.shape[0])
                    transformed_X_3d = preprocessor.fit_transform(single_X_3d)
                    single_X = transformed_X_3d.squeeze(axis=0).T

                algorithm_cfg_container = OmegaConf.to_container(
                    cfg.algorithm.instance,
                    resolve=True,
                    throw_on_missing=True,
                )
                trial_algorithm_cfg = OmegaConf.create(algorithm_cfg_container)
                supervision_overrides = (
                    build_supervision_param_overrides(single_y_true)
                    if is_supervised_mode
                    else {}
                )

                with open_dict(trial_algorithm_cfg):
                    if trial_window is not None and "window_size" in trial_algorithm_cfg:
                        trial_algorithm_cfg.window_size = trial_window
                    elif trial_window is not None:
                        _log_warn("window_size override skipped (parameter not declared in algorithm config).")
                    if supervision_overrides:
                        applied_params: Dict[str, int] = {}
                        for param_name, param_value in supervision_overrides.items():
                            if param_name in trial_algorithm_cfg:
                                trial_algorithm_cfg[param_name] = int(param_value)
                                applied_params[param_name] = int(param_value)
                        if applied_params:
                            applied_items = ", ".join(
                                f"{name}={value}" for name, value in sorted(applied_params.items())
                            )
                            _log_info(f"Supervision overrides: {applied_items}")
                            mlflow.log_params({f"trial_supervision_override_{k}": v for k, v in applied_params.items()})
                        else:
                            _log_warn("Supervision overrides did not match any algorithm parameters.")

                if trial_window is not None:
                    mlflow.log_param("trial_window_size", trial_window)
                if is_supervised_mode:
                    cp_count = count_change_points(single_y_true)
                    state_count = count_unique_states(single_y_true)
                    mlflow.log_param("trial_supervision_change_points", cp_count)
                    mlflow.log_param("trial_supervision_states", state_count)

                _log_info(f"Algorithm: {_BOLD}{cfg.algorithm.name}{_RESET}")
                algorithm = hydra.utils.instantiate(trial_algorithm_cfg)
                last_fitted_algorithm = algorithm

                _ensure_time_budget("algorithm fitting")
                if is_supervised_mode and supports_semi_supervised:
                    _log_info("Fitting (semi-supervised)...")
                elif is_supervised_mode:
                    _log_info("Fitting (unsupervised — algorithm lacks semi-supervised support)...")
                else:
                    _log_info("Fitting (unsupervised)...")

                algorithm.fit(single_X, axis=0)

                _ensure_time_budget("prediction post-processing")
                y_pred_raw = algorithm.predict(single_X, axis=0)
                seq_length = single_X.shape[0]

                try:
                    y_pred, change_points_pred, prediction_artifacts = extract_prediction_components(
                        y_pred_raw,
                        seq_length,
                        returns_dense=returns_dense,
                        algorithm_task=algorithm_task,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to normalize prediction output for trial {trial_idx+1}: {exc}"
                    ) from exc

                ground_truth_change_points = labels_to_change_points(single_y_true)

                _log_info("Evaluating metrics...")
                _ensure_time_budget("metric evaluation")
                metrics: Dict[str, float] = {}
                artifacts = {
                    "predicted_labels": y_pred,
                    "predicted_change_points": np.asarray(change_points_pred, dtype=int),
                    "ground_truth_change_points": np.asarray(ground_truth_change_points, dtype=int),
                }
                for extra_name, extra_value in prediction_artifacts.items():
                    artifacts[f"prediction_{extra_name}"] = extra_value

                for metric_name, metric_conf in cfg.metric.items():
                    if not isinstance(metric_conf, DictConfig) or "_target_" not in metric_conf:
                        continue

                    metric_task = str(metric_conf.get("task", "state")).lower()
                    if metric_task not in allowed_metric_tasks:
                        continue

                    compute_params = OmegaConf.to_container(metric_conf.get("compute_params", {}), resolve=True)
                    if compute_params is None:
                        compute_params = {}
                    instantiate_conf = {
                        k: v
                        for k, v in metric_conf.items()
                        if k not in {"compute_params", "task"}
                    }
                    metric = hydra.utils.instantiate(instantiate_conf, **compute_params)

                    result = metric.compute(single_y_true, y_pred)

                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            metrics[f"{metric_name}_{key}"] = value
                        else:
                            if key != "segmentation_prediction":
                                artifacts[f"{metric_name}_{key}"] = value

                mlflow.log_metrics(metrics)
                with tempfile.TemporaryDirectory() as tmpdir:
                    for name, data in artifacts.items():
                        if isinstance(data, (list, tuple)):
                            data = np.asarray(data)
                        if isinstance(data, np.ndarray):
                            artifact_path = os.path.join(tmpdir, f"{name}.npy")
                            np.save(artifact_path, data)
                    if os.listdir(tmpdir):
                        mlflow.log_artifacts(tmpdir)

                all_metrics.append(metrics)
                print(_format_metrics_table(metrics, title=f"Trial {trial_idx+1}"))
                mlflow.set_tag("status", "completed")
            except TimeoutException:
                mlflow.set_tag("status", "timeout")
                raise
            except Exception as exc:
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error_message", str(exc))
                raise
            finally:
                duration = float(max(time.monotonic() - trial_start_time, 0.0))
                mlflow.log_metric("execution_time_seconds", duration)

    final_metrics: Dict[str, float] = {}
    if all_metrics:
        metric_keys = set().union(*[m.keys() for m in all_metrics])
        for key in metric_keys:
            values = [m[key] for m in all_metrics if key in m]
            if values:
                final_metrics[key] = float(np.mean(values))

    if final_metrics:
        print(_format_metrics_table(final_metrics, title="Aggregated"))
    return final_metrics, last_fitted_algorithm


def run_default_pipeline(cfg: DictConfig) -> None:
    """Entry point for the default pipeline.

    Parameters
    ----------
    cfg: DictConfig
        Hydra configuration for the run.
    """

    configure_mlflow(cfg)

    # Resolve experiment config (support both hierarchical and flattened structures)
    if "experiment" in cfg:
        experiment_cfg = cfg.experiment
    else:
        experiment_cfg = cfg

    experiment_name = experiment_cfg.get("name", "tsseg-experiment-default")
    mlflow.set_experiment(experiment_name)

    supervision_mode_raw = experiment_cfg.get("supervision_mode", "unsupervised")
    supervision_mode = str(supervision_mode_raw if supervision_mode_raw is not None else "unsupervised")
    supervision_mode = supervision_mode.strip().lower().replace(" ", "_").replace("-", "_")
    if supervision_mode not in {"semi_supervised", "supervised"}:
        supervision_mode = "unsupervised"

    base_run_name = cfg.get("run_name", f"{cfg.algorithm.name}_{cfg.dataset.name}")
    run_name = f"{base_run_name}_{supervision_mode}"

    with mlflow.start_run(run_name=run_name) as run:
        _header(
            f"{cfg.algorithm.name}  {_ARROW}  {cfg.dataset.name}  [{supervision_mode}]",
            run_id=run.info.run_id,
        )
        run_start_time = time.monotonic()

        params_to_log: Dict[str, object] = {}
        algo_params = OmegaConf.to_container(cfg.algorithm.instance, resolve=True, throw_on_missing=True)
        params_to_log.update({f"algo_{k}": v for k, v in algo_params.items()})

        if "preprocessing" in cfg and cfg.preprocessing is not None:
            preproc_params = OmegaConf.to_container(cfg.preprocessing, resolve=True, throw_on_missing=True)
            params_to_log.update({f"preproc_{k}": v for k, v in preproc_params.items()})

        dataset_params = OmegaConf.to_container(cfg.dataset.loader, resolve=True, throw_on_missing=True)
        params_to_log.update({f"dataset_{k}": v for k, v in dataset_params.items()})

        params_to_log["experiment_supervision_mode"] = supervision_mode
        mlflow.log_params(params_to_log)
        mlflow.log_param("algorithm_name", cfg.algorithm.name)
        mlflow.set_tag("algorithm_name", cfg.algorithm.name)
        mlflow.log_param("supervision_mode", supervision_mode)
        mlflow.set_tag("supervision_mode", supervision_mode)

        timeout_seconds, deadline = resolve_parent_deadline(
            timeout_seconds=experiment_cfg.get("timeout_seconds"),
            timeout_hours=experiment_cfg.get("timeout_hours"),
        )
        if timeout_seconds is not None:
            hours = timeout_seconds / 3600.0
            _log_info(f"Time budget: {int(timeout_seconds)}s (~{hours:.2f}h)")
            mlflow.log_param("timeout_seconds", timeout_seconds)
            mlflow.log_param("timeout_hours", hours)

        try:
            metrics, _ = run_single_experiment(cfg, deadline=deadline)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("status", "completed")
        except TimeoutException:
            timeout_display = int(timeout_seconds) if timeout_seconds else "configured budget"
            _log_error(f"Run timed out after {timeout_display}s.")
            mlflow.set_tag("status", "timeout")
        except Exception as exc:  # pragma: no cover - logging for debugging
            _log_error(f"Run failed: {exc}")
            import traceback

            traceback.print_exc()
            mlflow.set_tag("status", "failed")
            mlflow.log_param("error_message", str(exc))
        finally:
            duration = float(max(time.monotonic() - run_start_time, 0.0))
            mlflow.log_metric("execution_time_seconds", duration)
            _footer(run_name, duration)


__all__ = ["run_default_pipeline", "TimeoutException"]
