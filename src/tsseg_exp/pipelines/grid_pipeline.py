"""Grid search pipeline for TS segmentation experiments."""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mlflow
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf

try:  # pragma: no cover - submitit is optional at runtime
    import submitit
except ImportError:  # pragma: no cover - handled gracefully at runtime
    submitit = None

from .default_pipeline import TimeoutException, run_single_experiment
from tsseg_exp.utils.main_helpers import (
    CHANGE_POINT_PARAM_ALIASES,
    SEGMENT_COUNT_PARAM_ALIASES,
    STATE_COUNT_PARAM_ALIASES,
    configure_mlflow,
    resolve_parent_deadline,
)


_SUPERVISION_ALIASES = {
    alias.lower()
    for alias in (
        list(CHANGE_POINT_PARAM_ALIASES)
        + list(SEGMENT_COUNT_PARAM_ALIASES)
        + list(STATE_COUNT_PARAM_ALIASES)
    )
}


def _is_supervised_mode(cfg: DictConfig) -> bool:
    if "experiment" not in cfg:
        # Fallback if experiment structure is flattened at root due to # @package _global_
        supervision_mode_raw = cfg.get("supervision_mode", "unsupervised")
    else:
        supervision_mode_raw = cfg.experiment.get("supervision_mode", "unsupervised")
    supervision_mode = str(supervision_mode_raw if supervision_mode_raw is not None else "unsupervised")
    normalized = supervision_mode.strip().lower().replace(" ", "_").replace("-", "_")
    return normalized in {"semi_supervised", "supervised"}


def _prune_supervision_parameters(
    cfg: DictConfig, tunable_spec: Dict[str, Any]
) -> Dict[str, Any]:
    if not tunable_spec or not _is_supervised_mode(cfg):
        return tunable_spec

    retained: Dict[str, Any] = {}
    removed: List[str] = []
    for raw_name, spec in tunable_spec.items():
        normalized = str(raw_name).strip().lower()
        if normalized in _SUPERVISION_ALIASES:
            removed.append(str(raw_name))
            continue
        retained[str(raw_name)] = spec

    if removed:
        removed_list = ", ".join(sorted(removed))
        print(
            f"[grid] Removing supervision-controlled parameters from grid search: {removed_list}."
        )

    return retained


def _normalise_tunable_parameters(spec: Any) -> Dict[str, Any]:
    if spec is None:
        return {}

    if isinstance(spec, (DictConfig, ListConfig)):
        spec = OmegaConf.to_container(spec, resolve=True, throw_on_missing=False)

    if isinstance(spec, dict):
        return spec

    if isinstance(spec, list):
        normalised: Dict[str, Any] = {}
        for entry in spec:
            if isinstance(entry, (DictConfig, ListConfig)):
                entry = OmegaConf.to_container(entry, resolve=True, throw_on_missing=False)
            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    "Each 'tunable_parameters' list entry must map a single name to a specification."
                )
            name, value = next(iter(entry.items()))
            normalised[str(name)] = value
        return normalised

    raise ValueError("Unsupported 'tunable_parameters' structure; expected dict or list of mappings.")


def _expand_parameter_axis(name: str, spec: Any) -> List[Any]:
    if isinstance(spec, (list, tuple)):
        return list(spec)
    if isinstance(spec, DictConfig):
        spec = OmegaConf.to_container(spec, resolve=True, throw_on_missing=False)

    if isinstance(spec, dict):
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


def _build_parameter_grid(cfg: DictConfig) -> List[Dict[str, Any]]:
    tunable_spec = _normalise_tunable_parameters(cfg.algorithm.get("tunable_parameters"))
    tunable_spec = _prune_supervision_parameters(cfg, tunable_spec)
    if not tunable_spec:
        return [{}]

    axes: List[List[Tuple[str, Any]]] = []
    for param_name, param_spec in tunable_spec.items():
        axis_values = _expand_parameter_axis(param_name, param_spec)
        if not axis_values:
            continue
        axes.append([(param_name, value) for value in axis_values])

    if not axes:
        return [{}]

    grid: List[Dict[str, Any]] = []
    for combination in _cartesian_product(*axes):
        candidate = {name: value for name, value in combination}
        grid.append(candidate)
    return grid


def _cartesian_product(*axes: Iterable[Tuple[str, Any]]) -> Iterable[List[Tuple[str, Any]]]:
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


def _format_run_suffix(params: Dict[str, Any]) -> str:
    if not params:
        return ""
    parts = []
    for name, value in sorted(params.items()):
        value_str = str(value).replace(" ", "")
        parts.append(f"{name}_{value_str}")
    return "_" + "_".join(parts)


def _apply_parameter_overrides(cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
    cfg_copy = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    for key, value in overrides.items():
        cfg_copy.algorithm.instance[key] = value
    return cfg_copy


def _sanitize_param_key(key: str) -> str:
    """Normalise Hydra keys before logging to MLflow."""

    return str(key).strip("_").replace(".", "_")


def _stringify_param_value(value: Any) -> Any:
    """Convert parameter values into MLflow-friendly types."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return json.dumps([_stringify_param_value(v) for v in value])
    if isinstance(value, dict):
        return json.dumps({str(k): _stringify_param_value(v) for k, v in value.items()})
    if isinstance(value, (set, frozenset)):
        return json.dumps(sorted(_stringify_param_value(v) for v in value))
    return value


def _prefixed_params(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """Attach a prefix while sanitising keys for logging."""

    prefixed: Dict[str, Any] = {}
    for raw_key, raw_value in values.items():
        clean_key = _sanitize_param_key(raw_key)
        prefixed[f"{prefix}_{clean_key}"] = raw_value
    return prefixed


class GridPipeline:
    """Coordinate distributed or sequential grid execution."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.grid_cfg = cfg.get("grid", {})
        self.parameter_grid = _build_parameter_grid(cfg)
        self.base_run_name = cfg.get("run_name", f"{cfg.algorithm.name}_{cfg.dataset.name}")
        self.execution_mode = self._resolve_execution_mode()

    def run(self) -> None:
        print(f"[grid] Generated {len(self.parameter_grid)} parameter combinations.")
        if self.execution_mode == "worker":
            self._run_worker()
            return

        if not self.parameter_grid:
            print("[grid] No tunable parameters detected; falling back to default pipeline.")
            self._run_single_variant(self.cfg, {}, role="worker")
            return

        executor = self._create_submitit_executor()
        if executor is None or self.grid_cfg.get("force_sequential", False):
            if executor is None:
                print("[grid] Submitit launcher unavailable; executing sequentially in-process.")
            else:
                print("[grid] Sequential execution forced; running combinations locally.")
            self._run_sequential()
            return

        self._run_submitit(executor)

    # ------------------------------------------------------------------
    # Controller helpers
    # ------------------------------------------------------------------
    def _create_submitit_executor(self) -> Optional[Any]:
        if submitit is None:
            return None

        try:
            hydra_cfg = HydraConfig.get()
        except Exception:  # pragma: no cover - outside Hydra runtime
            return None

        launcher_cfg = getattr(hydra_cfg, "launcher", None)
        if not launcher_cfg:
            return None

        launcher_dict = OmegaConf.to_container(launcher_cfg, resolve=True, throw_on_missing=False) or {}
        output_dir = hydra_cfg.runtime.output_dir or os.getcwd()
        submitit_dir = self.grid_cfg.get(
            "submitit_folder",
            str(Path(output_dir) / "submitit_grid_jobs"),
        )
        folder = Path(submitit_dir).resolve()
        folder.mkdir(parents=True, exist_ok=True)

        executor = submitit.AutoExecutor(folder=folder)
        exec_params = self._extract_submitit_parameters(launcher_dict)
        executor.update_parameters(**exec_params)
        return executor

    def _run_submitit(self, executor: Any) -> None:
        base_command = self._build_base_command()
        working_dir = self._working_directory()
        base_overrides = self._base_overrides()

        jobs: List[Tuple[Dict[str, Any], Any]] = []

        for index, combo in enumerate(self.parameter_grid):
            overrides = self._build_job_overrides(list(base_overrides), combo)
            command = base_command + overrides
            job_name = self._job_name(combo, index)
            executor.update_parameters(name=job_name)
            job = executor.submit(_run_cli_command, command, working_dir)
            jobs.append((combo, job))
            print(
                f"[grid] Submitted job '{job_name}' with overrides: {overrides}."
            )

        for combo, job in jobs:
            try:
                job.result()
                print(f"[grid] Job for {combo} completed successfully.")
            except Exception as exc:  # pragma: no cover - submitit error propagation
                print(f"[grid] Job for {combo} failed: {exc}.")
                raise

    def _extract_submitit_parameters(self, cfg_dict: Dict[str, Any]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        generic_keys = {
            "timeout_min",
            "cpus_per_task",
            "gpus_per_node",
            "tasks_per_node",
            "mem_gb",
            "nodes",
            "stderr_to_stdout",
            "name",
        }
        for key in generic_keys:
            if key in cfg_dict:
                params[key] = cfg_dict[key]

        slurm_specific_keys = {
            "partition",
            "qos",
            "account",
            "gres",
            "constraint",
            "comment",
            "mail_type",
            "mail_user",
            "mem",
            "mem_per_gpu",
            "mem_per_cpu",
            "num_gpus",
            "ntasks_per_node",
            "srun_args",
            "exclude",
            "time",
            "gpus_per_task",
            "signal_delay_s",
            "cpus_per_gpu",
            "dependency",
            "use_srun",
            "array_parallelism",
            "additional_parameters",
            "setup",
            "wckey",
            "exclusive",
            "nodelist",
        }

        for key in slurm_specific_keys:
            if key not in cfg_dict:
                continue
            value = cfg_dict[key]
            if key == "array_parallelism":
                params["slurm_array_parallelism"] = value
                continue

            if key == "setup":
                setup_cmds: List[str]
                if isinstance(value, (list, tuple)):
                    setup_cmds = [str(cmd) for cmd in value]
                elif value is None:
                    setup_cmds = []
                else:
                    setup_cmds = [str(value)]

                cleanup_env_cmds = ["unset SLURM_ARRAY_JOB_ID", "unset SLURM_ARRAY_TASK_ID"]
                for cmd in cleanup_env_cmds:
                    if cmd not in setup_cmds:
                        setup_cmds.append(cmd)

                params["slurm_setup"] = setup_cmds
                continue

            params[f"slurm_{key}"] = value

        return params

    def _build_base_command(self) -> List[str]:
        entrypoint = self.grid_cfg.get("entrypoint")
        if entrypoint:
            script_path = Path(entrypoint)
            if not script_path.is_absolute():
                script_path = Path(self._working_directory()) / script_path
            script = str(script_path)
        else:
            script = str(Path(__file__).resolve().parent.parent / "main.py")
        return [sys.executable, script]

    def _working_directory(self) -> str:
        try:
            return HydraConfig.get().runtime.cwd
        except Exception:  # pragma: no cover - fallback outside Hydra
            return os.getcwd()

    def _base_overrides(self) -> List[str]:
        try:
            overrides = list(HydraConfig.get().overrides.task)
        except Exception:  # pragma: no cover - fallback
            return []

        filtered: List[str] = []
        for override in overrides:
            if override.startswith("grid.execution_mode"):
                continue
            if override.startswith("grid.run_suffix"):
                continue
            if override.startswith("run_name=") or override.startswith("+run_name="):
                continue
            filtered.append(override)
        return filtered

    def _build_job_overrides(self, base_overrides: List[str], combo: Dict[str, Any]) -> List[str]:
        overrides = list(base_overrides)

        # if not any(ov.startswith("+pipeline=") or ov.startswith("pipeline=") for ov in overrides):
        #    overrides.append("+pipeline=grid")

        overrides.append("+grid.execution_mode=worker")

        run_suffix = _format_run_suffix(combo)
        target_run_name = f"{self.base_run_name}{run_suffix}" if run_suffix else self.base_run_name
        overrides.append(f"+run_name={self._quote_override_value(target_run_name)}")

        for name, value in combo.items():
            overrides.append(self._format_param_override(name, value))

        return overrides

    def _job_name(self, combo: Dict[str, Any], index: int) -> str:
        base_name = self.grid_cfg.get("job_name", self.base_run_name) or "grid"
        suffix = _format_run_suffix(combo)
        candidate = f"{base_name}{suffix or ''}_{index:03d}"
        return candidate.replace("/", "_")

    def _run_sequential(self) -> None:
        for combo in self.parameter_grid:
            variant_cfg = _apply_parameter_overrides(self.cfg, combo)
            self._run_single_variant(variant_cfg, combo, role="sequential")

    # ------------------------------------------------------------------
    # Worker helpers
    # ------------------------------------------------------------------
    def _run_worker(self) -> None:
        combo = self._extract_active_parameters()
        self._run_single_variant(self.cfg, combo, role="worker")

    def _run_single_variant(self, variant_cfg: DictConfig, combo: Dict[str, Any], role: str = "worker") -> None:
        configure_mlflow(variant_cfg)

        experiment_name = "default_experiment"
        if "experiment" in variant_cfg and variant_cfg.experiment and "name" in variant_cfg.experiment:
            experiment_name = variant_cfg.experiment.name
        elif "name" in variant_cfg:
            experiment_name = variant_cfg.name
        
        mlflow.set_experiment(experiment_name)

        run_suffix = _format_run_suffix(combo)
        run_name = variant_cfg.get("run_name", f"{variant_cfg.algorithm.name}_{variant_cfg.dataset.name}")
        if run_suffix and run_suffix not in run_name:
            run_name = f"{run_name}{run_suffix}"

        t_seconds = None
        t_hours = None
        if "experiment" in variant_cfg and variant_cfg.experiment:
            t_seconds = variant_cfg.experiment.get("timeout_seconds")
            t_hours = variant_cfg.experiment.get("timeout_hours")
        else:
            t_seconds = variant_cfg.get("timeout_seconds")
            t_hours = variant_cfg.get("timeout_hours")

        timeout_seconds, deadline = resolve_parent_deadline(
            timeout_seconds=t_seconds,
            timeout_hours=t_hours,
        )

        with mlflow.start_run(run_name=run_name) as run:
            print(f"--- Starting grid run: {run_name} (MLflow Run ID: {run.info.run_id}) ---")
            print("Config:")
            print(OmegaConf.to_yaml(variant_cfg))
            if timeout_seconds is not None:
                print(
                    f"Deadline set to {int(timeout_seconds)} seconds (~{timeout_seconds/3600:.2f} h) from start of grid run."
                )
            run_start_time = time.monotonic()

            params_to_log: Dict[str, Any] = {}

            algo_params = OmegaConf.to_container(
                variant_cfg.algorithm.instance,
                resolve=True,
                throw_on_missing=True,
            )
            prefixed_algo = _prefixed_params("algo", algo_params)
            params_to_log.update(prefixed_algo)

            prefixed_preproc: Dict[str, Any] = {}
            if "preprocessing" in variant_cfg and variant_cfg.preprocessing is not None:
                preproc_params = OmegaConf.to_container(
                    variant_cfg.preprocessing,
                    resolve=True,
                    throw_on_missing=True,
                )
                prefixed_preproc = _prefixed_params("preproc", preproc_params)
                params_to_log.update(prefixed_preproc)

            dataset_params = OmegaConf.to_container(
                variant_cfg.dataset.loader,
                resolve=True,
                throw_on_missing=True,
            )
            params_to_log.update(_prefixed_params("dataset", dataset_params))

            prefixed_grid: Dict[str, Any] = {}
            if combo:
                prefixed_grid = _prefixed_params("grid", combo)
                params_to_log.update(prefixed_grid)

            params_to_log["pipeline"] = "grid"
            params_to_log["grid_execution_role"] = role

            serialised_params = {
                key: _stringify_param_value(val) for key, val in params_to_log.items()
            }
            mlflow.log_params({key: str(value) for key, value in serialised_params.items()})
            mlflow.log_param("algorithm_name", variant_cfg.algorithm.name)
            mlflow.set_tag("algorithm_name", variant_cfg.algorithm.name)

            mlflow.set_tag("pipeline", "grid")
            mlflow.set_tag("grid_execution_role", role)
            if timeout_seconds is not None:
                mlflow.log_param("timeout_seconds", timeout_seconds)
                mlflow.log_param("timeout_hours", timeout_seconds / 3600.0)

            grid_parameters_payload = "{}"
            algo_target_value: Optional[str] = None
            preproc_target_value: Optional[str] = None

            if prefixed_algo:
                for key, value in prefixed_algo.items():
                    mlflow.set_tag(key, str(_stringify_param_value(value)))
                algo_target = prefixed_algo.get("algo_target") or prefixed_algo.get("algo_target_")
                if algo_target is not None:
                    algo_target_value = str(_stringify_param_value(algo_target))
                    mlflow.set_tag("algo_target", algo_target_value)

            if prefixed_preproc:
                for key, value in prefixed_preproc.items():
                    mlflow.set_tag(key, str(_stringify_param_value(value)))
                preproc_target = prefixed_preproc.get("preproc_target") or prefixed_preproc.get("preproc_target_")
                if preproc_target is not None:
                    preproc_target_value = str(_stringify_param_value(preproc_target))
                    mlflow.set_tag("preproc_target", preproc_target_value)

            if prefixed_grid:
                for key, value in prefixed_grid.items():
                    mlflow.set_tag(key, str(_stringify_param_value(value)))
                combo_payload = {
                    _sanitize_param_key(k): _stringify_param_value(v) for k, v in combo.items()
                }
                grid_parameters_payload = json.dumps(combo_payload)
                mlflow.set_tag("grid_parameters", grid_parameters_payload)
            else:
                mlflow.set_tag("grid_parameters", "{}")

            child_params: Dict[str, str] = {}
            for source in (prefixed_algo, prefixed_preproc, prefixed_grid):
                for key, value in source.items():
                    child_params[key] = str(_stringify_param_value(value))
            child_params["pipeline"] = "grid"

            child_tags: Dict[str, str] = {
                "pipeline": "grid",
                "grid_execution_role": role,
                "grid_parameters": grid_parameters_payload,
            }
            if algo_target_value is not None:
                child_tags["algo_target"] = algo_target_value
            if preproc_target_value is not None:
                child_tags["preproc_target"] = preproc_target_value
            for source in (prefixed_algo, prefixed_preproc, prefixed_grid):
                for key, value in source.items():
                    child_tags[key] = str(_stringify_param_value(value))

            try:
                metrics, _ = run_single_experiment(
                    variant_cfg,
                    extra_params=child_params,
                    extra_tags=child_tags,
                    run_suffix=run_suffix,
                    deadline=deadline,
                )
                mlflow.log_metrics(metrics)
                mlflow.set_tag("status", "completed")
            except TimeoutException:
                timeout_display = int(timeout_seconds) if timeout_seconds else "configured budget"
                print(f"!!! Grid run {run_name} timed out after {timeout_display} seconds. !!!")
                mlflow.set_tag("status", "timeout")
            except Exception as exc:  # pragma: no cover - logging for debugging
                print(f"!!! Grid run {run_name} failed with an exception: {exc} !!!")
                import traceback

                traceback.print_exc()
                mlflow.set_tag("status", "failed")
                mlflow.log_param("error_message", str(exc))
            finally:
                duration = float(max(time.monotonic() - run_start_time, 0.0))
                mlflow.log_metric("execution_time_seconds", duration)
                print(f"--- Finished grid run: {run_name} ---")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _resolve_execution_mode(self) -> str:
        mode_cfg = str(self.grid_cfg.get("execution_mode", "auto")).lower()
        if mode_cfg in {"worker", "controller"}:
            return mode_cfg

        try:
            hydra_mode = HydraConfig.get().mode
        except Exception:  # pragma: no cover
            hydra_mode = None

        if hydra_mode and str(hydra_mode).upper() == "MULTIRUN":
            return "worker"
        return "controller"

    def _extract_active_parameters(self) -> Dict[str, Any]:
        spec = _normalise_tunable_parameters(self.cfg.algorithm.get("tunable_parameters"))
        active: Dict[str, Any] = {}
        instance_cfg = self.cfg.algorithm.instance
        for name in spec.keys():
            if name in instance_cfg:
                active[name] = instance_cfg.get(name)
        return active

    def _format_param_override(self, name: str, value: Any) -> str:
        prefix = f"algorithm.instance.{name}="
        return prefix + self._quote_override_value(value)

    def _quote_override_value(self, value: Any) -> str:
        if isinstance(value, str):
            escaped = value.replace("\"", "\\\"")
            return f'"{escaped}"'
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)


def run_grid_pipeline(cfg: DictConfig) -> None:
    """Entry point bridging Hydra config with the grid pipeline."""

    configure_mlflow(cfg)
    GridPipeline(cfg).run()


def _run_cli_command(command: List[str], workdir: str) -> int:
    """Execute a CLI command for a grid worker via submitit."""

    print(f"[grid-worker] Executing command: {' '.join(command)} (cwd={workdir})")
    completed = subprocess.run(command, cwd=workdir)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Grid worker command failed with exit code {completed.returncode}: {' '.join(command)}"
        )
    return completed.returncode


__all__ = ["run_grid_pipeline"]
