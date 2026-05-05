"""Pipeline utilities for TS segmentation experiments."""

from typing import Callable

from omegaconf import DictConfig

from .default_pipeline import run_default_pipeline
from .default_pipeline_v2 import (
    ExperimentResult,
    RunContext,
    run_default_pipeline_v2,
)
from .grid_pipeline import run_grid_pipeline
from .grid_pipeline_v2 import run_grid_pipeline_v2
from .clustering_pipeline import run_state_evaluation_pipeline
from .tuning_pipeline import (
    DatasetTooSmallError,
    HyperparameterTuningPipeline,
    TrialSegment,
    run_tuning_pipeline,
)


PIPELINE_REGISTRY: dict[str, Callable[[DictConfig], None]] = {
    # v2 pipelines (per-trial timeout, partial metrics, audit fixes)
    "default": run_default_pipeline_v2,
    "default_v2": run_default_pipeline_v2,
    "grid": run_grid_pipeline_v2,
    "grid_v2": run_grid_pipeline_v2,
    # Legacy v1 (preserved for reproducibility of prior runs)
    "default_v1": run_default_pipeline,
    "grid_v1": run_grid_pipeline,
    # Other pipelines
    "clustering": run_state_evaluation_pipeline,
    "tuning": run_tuning_pipeline,
}


def get_pipeline_runner(name: str) -> Callable[[DictConfig], None]:
    """Return the callable associated with the requested pipeline name."""

    normalized = str(name).strip().lower()
    if normalized not in PIPELINE_REGISTRY:
        available = ", ".join(sorted(PIPELINE_REGISTRY))
        raise KeyError(f"Unknown pipeline '{name}'. Available pipelines: {available}.")
    return PIPELINE_REGISTRY[normalized]


__all__ = [
    "DatasetTooSmallError",
    "ExperimentResult",
    "HyperparameterTuningPipeline",
    "PIPELINE_REGISTRY",
    "RunContext",
    "TrialSegment",
    "get_pipeline_runner",
    "run_default_pipeline",
    "run_default_pipeline_v2",
    "run_grid_pipeline",
    "run_grid_pipeline_v2",
    "run_state_evaluation_pipeline",
    "run_tuning_pipeline",
]
