"""Pipeline utilities for TS segmentation experiments."""

from typing import Callable

from omegaconf import DictConfig

from .default_pipeline import run_default_pipeline
from .grid_pipeline import run_grid_pipeline
from .tuning_pipeline import (
    DatasetTooSmallError,
    HyperparameterTuningPipeline,
    TrialSegment,
    run_tuning_pipeline,
)


PIPELINE_REGISTRY: dict[str, Callable[[DictConfig], None]] = {
    "default": run_default_pipeline,
    "grid": run_grid_pipeline,
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
    "HyperparameterTuningPipeline",
    "PIPELINE_REGISTRY",
    "TrialSegment",
    "get_pipeline_runner",
    "run_default_pipeline",
    "run_grid_pipeline",
    "run_tuning_pipeline",
]
