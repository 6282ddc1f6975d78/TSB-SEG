"""Main entrypoint dispatching to the configured pipeline."""
from __future__ import annotations

import os
import warnings

# Suppress warnings and noisy logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Abseil (absl) logging commonly used by TensorFlow/JAX
os.environ["GLOG_minloglevel"] = "3"
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*Filesystem tracking backend.*")

import hydra
from omegaconf import DictConfig

from tsseg_exp.pipelines import PIPELINE_REGISTRY, get_pipeline_runner
from tsseg_exp.pipelines.clustering_pipeline import run_state_evaluation_pipeline


def _resolve_pipeline_name(cfg: DictConfig) -> str:
    if "pipeline" not in cfg:
        return "default"

    pipeline_value = cfg.pipeline
    if isinstance(pipeline_value, DictConfig):
        name = pipeline_value.get("name")
        if not name:
            return "default"
        return str(name)

    if pipeline_value is None:
        return "default"

    return str(pipeline_value)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint that resolves and runs the requested pipeline."""

    pipeline_name = _resolve_pipeline_name(cfg).strip().lower()

    try:
        # LOGIQUE DE DISPATCH
        if pipeline_name == "default":
            from tsseg_exp.pipelines.default_pipeline import run_default_pipeline

            run_default_pipeline(cfg)

        elif pipeline_name == "grid":
            from tsseg_exp.pipelines.grid_pipeline import run_grid_pipeline

            run_grid_pipeline(cfg)

        elif pipeline_name == "clustering":
            run_state_evaluation_pipeline(cfg)

        else:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()