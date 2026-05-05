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

from tsseg_exp.pipelines import get_pipeline_runner


def _resolve_pipeline_name(cfg: DictConfig) -> str:
    """Extract the pipeline name from config (string or sub-dict)."""
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
        runner = get_pipeline_runner(pipeline_name)
        runner(cfg)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()