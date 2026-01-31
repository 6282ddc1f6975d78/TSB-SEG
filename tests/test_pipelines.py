import pytest

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from tsseg_exp.pipelines.default_pipeline import run_default_pipeline

@pytest.fixture(scope="module")
def hydra_init():
    # Initialize hydra once for the module
    if not GlobalHydra.instance().is_initialized():
        # Point to the configs directory relative to this test file
        initialize(config_path="../configs", version_base=None)
    yield
    # GlobalHydra.instance().clear()

def run_pipeline_test(overrides):
    overrides.append("dataset.loader._target_=tsseg_exp.utils.test_utils.mock_loader")
    cfg = compose(config_name="config", overrides=overrides)
    
    try:
        run_default_pipeline(cfg)
    except ValueError as e:
        # Allow skipping if no compatible trials found (e.g. unsupervised run for supervised-only algo)
        if "No compatible trials found" in str(e):
            pytest.skip(f"Skipped due to capability mismatch: {e}")
        else:
            pytest.fail(f"Pipeline failed with error: {e}")
    except Exception as e:
        pytest.fail(f"Pipeline failed with error: {e}")

ALGORITHMS = [
    "amoc", "autoplait", "binseg", "bocd", "bottomup", 
    "clap", "clasp", "dynp", "e2usd", "eagglo", 
    "espresso", "fluss", "ggs", "hdp-hsmm", "hidalgo", 
    "icid", "igts", "kcpd", "patss", "pelt", 
    "prophet", "random", "tglad", "ticc", "time2state", 
    "tire", "vsax", "window"
]

EXPERIMENTS = ["default", "semi_supervised"]

@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_algorithm_pipeline(hydra_init, algorithm, experiment):
    run_pipeline_test([
        f"algorithm={algorithm}",
        "dataset=mocap",
        f"experiment={experiment}"
    ])
