import mlflow
import hydra
import logging
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import List, Dict, Any

from tsseg_exp.utils.main_helpers import configure_mlflow
from tsseg_exp.algorithms.cp_state_estimator import ChangePointToStateEstimator

log = logging.getLogger(__name__)

def _is_change_point_task(algo_name: str, config_root: Path) -> bool:
    """
    Checks if the algorithm is defined as a 'change_point' task in its YAML config.
    """
    algo_config_path = config_root / "algorithm" / f"{algo_name}.yaml"
    if not algo_config_path.exists():
        log.warning(f"Config for algorithm '{algo_name}' not found at {algo_config_path}. Assuming not eligible.")
        return False
        
    try:
        with open(algo_config_path, 'r') as f:
            # We use safe_load but need to handle potential Hydra specific tags if present (unlikely in basic task def)
            # Simple text parsing is safer against custom !tags if yaml loader fails
            content = f.read()
            if "task: change_point" in content or "task: \"change_point\"" in content:
                return True
            # Also try parsing to be sure
            data = yaml.safe_load(content)
            # handle cases where defaults list wrappers might exist, though unlikely for 'task' key which is usually top level
            if data and data.get("task") == "change_point":
                return True
    except Exception as e:
        log.warning(f"Failed to parse config for {algo_name}: {e}")
        
    return False

def run_state_evaluation_pipeline(cfg: DictConfig) -> None:
    """
    Pipeline to evaluate State Detection performance based on existing Change Point Detection runs.
    
    It iterates over completed runs from a source experiment, applies clustering on the segments,
    and computes state metrics.
    """
    # 1. Setup MLflow
    # Note: 'configure_mlflow' usually sets registry URI etc.
    configure_mlflow(cfg)
    
    # Ensure we use the proper DB URI. 
    # 1. Custom Config Override
    tracking_uri = cfg.experiment.get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # 2. If no remote URI set (by config or configure_mlflow), fallback to local SQLite
    current_uri = mlflow.get_tracking_uri()
    if not current_uri or str(current_uri).startswith("file:"):
        # Default fallback consistent with user request
        project_root = Path(hydra.utils.get_original_cwd())
        fallback_uri = f"sqlite:///{project_root}/mlflow.db"
        log.info(f"No remote MLflow URI configured. Falling back to local DB: {fallback_uri}")
        mlflow.set_tracking_uri(fallback_uri)
    
    experiment_name = cfg.experiment.name
    mlflow.set_experiment(experiment_name)
    log.info(f"Target Experiment: {experiment_name}")
    log.info(f"Tracking URI: {mlflow.get_tracking_uri()}")

    # 2. Fetch Source Runs
    source_experiment_id = str(cfg.experiment.get("source_experiment_id"))
    if not source_experiment_id:
        raise ValueError("cfg.experiment.source_experiment_id is required.")
        
    log.info(f"Scanning Source Experiment ID: {source_experiment_id} for completed runs...")
    
    # We fetch runs with minimal columns to filter efficiently
    base_query = f"attributes.status = 'FINISHED'"
    
    # Allow custom filter string from config
    custom_filter = cfg.experiment.get("source_run_filter")
    filter_parts = [base_query]
    if custom_filter:
        filter_parts.append(custom_filter)
    
    # OPTIMIZATION: Push down dataset filter to MLflow query to avoid fetching all runs
    target_dataset = cfg.dataset.get("name")
    if target_dataset:
        filter_parts.append(f"tags.dataset_name = '{target_dataset}'")
        
    query = " AND ".join(filter_parts)
        
    log.info(f"Searching runs with filter: {query}")
    
    # READ-ONLY OPTIMIZATION:
    # If using remote server (http), switch to local direct SQLite access for reading huge lists of runs
    current_uri = mlflow.get_tracking_uri()
    temp_uri = None
    if str(current_uri).startswith("http"):
        # Try to locate local DB file
        candidates = [
            Path("/scratch/fchavell/tsseg-exp/results/mlflow.db"), # Standard structure on cluster
            Path("/scratch/fchavell/tsseg-exp/mlflow.db"),         # Alternative root location
            Path(hydra.utils.get_original_cwd()) / "results" / "mlflow.db",
            Path("results/mlflow.db").resolve()
        ]
        
        local_db_path = None
        for cand in candidates:
            if cand.exists():
                local_db_path = cand
                break
        
        if local_db_path:
             temp_uri = f"sqlite:///{local_db_path}"
             log.info(f"Switching to direct local DB access for reading: {temp_uri}")
             mlflow.set_tracking_uri(temp_uri)
        else:
             log.warning(f"Could not find local mlflow.db to optimize read. Checked: {[str(p) for p in candidates]}")

    try:
        # Fetch runs
        # note: We could add max_results if needed, but per-dataset filtering should be enough
        # We add a safety limit to prevent Timeouts on SQLite when under load
        source_runs = mlflow.search_runs(
            experiment_ids=[source_experiment_id], 
            filter_string=query,
            max_results=500 
        )
    finally:
        if temp_uri:
            mlflow.set_tracking_uri(current_uri)
            log.info(f"Restored tracking URI to remote: {current_uri}")
    
    if source_runs.empty:
        log.error(f"No finished runs found in experiment {source_experiment_id} for query: {query}")
        return

    # 3. Create Lookup Index for Runs
    # Structure: index[(algo_name, dataset_name, trial_idx)] -> run_id
    # We need to rely on tags.
    required_tags = ['tags.algorithm_name', 'tags.dataset_name', 'tags.dataset_trial_index']
    
    # Verify tags exist
    missing_tags = [t for t in required_tags if t not in source_runs.columns]
    if missing_tags:
        log.warning(f"Missing logical tags in source runs: {missing_tags}. Trying to proceed usually fails run matching.")
        # Attempt minimal fallback for trial index if missing
        if 'tags.dataset_trial_index' in missing_tags:
            source_runs['tags.dataset_trial_index'] = "1"
            
    # Client-side filter is now redundant but harmless (except if dataset name in config differs from tag)
    # kept for safety if needed, or we can remove it. 
    if target_dataset and 'tags.dataset_name' in source_runs.columns:
         source_runs = source_runs[source_runs['tags.dataset_name'] == target_dataset]
         
    if source_runs.empty:
        log.warning(f"No runs found for dataset {target_dataset} in source experiment.")
        return

    # Identify valid Change Point algorithms
    config_root = Path(hydra.utils.get_original_cwd()) / "configs"
    unique_algos = source_runs['tags.algorithm_name'].unique()
    
    valid_cp_algos = []
    for algo in unique_algos:
        if _is_change_point_task(algo, config_root):
            valid_cp_algos.append(algo)
            
    log.info(f"Identified {len(valid_cp_algos)} valid Change Point algorithms for evaluation: {valid_cp_algos}")
    
    # Build lookup map
    # We only keep runs belonging to valid CP algos
    valid_runs = source_runs[source_runs['tags.algorithm_name'].isin(valid_cp_algos)]
    
    # Map: key -> run_id
    run_lookup = {}
    for _, row in valid_runs.iterrows():
        # Ensure string types for key
        key = (
            str(row['tags.algorithm_name']), 
            str(row['tags.dataset_name']), 
            str(row['tags.dataset_trial_index'])
        )
        run_lookup[key] = row.run_id
        
    log.info(f"Indexed {len(run_lookup)} valid source trials.")

    # 4. Processing Loop
    # We load the dataset ONCE (since we are likely running one dataset per job in cluster mode typically)
    log.info(f"Loading dataset: {target_dataset}")
    # cfg.dataset.loader is a partial object typically, we call it to get data
    X_list, y_list = hydra.utils.call(cfg.dataset.loader)
    
    # Ensure list format (some loaders might return single array if not fragmented)
    if not isinstance(X_list, list): 
        X_list, y_list = [X_list], [y_list]
        
    num_trials = len(X_list)
    log.info(f"Dataset loaded. Trials available (time series count): {num_trials}")
    
    # Iterate over Algorithms
    target_algos_cfg = cfg.experiment.get("target_algorithms", "all")
    algos_to_process = valid_cp_algos if target_algos_cfg == "all" else [a for a in valid_cp_algos if a in target_algos_cfg]

    for algo_name in algos_to_process:
        log.info(f">>> Evaluating State Detection for: {algo_name} <<<")
        
        # We start a parent run for this Algo-Dataset combination to keep structure clean
        parent_run_name = f"{algo_name}_{target_dataset}_StateEval"
        
        # Check if parent run already exists? (Optional, skipping for now)
        
        with mlflow.start_run(run_name=parent_run_name) as parent_run:
            # Tagging consistent with Default Pipeline expectations for querying
            mlflow.log_param("source_algorithm", algo_name)
            mlflow.log_param("dataset", target_dataset)
            mlflow.log_param("source_experiment_id", source_experiment_id)
            
            # Important: Tag metrics task type to differentiate results
            mlflow.set_tag("task", "state")
            mlflow.set_tag("algorithm_name", f"{algo_name}+Clustering")
            mlflow.set_tag("dataset_name", target_dataset)
            mlflow.set_tag("pipeline", "state_evaluation")
            
            # Loop over trials
            for trial_idx in range(1, num_trials + 1):
                s_trial_idx = str(trial_idx)
                
                # Fetch Data
                X_trial = X_list[trial_idx - 1]
                y_trial = y_list[trial_idx - 1]
                
                # Find Source Run ID
                source_key = (algo_name, target_dataset, s_trial_idx)
                source_run_id = run_lookup.get(source_key)
                
                if not source_run_id:
                    log.debug(f"  [Skipped] Trial {s_trial_idx}: No source run found.")
                    continue
                    
                child_run_name = f"{parent_run_name}_{s_trial_idx}"
                
                with mlflow.start_run(run_name=child_run_name, nested=True) as child_run:
                    mlflow.set_tag("dataset_trial_index", s_trial_idx)
                    mlflow.set_tag("source_run_id", source_run_id)
                    
                    try:
                        # 1. Instantiate and Fit Wrapper
                        # Note: We create a fresh estimator per trial
                        # We use 'auto' for n_clusters meaning we use Ground Truth to dictate N
                        n_clusters = cfg.algorithm.get("n_clusters", "auto")
                        
                        est = ChangePointToStateEstimator(
                            source_run_id=source_run_id,
                            n_clusters=n_clusters,
                            # Ideally we could hydrate the clustering_estimator from config here
                            # e.g. hydra.utils.instantiate(cfg.algorithm.clustering)
                        )
                        
                        est.fit(X_trial, y_trial) # y_trial needed for 'auto' K
                        
                        # 2. Predict
                        y_pred = est.predict(X_trial)
                        
                        # 3. Compute State Metrics
                        metrics_res = {}
                        
                        # We iterate over metric configurations
                        # cfg.metric is a DictConfig
                        if hasattr(cfg, "metric"):
                            for m_name, m_cfg in cfg.metric.items():
                                # Check task type
                                if m_cfg.get("task") != "state":
                                    continue
                                
                                # Instantiate metric
                                # Exclude metadata keys before instantiation
                                metric_params = {k: v for k, v in m_cfg.items() if k not in ['task', 'compute_params']}
                                if '_target_' in metric_params:
                                    metric_obj = hydra.utils.instantiate(metric_params)
                                    
                                    # Compute
                                    # Some metrics might return dicts, others scalars
                                    # Adapting to match typical library behavior
                                    try:
                                        val = metric_obj.compute(y_trial, y_pred)
                                        if isinstance(val, dict):
                                            for sub_k, sub_v in val.items():
                                                metrics_res[f"{m_name}_{sub_k}"] = sub_v
                                        else:
                                            metrics_res[m_name] = val
                                    except Exception as me:
                                        log.warning(f"Metric {m_name} failed: {me}")
                        
                        mlflow.log_metrics(metrics_res)
                        mlflow.set_tag("status", "completed")
                        
                    except Exception as e:
                        log.exception(f"Trial {s_trial_idx} failed.")
                        mlflow.set_tag("status", "failed")
                        mlflow.set_tag("error", str(e))

    log.info("State Evaluation Pipeline Completed.")
