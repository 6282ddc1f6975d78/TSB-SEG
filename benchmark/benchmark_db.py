import os
import yaml
import mlflow
import pandas as pd
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Union

class BenchmarkDataManager:
    """
    Manager to handle efficient data retrieval from MLflow (SQLite backend)
    and standardize terminology for the benchmark analysis.
    """

    def __init__(self, tracking_uri: str, project_root: Union[str, Path]):
        """
        Args:
            tracking_uri: The MLflow tracking URI (e.g., "sqlite:///mlflow.db")
            project_root: Path to the tsseg-exp project root.
        """
        self.tracking_uri = tracking_uri
        self.project_root = Path(project_root)
        mlflow.set_tracking_uri(self.tracking_uri)

    def get_experiment_names_from_configs(self) -> Dict[str, str]:
        """
        Parses YAML files in configs/experiment to retrieve the actual MLflow experiment names.
        
        Returns:
            Dict mapping 'config_key' (e.g., 'unsupervised') to 'experiment_name' 
            (e.g., 'tsseg-experiment-unsupervised-12-12').
        """
        config_path = self.project_root / "configs" / "experiment"
        experiments = {}
        
        target_configs = ["unsupervised", "semi_supervised", "grid_unsupervised", "grid_supervised"]
        
        for config_name in target_configs:
            yaml_file = config_path / f"{config_name}.yaml"
            if yaml_file.exists():
                with open(yaml_file, 'r') as f:
                    try:
                        data = yaml.safe_load(f)
                        if data and "name" in data:
                            experiments[config_name] = data["name"]
                    except yaml.YAMLError as e:
                        print(f"Warning: Could not parse {yaml_file}: {e}")
            else:
                print(f"Warning: Config file not found: {yaml_file}")
                
        return experiments

    def _resolve_experiment_ids(self, experiment_names: List[str] = None) -> List[mlflow.entities.Experiment]:
        """
        Helper to resolve experiment names to MLflow Experiment objects.
        If names is None, fetches all from config.
        """
        if experiment_names is None:
            experiment_map = self.get_experiment_names_from_configs()
            experiment_names = list(experiment_map.values())
            
        experiments = [
            mlflow.get_experiment_by_name(name) 
            for name in experiment_names 
            if mlflow.get_experiment_by_name(name) is not None
        ]
        
        return experiments

    def _determine_mode_from_name(self, exp_name: str, name_to_config_key: Dict[str, str]) -> str:
        """
        Helper to determine benchmark mode from experiment name.
        """
        # 1. Try exact match from config map
        config_key = name_to_config_key.get(exp_name)
        if config_key == "grid_unsupervised": return "grid_default"
        if config_key == "grid_supervised": return "grid_guided"
        if config_key == "unsupervised": return "default"
        if config_key == "semi_supervised": return "guided"
        
        # 2. Fallback to string matching (stricter)
        if "grid_unsupervised" in exp_name: return "grid_default"
        if "grid_supervised" in exp_name: return "grid_guided"
        if "unsupervised" in exp_name and "grid" not in exp_name: return "default"
        if ("semi_supervised" in exp_name or "supervised" in exp_name) and "grid" not in exp_name: return "guided"
        
        return "unknown"

    def fetch_runs(self, experiment_names: List[str] = None, max_results: int = 1000000) -> pd.DataFrame:
        """
        Fetches runs from MLflow. Optimized to select relevant columns.
        
        Args:
            experiment_names: List of experiment names to fetch. If None, fetches all 
                              found in configs.
            max_results: Maximum number of runs to fetch (default 100,000).
        """
        # Always get the map for accurate mode determination
        experiment_map = self.get_experiment_names_from_configs()
        
        if experiment_names is not None:
             print(f"Fetching runs for experiments: {experiment_names}")
        
        # Get Experiment IDs
        experiments = self._resolve_experiment_ids(experiment_names)
        
        if not experiments:
            return pd.DataFrame()
            
        exp_ids = [e.experiment_id for e in experiments]
        
        # Fetch runs (Pandas DataFrame)
        # We avoid fetching artifacts to save memory/time
        df = mlflow.search_runs(
            experiment_ids=exp_ids,
            filter_string="",
            run_view_type=mlflow.entities.ViewType.ALL,
            max_results=max_results
        )
        
        if df.empty:
            print("Warning: No runs found.")
            return df

        # --- Post-Processing & Standardization ---
        
        # 1. Extract critical tags/params if they are not top-level columns
        # (MLflow flattens them, e.g., 'params.algorithm', 'tags.dataset')
        
        # Ensure we have the basics
        required_cols = [
            'run_id', 'status', 'start_time', 'end_time', 
            'params.algorithm', 'params.dataset', 
            'tags.supervision_mode', 'tags.error_message'
        ]
        
        # 2. Standardize Benchmark Mode
        
        name_to_config_key = {v: k for k, v in experiment_map.items()}
        exp_id_name_map = {e.experiment_id: e.name for e in experiments}
        
        def get_mode(exp_id):
            exp_name = exp_id_name_map.get(exp_id, "")
            return self._determine_mode_from_name(exp_name, name_to_config_key)
            
        df['benchmark_mode'] = df['experiment_id'].apply(get_mode)

        # 3. Calculate Duration
        if 'start_time' in df.columns and 'end_time' in df.columns:
            # Ensure datetime format and UTC timezone to avoid "Cannot subtract tz-naive and tz-aware"
            df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
            df['end_time'] = pd.to_datetime(df['end_time'], utc=True)
            df['duration_s'] = (df['end_time'] - df['start_time']).dt.total_seconds()
        
        return df

    def get_sqlite_connection(self):
        """Returns a raw connection to the SQLite DB for custom SQL queries."""
        if self.tracking_uri.startswith("sqlite:///"):
            db_path = self.tracking_uri.replace("sqlite:///", "")
            return sqlite3.connect(db_path)
        else:
            raise ValueError("Tracking URI is not SQLite.")
        
    def fetch_heatmap_data_sql(self, experiment_names: List[str] = None) -> pd.DataFrame:
        """
        Fetches counts of FINISHED runs grouped by (experiment_id, algorithm, dataset).
        Filters for child runs (runs with 'mlflow.parentRunId' tag).
        Deduplicates by keeping only children of the LATEST parent run if duplicates exist.
        """
        # Always get the map for accurate mode determination
        experiment_map = self.get_experiment_names_from_configs()
        
        # Get Experiment IDs
        experiments = self._resolve_experiment_ids(experiment_names)
        
        if not experiments:
            return pd.DataFrame()
            
        exp_ids = [str(e.experiment_id) for e in experiments]
        exp_ids_str = ",".join(f"'{eid}'" for eid in exp_ids) # Quote IDs just in case
        
        conn = self.get_sqlite_connection()
        try:
            # 1. Identify Valid Parents (Latest per run_name)
            # Fetch all identified parents (either by having children or being a grid parent)
            # Typically parent runs don't have parentRunId. 
            # We filter parents by ensuring they are NOT children themselves (optional but safe).
            q_parents = f"""
            SELECT 
                r.run_uuid as parent_id,
                r.experiment_id,
                r.start_time,
                t_name.value as run_name
            FROM runs r
            LEFT JOIN tags t_name ON r.run_uuid = t_name.run_uuid AND t_name.key = 'mlflow.runName'
            WHERE r.experiment_id IN ({exp_ids_str})
            AND NOT EXISTS (
                SELECT 1 FROM tags t_check 
                WHERE t_check.run_uuid = r.run_uuid 
                AND t_check.key = 'mlflow.parentRunId'
            )
            """
            parents_df = pd.read_sql_query(q_parents, conn)
            
            valid_parent_ids = set()
            if not parents_df.empty:
                # Sort by start_time descending
                parents_df = parents_df.sort_values('start_time', ascending=False)
                # Drop duplicates based on experiment + run_name, keeping first (latest)
                unique_parents = parents_df.drop_duplicates(subset=['experiment_id', 'run_name'], keep='first')
                valid_parent_ids = set(unique_parents['parent_id'])
            
            # 2. Fetch Child Runs Info (Raw list, not aggregated yet)
            # We fetch all finished children, then filter in Python
            q_children = f"""
            SELECT
                r.experiment_id,
                r.run_uuid,
                t_parent.value as parent_id,
                COALESCE(p_algo.value, t_algo.value) as algorithm,
                COALESCE(p_data.value, t_data.value) as dataset
            FROM runs r
            JOIN tags t_parent ON r.run_uuid = t_parent.run_uuid AND t_parent.key = 'mlflow.parentRunId'
            LEFT JOIN params p_algo ON r.run_uuid = p_algo.run_uuid AND p_algo.key = 'algorithm_name'
            LEFT JOIN tags t_algo ON r.run_uuid = t_algo.run_uuid AND t_algo.key = 'algorithm_name'
            LEFT JOIN params p_data ON r.run_uuid = p_data.run_uuid AND p_data.key = 'dataset_name'
            LEFT JOIN tags t_data ON r.run_uuid = t_data.run_uuid AND t_data.key = 'dataset_name'
            WHERE r.status = 'FINISHED'
            AND r.experiment_id IN ({exp_ids_str})
            """
            
            children_df = pd.read_sql_query(q_children, conn)
            
        finally:
            conn.close()
            
        if children_df.empty:
            return pd.DataFrame()

        # 3. Filter children by valid parents (if we found parents)
        # If valid_parent_ids is empty string or None (rare), this might filter everything.
        # But if we have valid parents, we apply the filter.
        if valid_parent_ids:
            children_df = children_df[children_df['parent_id'].isin(valid_parent_ids)]
        
        if children_df.empty:
            return pd.DataFrame()

        # 4. Aggregate
        df = children_df.groupby(['experiment_id', 'algorithm', 'dataset']).size().reset_index(name='count')
            
        # Map experiment_id to mode
        name_to_config_key = {v: k for k, v in experiment_map.items()}
        exp_id_name_map = {e.experiment_id: e.name for e in experiments}
        
        def get_mode(exp_id):
            exp_name = exp_id_name_map.get(str(exp_id), "")
            return self._determine_mode_from_name(exp_name, name_to_config_key)
            
        if not df.empty:
            df['benchmark_mode'] = df['experiment_id'].apply(get_mode)
            
        return df

    def fetch_error_stats_sql(self, experiment_names: List[str] = None) -> pd.DataFrame:
        """
        Fetches error message counts for failed runs using direct SQL.
        Uses LEFT JOIN to include failed runs that might not have an error_message tag.
        Checks both tags and params for 'error_message'.
        """
        experiments = self._resolve_experiment_ids(experiment_names)
        
        if not experiments:
            return pd.DataFrame()
            
        exp_ids = [str(e.experiment_id) for e in experiments]
        exp_ids_str = ",".join(f"'{eid}'" for eid in exp_ids)
        
        query = f"""
        SELECT
            COALESCE(t.value, p.value, 'Unknown Error (No tag)') as error_message,
            COUNT(r.run_uuid) as count
        FROM runs r
        LEFT JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'error_message'
        LEFT JOIN params p ON r.run_uuid = p.run_uuid AND p.key = 'error_message'
        WHERE r.status != 'FINISHED'
        AND r.experiment_id IN ({exp_ids_str})
        GROUP BY COALESCE(t.value, p.value, 'Unknown Error (No tag)')
        ORDER BY count DESC
        """
        
        conn = self.get_sqlite_connection()
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()
            
        return df

    def fetch_integrity_stats_sql(self, experiment_names: List[str] = None) -> Dict[str, Union[int, pd.DataFrame]]:
        """
        Checks for data integrity issues using SQL (Missing params, Suspicious durations).
        Returns a dictionary with counts and samples.
        """
        experiments = self._resolve_experiment_ids(experiment_names)
        
        if not experiments:
            return {}
            
        exp_ids = [str(e.experiment_id) for e in experiments]
        exp_ids_str = ",".join(f"'{eid}'" for eid in exp_ids)
        
        conn = self.get_sqlite_connection()
        stats = {}
        
        try:
            # 1. Missing Algorithm Name
            q_missing_algo = f"""
            SELECT COUNT(*) as count FROM runs r
            LEFT JOIN params p ON r.run_uuid = p.run_uuid AND p.key = 'algorithm_name'
            WHERE r.experiment_id IN ({exp_ids_str}) AND p.value IS NULL
            """
            stats['missing_algo_count'] = pd.read_sql_query(q_missing_algo, conn).iloc[0]['count']
            
            # 2. Missing Dataset Name
            q_missing_data = f"""
            SELECT COUNT(*) as count FROM runs r
            LEFT JOIN params p ON r.run_uuid = p.run_uuid AND p.key = 'dataset_name'
            WHERE r.experiment_id IN ({exp_ids_str}) AND p.value IS NULL
            """
            stats['missing_dataset_count'] = pd.read_sql_query(q_missing_data, conn).iloc[0]['count']
            
            # 3. Suspicious Duration (< 1s)
            # MLflow stores time in ms (BIGINT)
            q_short_duration = f"""
            SELECT COUNT(*) as count FROM runs r
            WHERE r.experiment_id IN ({exp_ids_str}) 
            AND r.status = 'FINISHED' 
            AND (r.end_time - r.start_time) < 1000
            """
            stats['short_duration_count'] = pd.read_sql_query(q_short_duration, conn).iloc[0]['count']
            
            # 4. Sample of short duration runs
            q_sample_short = f"""
            SELECT r.run_uuid, r.experiment_id, (r.end_time - r.start_time) as duration_ms
            FROM runs r
            WHERE r.experiment_id IN ({exp_ids_str}) 
            AND r.status = 'FINISHED' 
            AND (r.end_time - r.start_time) < 1000
            LIMIT 5
            """
            stats['sample_short_duration'] = pd.read_sql_query(q_sample_short, conn)
            
        finally:
            conn.close()
            
        return stats
    
    def fetch_run_details_sql(self, experiment_name: str, dataset_name: str = None, algorithm_name: str = None, limit: int = 1000000, require_artifacts: bool = False) -> pd.DataFrame:
        """
        Fetches detailed run info for inspection, including parent/child relationship.
        
        Args:
            require_artifacts: If True, filters runs that likely have artifacts (Proxy: Has metrics logged).
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return pd.DataFrame()
        
        exp_id = str(experiment.experiment_id)
        
        # Base query
        query = f"""
        SELECT 
            r.run_uuid,
            r.experiment_id,
            r.start_time,
            r.status,
            r.artifact_uri,
            (r.end_time - r.start_time) / 1000.0 as duration_s,
            COALESCE(p_algo.value, t_algo.value, p_algo2.value, t_algo2.value) as algorithm,
            COALESCE(p_data.value, t_data.value, p_data2.value, t_data2.value) as dataset,
            t_parent.value as parent_run_id,
            COALESCE(t_error.value, '') as error_message
        FROM runs r
        LEFT JOIN tags t_parent ON r.run_uuid = t_parent.run_uuid AND t_parent.key = 'mlflow.parentRunId'
        LEFT JOIN params p_algo ON r.run_uuid = p_algo.run_uuid AND p_algo.key = 'algorithm_name'
        LEFT JOIN tags t_algo ON r.run_uuid = t_algo.run_uuid AND t_algo.key = 'algorithm_name'
        LEFT JOIN params p_algo2 ON r.run_uuid = p_algo2.run_uuid AND p_algo2.key = 'algorithm'
        LEFT JOIN tags t_algo2 ON r.run_uuid = t_algo2.run_uuid AND t_algo2.key = 'algorithm'
        
        LEFT JOIN params p_data ON r.run_uuid = p_data.run_uuid AND p_data.key = 'dataset_name'
        LEFT JOIN tags t_data ON r.run_uuid = t_data.run_uuid AND t_data.key = 'dataset_name'
        LEFT JOIN params p_data2 ON r.run_uuid = p_data2.run_uuid AND p_data2.key = 'dataset'
        LEFT JOIN tags t_data2 ON r.run_uuid = t_data2.run_uuid AND t_data2.key = 'dataset'

        LEFT JOIN tags t_error ON r.run_uuid = t_error.run_uuid AND t_error.key = 'error_message'
        WHERE r.experiment_id = '{exp_id}'
        """
        
        params = []
        if dataset_name and dataset_name != "All":
            query += " AND (p_data.value = ? OR t_data.value = ? OR p_data2.value = ? OR t_data2.value = ?)"
            params.extend([dataset_name, dataset_name, dataset_name, dataset_name])
            
        if algorithm_name and algorithm_name != "All":
            query += " AND (p_algo.value = ? OR t_algo.value = ? OR p_algo2.value = ? OR t_algo2.value = ?)"
            params.extend([algorithm_name, algorithm_name, algorithm_name, algorithm_name])
            
        if require_artifacts:
            # Proxy: A run with artifacts usually has metrics logged. 
            # We check if there's at least one metric for this run.
            query += " AND (EXISTS (SELECT 1 FROM metrics m WHERE m.run_uuid = r.run_uuid) OR r.status = 'FINISHED')"
            
        query += " ORDER BY r.start_time DESC LIMIT ?"
        params.append(limit)
        
        conn = self.get_sqlite_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
            
        return df

    def fetch_failure_analysis_sql(self, experiment_names: List[str] = None) -> pd.DataFrame:
        """
        Analyzes failure modes by aggregating Child Run stats onto Parent Runs.
        Optimized to avoid slow joins on unindexed tag values.
        """
        experiments = self._resolve_experiment_ids(experiment_names)
        
        if not experiments:
            return pd.DataFrame()
            
        exp_ids = [str(e.experiment_id) for e in experiments]
        exp_ids_str = ",".join(f"'{eid}'" for eid in exp_ids)
        
        conn = self.get_sqlite_connection()
        try:
            # 1. Aggregate Child Stats (Group by Parent ID)
            # This is fast because we filter children by experiment_id first, 
            # and looking up parentRunId tag is fast (PK).
            q_children = f"""
            SELECT
                t.value as parent_id,
                COUNT(r.run_uuid) as total_children,
                SUM(CASE WHEN r.status = 'FINISHED' THEN 1 ELSE 0 END) as children_finished,
                SUM(CASE WHEN r.status != 'FINISHED' THEN 1 ELSE 0 END) as children_failed,
                SUM(CASE WHEN t_err.value LIKE '%Unable to allocate%' OR t_err.value LIKE '%Killed%' OR t_err.value LIKE '%MemoryError%' THEN 1 ELSE 0 END) as child_oom,
                SUM(CASE WHEN t_err.value LIKE '%timed-out%' OR t_err.value LIKE '%time limit%' OR t_err.value LIKE '%Timeout%' THEN 1 ELSE 0 END) as child_timeout
            FROM runs r
            JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'mlflow.parentRunId'
            LEFT JOIN tags t_err ON r.run_uuid = t_err.run_uuid AND t_err.key = 'error_message'
            WHERE r.experiment_id IN ({exp_ids_str})
            GROUP BY t.value
            """
            df_children = pd.read_sql_query(q_children, conn)
            
            # 2. Fetch Parent Metadata
            # We fetch runs that are NOT children (don't have parentRunId tag)
            q_parents = f"""
            SELECT
                r.experiment_id,
                r.run_uuid as parent_id,
                r.start_time,
                r.status as parent_status,
                (r.end_time - r.start_time) as parent_duration_ms,
                COALESCE(p_algo.value, t_algo.value) as algorithm,
                COALESCE(p_data.value, t_data.value) as dataset,
                COALESCE(t_err.value, '') as parent_error,
                t_name.value as run_name
            FROM runs r
            LEFT JOIN params p_algo ON r.run_uuid = p_algo.run_uuid AND p_algo.key = 'algorithm_name'
            LEFT JOIN tags t_algo ON r.run_uuid = t_algo.run_uuid AND t_algo.key = 'algorithm_name'
            LEFT JOIN params p_data ON r.run_uuid = p_data.run_uuid AND p_data.key = 'dataset_name'
            LEFT JOIN tags t_data ON r.run_uuid = t_data.run_uuid AND t_data.key = 'dataset_name'
            LEFT JOIN tags t_err ON r.run_uuid = t_err.run_uuid AND t_err.key = 'error_message'
            LEFT JOIN tags t_name ON r.run_uuid = t_name.run_uuid AND t_name.key = 'mlflow.runName'
            WHERE r.experiment_id IN ({exp_ids_str})
            AND NOT EXISTS (
                SELECT 1 FROM tags t_check 
                WHERE t_check.run_uuid = r.run_uuid 
                AND t_check.key = 'mlflow.parentRunId'
            )
            """
            df_parents = pd.read_sql_query(q_parents, conn)
            
        finally:
            conn.close()
            
        # Deduplicate parents: keep only the latest start_time for each (experiment_id, run_name)
        if not df_parents.empty and 'run_name' in df_parents.columns:
            df_parents = df_parents.sort_values(by='start_time', ascending=False)
            df_parents = df_parents.drop_duplicates(subset=['experiment_id', 'run_name'], keep='first')
            
        if df_parents.empty:
            return pd.DataFrame()

        # 3. Merge in Python
        df = pd.merge(df_parents, df_children, on='parent_id', how='left')
        
        # Fill NaNs for parents with no children (e.g. Incompatible)
        fill_cols = ['total_children', 'children_finished', 'children_failed', 'child_oom', 'child_timeout']
        df[fill_cols] = df[fill_cols].fillna(0)
        
        # Map experiment IDs to names
        id_map = {str(e.experiment_id): e.name for e in experiments}
        df['experiment_name'] = df['experiment_id'].astype(str).map(id_map)
        
        # Determine Global Status
        def classify_row(row):
            # 1. Structural Incompatibility
            if row['total_children'] == 0 and "No compatible trials" in row['parent_error']:
                return "Incompatible"
            
            # 2. Global Timeout 
            # Condition A: Explicit long duration (> 22h approx)
            # Condition B: Missing duration (NaN) + Not Finished (implies interrupted/timeout)
            duration = row['parent_duration_ms']
            is_long_duration = (pd.notna(duration) and duration > 80000000)
            is_interrupted = (pd.isna(duration) and row['parent_status'] != 'FINISHED')
            
            if is_long_duration or is_interrupted:
                return "Global Timeout"
            
            # 3. All Good
            if row['parent_status'] == 'FINISHED' and row['children_failed'] == 0:
                return "Complete"
            
            # 4. Partial / Mixed
            if row['children_failed'] > 0:
                return "Partial Failures"
            
            if row['parent_status'] != 'FINISHED':
                return "Parent Failed (Other)"
                
            return "Unknown"

        df['audit_status'] = df.apply(classify_row, axis=1)
            
        return df
    
    def fetch_child_metrics_sql(self, parent_ids: List[str], metrics: List[str] = None) -> pd.DataFrame:
        """
        Fetches metrics for all children of the specified parent runs.
        Returns a DataFrame with columns: [parent_id, child_id, metric_name, metric_value]
        """
        if not parent_ids:
            return pd.DataFrame()
            
        # Sanitize IDs
        parent_ids_str = ",".join(f"'{pid}'" for pid in parent_ids)
        
        metric_filter = ""
        if metrics:
            metrics_str = ",".join(f"'{m}'" for m in metrics)
            metric_filter = f"AND m.key IN ({metrics_str})"
            
        query = f"""
        SELECT 
            t_parent.value as parent_id,
            r.run_uuid as child_id,
            m.key as metric_name,
            m.value as metric_value
        FROM runs r
        JOIN tags t_parent ON r.run_uuid = t_parent.run_uuid AND t_parent.key = 'mlflow.parentRunId'
        JOIN metrics m ON r.run_uuid = m.run_uuid
        WHERE t_parent.value IN ({parent_ids_str})
        {metric_filter}
        """
        
        conn = self.get_sqlite_connection()
        try:
            df = pd.read_sql_query(query, conn)
        finally:
            conn.close()
            
        return df

    def get_available_metrics(self, experiment_names: List[str] = None) -> List[str]:
        """Returns a sorted list of all unique metric keys available in the experiments."""
        experiments = self._resolve_experiment_ids(experiment_names)
        if not experiments:
            return []
        
        exp_ids = [str(e.experiment_id) for e in experiments]
        exp_ids_str = ",".join(f"'{eid}'" for eid in exp_ids)
        
        query = f"""
        SELECT DISTINCT key 
        FROM metrics m
        JOIN runs r ON m.run_uuid = r.run_uuid
        WHERE r.experiment_id IN ({exp_ids_str})
        """
        
        conn = self.get_sqlite_connection()
        try:
            df = pd.read_sql_query(query, conn)
            metrics = df['key'].tolist()
            return sorted(metrics)
        finally:
            conn.close()

    def fetch_strict_comparison_data(self, metric_key='covering_score') -> pd.DataFrame:
        """
        Fetches comparison data with strict series matching logic.
        Returns: DataFrame with columns [algorithm, dataset, mode, series_name, metric_value]
        """
        exp_map = self.get_experiment_names_from_configs()
        
        groups = {
            'default': ['unsupervised'],
            'guided': ['semi_supervised'],
            'grid_default': ['grid_unsupervised'],
            'grid_guided': ['grid_supervised']
        }
        
        conn = self.get_sqlite_connection()
        all_series_data = []
        
        try:
            for mode_group, config_keys in groups.items():
                exp_names = [exp_map.get(k) for k in config_keys if k in exp_map]
                exp_ids = []
                for ename in exp_names:
                    if ename:
                        e = mlflow.get_experiment_by_name(ename)
                        if e: exp_ids.append(str(e.experiment_id))
                
                if not exp_ids:
                    continue
                    
                exp_ids_str = ",".join(f"'{e}'" for e in exp_ids)
                is_grid = 'grid' in mode_group
                is_time_metric = any(k in metric_key.lower() for k in ['time', 'duration', 'seconds'])

                # --- Step 1: Identify Target Parents ---
                
                if is_grid:
                    # Optimized Grid Selection
                    
                    if is_time_metric:
                        val_expr = "COALESCE(m.value, (r.end_time - r.start_time)/1000.0)"
                        join_type = "LEFT JOIN"
                    else:
                        val_expr = "m.value"
                        join_type = "LEFT JOIN"

                    q_candidates = f"""
                    SELECT 
                        r.run_uuid as parent_id,
                        t_algo.value as algorithm,
                        t_data.value as dataset,
                        r.start_time,
                        {val_expr} as parent_metric
                    FROM runs r
                    JOIN tags t_algo ON r.run_uuid = t_algo.run_uuid AND t_algo.key = 'algorithm'
                    JOIN tags t_data ON r.run_uuid = t_data.run_uuid AND t_data.key = 'dataset'
                    {join_type} metrics m ON r.run_uuid = m.run_uuid AND m.key = '{metric_key}'
                    WHERE r.experiment_id IN ({exp_ids_str})
                    AND r.status = 'FINISHED'
                    """
                    df_candidates = pd.read_sql_query(q_candidates, conn)
                    
                    if df_candidates.empty:
                        continue
                        
                    # Aggregate from children if needed
                    if df_candidates['parent_metric'].isna().sum() > 0:
                         if is_time_metric:
                             q_child_agg = f"""
                             SELECT 
                                t_p.value as parent_id,
                                AVG(COALESCE(m.value, (r.end_time - r.start_time)/1000.0)) as child_mean_score
                             FROM runs r
                             JOIN tags t_p ON r.run_uuid = t_p.run_uuid AND t_p.key = 'mlflow.parentRunId'
                             LEFT JOIN metrics m ON r.run_uuid = m.run_uuid AND m.key = '{metric_key}'
                             WHERE r.experiment_id IN ({exp_ids_str})
                             GROUP BY t_p.value
                             """
                         else:
                             q_child_agg = f"""
                             SELECT 
                                t_p.value as parent_id,
                                AVG(m.value) as child_mean_score
                             FROM runs r
                             JOIN tags t_p ON r.run_uuid = t_p.run_uuid AND t_p.key = 'mlflow.parentRunId'
                             JOIN metrics m ON r.run_uuid = m.run_uuid AND m.key = '{metric_key}'
                             WHERE r.experiment_id IN ({exp_ids_str})
                             GROUP BY t_p.value
                             """
                         
                         df_agg = pd.read_sql_query(q_child_agg, conn)
                         df_candidates = pd.merge(df_candidates, df_agg, on='parent_id', how='left')
                         df_candidates['final_score'] = df_candidates['parent_metric'].fillna(df_candidates['child_mean_score'])
                    else:
                         df_candidates['final_score'] = df_candidates['parent_metric']
                    
                    df_candidates = df_candidates.dropna(subset=['final_score'])
                    if df_candidates.empty:
                        continue

                    # Select Best Config (Higher is Better, unless time)
                    ascending = True if is_time_metric else False
                    target_parents = df_candidates.sort_values('final_score', ascending=ascending) \
                                                .drop_duplicates(subset=['algorithm', 'dataset'], keep='first')
                else:
                    # Non-Grid: Latest Parent
                    q_parents = f"""
                    SELECT 
                        r.run_uuid as parent_id,
                        t_algo.value as algorithm,
                        t_data.value as dataset,
                        r.start_time
                    FROM runs r
                    JOIN tags t_algo ON r.run_uuid = t_algo.run_uuid AND t_algo.key = 'algorithm'
                    JOIN tags t_data ON r.run_uuid = t_data.run_uuid AND t_data.key = 'dataset'
                    WHERE r.experiment_id IN ({exp_ids_str})
                    """
                    df_parents = pd.read_sql_query(q_parents, conn)
                    if df_parents.empty: continue
                        
                    target_parents = df_parents.sort_values('start_time', ascending=False) \
                                               .drop_duplicates(subset=['algorithm', 'dataset'], keep='first')

                parent_ids = target_parents['parent_id'].tolist()
                if not parent_ids:
                    continue

                # --- Step 2: Fetch Children Data ---
                
                chunk_size = 500
                for i in range(0, len(parent_ids), chunk_size):
                    chunk = parent_ids[i:i+chunk_size]
                    chunk_str = ",".join(f"'{p}'" for p in chunk)
                    
                    if is_time_metric:
                        q_children = f"""
                        SELECT 
                            t_parent.value as parent_id,
                            t_name.value as series_name,
                            COALESCE(m.value, (r.end_time - r.start_time)/1000.0) as metric_value
                        FROM runs r
                        JOIN tags t_parent ON r.run_uuid = t_parent.run_uuid AND t_parent.key = 'mlflow.parentRunId'
                        JOIN tags t_name   ON r.run_uuid = t_name.run_uuid   AND t_name.key = 'mlflow.runName'
                        LEFT JOIN metrics m ON r.run_uuid = m.run_uuid     AND m.key = '{metric_key}'
                        WHERE t_parent.value IN ({chunk_str})
                        AND (m.value IS NOT NULL OR (r.end_time IS NOT NULL AND r.start_time IS NOT NULL))
                        """
                    else:
                        q_children = f"""
                        SELECT 
                            t_parent.value as parent_id,
                            t_name.value as series_name,
                            m.value as metric_value
                        FROM runs r
                        JOIN tags t_parent ON r.run_uuid = t_parent.run_uuid AND t_parent.key = 'mlflow.parentRunId'
                        JOIN tags t_name   ON r.run_uuid = t_name.run_uuid   AND t_name.key = 'mlflow.runName'
                        JOIN metrics m     ON r.run_uuid = m.run_uuid     AND m.key = '{metric_key}'
                        WHERE t_parent.value IN ({chunk_str})
                        """
                    
                    df_children = pd.read_sql_query(q_children, conn)
                    merged = pd.merge(df_children, target_parents[['parent_id', 'algorithm', 'dataset']], on='parent_id')
                    merged['mode'] = mode_group
                    all_series_data.append(merged)
            
            if not all_series_data:
                return pd.DataFrame()
                
            final_df = pd.concat(all_series_data, ignore_index=True)
            final_df['dataset'] = final_df['dataset'].str.lower()
            return final_df
            
        finally:
            conn.close()
