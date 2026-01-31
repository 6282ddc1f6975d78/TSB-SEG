import os
import glob
import yaml
import mlflow
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

class MLflowBenchmarkManager:
    """
    Cleaner, config-driven manager for accessing MLflow data.
    """

    def __init__(self, config_path: Union[str, Path], tracking_uri: str = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.tracking_uri = tracking_uri or mlflow.get_tracking_uri()
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            
        self.exp_name_map = self._resolve_experiment_names()

    def _resolve_experiment_names(self) -> Dict[str, List[str]]:
        """
        Resolves config experiment patterns (e.g. 'exp-*') to actual existing MLflow experiment names.
        Returns: Dict[logical_key -> List[actual_names]]
        """
        resolved = {}
        all_exps = mlflow.search_experiments(view_type=mlflow.entities.ViewType.ALL)
        all_exp_names = [e.name for e in all_exps]

        for key, pattern in self.config.get('experiments', {}).items():
            if '*' in pattern:
                import fnmatch
                matches = fnmatch.filter(all_exp_names, pattern)
                resolved[key] = matches
            else:
                if pattern in all_exp_names:
                    resolved[key] = [pattern]
                else:
                    resolved[key] = []
        return resolved

    def get_experiment_ids(self, keys: List[str] = None) -> List[str]:
        """Returns list of experiment IDs for given logical keys (e.g. ['unsupervised'])."""
        if keys is None:
            keys = list(self.exp_name_map.keys())
            
        target_names = []
        for k in keys:
            target_names.extend(self.exp_name_map.get(k, []))
            
        if not target_names:
            return []
            
        # Bulk fetch IDs
        all_exps = mlflow.search_experiments(filter_string="", view_type=mlflow.entities.ViewType.ALL)
        return [e.experiment_id for e in all_exps if e.name in target_names]

    def fetch_runs(self, experiment_keys: List[str] = None, filter_string: str = "", limit: int = 10000) -> pd.DataFrame:
        """
        Fetches runs using mlflow.search_runs and standardizes columns based on config aliases.
        """
        exp_ids = self.get_experiment_ids(experiment_keys)
        if not exp_ids:
            return pd.DataFrame()

        # Fetch using standard API
        df = mlflow.search_runs(
            experiment_ids=exp_ids,
            filter_string=filter_string,
            max_results=limit,
            run_view_type=mlflow.entities.ViewType.ALL 
        )

        if df.empty:
            return df

        # --- standardization ---
        aliases = self.config.get('aliases', {})
        for target_col, candidate_cols in aliases.items():
            # Find which candidates exist in this DF
            valid_cols = [c for c in candidate_cols if c in df.columns]
            
            if not valid_cols:
                df[target_col] = None
                continue
                
            # Coalesce: Use bfill (backfill) or iterate. 
            # ffill/bfill works if we subset properties. 
            # A cleaner way using combine_first chain:
            df[target_col] = df[valid_cols[0]]
            for c in valid_cols[1:]:
                df[target_col] = df[target_col].combine_first(df[c])

        # --- Time calculations ---
        if 'start_time' in df.columns and 'end_time' in df.columns:
            # MLflow returns Timestamp objects usually, but let's ensure
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
            df['duration_s'] = (df['end_time'] - df['start_time']).dt.total_seconds()

        # --- Cleanup ---
        # Keep base columns + aliased columns + metrics
        base_cols = self.config.get('base_columns', [])
        aliased_cols = list(aliases.keys())
        
        # We also want to keep metrics (cols starting with 'metrics.')
        metric_cols = [c for c in df.columns if c.startswith('metrics.')]
        
        # And params? generally yes, or just the standardized ones?
        # Let's keep standardized ones primarily, but allow access to everything 
        # via the original DF if needed. 
        # But for 'clean' output, we select specific ones.
        
        cols_to_return = base_cols + aliased_cols + ['duration_s'] + metric_cols
        
        # Ensure exist
        cols_to_return = [c for c in cols_to_return if c in df.columns]
        
        return df[cols_to_return].copy()

    def fetch_metrics(self, run_ids: List[str], metrics: List[str] = None) -> pd.DataFrame:
        """
        Fetches full metric history for specific runs.
        Note: mlflow.search_runs only returns the *latest* metric value.
        For history (plots), we need explicit API calls (slow loop) or standard search if latest is enough.
        """
        # If the user wants full history, we must loop.
        # This is where the SQL approach was faster in benchmark_db.py.
        # But to use "tools" properly:
        data = []
        client = mlflow.MlflowClient()
        for rid in run_ids:
            if metrics:
                for m_key in metrics:
                    try:
                        history = client.get_metric_history(rid, m_key)
                        for point in history:
                            data.append({
                                'run_id': rid,
                                'key': m_key,
                                'value': point.value,
                                'step': point.step,
                                'timestamp': point.timestamp
                            })
                    except:
                        pass
        return pd.DataFrame(data)

    def get_db_connection(self):
        """Returns a raw connection to the SQLite DB for custom SQL queries."""
        import sqlite3
        if self.tracking_uri.startswith("sqlite:///"):
            db_path = self.tracking_uri.replace("sqlite:///", "")
            return sqlite3.connect(db_path)
        else:
            return None

    def fetch_parent_stats_sql(self, experiment_keys: List[str] = None) -> pd.DataFrame:
        """
        Fetches parent run statistics (children counts) using fast SQL aggregation.
        Replaces the slow Python-side aggregation.
        """
        conn = self.get_db_connection()
        if not conn:
            return pd.DataFrame()

        exp_ids = self.get_experiment_ids(experiment_keys)
        if not exp_ids:
            return pd.DataFrame()
            
        exp_ids_str = ",".join(f"'{eid}'" for eid in exp_ids)
        
        # 1. Aggregate Child Stats (Group by Parent ID)
        # We assume children are tagged with 'parentRunId'
        q_children = f"""
        SELECT
            t.value as parent_run_id,
            COUNT(r.run_uuid) as total_children,
            SUM(CASE WHEN r.status = 'FINISHED' THEN 1 ELSE 0 END) as children_finished,
            SUM(CASE WHEN r.status != 'FINISHED' THEN 1 ELSE 0 END) as children_failed
        FROM runs r
        JOIN tags t ON r.run_uuid = t.run_uuid AND t.key = 'mlflow.parentRunId'
        WHERE r.experiment_id IN ({exp_ids_str})
        GROUP BY t.value
        """

        try:
            df_children = pd.read_sql_query(q_children, conn)
            
            # 2. Fetch Parent Metadata
            # Parents are runs in these experiments that are NOT children themselves
            # AND match the aggregated IDs (or we fetch all potential parents)
            q_parents = f"""
            SELECT
                r.run_uuid as run_id,
                r.experiment_id,
                r.start_time,
                r.status,
                COALESCE(p_algo.value, t_algo.value) as algorithm,
                COALESCE(p_data.value, t_data.value) as dataset
            FROM runs r
            LEFT JOIN params p_algo ON r.run_uuid = p_algo.run_uuid AND p_algo.key = 'algorithm_name'
            LEFT JOIN tags t_algo ON r.run_uuid = t_algo.run_uuid AND t_algo.key = 'algorithm_name'
            LEFT JOIN params p_data ON r.run_uuid = p_data.run_uuid AND p_data.key = 'dataset_name'
            LEFT JOIN tags t_data ON r.run_uuid = t_data.run_uuid AND t_data.key = 'dataset_name'
            WHERE r.experiment_id IN ({exp_ids_str})
            AND NOT EXISTS (
                SELECT 1 FROM tags t_check 
                WHERE t_check.run_uuid = r.run_uuid 
                AND t_check.key = 'mlflow.parentRunId'
            )
            """
            
            df_parents = pd.read_sql_query(q_parents, conn)
            
            if df_parents.empty:
                return pd.DataFrame()
                
            # Merge
            df_merged = df_parents.merge(df_children, left_on='run_id', right_on='parent_run_id', how='left')
            
            # Fill NaNs
            cols = ['total_children', 'children_finished', 'children_failed']
            df_merged[cols] = df_merged[cols].fillna(0).astype(int)
            
            # Ensure valid run_id/parent_run_id consistency
            df_merged['parent_run_id'] = df_merged['run_id']
            
            return df_merged

        except Exception as e:
            print(f"SQL Parent Stats Fetch Error: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

    def fetch_metrics_sql(self, parent_ids: List[str], metric_keys: List[str] = None) -> pd.DataFrame:
        """
        Fast SQL-based fetch for metrics of child runs.
        Returns a DataFrame with [run_id, parent_run_id, key, value, start_time, trial_index].
        """
        conn = self.get_db_connection()
        if not conn:
            print("Warning: SQL acceleration not available (URI is not sqlite).")
            return pd.DataFrame() # Fallback or Raise

        if not parent_ids:
            return pd.DataFrame()

        # Chunking for SQL limits (SQLite typically handles many, but safe to batch huge lists)
        CHUNK_SIZE = 500
        all_dfs = []
        
        for i in range(0, len(parent_ids), CHUNK_SIZE):
            chunk_ids = parent_ids[i:i+CHUNK_SIZE]
            
            parent_ids_quoted = [f"'{pid}'" for pid in chunk_ids]
            ids_str = ",".join(parent_ids_quoted)
            
            query = f"""
            SELECT 
                r.run_uuid as run_id,
                t.value as parent_run_id,
                m.key,
                m.value,
                r.start_time,
                t_idx.value as trial_index
            FROM runs r
            JOIN tags t ON r.run_uuid = t.run_uuid
            JOIN metrics m ON r.run_uuid = m.run_uuid
            LEFT JOIN tags t_idx ON r.run_uuid = t_idx.run_uuid AND t_idx.key = 'dataset_trial_index'
            WHERE t.key = 'mlflow.parentRunId'
            AND t.value IN ({ids_str})
            """
            
            if metric_keys:
                keys_quoted = [f"'{k}'" for k in metric_keys]
                keys_str = ",".join(keys_quoted)
                query += f" AND m.key IN ({keys_str})"

            try:
                df_chunk = pd.read_sql_query(query, conn)
                all_dfs.append(df_chunk)
            except Exception as e:
                print(f"SQL Fetch Error on chunk {i}: {e}")

        conn.close()

        if not all_dfs:
            return pd.DataFrame()

        df = pd.concat(all_dfs, ignore_index=True)
        # Prefix metric keys with 'metrics.' to match mlflow.search_runs format
        if not df.empty:
            df['key'] = 'metrics.' + df['key']
            
        return df

    def filter_with_artifacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filters DataFrame to rows that likely have artifacts/metrics."""
        if df.empty:
            return df
            
        # Strategy: Run is FINISHED OR has at least one metric
        # Check if any 'metrics.*' column is not NaN
        metric_cols = [c for c in df.columns if c.startswith('metrics.')]
        
        if not metric_cols:
            return df[df['status'] == 'FINISHED']
            
        has_metrics = df[metric_cols].notna().any(axis=1)
        is_finished = df['status'] == 'FINISHED'
        
        return df[has_metrics | is_finished]
