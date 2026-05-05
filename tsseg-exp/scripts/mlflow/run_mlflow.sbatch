#!/bin/bash
#SBATCH --job-name=mlflow
#SBATCH --partition=cpu_devel
#SBATCH --time=150:00:00               # < 1 week (cpu_devel partition limit)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=server/mlflow.sbatch.%j.out
#SBATCH --error=server/mlflow.sbatch.%j.err

set -euo pipefail

# --- CONFIG ---
BASE_DIR="${CLUSTER_WORKDIR:-/scratch/$USER/tsseg-exp}"
WORKSPACE="${TSSEG_WORKSPACE:-main}"
MLFLOW_DIR="$BASE_DIR/workspaces/$WORKSPACE"
MLFLOW_DB="$MLFLOW_DIR/mlflow.db"
ARTIFACTS_DIR="$MLFLOW_DIR/mlartifacts"
MLFLOW_PORT="${MLFLOW_PORT:-15050}"
MLFLOW_HOST=0.0.0.0
TMUX_SESSION="mlflow-${WORKSPACE}"
# ----------------

mkdir -p "$MLFLOW_DIR" "$ARTIFACTS_DIR" "$MLFLOW_DIR/outputs" "$MLFLOW_DIR/multirun"

# Activate conda environment
CONDA_ENV="${CONDA_ENV:-tsseg-env}"
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV"

echo "Job started on $(hostname) at $(date)"
echo "Workspace: $WORKSPACE"
echo "MLflow DB: $MLFLOW_DB"
echo "Artifacts: $ARTIFACTS_DIR"

# write the hostname to a known file so that the submitter can read it later
echo "$(hostname)" > "$MLFLOW_DIR/mlflow_node.txt"
echo "Started by sbatch job $SLURM_JOB_ID on $(hostname) at $(date)" >> "$MLFLOW_DIR/mlflow_node.txt"

# Ensure SQLite uses WAL mode for better concurrency on Lustre.
# WAL (Write-Ahead Logging) allows concurrent reads during writes and avoids
# exclusive locks on the entire DB file — critical for multi-worker servers on
# network filesystems.  The setting is persistent once applied.
python3 -c "
import sqlite3, sys, os
db = '$MLFLOW_DB'
if os.path.exists(db):
    conn = sqlite3.connect(db, timeout=30)
    mode = conn.execute('PRAGMA journal_mode=WAL').fetchone()[0]
    print(f'SQLite journal_mode set to: {mode}', file=sys.stderr)
    # Ensure performance indexes exist (MLflow does not create them by default)
    for sql in [
        'CREATE INDEX IF NOT EXISTS idx_runs_experiment_id ON runs (experiment_id)',
        'CREATE INDEX IF NOT EXISTS idx_runs_status ON runs (status)',
        'CREATE INDEX IF NOT EXISTS idx_runs_experiment_id_status ON runs (experiment_id, status)',
        'CREATE INDEX IF NOT EXISTS idx_runs_experiment_id_start_time ON runs (experiment_id, start_time)',
        'CREATE INDEX IF NOT EXISTS idx_runs_start_time ON runs (start_time)',
    ]:
        conn.execute(sql)
    conn.commit()
    print('Performance indexes ensured', file=sys.stderr)
    conn.close()
"

# start mlflow in a detached tmux session so it survives if the job's shell exits
# the tmux session lifetime is tied to the job; when the SBATCH job ends, tmux/session ends.
tmux new-session -d -s "$TMUX_SESSION" \
  "cd '$MLFLOW_DIR' && \
   mlflow server \
     --backend-store-uri sqlite:///$MLFLOW_DB \
     --default-artifact-root $ARTIFACTS_DIR \
     --host $MLFLOW_HOST \
     --port $MLFLOW_PORT \
     --workers 4 \
     --disable-security-middleware > '$MLFLOW_DIR/mlflow.log' 2>&1"

sleep 1
tmux ls || true
echo "MLflow should be running in tmux session '$TMUX_SESSION' on $(hostname):$MLFLOW_HOST:$MLFLOW_PORT"
echo "See logs: $MLFLOW_DIR/mlflow.log"
echo "Tmux session: $TMUX_SESSION"

# Keep the job alive while tmux session exists. This makes the sbatch allocation stay active.
# If tmux is killed, the script will exit and the job ends.
while tmux has-session -t "$TMUX_SESSION" 2>/dev/null; do
  sleep 30
done

echo "tmux session ended, exiting job. $(date)"
