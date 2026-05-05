# Scripts for Cluster Environment

This directory contains scripts to manage **MLflow workspaces**, **tracking servers**, **databases**, **result synchronization**, and **experiment submission** for the tsseg-exp benchmark.

## Workspace Architecture

All MLflow data is organized into **isolated workspaces** under `workspaces/<origin>/<name>/`. Two origin prefixes are used:

- **`local/`** — workspaces created and used locally
- **`cluster/`** — workspaces pulled from the remote cluster

This avoids name conflicts between local and remote workspaces with the same name, and eliminates the need for `results/` or `mlruns/` at the project root.

```
workspaces/
├── local/
│   ├── main/               ← local production experiments
│   │   ├── mlflow.db
│   │   ├── mlartifacts/
│   │   ├── outputs/
│   │   └── multirun/
│   └── dev/                ← local development
│       └── ...
└── cluster/
    ├── main/               ← pulled from cluster
    │   ├── mlflow.db       (direct copy, no symlinks)
    │   └── mlartifacts/
    └── dev/
        └── ...
```

On the cluster, the layout is flat: `<user>/tsseg-exp/workspaces/<name>/`.

The active workspace is set via:
- `TSSEG_WORKSPACE` in `.env` (default: `local/main`)
- `-w`/`--workspace` flag on any command (overrides `.env`)
- A bare name like `main` is automatically treated as `local/main`

---

## File Overview

| Script | Purpose |
|--------|---------|
| **`run_experiments.sh`** | **Core experiment launcher** (single algo/dataset combos via Hydra `--multirun`) |
| **`run_baseline_benchmark.sh`** | **Baseline benchmark** — all algos × datasets with default params (no tuning) |
| **`run_grid_experiments.sh`** | **Grid-search benchmark** — Cartesian product of tunable parameters per algo |
| `submit_mlflow.sh` | Start MLflow server on SLURM + SSH tunnel |
| `run_mlflow.sbatch` | SLURM job for SQLite-backed server |
| `submit_mlflow_pg.sh` | Start MLflow server (PostgreSQL variant) |
| `run_mlflow_pg.sbatch` | SLURM job for Postgres-backed server |
| **`mlflow_db.sh`** | **Create, compact, snapshot, inspect, reset workspaces** |
| **`sync_results.sh`** | **Synchronize DB & artifacts between cluster ↔ local** |

---

## Quick Start

### Create a workspace
```bash
./scripts/mlflow/mlflow_db.sh new                    # create "main" on cluster
./scripts/mlflow/mlflow_db.sh new -w dev             # create "dev" on cluster
./scripts/mlflow/mlflow_db.sh new -w dev --local     # create "dev" locally
```

### Start the MLflow server
```bash
submit_mlflow                    # start on "main" workspace
submit_mlflow -w dev             # start on "dev" workspace
submit_mlflow -v                 # verbose mode
```

### Run experiments
```bash
# Single algo/dataset combo
./scripts/run_experiments.sh -a clasp -d mocap
./scripts/run_experiments.sh -w dev -a clasp -d mocap

# Dry-run of 1 algo, 1 dataset, 1 experiment, on local custom workspace
./scripts/run_experiments.sh -w workspace -a clasp -e unsupervised -d mocap --local --dry-run
```

### Run the baseline benchmark (default parameters, no tuning)
```bash
./scripts/run_baseline_benchmark.sh                 # all algos × datasets × 2 modes
./scripts/run_baseline_benchmark.sh --dry-run       # preview
./scripts/run_baseline_benchmark.sh -w my_workspace # custom workspace
```

### Run the grid-search benchmark (parameter tuning)
```bash
# Preview what will be submitted
./scripts/run_grid_experiments.sh --dry-run

# Launch batch 1 (light algos, grids ≤ 18) — both modes
./scripts/run_grid_experiments.sh --batch 1

# Launch batch 2 (heavy algos, grids = 27) — both modes
./scripts/run_grid_experiments.sh --batch 2

# Specific algo or mode
./scripts/run_grid_experiments.sh -a clasp,ticc -m non-guided
```

### Sync results to your local machine
```bash
# Check what's available (local + remote)
./scripts/mlflow/sync_results.sh status
./scripts/mlflow/sync_results.sh status -w dev

# Pull a DB snapshot from the cluster
./scripts/mlflow/sync_results.sh pull
./scripts/mlflow/sync_results.sh pull -w dev

# Pull DB + prediction artifacts only
./scripts/mlflow/sync_results.sh pull --artifacts predictions

# Pull everything (DB + all artifacts — can be large)
./scripts/mlflow/sync_results.sh pull --artifacts all
```

### Manage databases
```bash
# List all workspaces (local + remote)
./scripts/mlflow/mlflow_db.sh workspaces

# Inspect a workspace's DB
./scripts/mlflow/mlflow_db.sh info
./scripts/mlflow/mlflow_db.sh info -w dev
./scripts/mlflow/mlflow_db.sh info --db workspaces/cluster/main/mlflow.db

# List available snapshots
./scripts/mlflow/mlflow_db.sh list

# Compact: keep only active/selected experiments
./scripts/mlflow/mlflow_db.sh compact -w main
./scripts/mlflow/mlflow_db.sh compact --keep "exp-unsupervised-09-02,exp-supervised-09-02"

# Reset a workspace (purge DB + artifacts + outputs)
./scripts/mlflow/mlflow_db.sh reset -w dev

# Clean up old snapshots (keep the 3 most recent)
./scripts/mlflow/mlflow_db.sh prune --keep-n 3
```

### Push a compacted DB back to the cluster
```bash
# 1. Compact locally
./scripts/mlflow/mlflow_db.sh compact -w main

# 2. Push the compacted DB to replace the remote one
./scripts/mlflow/sync_results.sh push -w main --db workspaces/cluster/main/mlflow_compact_20260209_143000.db
```

---

## Typical Workflow

```
  ┌──────────────┐         ┌──────────────────┐         ┌────────────────┐
  │   Cluster    │         │   sync_results   │         │     Local      │
  │  ws/main/    │──pull──▶│   .sh pull       │────────▶│ ws/cluster/    │
  │  mlflow.db   │         │   -w main        │         │ main/mlflow.db │
  │  mlartifacts/│         │                  │         │                │
  └──────────────┘         └──────────────────┘         └───────┬────────┘
                                                                │
                                                      mlflow_db.sh info
                                                      mlflow_db.sh compact
                                                                │
                          ┌─────────────────┐          ┌──────▼───────┐
  ┌─────────────┐         │   sync_results  │          │  compact.db  │
  │   Cluster   │◀──push──│   .sh push      │◀─────────│  (smaller)   │
  │  ws/main/   │         │   -w main       │          └──────────────┘
  │  mlflow.db  │         │                 │
  └─────────────┘         └─────────────────┘
```

**When the DB gets too large:**
1. `sync_results.sh pull -w main` — download latest snapshot to `workspaces/cluster/main/`
2. `mlflow_db.sh info --db workspaces/cluster/main/mlflow.db` — see what's inside
3. `mlflow_db.sh compact --keep "exp1,exp2"` — keep only current experiments
4. Stop the MLflow server on the cluster
5. `sync_results.sh push -w main --db workspaces/cluster/main/mlflow_compact_XXX.db` — replace remote DB
6. Restart the server

---

## Experiment Scripts

### Baseline Benchmark (`run_baseline_benchmark.sh`)

Wrapper around `run_experiments.sh` that launches **all algorithms × all datasets** for both unsupervised and semi_supervised modes using **default parameters** (no parameter tuning).

Each (algorithm, dataset, experiment) triple is submitted as a single SLURM job via Hydra `--multirun`.

```bash
./scripts/run_baseline_benchmark.sh                 # submit all
./scripts/run_baseline_benchmark.sh --dry-run       # preview only
```

### Grid-Search Benchmark (`run_grid_experiments.sh`)

Launches **grid-search experiments** that explore the Cartesian product of each algorithm's `tunable_parameters`. For each (algorithm × dataset) pair, a **grid_v2 controller** is started on the login node. The controller expands the parameter grid and submits one SLURM sub-job per grid combo via submitit in **fire-and-forget** mode.

**Key design: no `--multirun`.** Each invocation runs as a standard Hydra single-run, which resolves to `"controller"` mode in `grid_pipeline_v2.py`. Using `--multirun` would resolve to `"worker"` mode and skip grid expansion entirely.

#### Dimensions (current)

| | Algos | Datasets | Modes | Grid jobs (total) |
|-|------:|---------:|------:|------------------:|
| Batch 1 (grid ≤ 18) | 15 | 9 | 2 | ~2 151 |
| Batch 2 (grid = 27)  | 8  | 9 | 2 | ~3 834 |
| **Total** | **23** | **9** | **2** | **~5 985** |

Estimated compute: **~10 500h**, wall-time **~6 days** with 74 parallel slots.

**Excluded algorithms** (timeout on some datasets, not comparable across all 9):
`bocd`, `tglad`, `tscp2`, `eagglo`.

**Excluded timeout combinations** (from previous experiments 6 & 7):
`clap:pamap2`, `clasp:pamap2`, `icid:pamap2`, `kcpd:has`, `kcpd:pump`, `kcpd:usc-had`, `prophet:pump`.

#### Timeout hierarchy

Two timeout mechanisms are in play:

1. **SLURM hard timeout** — `hydra/launcher/slurm.yaml` → `timeout_min: 1440` (24h).
   SLURM sends SIGTERM then kills the job. This is the **real ceiling**.
2. **Application soft timeout** — `configs/experiment/grid_*_v3.yaml` → `timeout_seconds: 86400` (24h).
   Python-level graceful stop (saves partial metrics before exiting). Aligned with SLURM.

#### Queue throttling

The cluster has a **2000-job pending limit** and **75 simultaneous slots** (74 usable, 1 reserved for MLflow).

The script monitors the SLURM queue via SSH and **pauses submissions** when the queue approaches saturation:

```
┌─────────────────────────────────────────────────────────┐
│  For each (algo, dataset) pair:                         │
│    1. Check squeue -u $USER | wc -l                     │
│    2. If queue_size >= MAX_QUEUE - grid_size:            │
│       → wait QUEUE_POLL_SECONDS (default: 60s)          │
│       → re-check, loop until space available             │
│    3. SSH to login node → python main.py ...             │
│       → controller submits grid_size sub-jobs            │
│       → returns immediately (fire-and-forget)            │
│    4. sleep 2s → next pair                               │
└─────────────────────────────────────────────────────────┘
```

- `--max-queue N` — threshold (default: **1800**, leaves 200-job headroom)
- `--queue-poll N` — seconds between checks when throttled (default: 60)

This means **both batches can be run with both modes** without manual splitting — the queue throttle handles backpressure automatically.

#### Options

| Flag | Description |
|------|-------------|
| `-a, --algorithms` | Comma-separated algorithm names |
| `-d, --datasets` | Comma-separated dataset names |
| `-m, --modes` | `non-guided`, `guided`, or both (default: both) |
| `--batch N` | Run only batch 1 (light) or 2 (heavy) |
| `--max-queue N` | Queue throttle threshold (default: 1800) |
| `--include-timeouts` | Don't skip known timeout combinations |
| `--local` | Run locally without SLURM |
| `--dry-run` | Preview commands without submitting |
| `-w, --workspace` | MLflow workspace |
| `-x, --exclude-ds` | Regex of datasets to exclude |
| `-X, --exclude-algo` | Regex of algorithms to exclude |

#### Recommended workflow

```bash
# 1. Start MLflow server
submit_mlflow

# 2. Batch 1: light algos (~2150 jobs, ~1.5 days)
./scripts/run_grid_experiments.sh --batch 1

# 3. Batch 2: heavy algos (~3834 jobs, ~4.5 days)
./scripts/run_grid_experiments.sh --batch 2

# Queue throttle handles backpressure — no manual splitting needed.
# Restart MLflow server if its 1-week SLURM allocation expires.
```

---

## Database Management (`mlflow_db.sh`)

### Commands

| Command | Description |
|---------|-------------|
| `new` | Create a fresh workspace (empty DB + directory structure) |
| `reset` | Purge a workspace (remove DB, artifacts, outputs, multirun) |
| `compact` | Copy only active/selected experiments into a new clean DB |
| `snapshot` | Create a timestamped backup (local or remote) |
| `info` | Show experiment list, run counts, DB size, workspace contents |
| `list` | List all snapshots across workspaces |
| `prune` | Remove old snapshots, keep the N most recent |
| `workspaces` | List all workspaces (local + remote) with stats |

### Options
- `-w, --workspace NAME` — workspace to operate on (default: `$TSSEG_WORKSPACE` or `main`)
- `--db PATH` — target database path (overrides workspace DB for `info`/`compact`)
- `--keep "name1,name2"` — experiment names to keep during compact
- `--keep-n N` — number of snapshots to keep during prune (default: 3)
- `--remote` / `--local` — operate on remote cluster or local workspace

---

## Result Sync (`sync_results.sh`)

### Commands

| Command | Description |
|---------|-------------|
| `pull` | Download DB snapshot + optional artifacts from cluster |
| `push` | Upload a local DB to replace the remote one |
| `status` | Compare local vs remote (DB sizes, timestamps, runs) |

### Artifact modes (`--artifacts`)
- `none` — DB only (default, fast)
- `predictions` — DB + prediction files only (`.npy`, change points, labels)
- `all` — full `mlartifacts/` sync via rsync (can be very large)

### Options
- `-w, --workspace NAME` — workspace to sync (default: `$TSSEG_WORKSPACE` or `main`)
- `--artifacts MODE` — `none` / `predictions` / `all`
- `--experiments "e1,e2"` — filter artifact sync to specific experiments
- `--db PATH` — local DB path (for `push`)
- `--dry-run` — show what would be transferred

---

## Environment Variables

All scripts respect these (set in `.env` or export):

| Variable | Default | Description |
|----------|---------|-------------|
| `TSSEG_WORKSPACE` | `local/main` | Active workspace (format: origin/name) |
| `CLUSTER_USER` | `$USER` | SSH username |
| `CLUSTER_HOST` | `cleps.inria.fr` | SSH hostname |
| `CLUSTER_WORKDIR` | `/scratch/$USER/tsseg-exp` | Remote working directory |
| `TSSEG_MLFLOW_DIR` | — | Base dir for auto-discovery (reads `workspaces/<ws>/mlflow_node.txt`) |
| `MLFLOW_TRACKING_URI` | — | Explicit MLflow URI (bypasses auto-discovery) |

---

## Server Configuration

### SQLite (Default)
- **`run_mlflow.sbatch`**: SLURM job script
  - Reads `TSSEG_WORKSPACE` to select workspace (default: `main`)
  - SQLite backend (`workspaces/<ws>/mlflow.db`), WAL mode
  - tmux session named `mlflow-<workspace>` for persistence
  - Writes `workspaces/<ws>/mlflow_node.txt` for auto-discovery
- **`submit_mlflow.sh`**: Client script
  - Submits job, waits for readiness, creates SSH tunnel
  - Auto-syncs sbatch script via MD5 comparison
  - Port **15050**
  - `-w`/`--workspace` to target a specific workspace

### PostgreSQL (Advanced)
- **`run_mlflow_pg.sbatch`** + **`submit_mlflow_pg.sh`**
  - Apptainer/Singularity PostgreSQL container
  - Port **15051**

## Shortcut Installation
```bash
ln -sf $(pwd)/submit_mlflow.sh ~/.local/bin/submit_mlflow
```
Ensure `~/.local/bin` is in your `$PATH`.
