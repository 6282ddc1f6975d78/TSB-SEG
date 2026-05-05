#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# run_baseline_benchmark.sh — Launch the baseline benchmark on SLURM
#
# Runs all algorithms (except excluded ones) on all datasets, for both
# unsupervised and semi_supervised experiments, using DEFAULT parameters
# (no grid search / parameter tuning).
#
# For grid-search experiments with parameter tuning, use:
#   ./scripts/run_grid_experiments.sh
#
# Excluded algorithms:
#   patss, hmm, hdp-hsmm-legacy, vsax, vqtss, time2feat, tirex-*
#
# Cluster constraints:
#   - 75 simultaneous jobs (1 reserved for MLflow → 74 available)
#   - 2000 maximum pending jobs
#
# Usage:
#   ./scripts/run_baseline_benchmark.sh                 # submit
#   ./scripts/run_baseline_benchmark.sh --dry-run       # preview only
#   ./scripts/run_baseline_benchmark.sh -w my_workspace # custom workspace
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$SCRIPT_DIR/run_experiments.sh" \
  -e unsupervised,semi_supervised \
  -X 'patss|hdp-hsmm-legacy|vsax|vqtss|time2feat|tirex-.*' \
  "$@" \
  -- hydra.launcher.array_parallelism=74
