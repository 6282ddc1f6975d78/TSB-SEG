#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# run_grid_experiments.sh — Submit grid-search experiments to SLURM
#
# For each (algorithm × dataset) pair, launches a grid_v2 controller
# that expands tunable_parameters into a Cartesian product and submits
# one SLURM sub-job per grid combo via submitit (fire-and-forget).
#
# The script runs from the LOGIN NODE and monitors queue size to avoid
# exceeding the 2000 pending job limit.
#
# Combinations known to timeout from non-guided-full-time-3 (exp 6)
# and guided-full-time-3 (exp 7) are excluded by default.
#
# Usage:
#   ./scripts/run_grid_experiments.sh                      # all defaults
#   ./scripts/run_grid_experiments.sh --dry-run            # preview
#   ./scripts/run_grid_experiments.sh -m non-guided        # non-guided only
#   ./scripts/run_grid_experiments.sh -m guided            # guided only
#   ./scripts/run_grid_experiments.sh --batch 1            # batch 1 only
#   ./scripts/run_grid_experiments.sh --batch 2            # batch 2 only
#   ./scripts/run_grid_experiments.sh -a clasp,ticc        # specific algos
#   ./scripts/run_grid_experiments.sh --include-timeouts   # include timeout combos
#   ./scripts/run_grid_experiments.sh --max-queue 1800     # queue threshold
#
# Architecture:
#   Login node → for each (algo, ds):
#     python main.py algorithm=X dataset=Y experiment=grid_*_v3 \
#       hydra/launcher=slurm +grid.wait_for_results=false
#     → grid_v2 controller (no --multirun → "controller" mode)
#     → submits grid_size SLURM sub-jobs via submitit
#     → returns immediately (fire-and-forget)
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Colours ───────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Pre-parse: extract --profile before loading .env ──────────────────
_prev=""
for _arg in "$@"; do
  if [[ "$_prev" == "-p" || "$_prev" == "--profile" ]]; then
    export CLUSTER_PROFILE="$_arg"; break
  fi
  _prev="$_arg"
done
unset _prev _arg

# ── Load .env ─────────────────────────────────────────────────────────
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

# ── Defaults ──────────────────────────────────────────────────────────
ALGORITHMS=""
DATASETS=""
MODES="non-guided,guided"     # guided = semi_supervised, non-guided = unsupervised
EXCLUDE_DATASETS="suturing|needle-passing|knot-tying"
EXCLUDE_ALGOS="patss|hdp-hsmm-legacy|vsax|time2feat|tirex-.*|bocd|tglad|tscp2|eagglo"
BASE_MLFLOW_PORT=15050
WORKSPACE="${TSSEG_WORKSPACE:-main}"
CLUSTER_USER="${CLUSTER_USER:-$USER}"
CLUSTER_HOST="${CLUSTER_HOST:-cleps.inria.fr}"
CLUSTER_WORKDIR="${CLUSTER_WORKDIR:-/scratch/$USER/tsseg-exp}"
CONDA_ENV="${CONDA_ENV:-tsseg-env}"
LAUNCHER=""                     # empty = default (slurm for baseline, submitit_gpu_h100 for FM)
DRY_RUN=false
LOCAL=false
INCLUDE_TIMEOUTS=false
BATCH=""                      # empty = all, "1" or "2" = specific batch
MAX_QUEUE=1800                # pause when queue >= this (2000 hard limit)
QUEUE_POLL_SECONDS=60         # how often to check queue
START_FROM=1                  # resume: skip combos before this index
EXTRA_ARGS=()

# ── Known timeout combinations from non-guided-full-time-3 (exp 6) ──
# NOTE: bocd, tglad, tscp2, eagglo entirely excluded (can't compare across all datasets)
TIMEOUT_NON_GUIDED=(
    "clap:pamap2"
    "clasp:pamap2"
    "icid:pamap2"
    "kcpd:has"
    "kcpd:pump"
    "kcpd:usc-had"
)

# ── Known timeout combinations from guided-full-time-3 (exp 7) ──────
# NOTE: bocd, tglad, tscp2, eagglo entirely excluded (can't compare across all datasets)
TIMEOUT_GUIDED=(
    "clap:pamap2"
    "clasp:pamap2"
    "icid:pamap2"
    "kcpd:has"
    "kcpd:usc-had"
    "prophet:pump"
)

# ── Approximate grid sizes (for batch splitting & job counting) ──────
declare -A GRID_SIZES=(
    [amoc]=3 [autoplait]=1 [binseg]=9 [bottomup]=9
    [clap]=6 [clasp]=18 [dynp]=9 [e2usd]=27
    [espresso]=9 [fluss]=18 [ggs]=9 [hdp-hsmm]=27 [hidalgo]=27
    [icid]=27 [igts]=9 [kcpd]=9 [pelt]=9 [prophet]=6
    [random]=1 [ticc]=27 [time2state]=27 [tire]=27
    [window]=27
    [changefinder]=54
    # Foundation model detectors (tsseg-fm)
    [fm-agglom]=72 [fm-binseg]=72 [fm-bottomup-stab]=48 [fm-bottomup]=48
    [fm-clasp]=72 [fm-cosim]=54 [fm-derivative]=54 [fm-distprofile]=48
    [fm-dynp]=48 [fm-kernel]=72 [fm-l2]=54
    [fm-state-gmm]=36 [fm-state-kmeans]=36
)

# ── Batch 1 algos: grid_size <= 18 (lighter, ~900 jobs per mode) ────
BATCH1_ALGOS="amoc|autoplait|binseg|bottomup|clap|clasp|dynp|espresso|fluss|ggs|igts|kcpd|pelt|prophet|random"
# ── Batch 2 algos: grid_size = 27 (heavier, ~1900 jobs per mode) ────
BATCH2_ALGOS="e2usd|hdp-hsmm|hidalgo|icid|ticc|time2state|tire|window"
# ── Batch 2a: first half of heavy algos ─────────────────────────────
BATCH2A_ALGOS="e2usd|hdp-hsmm|hidalgo|icid"
# ── Batch 2b: second half of heavy algos ────────────────────────────
BATCH2B_ALGOS="ticc|time2state|tire|window"

# ── Help ──────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
${BOLD}Usage:${NC} $(basename "$0") [options] [-- extra_hydra_args...]

${BOLD}Submit grid-search experiments to the SLURM cluster.${NC}

Each (algorithm x dataset) pair is launched as a grid_v2 controller
that submits one SLURM job per grid combo via submitit (fire-and-forget).

The script runs on the LOGIN NODE, monitors queue size, and pauses
submissions when the queue approaches the 2000-job cluster limit.

Timeout combinations from non-guided-full-time-3 / guided-full-time-3
are excluded by default.

${BOLD}Options:${NC}
  -a, --algorithms ALGOS       Comma-separated algorithm names (default: all eligible)
  -d, --datasets DATASETS      Comma-separated dataset names (default: all eligible)
  -m, --modes MODES            Comma-separated: non-guided,guided (default: both)
  -x, --exclude-ds REGEX       Regex of dataset names to exclude
  -X, --exclude-algo REGEX     Regex of algorithm names to exclude
  -w, --workspace WS           MLflow workspace (default: \$TSSEG_WORKSPACE or "main")
      --batch N                 Run only batch N (1=light, 2=all heavy, 2a/2b=split heavy)
      --max-queue N             Pause when queue >= N jobs (default: 1800)
      --start-from N            Resume from combo N (skip 1..N-1)
      --include-timeouts        Include known timeout combinations
      --launcher NAME           Hydra launcher config (default: slurm)
                GPU options: submitit_gpu_available (fastest queue),
                submitit_gpu_any (>=24GB), submitit_gpu_h200,
                submitit_gpu_h100, submitit_gpu_a100, submitit_gpu_rtx8000
      --local                   Run locally (no SLURM)
      --dry-run                 Print commands without executing
  -h, --help                   Show this help

${BOLD}Batching:${NC}
  Batch 1:  algos with grid <= 18 points  (~1075 jobs per mode)
  Batch 2:  all algos with grid = 27       (~1917 jobs per mode)
  Batch 2a: e2usd, hdp-hsmm, hidalgo, icid (~972 jobs per mode)
  Batch 2b: ticc, time2state, tire, window  (~972 jobs per mode)

  Excluded algos (timeout on some datasets, not comparable):
    bocd, tglad, tscp2, eagglo

  Total: 23 algos × 9 datasets × 2 modes = ~5985 SLURM jobs
  Estimated compute: ~10500h, wall-time ~6 days @74 slots

  Recommended workflow:
    1. Start MLflow server   (max 7-day node limit!)
    2. ./run_grid_experiments.sh --batch 1              # ~2150 jobs, ~1.5 days
    3. ./run_grid_experiments.sh --batch 2a             # ~1944 jobs, ~2.5 days
    4. ./run_grid_experiments.sh --batch 2b             # ~1944 jobs, ~2.5 days
    (Queue monitoring throttles at --max-queue 1800 automatically)
    (Restart MLflow between batch 2a and 2b if needed)

EOF
  exit 0
}

# ── Argument parsing ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--algorithms)       ALGORITHMS="$2";        shift 2 ;;
    -d|--datasets)         DATASETS="$2";          shift 2 ;;
    -m|--modes)            MODES="$2";             shift 2 ;;
    -x|--exclude-ds)       EXCLUDE_DATASETS="$2";  shift 2 ;;
    -X|--exclude-algo)     EXCLUDE_ALGOS="$2";     shift 2 ;;
    -w|--workspace)        WORKSPACE="$2";         shift 2 ;;
    -p|--profile)          shift 2 ;;  # already handled in pre-parse
    --batch)               BATCH="$2";             shift 2 ;;
    --max-queue)           MAX_QUEUE="$2";         shift 2 ;;
    --start-from)          START_FROM="$2";         shift 2 ;;
    --include-timeouts)    INCLUDE_TIMEOUTS=true;   shift ;;
    --launcher)            LAUNCHER="$2";           shift 2 ;;
    --local)               LOCAL=true;              shift ;;
    --dry-run)             DRY_RUN=true;           shift ;;
    -h|--help)             usage ;;
    --)                    shift; EXTRA_ARGS=("$@"); break ;;
    *)                     echo -e "${RED}Unknown option: $1${NC}"; usage ;;
  esac
done

# ── Workspace normalization ────────────────────────────────────────────
_normalize_workspace() {
  local ws="$1"; [[ "$ws" != */* ]] && ws="local/$ws"; echo "$ws"
}
_ws_name() { local ws; ws=$(_normalize_workspace "$1"); echo "${ws#*/}"; }

WORKSPACE=$(_normalize_workspace "$WORKSPACE")
WS_NAME=$(_ws_name "$WORKSPACE")

# ── Derive per-workspace MLflow port ─────────────────────────────────
# Hash includes username to avoid port conflicts when two users share a cluster
_port_key="${CLUSTER_USER}:${WS_NAME}"
_hash=$(echo -n "$_port_key" | cksum | awk '{print $1}')
MLFLOW_PORT=$(( BASE_MLFLOW_PORT + (_hash % 200) + 1 ))

# ── Discover MLflow server ────────────────────────────────────────────
discover_mlflow() {
  if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    echo -e "${GREEN}✓${NC} MLflow URI (explicit): ${BOLD}${MLFLOW_TRACKING_URI}${NC}"
    return 0
  fi
  if [[ "$LOCAL" == "true" ]]; then
    local ws_dir="${PROJECT_DIR}/workspaces/${WORKSPACE}"
    mkdir -p "$ws_dir"
    export MLFLOW_TRACKING_URI="sqlite:///${ws_dir}/mlflow.db"
    echo -e "${GREEN}✓${NC} MLflow URI (local): ${BOLD}${MLFLOW_TRACKING_URI}${NC}"
    return 0
  fi
  local mlflow_dir="${TSSEG_MLFLOW_DIR:-}"
  if [[ -z "$mlflow_dir" ]]; then
    echo -e "${YELLOW}⚠${NC} TSSEG_MLFLOW_DIR not set — MLflow will use local fallback."
    return 0
  fi
  local remote="${CLUSTER_USER}@${CLUSTER_HOST}"
  local node_file="$mlflow_dir/workspaces/${WS_NAME}/mlflow_node.txt"
  local raw_node
  raw_node=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$remote" \
    "cat '$node_file' 2>/dev/null | head -n1" 2>/dev/null | tr -d '\r') || true
  if [[ -z "$raw_node" ]]; then
    if [[ "$DRY_RUN" == "true" ]]; then
      echo -e "${YELLOW}⚠${NC} MLflow node file not found — ignored in dry-run."
      return 0
    fi
    echo -e "${RED}✗${NC} MLflow server not running."
    exit 1
  fi
  local node_ip
  node_ip=$(ssh -o BatchMode=yes "$remote" \
    "getent hosts '$raw_node' 2>/dev/null | awk '{print \$1}'" 2>/dev/null | tr -d '\r') || true
  local target="${node_ip:-$raw_node}"
  export MLFLOW_TRACKING_URI="http://${target}:${MLFLOW_PORT}"
  echo -e "${GREEN}✓${NC} MLflow URI (auto-discovered): ${BOLD}${MLFLOW_TRACKING_URI}${NC}"
}

# ── Resolve algorithms ────────────────────────────────────────────────
resolve_algorithms() {
  local list
  if [[ -n "$ALGORITHMS" ]]; then
    list=$(echo "$ALGORITHMS" | tr ',' '\n')
  else
    list=$(find "$PROJECT_DIR/configs/algorithm" -name "*.yaml" -exec basename {} .yaml \; \
      | grep -vE "^(${EXCLUDE_ALGOS})$" \
      | sort)
  fi
  # Apply batch filter
  if [[ -n "$BATCH" ]]; then
    case "$BATCH" in
      1)  echo "$list" | grep -E "^(${BATCH1_ALGOS})$" ;;
      2)  echo "$list" | grep -E "^(${BATCH2_ALGOS})$" ;;
      2a) echo "$list" | grep -E "^(${BATCH2A_ALGOS})$" ;;
      2b) echo "$list" | grep -E "^(${BATCH2B_ALGOS})$" ;;
      *) echo -e "${RED}Invalid batch: $BATCH (must be 1, 2, 2a or 2b)${NC}" >&2; exit 1 ;;
    esac
  else
    echo "$list"
  fi
}

# ── Resolve datasets ─────────────────────────────────────────────────
resolve_datasets() {
  if [[ -n "$DATASETS" ]]; then
    echo "$DATASETS" | tr ',' '\n'
  else
    find "$PROJECT_DIR/configs/dataset" -name "*.yaml" -exec basename {} .yaml \; \
      | grep -vE "^(${EXCLUDE_DATASETS})$" \
      | sort
  fi
}

# ── Check if a combo is in a timeout list ─────────────────────────────
is_timeout() {
  local algo="$1" ds="$2" mode="$3"
  local -n timeout_list
  if [[ "$mode" == "non-guided" ]]; then
    timeout_list=TIMEOUT_NON_GUIDED
  else
    timeout_list=TIMEOUT_GUIDED
  fi
  local key="${algo}:${ds}"
  for entry in "${timeout_list[@]}"; do
    [[ "$entry" == "$key" ]] && return 0
  done
  return 1
}

# ── Map mode to experiment preset ─────────────────────────────────────
experiment_for_mode() {
  case "$1" in
    non-guided) echo "grid_unsupervised_v3" ;;
    guided)     echo "grid_supervised_v3" ;;
    *)          echo -e "${RED}Unknown mode: $1${NC}" >&2; exit 1 ;;
  esac
}

# ── Queue monitoring (cluster only) ──────────────────────────────────
get_queue_size() {
  if [[ "$LOCAL" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    echo 0
    return
  fi
  local remote="${CLUSTER_USER}@${CLUSTER_HOST}"
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$remote" \
    "squeue -u $CLUSTER_USER -h 2>/dev/null | wc -l" 2>/dev/null || echo 0
}

wait_for_queue_space() {
  local grid_size="$1"
  if [[ "$LOCAL" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
    return
  fi
  local threshold=$((MAX_QUEUE - grid_size))
  while true; do
    local qsize
    qsize=$(get_queue_size)
    if [[ "$qsize" -lt "$threshold" ]]; then
      return
    fi
    echo -e "  ${YELLOW}⏳ Queue: ${qsize}/${MAX_QUEUE} — waiting ${QUEUE_POLL_SECONDS}s...${NC}"
    sleep "$QUEUE_POLL_SECONDS"
  done
}

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

cd "$PROJECT_DIR"

echo -e "${BOLD}═══ tsseg-exp: Grid-Search Experiment Submission ═══${NC}"
echo -e "${BLUE}Workspace:${NC}   ${BOLD}${WORKSPACE}${NC}"
[[ -n "$BATCH" ]] && echo -e "${BLUE}Batch:${NC}       ${BOLD}${BATCH}${NC}"
echo ""

# 1. MLflow
discover_mlflow

# 2. Resolve dimensions
mapfile -t ALGO_LIST < <(resolve_algorithms)
mapfile -t DS_LIST < <(resolve_datasets)
IFS=',' read -ra MODE_LIST <<< "$MODES"

echo ""
echo -e "${BLUE}Modes:${NC}       ${MODES}"
echo -e "${BLUE}Algorithms:${NC}  (${#ALGO_LIST[@]}) ${ALGO_LIST[*]}"
echo -e "${BLUE}Datasets:${NC}    (${#DS_LIST[@]}) ${DS_LIST[*]}"
echo -e "${BLUE}Max queue:${NC}   ${MAX_QUEUE}"

# 3. Build combo list
declare -a COMBOS=()
n_excluded=0
n_total=0
n_slurm_jobs=0

for mode in "${MODE_LIST[@]}"; do
  experiment=$(experiment_for_mode "$mode")
  for algo in "${ALGO_LIST[@]}"; do
    for ds in "${DS_LIST[@]}"; do
      n_total=$((n_total + 1))
      if [[ "$INCLUDE_TIMEOUTS" != "true" ]] && is_timeout "$algo" "$ds" "$mode"; then
        n_excluded=$((n_excluded + 1))
        continue
      fi
      grid_size=${GRID_SIZES[$algo]:-1}
      n_slurm_jobs=$((n_slurm_jobs + grid_size))
      COMBOS+=("${mode}:${algo}:${ds}:${experiment}:${grid_size}")
    done
  done
done

echo ""
echo -e "${BLUE}(algo x dataset) pairs:${NC}  ${BOLD}${#COMBOS[@]}${NC} (excl. ${n_excluded} timeout)"
echo -e "${BLUE}Total SLURM grid jobs:${NC}   ${BOLD}${n_slurm_jobs}${NC} (after grid expansion)"
echo ""

if [[ ${#COMBOS[@]} -eq 0 ]]; then
  echo -e "${YELLOW}No combinations to submit.${NC}"
  exit 0
fi

# 4. Show excluded combos
if [[ "$n_excluded" -gt 0 ]]; then
  echo -e "${YELLOW}Excluded timeout combinations:${NC}"
  for mode in "${MODE_LIST[@]}"; do
    local_list=()
    if [[ "$mode" == "non-guided" ]]; then
      ref_list=("${TIMEOUT_NON_GUIDED[@]}")
    else
      ref_list=("${TIMEOUT_GUIDED[@]}")
    fi
    for entry in "${ref_list[@]}"; do
      IFS=':' read -r a d <<< "$entry"
      algo_match=false; ds_match=false
      for al in "${ALGO_LIST[@]}"; do [[ "$al" == "$a" ]] && algo_match=true; done
      for dl in "${DS_LIST[@]}"; do [[ "$dl" == "$d" ]] && ds_match=true; done
      if $algo_match && $ds_match; then
        local_list+=("$entry")
      fi
    done
    if [[ ${#local_list[@]} -gt 0 ]]; then
      echo -e "  ${DIM}${mode}:${NC} ${local_list[*]}"
    fi
  done
  echo ""
fi

# 5. Build common Hydra overrides
# NOTE: NO --multirun! Each invocation is a single run in "controller" mode.
# The controller uses submitit to submit one SLURM job per grid combo.
HYDRA_OVERRIDES=""
if [[ "$LOCAL" != "true" ]]; then
  _launcher="${LAUNCHER:-slurm}"
  HYDRA_OVERRIDES+=" hydra/launcher=${_launcher}"
  SETUP='"export SLURM_CPU_BIND=none"'
  SETUP+=', "export TSSEG_WORKSPACE='"${WS_NAME}"'"'
  if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    SETUP+=', "export MLFLOW_TRACKING_URI='"${MLFLOW_TRACKING_URI}"'"'
  fi
  HYDRA_OVERRIDES+=" 'hydra.launcher.setup=[${SETUP}]'"
  _ws_dir="${CLUSTER_WORKDIR}/workspaces/${WS_NAME}"
  HYDRA_OVERRIDES+=" 'hydra.run.dir=${_ws_dir}/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}'"
  HYDRA_OVERRIDES+=" 'hydra.sweep.dir=${_ws_dir}/multirun/\${now:%Y-%m-%d}/\${now:%H-%M-%S}'"
fi

# Fire-and-forget: controller submits SLURM sub-jobs and returns immediately
HYDRA_OVERRIDES+=" +grid.wait_for_results=false"

EXTRA=""
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  EXTRA=" ${EXTRA_ARGS[*]}"
fi

# 6. Submit one controller per (algo, dataset) pair
n_submitted=0
n_total_combos=${#COMBOS[@]}
submitted_jobs=0

echo -e "${BOLD}Launching ${n_total_combos} grid controllers (${n_slurm_jobs} total sub-jobs)...${NC}"
echo ""

for combo in "${COMBOS[@]}"; do
  IFS=':' read -r mode algo ds experiment grid_size <<< "$combo"
  n_submitted=$((n_submitted + 1))
  submitted_jobs=$((submitted_jobs + grid_size))

  # Skip combos before --start-from
  if [[ "$n_submitted" -lt "$START_FROM" ]]; then
    echo -e "  ${DIM}[${n_submitted}/${n_total_combos}] SKIP (--start-from ${START_FROM})${NC}"
    continue
  fi

  # Wait if queue is getting full
  wait_for_queue_space "$grid_size"

  # Use -m to avoid adding src/tsseg_exp/ to sys.path[0]
  # (prevents local datasets/ from shadowing HuggingFace datasets)
  CMD="python -m tsseg_exp.main"
  CMD+=" experiment=${experiment}"
  CMD+=" algorithm=${algo}"
  CMD+=" dataset=${ds}"
  CMD+="${HYDRA_OVERRIDES}"
  CMD+="${EXTRA}"

  echo -e "${BLUE}[${n_submitted}/${n_total_combos}]${NC} ${BOLD}${mode}${NC} ${algo} x ${ds} (${grid_size} grid combos) [cumul: ${submitted_jobs}/${n_slurm_jobs}]"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "  ${DIM}${CMD}${NC}"
    continue
  fi

  if [[ "$LOCAL" == "true" ]]; then
    echo -e "  ${GREEN}▶ Running locally...${NC}"
    eval "$CMD"
  else
    local_remote="${CLUSTER_USER}@${CLUSTER_HOST}"
    ssh -o BatchMode=yes "$local_remote" bash -s <<REMOTE_SCRIPT
set +x
source ~/.bashrc 2>/dev/null
cd '${CLUSTER_WORKDIR}'
conda activate '${CONDA_ENV}'
export MLFLOW_TRACKING_URI='${MLFLOW_TRACKING_URI}'
${CMD}
REMOTE_SCRIPT
  fi

  # Small delay between controller invocations
  if [[ "$DRY_RUN" != "true" ]]; then
    sleep 2
  fi
done

echo ""
echo -e "${BOLD}═══ Done: ${n_submitted} controllers launched (${submitted_jobs} grid sub-jobs) ═══${NC}"
