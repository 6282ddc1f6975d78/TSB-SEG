#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# run_experiments.sh — Submit batch experiments to the SLURM cluster
#
# Discovers the MLflow tracking server automatically, resolves datasets
# and algorithms from the config directory, and launches a Hydra
# multirun via the SLURM launcher.
#
# Usage:
#   ./scripts/run_experiments.sh                          # all defaults
#   ./scripts/run_experiments.sh -a clasp,ticc            # specific algos
#   ./scripts/run_experiments.sh -d mocap,utsa            # specific datasets
#   ./scripts/run_experiments.sh -e unsupervised          # specific experiment
#   ./scripts/run_experiments.sh --dry-run                # show command only
#   ./scripts/run_experiments.sh -a tirex-l2 -d utsa -e unsupervised
#
# See ./scripts/run_experiments.sh --help for all options.
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
EXPERIMENTS="unsupervised,semi_supervised"
EXCLUDE_DATASETS="suturing|needle-passing|knot-tying"
EXCLUDE_ALGOS="vsax"
BASE_MLFLOW_PORT=15050
# Par défaut, WORKSPACE="main" (et non "local/main")
WORKSPACE="${TSSEG_WORKSPACE:-main}"
CLUSTER_USER="${CLUSTER_USER:-$USER}"
CLUSTER_HOST="${CLUSTER_HOST:-cleps.inria.fr}"
CLUSTER_WORKDIR="${CLUSTER_WORKDIR:-/scratch/$USER/tsseg-exp}"
CONDA_ENV="${CONDA_ENV:-tsseg-env}"
DRY_RUN=false
LOCAL=false
EXTRA_ARGS=()

# ── Help ──────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
${BOLD}Usage:${NC} $(basename "$0") [options] [-- extra_hydra_args...]

${BOLD}Submit a batch of experiments to the SLURM cluster.${NC}

The script auto-discovers the MLflow server, resolves available datasets
and algorithms from the config directory, and launches a Hydra multirun.

${BOLD}Options:${NC}
  -a, --algorithms ALGOS   Comma-separated algorithm names (default: all in configs/)
  -d, --datasets DATASETS  Comma-separated dataset names (default: all in configs/)
  -e, --experiments EXPS   Comma-separated experiment presets (default: unsupervised,semi_supervised)
  -x, --exclude-ds REGEX   Regex of dataset names to exclude (default: ${EXCLUDE_DATASETS})
  -X, --exclude-algo REGEX Regex of algorithm names to exclude (default: ${EXCLUDE_ALGOS})
  -w, --workspace WS       MLflow workspace to use (default: \$TSSEG_WORKSPACE or "main")
      --local              Run locally (no SLURM), useful for debugging
      --dry-run            Print the command without executing
  -h, --help               Show this help

${BOLD}Examples:${NC}
  $(basename "$0")                                      # full benchmark
  $(basename "$0") -a clasp,ticc -d mocap               # subset
  $(basename "$0") -e unsupervised -a tirex-l2           # single experiment
  $(basename "$0") --local -a clasp -d mocap             # debug locally
  $(basename "$0") -- metric=custom_metric               # extra Hydra overrides

EOF
  exit 0
}

# ── Argument parsing ──────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--algorithms)  ALGORITHMS="$2";      shift 2 ;;
    -d|--datasets)    DATASETS="$2";        shift 2 ;;
    -e|--experiments) EXPERIMENTS="$2";     shift 2 ;;
    -x|--exclude-ds)  EXCLUDE_DATASETS="$2"; shift 2 ;;
    -X|--exclude-algo) EXCLUDE_ALGOS="$2";  shift 2 ;;
    -w|--workspace)   WORKSPACE="$2";       shift 2 ;;
    -p|--profile)     shift 2 ;;  # already handled in pre-parse
    --local)          LOCAL=true;            shift ;;
    --dry-run)        DRY_RUN=true;         shift ;;
    -h|--help)        usage ;;
    --)               shift; EXTRA_ARGS=("$@"); break ;;
    *)                echo -e "${RED}Unknown option: $1${NC}"; usage ;;
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
  # 1. Already set explicitly
  if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    echo -e "${GREEN}✓${NC} MLflow URI (explicit): ${BOLD}${MLFLOW_TRACKING_URI}${NC}"
    return 0
  fi

  # 2. Running locally → use local workspace SQLite DB directly
  if [[ "$LOCAL" == "true" ]]; then
    local ws_dir="${PROJECT_DIR}/workspaces/${WORKSPACE}"
    mkdir -p "$ws_dir"
    export MLFLOW_TRACKING_URI="sqlite:///${ws_dir}/mlflow.db"
    echo -e "${GREEN}✓${NC} MLflow URI (local): ${BOLD}${MLFLOW_TRACKING_URI}${NC}"
    return 0
  fi

  # 3. Auto-discover from node file on the remote cluster
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
      echo -e "${YELLOW}⚠${NC} MLflow node file not found on ${CLUSTER_HOST} — ignored in dry-run."
      return 0
    fi
    echo -e "${RED}✗${NC} MLflow server not running (${node_file} not found on ${CLUSTER_HOST})."
    echo -e "  Start it with: ${BOLD}submit_mlflow${NC}"
    exit 1
  fi

  # Resolve hostname → IP on the cluster (avoids "Invalid Host Header")
  local node_ip
  node_ip=$(ssh -o BatchMode=yes "$remote" \
    "getent hosts '$raw_node' 2>/dev/null | awk '{print \$1}'" 2>/dev/null | tr -d '\r') || true
  local target="${node_ip:-$raw_node}"

  export MLFLOW_TRACKING_URI="http://${target}:${MLFLOW_PORT}"
  echo -e "${GREEN}✓${NC} MLflow URI (auto-discovered from ${CLUSTER_HOST}): ${BOLD}${MLFLOW_TRACKING_URI}${NC}"
}

# ── Resolve algorithms ────────────────────────────────────────────────
resolve_algorithms() {
  if [[ -n "$ALGORITHMS" ]]; then
    echo "$ALGORITHMS"
    return
  fi
  find "$PROJECT_DIR/configs/algorithm" -name "*.yaml" -exec basename {} .yaml \; \
    | grep -vE "^(${EXCLUDE_ALGOS})$" \
    | sort \
    | paste -sd "," -
}

# ── Resolve datasets ─────────────────────────────────────────────────
resolve_datasets() {
  if [[ -n "$DATASETS" ]]; then
    echo "$DATASETS"
    return
  fi
  find "$PROJECT_DIR/configs/dataset" -name "*.yaml" -exec basename {} .yaml \; \
    | grep -vE "^(${EXCLUDE_DATASETS})$" \
    | sort \
    | paste -sd "," -
}

# ── Show SLURM queue status ──────────────────────────────────────────
show_queue_status() {
  if command -v squeue &>/dev/null; then
    local running pending
    running=$(squeue -u "$USER" -t R -h 2>/dev/null | wc -l)
    pending=$(squeue -u "$USER" -t PD -h 2>/dev/null | wc -l)
    echo -e "${DIM}  SLURM queue: ${running} running, ${pending} pending${NC}"
  fi
}

# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

cd "$PROJECT_DIR"

echo -e "${BOLD}═══ tsseg-exp: Batch Experiment Submission ═══${NC}"
echo -e "${BLUE}Workspace:${NC}   ${BOLD}${WORKSPACE}${NC}"
echo ""

# 1. MLflow
discover_mlflow

# 2. Resolve dimensions
ALGOS=$(resolve_algorithms)
DS=$(resolve_datasets)
algo_count=$(echo "$ALGOS" | tr ',' '\n' | wc -l)
ds_count=$(echo "$DS" | tr ',' '\n' | wc -l)
exp_count=$(echo "$EXPERIMENTS" | tr ',' '\n' | wc -l)
total=$((algo_count * ds_count * exp_count))

echo ""
echo -e "${BLUE}Experiments:${NC} $EXPERIMENTS"
echo -e "${BLUE}Algorithms:${NC}  $ALGOS"
echo -e "${BLUE}Datasets:${NC}    $DS"
echo -e "${BLUE}Total:${NC}       ${BOLD}${total} combinations${NC} (${algo_count} algos × ${ds_count} datasets × ${exp_count} experiments)"
show_queue_status
echo ""

# 3. Build command (will be executed on the cluster)
CMD="python -m tsseg_exp.main --multirun"
CMD+=" experiment=$EXPERIMENTS"
CMD+=" algorithm=$ALGOS"
CMD+=" dataset=$DS"

if [[ "$LOCAL" == "true" ]]; then
  echo -e "${YELLOW}▶ Running locally (no SLURM)${NC}"
else
  CMD+=" hydra/launcher=slurm"
  # Each setup command must be a separate element in the YAML list
  SETUP='"export SLURM_CPU_BIND=none"'
  SETUP+=', "export TSSEG_WORKSPACE='"${WS_NAME}"'"'
  if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    SETUP+=', "export MLFLOW_TRACKING_URI='"${MLFLOW_TRACKING_URI}"'"'
  fi
  CMD+=" 'hydra.launcher.setup=[${SETUP}]'"
  # Direct Hydra outputs to workspace (remote uses bare name)
  _ws_dir="${CLUSTER_WORKDIR}/workspaces/${WS_NAME}"
  CMD+=" 'hydra.run.dir=${_ws_dir}/outputs/\${now:%Y-%m-%d}/\${now:%H-%M-%S}'"
  CMD+=" 'hydra.sweep.dir=${_ws_dir}/multirun/\${now:%Y-%m-%d}/\${now:%H-%M-%S}'"
fi

# Append any extra Hydra arguments
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=" ${EXTRA_ARGS[*]}"
fi

# 4. Execute or print
local_remote="${CLUSTER_USER}@${CLUSTER_HOST}"

if [[ "$DRY_RUN" == "true" ]]; then
  if [[ "$LOCAL" == "true" ]]; then
    echo -e "${YELLOW}[dry-run]${NC} Would execute ${BOLD}locally${NC}:"
  else
    echo -e "${YELLOW}[dry-run]${NC} Would execute on ${BOLD}${CLUSTER_HOST}${NC}:"
  fi
  echo ""
  echo "  ${CMD}"
  echo ""
  exit 0
fi

if [[ "$LOCAL" == "true" ]]; then
  # ── Local execution ───────────────────────────────────────────────
  echo -e "${GREEN}▶ Running locally...${NC}"
  echo -e "${DIM}  ${CMD}${NC}"
  echo ""
  # Export workspace / tracking URI so the Python process uses them
  export TSSEG_WORKSPACE="${WORKSPACE}"
  exec bash -c "${CMD}"
else
  # ── Remote execution via SSH ──────────────────────────────────────
  echo -e "${GREEN}▶ Launching on ${CLUSTER_HOST}...${NC}"
  echo -e "${DIM}  ${CMD}${NC}"
  echo ""
  # Use heredoc via stdin to avoid nested-quoting issues with ssh + bash -c.
  # bash -s  : read commands from stdin (no login MOTD).
  exec ssh "$local_remote" bash -s <<REMOTE_SCRIPT
set +x
source ~/.bashrc 2>/dev/null
cd '${CLUSTER_WORKDIR}'
conda activate '${CONDA_ENV}'
export MLFLOW_TRACKING_URI='${MLFLOW_TRACKING_URI}'
${CMD}
REMOTE_SCRIPT
fi
