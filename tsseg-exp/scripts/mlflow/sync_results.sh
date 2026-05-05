#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# sync_results.sh — Synchronize MLflow DB and artifacts between
#                    remote cluster and local machine
#
# Operations:
#   pull       Download DB snapshot + artifacts from remote cluster
#   push       Upload a local DB to the remote cluster
#   status     Show sync status (local vs remote sizes, timestamps)
#
# Artifact sync modes:
#   --artifacts all          Full sync of mlartifacts/ (can be large)
#   --artifacts predictions  Sync only predicted change points & states
#   --artifacts none         Skip artifacts entirely (default for pull-db)
#
# Usage:
#   ./sync_results.sh pull                     # DB only (fast)
#   ./sync_results.sh pull --artifacts all     # DB + all artifacts
#   ./sync_results.sh pull --artifacts predictions  # DB + predictions only
#   ./sync_results.sh pull --experiments "exp1,exp2" # only specific exps' artifacts
#   ./sync_results.sh push --db path/to/db     # upload a compacted DB
#   ./sync_results.sh status                   # compare local vs remote
#
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail
trap 'echo -e "\n${RED}✗ Script failed at line $LINENO (exit code $?)${NC}"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
REMOTE_USER="${CLUSTER_USER:-${USER}}"
REMOTE_HOST="${CLUSTER_HOST:-cleps.inria.fr}"
REMOTE_WORKDIR="${CLUSTER_WORKDIR:-/scratch/${USER}/tsseg-exp}"
REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

ARTIFACT_MODE="none"
EXPERIMENT_FILTER=""
DB_PATH=""
DRY_RUN=false
WORKSPACE="${TSSEG_WORKSPACE:-local/main}"

# ── Workspace naming convention ────────────────────────────────────────
# WORKSPACE is <origin>/<name>.  A bare name is treated as local/<name>.
_normalize_workspace() {
  local ws="$1"
  if [[ "$ws" != */* ]]; then ws="local/$ws"; fi
  echo "$ws"
}
_ws_name() {
  local ws; ws=$(_normalize_workspace "$1"); echo "${ws#*/}"
}
_ws_origin() {
  local ws; ws=$(_normalize_workspace "$1"); echo "${ws%%/*}"
}


# ── Help ──────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
${BOLD}Usage:${NC} $(basename "$0") <command> [options]

${BOLD}Commands:${NC}
  pull       Download DB snapshot and/or artifacts from the remote cluster
  push       Upload a local (compacted) DB to the remote cluster
  status     Compare local and remote DB sizes and timestamps

${BOLD}Options:${NC}
  -w, --workspace NAME   Workspace to sync (default: \$TSSEG_WORKSPACE or "main")
  --artifacts MODE       Artifact sync mode: none|predictions|all (default: none)
  --experiments "e1,e2"  Only sync artifacts for these experiments
  --db PATH              Local DB path (for push)
  --dry-run              Show what would be transferred
  -h, --help             Show this help

${BOLD}Pull examples:${NC}
  $(basename "$0") pull                                  # DB snapshot only
  $(basename "$0") pull --artifacts predictions           # DB + prediction files
  $(basename "$0") pull --artifacts all                   # DB + everything
  $(basename "$0") pull --artifacts predictions --experiments "tsseg-experiment-unsupervised-09-02"

${BOLD}Push examples:${NC}
  $(basename "$0") push --db results/mlflow_compact.db   # replace remote DB

${BOLD}Environment variables:${NC}
  CLUSTER_USER    Remote username   (default: \$USER)
  CLUSTER_HOST    Remote hostname   (default: cleps.inria.fr)
  CLUSTER_WORKDIR Remote work dir   (default: /scratch/\$USER/tsseg-exp)

EOF
  exit 0
}

# ── Helpers ───────────────────────────────────────────────────────────
human_size() {
  local bytes=$1
  if [[ $bytes -gt 1073741824 ]]; then
    echo "$(echo "scale=2; $bytes / 1073741824" | bc) GB"
  elif [[ $bytes -gt 1048576 ]]; then
    echo "$(echo "scale=1; $bytes / 1048576" | bc) MB"
  elif [[ $bytes -gt 1024 ]]; then
    echo "$(echo "scale=0; $bytes / 1024" | bc) KB"
  else
    echo "$bytes B"
  fi
}

remote_exec() {
  ssh -o BatchMode=yes -o ConnectTimeout=10 \
      -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
      "$REMOTE" "$@"
}

# ══════════════════════════════════════════════════════════════════════
#  COMMAND: status
# ══════════════════════════════════════════════════════════════════════
cmd_status() {
  echo -e "${BOLD}═══ Sync Status ═══${NC}"
  echo -e "${BLUE}Workspace:${NC} ${BOLD}${WORKSPACE}${NC}  (remote name: ${WS_NAME})"
  echo ""

  # ── Remote DB (fast: stat + 2 small SQL queries) ──
  echo -e "${BLUE}Remote (${REMOTE_HOST}):${NC}"
  local db_info
  db_info=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=10 \
      -o ServerAliveInterval=5 -o ServerAliveCountMax=3 \
      "$REMOTE" bash -s <<RSCRIPT
DB="${REMOTE_WS_DIR}/mlflow.db"
if [[ -f "\$DB" ]]; then
  SIZE=\$(stat -c%s "\$DB" 2>/dev/null)
  MTIME_H=\$(stat -c%y "\$DB" 2>/dev/null | cut -d'.' -f1)
  N_RUNS=\$(sqlite3 "\$DB" "SELECT COUNT(*) FROM runs;" 2>/dev/null || echo "?")
  N_FINISHED=\$(sqlite3 "\$DB" "SELECT COUNT(*) FROM runs WHERE status='FINISHED';" 2>/dev/null || echo "?")
  N_EXP=\$(sqlite3 "\$DB" "SELECT COUNT(*) FROM experiments WHERE lifecycle_stage='active';" 2>/dev/null || echo "?")
  echo "FOUND|\$SIZE|\$MTIME_H|\$N_EXP|\$N_RUNS|\$N_FINISHED"
else
  echo "NOT_FOUND"
fi
RSCRIPT
  ) || { echo -e "  ${RED}Cannot connect to ${REMOTE_HOST} (timeout or SSH error)${NC}"; db_info="CONN_FAIL"; }

  if echo "$db_info" | grep -q "^FOUND"; then
    local fields
    IFS='|' read -ra fields <<< "$(echo "$db_info" | head -n1)"
    echo -e "  DB:          $(human_size "${fields[1]}") (${fields[1]} bytes)"
    echo -e "  Modified:    ${fields[2]}"
    echo -e "  Experiments: ${fields[3]} active"
    echo -e "  Runs:        ${fields[5]} finished / ${fields[4]} total"
  elif echo "$db_info" | grep -q "NOT_FOUND"; then
    echo -e "  ${YELLOW}No mlflow.db found on remote${NC}"
  elif echo "$db_info" | grep -q "CONN_FAIL"; then
    true  # already printed
  fi

  # ── Remote artifacts (slow: separate call, don't block on it) ──
  if [[ "$db_info" != "CONN_FAIL" ]]; then
    echo -ne "  Artifacts:   ${DIM}scanning...${NC}"
    local art_info
    art_info=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=5 \
        "$REMOTE" bash -s <<RSCRIPT
ART_DIR="${REMOTE_WS_DIR}/mlartifacts"
if [[ -d "\$ART_DIR" ]]; then
  # Count top-level experiment dirs only (instant)
  N_EXP_DIRS=\$(ls -1d "\$ART_DIR"/*/ 2>/dev/null | wc -l || echo 0)
  # Fast size estimate: stat the directory itself (no recursion)
  # For accurate size, du is too slow on large dirs — use df or skip
  DB_SIZE=\$(stat -c%s "${REMOTE_WORKDIR}/mlflow.db" 2>/dev/null || echo 0)
  echo "ART|\$N_EXP_DIRS experiment dirs"
else
  echo "ART_NONE"
fi
RSCRIPT
    ) 2>/dev/null || art_info="ART_TIMEOUT"

    # Clear the "scanning..." text
    echo -ne "\r"

    if echo "$art_info" | grep -q "^ART|"; then
      IFS='|' read -ra afields <<< "$art_info"
      echo -e "  Artifacts:   ${afields[1]}                    "
    elif echo "$art_info" | grep -q "ART_NONE"; then
      echo -e "  Artifacts:   ${DIM}none${NC}                    "
    else
      echo -e "  Artifacts:   ${DIM}(scan timed out)${NC}                    "
    fi
  fi

  echo ""

  # ── Local snapshots ──
  echo -e "${BLUE}Local (workspaces/cluster/${WS_NAME}/):${NC}"
  local found=false
  local cluster_db="$CLUSTER_LOCAL_DIR/mlflow.db"
  if [[ -f "$cluster_db" ]]; then
    found=true
    local size mtime
    size=$(stat -c%s "$cluster_db" 2>/dev/null || stat -f%z "$cluster_db" 2>/dev/null)
    mtime=$(stat -c%y "$cluster_db" 2>/dev/null | cut -d'.' -f1 || stat -f%Sm "$cluster_db" 2>/dev/null)
    echo -e "  mlflow.db  $(human_size "$size")  ($mtime)"
  fi
  # Also list any snapshots in the workspace dir
  while IFS= read -r f; do
    found=true
    local fname size mtime
    fname=$(basename "$f")
    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
    mtime=$(stat -c%y "$f" 2>/dev/null | cut -d'.' -f1 || stat -f%Sm "$f" 2>/dev/null)
    echo -e "  $fname  $(human_size "$size")  ($mtime)"
  done < <(find "$CLUSTER_LOCAL_DIR" -maxdepth 1 -name "mlflow_snapshot*.db" -not -name "*-wal" -not -name "*-shm" 2>/dev/null | sort -r)

  if [[ "$found" == false ]]; then
    echo -e "  ${DIM}No local copy${NC}"
  fi

  # Local artifacts
  echo ""
  if [[ -d "$ARTIFACTS_LOCAL" ]]; then
    local local_art_size local_art_count
    local_art_size=$(du -sb "$ARTIFACTS_LOCAL" 2>/dev/null | cut -f1 || echo 0)
    local_art_count=$(find "$ARTIFACTS_LOCAL" -type f 2>/dev/null | wc -l)
    echo -e "${BLUE}Local artifacts:${NC}  $(human_size "$local_art_size") ($local_art_count files)"
  else
    echo -e "${BLUE}Local artifacts:${NC}  ${DIM}none${NC}"
  fi
}

# ══════════════════════════════════════════════════════════════════════
#  COMMAND: pull
# ══════════════════════════════════════════════════════════════════════
cmd_pull() {
  echo -e "${BOLD}═══ Pull Results from ${REMOTE_HOST} (workspace: ${WS_NAME}) ═══${NC}"
  echo ""

  mkdir -p "$CLUSTER_LOCAL_DIR"

  # ── 1. Pull DB snapshot ──
  local snap_path="$CLUSTER_LOCAL_DIR/mlflow.db"

  echo -e "${BLUE}[1/2] Pulling DB snapshot...${NC}"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${DIM}  Would: sqlite3 backup → gzip → scp → gunzip${NC}"
    echo -e "${DIM}  Destination: $snap_path${NC}"
  else
    # Hot backup on remote + compress
    local remote_snap="/tmp/mlflow_snap_$$"
    echo -e "${DIM}  Remote backup + compress (this may take a few minutes for large DBs)...${NC}"
    if ! timeout 600 ssh -o BatchMode=yes -o ConnectTimeout=10 \
        -o ServerAliveInterval=10 -o ServerAliveCountMax=6 \
        "$REMOTE" bash -s <<RSCRIPT
set -e
echo "Backing up DB..."
sqlite3 "${REMOTE_WS_DIR}/mlflow.db" ".backup '${remote_snap}.db'"
SIZE=\$(stat -c%s "${remote_snap}.db" 2>/dev/null || echo "?")
echo "Backup done (\${SIZE} bytes). Compressing..."
gzip -1 "${remote_snap}.db"
ls -lh "${remote_snap}.db.gz"
echo "Compression done."
RSCRIPT
    then
      echo -e "${RED}✗ Remote backup/compress failed or timed out (10 min limit).${NC}"
      echo -e "${DIM}  Try running manually: ssh $REMOTE 'ls -lh ${REMOTE_WORKDIR}/mlflow.db'${NC}"
      exit 1
    fi

    # Download
    echo -e "${DIM}  Downloading snapshot...${NC}"
    if ! scp -C "${REMOTE}:${remote_snap}.db.gz" "${snap_path}.gz"; then
      echo -e "${RED}✗ Download failed.${NC}"
      remote_exec "rm -f ${remote_snap}.db.gz" 2>/dev/null || true
      exit 1
    fi

    # Decompress
    echo -e "${DIM}  Decompressing...${NC}"
    gunzip -f "${snap_path}.gz"
    rm -f "${snap_path}-wal" "${snap_path}-shm"

    # Cleanup remote
    remote_exec "rm -f ${remote_snap}.db.gz" 2>/dev/null || true

    local size
    size=$(stat -c%s "$snap_path" 2>/dev/null || stat -f%z "$snap_path" 2>/dev/null)
    echo -e "${GREEN}✓${NC} DB snapshot: $snap_path ($(human_size "$size"))"
    echo -e "${DIM}  Workspace: workspaces/cluster/${WS_NAME}/${NC}"
  fi

  # ── 2. Pull artifacts ──
  if [[ "$ARTIFACT_MODE" == "none" ]]; then
    echo -e "${DIM}[2/2] Skipping artifacts (use --artifacts predictions|all)${NC}"
    echo ""
    echo -e "${GREEN}✓ Pull complete.${NC}"
    return
  fi

  echo ""
  echo -e "${BLUE}[2/2] Pulling artifacts (mode: $ARTIFACT_MODE)...${NC}"
  mkdir -p "$ARTIFACTS_LOCAL"

  local rsync_opts=(-avz --progress --partial --human-readable)

  case "$ARTIFACT_MODE" in
    all)
      # Full artifact sync
      if [[ "$DRY_RUN" == "true" ]]; then
        rsync_opts+=(--dry-run)
      fi
      rsync "${rsync_opts[@]}" \
        "${REMOTE}:${REMOTE_WS_DIR}/mlartifacts/" \
        "$ARTIFACTS_LOCAL/"
      ;;

    predictions)
      # Only sync prediction files (*.npy, *.json, *.csv with predictions/changepoints)
      # MLflow artifacts structure: mlartifacts/<exp_id>/<run_id>/artifacts/
      local include_patterns=()
      include_patterns+=(--include='*/')
      include_patterns+=(--include='predicted_change_points.*')
      include_patterns+=(--include='predicted_labels.*')
      include_patterns+=(--include='predicted_states.*')
      include_patterns+=(--include='change_points_*')
      include_patterns+=(--include='labels_*')
      include_patterns+=(--include='scores_*')
      include_patterns+=(--include='*.npy')
      include_patterns+=(--exclude='*')

      if [[ -n "$EXPERIMENT_FILTER" ]]; then
        # Resolve experiment IDs for the filter
        echo -e "${DIM}  Filtering experiments: $EXPERIMENT_FILTER${NC}"
        local exp_ids
        exp_ids=$(remote_exec bash -s <<RSCRIPT
sqlite3 "${REMOTE_WS_DIR}/mlflow.db" "
  SELECT experiment_id FROM experiments
  WHERE name IN ($(echo "$EXPERIMENT_FILTER" | sed "s/,/','/g" | sed "s/^/'/" | sed "s/$/'/" ))
  AND lifecycle_stage = 'active';
"
RSCRIPT
        )

        # Sync per experiment ID
        while IFS= read -r eid; do
          [[ -z "$eid" ]] && continue
          echo -e "${DIM}  Syncing experiment $eid...${NC}"
          if [[ "$DRY_RUN" == "true" ]]; then
            rsync_opts+=(--dry-run)
          fi
          rsync "${rsync_opts[@]}" "${include_patterns[@]}" \
            "${REMOTE}:${REMOTE_WS_DIR}/mlartifacts/${eid}/" \
            "$ARTIFACTS_LOCAL/${eid}/" 2>/dev/null || true
        done <<< "$exp_ids"
      else
        # Sync all experiments' predictions
        if [[ "$DRY_RUN" == "true" ]]; then
          rsync_opts+=(--dry-run)
        fi
        rsync "${rsync_opts[@]}" "${include_patterns[@]}" \
          "${REMOTE}:${REMOTE_WS_DIR}/mlartifacts/" \
          "$ARTIFACTS_LOCAL/"
      fi
      ;;
  esac

  echo ""
  echo -e "${GREEN}✓ Pull complete.${NC}"
}

# ══════════════════════════════════════════════════════════════════════
#  COMMAND: push
# ══════════════════════════════════════════════════════════════════════
cmd_push() {
  local db="$1"

  if [[ ! -f "$db" ]]; then
    echo -e "${RED}✗${NC} Database not found: $db"
    exit 1
  fi

  local size
  size=$(stat -c%s "$db" 2>/dev/null || stat -f%z "$db" 2>/dev/null)

  echo -e "${BOLD}═══ Push DB to ${REMOTE_HOST} (workspace: ${WS_NAME}) ═══${NC}"
  echo ""
  echo -e "${BLUE}Source:${NC}       $db ($(human_size "$size"))"
  echo -e "${BLUE}Destination:${NC}  ${REMOTE_WS_DIR}/mlflow.db"
  echo ""
  echo -e "${YELLOW}⚠  This will REPLACE the remote DB.${NC}"
  echo -e "${YELLOW}   Make sure the MLflow server is STOPPED first.${NC}"
  echo ""

  if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${DIM}[dry-run] Would upload $db → ${REMOTE_WORKDIR}/mlflow.db${NC}"
    return
  fi

  read -r -p "  Proceed? [y/N] " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 0
  fi

  # Backup current remote DB first
  echo -e "${BLUE}Backing up current remote DB...${NC}"
  remote_exec bash -s <<RSCRIPT
set -e
if [[ -f "${REMOTE_WS_DIR}/mlflow.db" ]]; then
  cp "${REMOTE_WS_DIR}/mlflow.db" "${REMOTE_WS_DIR}/mlflow.db.bak"
  echo "Backup: mlflow.db.bak"
fi
RSCRIPT

  # Upload
  echo -e "${BLUE}Uploading...${NC}"
  scp "$db" "${REMOTE}:${REMOTE_WS_DIR}/mlflow.db"

  # Remove any leftover WAL/SHM on remote
  remote_exec "rm -f ${REMOTE_WS_DIR}/mlflow.db-wal ${REMOTE_WS_DIR}/mlflow.db-shm" 2>/dev/null || true

  echo -e "${GREEN}✓${NC} DB pushed successfully."
  echo -e "${DIM}  Previous DB saved as mlflow.db.bak on remote.${NC}"
  echo -e "${DIM}  You can now restart the MLflow server.${NC}"
}


# ══════════════════════════════════════════════════════════════════════
#  MAIN — Argument Parsing
# ══════════════════════════════════════════════════════════════════════

# Handle --help / -h as first argument
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
fi

COMMAND="${1:-}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workspace)  WORKSPACE="$2";         shift 2 ;;
    -p|--profile)    shift 2 ;;  # already handled in pre-parse
    --artifacts)     ARTIFACT_MODE="$2";       shift 2 ;;
    --experiments)   EXPERIMENT_FILTER="$2";   shift 2 ;;
    --db)            DB_PATH="$2";             shift 2 ;;
    --dry-run)       DRY_RUN=true;             shift ;;
    -h|--help)       usage ;;
    *)               echo -e "${RED}Unknown option: $1${NC}"; usage ;;
  esac
done

# Normalize workspace
WORKSPACE=$(_normalize_workspace "$WORKSPACE")
WS_NAME=$(_ws_name "$WORKSPACE")

# Derived paths
# Remote always uses the bare name (no origin prefix).
REMOTE_WS_DIR="${REMOTE_WORKDIR}/workspaces/${WS_NAME}"
# Pull always writes into workspaces/cluster/<name>/
CLUSTER_LOCAL_DIR="$PROJECT_DIR/workspaces/cluster/${WS_NAME}"
ARTIFACTS_LOCAL="$CLUSTER_LOCAL_DIR/mlartifacts"

case "$COMMAND" in
  pull)
    cmd_pull
    ;;
  push)
    db="${DB_PATH:-}"
    if [[ -z "$db" ]]; then
      echo -e "${RED}✗${NC} --db required for push. Specify the DB to upload."
      exit 1
    fi
    cmd_push "$db"
    ;;
  status)
    cmd_status
    ;;
  ""|help)
    usage
    ;;
  *)
    echo -e "${RED}Unknown command: $COMMAND${NC}"
    usage
    ;;
esac
