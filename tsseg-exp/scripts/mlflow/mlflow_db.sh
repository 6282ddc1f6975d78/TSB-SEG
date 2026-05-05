#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# mlflow_db.sh — Manage MLflow workspaces and SQLite databases
#
# Workspaces are organized under workspaces/<origin>/<name>/:
#   workspaces/local/<name>/    — created locally
#   workspaces/cluster/<name>/  — pulled from the remote cluster
#
# On the remote cluster the layout is flat:
#   $CLUSTER_WORKDIR/workspaces/<name>/
#
# Each workspace contains:
#   mlflow.db        SQLite backend
#   mlartifacts/     Run artifacts
#   outputs/         Hydra outputs
#   multirun/        Hydra multirun sweeps
#
# The workspace is identified as <origin>/<name> (e.g. local/dev).
# A bare name like "main" is treated as "local/main".
#
# Operations:
#   new         Create a fresh workspace (empty DB + directories)
#   reset       Purge a workspace (remove DB, artifacts, outputs…)
#   compact     Create a clean DB keeping only active experiments
#   snapshot    Create a timestamped snapshot of a workspace's DB
#   info        Show stats about a workspace's DB
#   list        List available snapshots across all workspaces
#   prune       Remove old snapshots, keeping the N most recent
#   workspaces  List all existing workspaces (local and remote)
#
# Every command accepts --workspace <origin/name> (default:
# $TSSEG_WORKSPACE from .env, or "local/main").
#
# Usage:
#   ./mlflow_db.sh new                                   # new "main" workspace on cluster
#   ./mlflow_db.sh new --workspace dev                   # new "dev" workspace on cluster
#   ./mlflow_db.sh reset --workspace dev                 # purge dev
#   ./mlflow_db.sh compact --workspace local/main        # prune main DB
#   ./mlflow_db.sh info                                  # stats on active ws
#   ./mlflow_db.sh workspaces                            # list all workspaces
#
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/results"

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
CONDA_ENV="${CONDA_ENV:-tsseg-env}"
DB_PATH=""
KEEP_EXPERIMENTS=""
KEEP_N=3

# Default workspace (overridable via .env or --workspace)
WORKSPACE="${TSSEG_WORKSPACE:-local/main}"

# ── Workspace naming convention ────────────────────────────────────────
# WORKSPACE is <origin>/<name>.  A bare name is treated as local/<name>.
_normalize_workspace() {
  local ws="$1"
  if [[ "$ws" != */* ]]; then
    ws="local/$ws"
  fi
  echo "$ws"
}

# Extract the bare name (without origin prefix)
_ws_name() {
  local ws="$1"
  ws=$(_normalize_workspace "$ws")
  echo "${ws#*/}"
}

# Extract the origin (local or cluster)
_ws_origin() {
  local ws="$1"
  ws=$(_normalize_workspace "$ws")
  echo "${ws%%/*}"
}

# ── Workspace path helpers ──────────────────────────────────────────────
# Local: workspaces/<origin>/<name>/
ws_local_dir()  {
  local ws
  ws=$(_normalize_workspace "$1")
  echo "$PROJECT_DIR/workspaces/$ws"
}
# Remote (cluster): workspaces/<name>/  (flat, no origin prefix)
ws_remote_dir() { echo "${REMOTE_WORKDIR}/workspaces/$(_ws_name "$1")"; }
ws_db_path()    { echo "$(ws_local_dir "$1")/mlflow.db"; }

# ── Help ──────────────────────────────────────────────────────────────
usage() {
  cat <<EOF
${BOLD}Usage:${NC} $(basename "$0") <command> [options]

${BOLD}Commands:${NC}
  new          Create a fresh workspace (empty DB + directory structure)
  reset        Purge a workspace (remove DB, artifacts, outputs, multirun)
  compact      Create a new DB keeping only active/selected experiments
  snapshot     Create a timestamped snapshot of the workspace's DB
  info         Show statistics about a workspace's DB
  list         List available snapshots across workspaces
  prune        Remove old snapshots, keeping the N most recent
  workspaces   List all existing workspaces (local and remote)

${BOLD}Options:${NC}
  -w, --workspace NAME  Workspace to operate on.  Format: <origin>/<name>
                        (e.g. local/dev, cluster/main).  A bare name is
                        treated as local/<name>.  Default: \$TSSEG_WORKSPACE
                        or "local/main".
  --db PATH             Path to a specific DB (overrides workspace DB for info/compact)
  --keep "e1,e2"        Experiment names to keep during compact (default: all active)
  --keep-n N            Number of snapshots to keep during prune (default: 3)
  --remote              Operate on the remote cluster workspace (default for new/reset)
  --local               Force local operation
  -h, --help            Show this help

${BOLD}Examples:${NC}
  $(basename "$0") new                                    # new "main" workspace on cluster
  $(basename "$0") new -w dev                             # new "dev" workspace on cluster
  $(basename "$0") new -w dev --local                     # new "dev" workspace locally
  $(basename "$0") reset -w dev                           # purge "dev" on cluster
  $(basename "$0") info -w local/main                     # stats on local/main
  $(basename "$0") compact -w local/main                  # prune old experiments
  $(basename "$0") snapshot -w local/main                 # backup main DB
  $(basename "$0") workspaces                             # list all workspaces

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

# ── Python-based sqlite3 wrappers (no CLI dependency) ────────────────
_sqlite3() {
  local db="$1" sql="$2"
  python3 -c "
import sqlite3, sys
conn = sqlite3.connect('$db')
try:
    conn.executescript('''$sql''')
    conn.commit()
except Exception as e:
    print(f'SQLite error: {e}', file=sys.stderr)
    sys.exit(1)
finally:
    conn.close()
"
}

_sqlite3_query() {
  local db="$1" sql="$2"
  python3 << PYEOF
import sqlite3, sys
conn = sqlite3.connect('$db')
try:
    cur = conn.execute('''$sql''')
    rows = cur.fetchall()
    if cur.description:
        headers = [d[0] for d in cur.description]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))
        print('  '.join(h.ljust(w) for h, w in zip(headers, widths)))
        print('  '.join('-' * w for w in widths))
        for row in rows:
            print('  '.join(str(v).ljust(w) for v, w in zip(row, widths)))
except Exception as e:
    print(f'SQLite error: {e}', file=sys.stderr)
    sys.exit(1)
finally:
    conn.close()
PYEOF
}

_sqlite3_scalar() {
  local db="$1" sql="$2"
  python3 -c "
import sqlite3
conn = sqlite3.connect('$db')
result = conn.execute('''$sql''').fetchone()
print(result[0] if result else '')
conn.close()
"
}

_sqlite3_backup() {
  local src="$1" dst="$2"
  python3 -c "
import sqlite3
src = sqlite3.connect('$src')
dst = sqlite3.connect('$dst')
src.backup(dst)
dst.close()
src.close()
"
}


# ══════════════════════════════════════════════════════════════════════
#  COMMAND: workspaces — list all workspaces
# ══════════════════════════════════════════════════════════════════════
cmd_workspaces() {
  echo -e "${BOLD}═══ MLflow Workspaces ═══${NC}"
  echo ""

  # ── Local ──
  local active_ws
  active_ws=$(_normalize_workspace "$WORKSPACE")
  for origin in local cluster; do
    echo -e "${BLUE}Local — ${origin}/ (${PROJECT_DIR}/workspaces/${origin}/):${NC}"
    local ws_root="$PROJECT_DIR/workspaces/${origin}"
    if [[ -d "$ws_root" ]]; then
      local found=false
      for d in "$ws_root"/*/; do
        [[ ! -d "$d" ]] && continue
        found=true
        local name
        name=$(basename "$d")
        local full_ws="${origin}/${name}"
        local marker=""
        [[ "$full_ws" == "$active_ws" ]] && marker=" ${GREEN}← active${NC}"
        local db="$d/mlflow.db"
        if [[ -f "$db" ]]; then
          local size n_runs
          size=$(stat -c%s "$db" 2>/dev/null || stat -f%z "$db" 2>/dev/null)
          n_runs=$(_sqlite3_scalar "$db" "SELECT COUNT(*) FROM runs WHERE status='FINISHED'" 2>/dev/null || echo "?")
          echo -e "  ${BOLD}${full_ws}${NC}  $(human_size "$size")  ${DIM}($n_runs finished runs)${NC}$marker"
        else
          echo -e "  ${BOLD}${full_ws}${NC}  ${DIM}(no DB)${NC}$marker"
        fi
      done
      if [[ "$found" == false ]]; then
        echo -e "  ${DIM}No workspaces found.${NC}"
      fi
    else
      echo -e "  ${DIM}No ${origin}/ directory.${NC}"
    fi
    echo ""
  done

  # ── Remote ──
  echo -e "${BLUE}Remote (${REMOTE_HOST}:${REMOTE_WORKDIR}/workspaces/):${NC}"
  local remote_info
  if ! remote_info=$(
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
      bash -s "${REMOTE_WORKDIR}" 2>/dev/null <<'RSCRIPT'
WS_DIR="$1/workspaces"
if [[ -d "$WS_DIR" ]]; then
  found=false
  for d in "$WS_DIR"/*/; do
    [[ ! -d "$d" ]] && continue
    found=true
    name=$(basename "$d")
    db="$d/mlflow.db"
    if [[ -f "$db" ]]; then
      size=$(stat -c%s "$db" 2>/dev/null || echo "0")
      n_runs=$(python3 -c "
import sqlite3
c = sqlite3.connect('$db')
print(c.execute('SELECT COUNT(*) FROM runs WHERE status=\"FINISHED\"').fetchone()[0])
c.close()
" 2>/dev/null || echo "?")
      echo "$name|$size|$n_runs"
    else
      echo "$name|0|none"
    fi
  done
  if [[ "$found" == false ]]; then echo "EMPTY"; fi
else
  echo "NONE"
fi
RSCRIPT
  ); then
    echo -e "  ${RED}Cannot connect to ${REMOTE_HOST}${NC}"
    return
  fi

  if [[ "$remote_info" == "NONE" || "$remote_info" == "EMPTY" ]]; then
    echo -e "  ${DIM}No workspaces found.${NC}"
  else
    while IFS='|' read -r name size n_runs; do
      [[ -z "$name" ]] && continue
      local marker=""
      [[ "$name" == "$(_ws_name "$WORKSPACE")" ]] && marker=" ${GREEN}← active${NC}"
      if [[ "$n_runs" == "none" ]]; then
        echo -e "  ${BOLD}$name${NC}  ${DIM}(no DB)${NC}$marker"
      else
        echo -e "  ${BOLD}$name${NC}  $(human_size "$size")  ${DIM}($n_runs finished runs)${NC}$marker"
      fi
    done <<< "$remote_info"
  fi
}


# ══════════════════════════════════════════════════════════════════════
#  COMMAND: info
# ══════════════════════════════════════════════════════════════════════
cmd_info() {
  local db="$1" ws="$2"
  if [[ ! -f "$db" ]]; then
    echo -e "${RED}✗${NC} Database not found: $db"
    exit 1
  fi

  local size
  size=$(stat -c%s "$db" 2>/dev/null || stat -f%z "$db" 2>/dev/null)

  echo -e "${BOLD}═══ MLflow DB Info ═══${NC}"
  echo -e "${BLUE}Workspace:${NC} $ws"
  echo -e "${BLUE}Path:${NC}      $db"
  echo -e "${BLUE}Size:${NC}      $(human_size "$size") ($size bytes)"

  # WAL/SHM
  for ext in wal shm; do
    if [[ -f "${db}-${ext}" ]]; then
      local ext_size
      ext_size=$(stat -c%s "${db}-${ext}" 2>/dev/null || stat -f%z "${db}-${ext}" 2>/dev/null)
      echo -e "${DIM}  ${ext^^}: $(human_size "$ext_size")${NC}"
    fi
  done

  # Workspace directory contents
  local ws_dir
  ws_dir=$(dirname "$db")
  echo ""
  echo -e "${BOLD}Workspace contents:${NC}"
  for subdir in mlartifacts outputs multirun; do
    if [[ -d "$ws_dir/$subdir" ]]; then
      local count
      count=$(find "$ws_dir/$subdir" -type f 2>/dev/null | wc -l)
      echo -e "  ${subdir}/  ${DIM}($count files)${NC}"
    else
      echo -e "  ${subdir}/  ${DIM}(absent)${NC}"
    fi
  done

  echo ""
  echo -e "${BOLD}Experiments:${NC}"
  _sqlite3_query "$db" "
SELECT
  e.experiment_id                                          AS id,
  e.name                                                   AS experiment_name,
  e.lifecycle_stage                                        AS stage,
  COUNT(r.run_uuid)                                        AS total_runs,
  SUM(CASE WHEN r.status = 'FINISHED' THEN 1 ELSE 0 END)  AS finished,
  SUM(CASE WHEN r.status = 'FAILED'   THEN 1 ELSE 0 END)  AS failed,
  SUM(CASE WHEN r.status = 'RUNNING'  THEN 1 ELSE 0 END)  AS running
FROM experiments e
LEFT JOIN runs r ON e.experiment_id = r.experiment_id
GROUP BY e.experiment_id, e.name, e.lifecycle_stage
ORDER BY e.experiment_id;
"

  echo ""
  echo -e "${BOLD}Summary:${NC}"
  _sqlite3_query "$db" "
SELECT
  (SELECT COUNT(*) FROM experiments WHERE lifecycle_stage = 'active')  AS active_experiments,
  (SELECT COUNT(*) FROM experiments WHERE lifecycle_stage = 'deleted') AS deleted_experiments,
  (SELECT COUNT(*) FROM runs)                                         AS total_runs,
  (SELECT COUNT(*) FROM runs WHERE status = 'FINISHED')               AS finished_runs,
  (SELECT COUNT(*) FROM metrics)                                      AS total_metrics,
  (SELECT COUNT(*) FROM params)                                       AS total_params,
  (SELECT COUNT(*) FROM tags)                                         AS total_tags;
"
}


# ══════════════════════════════════════════════════════════════════════
#  COMMAND: new
# ══════════════════════════════════════════════════════════════════════
cmd_new() {
  local ws="$1" is_remote="$2"

  if [[ "$is_remote" == "true" ]]; then
    local remote_ws
    remote_ws=$(ws_remote_dir "$ws")
    local remote_db="${remote_ws}/mlflow.db"

    echo -e "${BOLD}═══ Create Workspace \"${ws}\" on ${REMOTE_HOST} ═══${NC}"
    echo -e "${BLUE}Target:${NC} ${REMOTE}:${remote_ws}/"
    echo ""

    local exists
    exists=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
      "[[ -f '${remote_db}' ]] && echo yes || echo no" 2>/dev/null) \
      || { echo -e "${RED}✗ Cannot connect to ${REMOTE_HOST}${NC}"; exit 1; }

    if [[ "$exists" == "yes" ]]; then
      local remote_runs
      remote_runs=$(ssh "$REMOTE" "python3 -c \"
import sqlite3
c = sqlite3.connect('${remote_db}')
print(c.execute('SELECT COUNT(*) FROM runs WHERE status=\\\"FINISHED\\\"').fetchone()[0])
c.close()
\"" 2>/dev/null || echo "?")

      echo -e "${YELLOW}⚠  Workspace \"${ws}\" already exists on the cluster:${NC}"
      echo -e "   Finished runs: ${remote_runs}"
      echo ""
      echo -e "   ${BOLD}1)${NC} Reset workspace (backup DB → create fresh)"
      echo -e "   ${BOLD}2)${NC} Abort"
      echo ""
      read -r -p "  Choice [1/2]: " choice
      case "$choice" in
        1)
          echo -e "${BLUE}Backing up existing workspace...${NC}"
          ssh "$REMOTE" "cp '${remote_db}' '${remote_db}.bak' && rm -f '${remote_db}-wal' '${remote_db}-shm'"
          echo -e "${GREEN}✓${NC} Backup saved as mlflow.db.bak"
          ;;
        *) echo "Aborted."; exit 0 ;;
      esac
    fi

    echo -e "${BLUE}Initializing workspace on cluster...${NC}"
    ssh "$REMOTE" bash -s "$remote_ws" "$CONDA_ENV" << 'RSCRIPT'
set -e
WS_DIR="$1"
CONDA_ENV_NAME="$2"

# Activate conda env for mlflow
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "$CONDA_ENV_NAME" 2>/dev/null || true

mkdir -p "$WS_DIR"/{mlartifacts,outputs,multirun}

rm -f "$WS_DIR"/mlflow.db "$WS_DIR"/mlflow.db-wal "$WS_DIR"/mlflow.db-shm

python3 -c "
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri('sqlite:///$1/mlflow.db')
client = MlflowClient()
client.search_experiments()
print('Schema initialized.')
"

python3 -c "
import sqlite3
conn = sqlite3.connect('$1/mlflow.db')
conn.execute('PRAGMA journal_mode=WAL')
conn.execute('PRAGMA busy_timeout=30000')
conn.close()
print('WAL mode enabled.')
"

echo "Contents:"
ls -la "$WS_DIR"/
RSCRIPT

    echo ""
    echo -e "${GREEN}✓${NC} Workspace \"${ws}\" created on ${REMOTE_HOST}:${remote_ws}/"
    echo ""
    echo -e "${DIM}Start MLflow server: ./submit_mlflow.sh --workspace ${ws}${NC}"

  else
    local ws_dir
    ws_dir=$(ws_local_dir "$ws")
    local db
    db=$(ws_db_path "$ws")

    echo -e "${BOLD}═══ Create Local Workspace \"${ws}\" ═══${NC}"
    echo -e "${BLUE}Target:${NC} $ws_dir/"
    echo ""

    if [[ -f "$db" ]]; then
      echo -e "${YELLOW}⚠${NC}  Workspace \"${ws}\" already exists: $ws_dir/"
      read -r -p "  Reset it? [y/N] " confirm
      if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo "Aborted."; exit 0
      fi
      rm -f "$db" "${db}-wal" "${db}-shm"
    fi

    mkdir -p "$ws_dir"/{mlartifacts,outputs,multirun}

    echo -e "${BLUE}Creating MLflow DB:${NC} $db"
    python3 -c "
import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri('sqlite:///${db}')
client = MlflowClient()
client.search_experiments()
print('Schema initialized.')
"
    _sqlite3 "$db" "PRAGMA journal_mode=WAL; PRAGMA busy_timeout=30000;"

    local size
    size=$(stat -c%s "$db" 2>/dev/null || stat -f%z "$db" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Workspace \"${ws}\" created: $ws_dir/ (DB: $(human_size "$size"))"
    echo ""
    local full_ws
    full_ws=$(_normalize_workspace "$ws")
    echo -e "${DIM}Activate it:  export TSSEG_WORKSPACE=${full_ws}${NC}"
    echo -e "${DIM}Or in .env:   TSSEG_WORKSPACE=${full_ws}${NC}"
  fi
}


# ══════════════════════════════════════════════════════════════════════
#  COMMAND: reset
# ══════════════════════════════════════════════════════════════════════
cmd_reset() {
  local ws="$1" is_remote="$2"

  if [[ "$is_remote" == "true" ]]; then
    local remote_ws
    remote_ws=$(ws_remote_dir "$ws")

    echo -e "${BOLD}═══ Reset Workspace \"${ws}\" on ${REMOTE_HOST} ═══${NC}"
    echo -e "${BLUE}Target:${NC} ${REMOTE}:${remote_ws}/"
    echo ""

    local remote_info
    if ! remote_info=$(
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE" \
        bash -s "$remote_ws" 2>/dev/null <<'RSCRIPT'
WS="$1"
if [[ ! -d "$WS" ]]; then echo "NOT_FOUND"; exit 0; fi
db_size=0; art_files=0; out_files=0; mr_files=0
[[ -f "$WS/mlflow.db" ]] && db_size=$(stat -c%s "$WS/mlflow.db" 2>/dev/null || echo 0)
[[ -d "$WS/mlartifacts" ]] && art_files=$(find "$WS/mlartifacts" -type f 2>/dev/null | wc -l)
[[ -d "$WS/outputs" ]] && out_files=$(find "$WS/outputs" -type f 2>/dev/null | wc -l)
[[ -d "$WS/multirun" ]] && mr_files=$(find "$WS/multirun" -type f 2>/dev/null | wc -l)
echo "FOUND|$db_size|$art_files|$out_files|$mr_files"
RSCRIPT
    ); then
      echo -e "${RED}✗ Cannot connect to ${REMOTE_HOST}${NC}"; exit 1
    fi

    if [[ "$remote_info" == "NOT_FOUND" ]]; then
      echo -e "${YELLOW}Workspace \"${ws}\" does not exist on remote.${NC}"
      exit 0
    fi

    IFS='|' read -r _ db_size art_files out_files mr_files <<< "$remote_info"
    echo -e "  DB:          $(human_size "$db_size")"
    echo -e "  Artifacts:   $art_files files"
    echo -e "  Outputs:     $out_files files"
    echo -e "  Multirun:    $mr_files files"
    echo ""
    echo -e "${YELLOW}⚠  This will DELETE everything in this workspace.${NC}"
    read -r -p "  Proceed? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
      echo "Aborted."; exit 0
    fi

    ssh "$REMOTE" bash -s "$remote_ws" <<'RSCRIPT'
set -e
WS="$1"
rm -f "$WS"/mlflow.db "$WS"/mlflow.db-wal "$WS"/mlflow.db-shm
rm -rf "$WS"/mlartifacts "$WS"/outputs "$WS"/multirun
mkdir -p "$WS"/{mlartifacts,outputs,multirun}
echo "Workspace purged and directories recreated."
RSCRIPT

    echo -e "${GREEN}✓${NC} Workspace \"${ws}\" reset on ${REMOTE_HOST}."
    echo -e "${DIM}Run 'new -w ${ws}' to reinitialize the DB.${NC}"

  else
    local ws_dir
    ws_dir=$(ws_local_dir "$ws")

    echo -e "${BOLD}═══ Reset Local Workspace \"${ws}\" ═══${NC}"
    echo -e "${BLUE}Target:${NC} $ws_dir/"
    echo ""

    if [[ ! -d "$ws_dir" ]]; then
      echo -e "${YELLOW}Workspace \"${ws}\" does not exist locally.${NC}"
      exit 0
    fi

    local db_size=0 art_files=0 out_files=0 mr_files=0
    [[ -f "$ws_dir/mlflow.db" ]] && db_size=$(stat -c%s "$ws_dir/mlflow.db" 2>/dev/null || echo 0)
    [[ -d "$ws_dir/mlartifacts" ]] && art_files=$(find "$ws_dir/mlartifacts" -type f 2>/dev/null | wc -l)
    [[ -d "$ws_dir/outputs" ]] && out_files=$(find "$ws_dir/outputs" -type f 2>/dev/null | wc -l)
    [[ -d "$ws_dir/multirun" ]] && mr_files=$(find "$ws_dir/multirun" -type f 2>/dev/null | wc -l)

    echo -e "  DB:          $(human_size "$db_size")"
    echo -e "  Artifacts:   $art_files files"
    echo -e "  Outputs:     $out_files files"
    echo -e "  Multirun:    $mr_files files"
    echo ""
    echo -e "${YELLOW}⚠  This will DELETE everything in this workspace.${NC}"
    read -r -p "  Proceed? [y/N] " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
      echo "Aborted."; exit 0
    fi

    rm -f "$ws_dir"/mlflow.db "$ws_dir"/mlflow.db-wal "$ws_dir"/mlflow.db-shm
    rm -rf "$ws_dir"/mlartifacts "$ws_dir"/outputs "$ws_dir"/multirun
    mkdir -p "$ws_dir"/{mlartifacts,outputs,multirun}

    echo -e "${GREEN}✓${NC} Workspace \"${ws}\" reset."
    echo -e "${DIM}Run 'new -w ${ws} --local' to reinitialize the DB.${NC}"
  fi
}


# ══════════════════════════════════════════════════════════════════════
#  COMMAND: compact
# ══════════════════════════════════════════════════════════════════════
cmd_compact() {
  local src_db="$1" keep_csv="$2" ws="$3"

  if [[ ! -f "$src_db" ]]; then
    echo -e "${RED}✗${NC} Source database not found: $src_db"
    exit 1
  fi

  echo -e "${BOLD}═══ Compact Workspace \"${ws}\" ═══${NC}"
  echo -e "${BLUE}Source:${NC} $src_db"
  echo ""

  local keep_clause=""
  if [[ -n "$keep_csv" ]]; then
    local in_list=""
    IFS=',' read -ra NAMES <<< "$keep_csv"
    for name in "${NAMES[@]}"; do
      name=$(echo "$name" | xargs)
      if [[ -n "$in_list" ]]; then in_list+=","; fi
      in_list+="'$name'"
    done
    keep_clause="AND e.name IN ($in_list)"
    echo -e "${BLUE}Keeping experiments:${NC} $keep_csv"
  else
    echo -e "${BLUE}Keeping:${NC} all active (non-deleted) experiments"
    keep_clause="AND e.lifecycle_stage = 'active'"
  fi

  echo ""
  echo -e "${BOLD}Experiments to keep:${NC}"
  _sqlite3_query "$src_db" "
    SELECT e.experiment_id AS id, e.name, COUNT(r.run_uuid) as runs
    FROM experiments e
    LEFT JOIN runs r ON e.experiment_id = r.experiment_id
    WHERE 1=1 $keep_clause
    GROUP BY e.experiment_id, e.name;
  "

  echo ""
  echo -e "${BOLD}Experiments to DROP:${NC}"
  _sqlite3_query "$src_db" "
    SELECT e.experiment_id AS id, e.name, COUNT(r.run_uuid) as runs
    FROM experiments e
    LEFT JOIN runs r ON e.experiment_id = r.experiment_id
    WHERE NOT (1=1 $keep_clause)
    GROUP BY e.experiment_id, e.name;
  "

  echo ""
  read -r -p "  Proceed with compaction? [y/N] " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."; exit 0
  fi

  local timestamp
  timestamp=$(date +%Y%m%d_%H%M%S)
  local compact_db
  compact_db="$(dirname "$src_db")/mlflow_compact_${timestamp}.db"

  echo -e "${BLUE}Creating compacted DB:${NC} $compact_db"
  rm -f "$compact_db" "${compact_db}-wal" "${compact_db}-shm"

  python3 << PYEOF
import sqlite3, sys

src = "$src_db"
dst = "$compact_db"
keep_clause = """$keep_clause"""

src_conn = sqlite3.connect(src)
schema_rows = src_conn.execute(
    "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type DESC"
).fetchall()

dst_conn = sqlite3.connect(dst)
dst_conn.execute("PRAGMA journal_mode=WAL")
dst_conn.execute("PRAGMA busy_timeout=30000")

for (ddl,) in schema_rows:
    try:
        dst_conn.execute(ddl)
    except sqlite3.OperationalError:
        pass
dst_conn.commit()
print("  Schema cloned.")

try:
    exp_query = f"SELECT experiment_id FROM experiments e WHERE 1=1 {keep_clause}"
    exp_ids = [row[0] for row in src_conn.execute(exp_query).fetchall()]
    if not exp_ids:
        print("No experiments match the filter. Aborting.")
        sys.exit(1)

    placeholders = ",".join("?" for _ in exp_ids)
    print(f"  Copying {len(exp_ids)} experiments...")

    rows = src_conn.execute(
        f"SELECT * FROM experiments WHERE experiment_id IN ({placeholders})", exp_ids,
    ).fetchall()
    cols = [d[0] for d in src_conn.execute("SELECT * FROM experiments LIMIT 0").description]
    ph = ",".join("?" for _ in range(len(cols)))
    dst_conn.executemany(f"INSERT OR REPLACE INTO experiments VALUES ({ph})", rows)
    print(f"    ✓ {len(rows)} experiments")

    runs = src_conn.execute(
        f"SELECT * FROM runs WHERE experiment_id IN ({placeholders})", exp_ids,
    ).fetchall()
    cols_r = [d[0] for d in src_conn.execute("SELECT * FROM runs LIMIT 0").description]
    ph_r = ",".join("?" for _ in range(len(cols_r)))
    dst_conn.executemany(f"INSERT OR REPLACE INTO runs VALUES ({ph_r})", runs)
    run_ids = [r[cols_r.index('run_uuid')] for r in runs]
    print(f"    ✓ {len(runs)} runs")

    for table in ['params', 'tags', 'metrics', 'latest_metrics']:
        try:
            cols_t = [d[0] for d in src_conn.execute(f"SELECT * FROM {table} LIMIT 0").description]
        except sqlite3.OperationalError:
            continue
        ph_t = ",".join("?" for _ in range(len(cols_t)))
        if 'run_uuid' not in cols_t:
            continue
        total = 0
        chunk_size = 5000
        for i in range(0, len(run_ids), chunk_size):
            chunk = run_ids[i:i+chunk_size]
            chunk_ph = ",".join("?" for _ in chunk)
            rows_t = src_conn.execute(
                f"SELECT * FROM {table} WHERE run_uuid IN ({chunk_ph})", chunk,
            ).fetchall()
            if rows_t:
                dst_conn.executemany(f"INSERT OR REPLACE INTO {table} VALUES ({ph_t})", rows_t)
                total += len(rows_t)
        print(f"    ✓ {total:,} rows in {table}")

    try:
        et_cols = [d[0] for d in src_conn.execute("SELECT * FROM experiment_tags LIMIT 0").description]
        et_ph = ",".join("?" for _ in range(len(et_cols)))
        et_rows = src_conn.execute(
            f"SELECT * FROM experiment_tags WHERE experiment_id IN ({placeholders})", exp_ids,
        ).fetchall()
        if et_rows:
            dst_conn.executemany(f"INSERT OR REPLACE INTO experiment_tags VALUES ({et_ph})", et_rows)
            print(f"    ✓ {len(et_rows)} experiment_tags")
    except sqlite3.OperationalError:
        pass

    dst_conn.commit()
    print("\n  Compaction complete.")
finally:
    src_conn.close()
    dst_conn.close()
PYEOF

  echo -e "${BLUE}Vacuuming...${NC}"
  _sqlite3 "$compact_db" "VACUUM;"

  local src_size dst_size
  src_size=$(stat -c%s "$src_db" 2>/dev/null || stat -f%z "$src_db" 2>/dev/null)
  dst_size=$(stat -c%s "$compact_db" 2>/dev/null || stat -f%z "$compact_db" 2>/dev/null)

  echo ""
  echo -e "${GREEN}✓${NC} Compacted DB created:"
  echo -e "  ${BOLD}Source:${NC}  $(human_size "$src_size")  →  ${BOLD}Output:${NC}  $(human_size "$dst_size")"
  echo -e "  ${DIM}Reduction: $(echo "scale=0; 100 - ($dst_size * 100 / $src_size)" | bc)%${NC}"
  echo -e "  ${BOLD}Path:${NC}  $compact_db"
}


# ══════════════════════════════════════════════════════════════════════
#  COMMAND: snapshot
# ══════════════════════════════════════════════════════════════════════
cmd_snapshot() {
  local db="$1" remote="$2" ws="$3"

  local ws_dir
  ws_dir=$(ws_local_dir "$ws")
  mkdir -p "$ws_dir"

  if [[ "$remote" == "true" ]]; then
    local remote_ws
    remote_ws=$(ws_remote_dir "$ws")
    local timestamp
    timestamp=$(date +%d_%m)
    local ws_name
    ws_name=$(_ws_name "$ws")
    local snap_name="mlflow_snapshot_${ws_name}_${timestamp}.db"
    local snap_path="$ws_dir/$snap_name"

    echo -e "${BLUE}Creating snapshot of workspace \"${ws}\" from remote...${NC}"
    echo -e "${DIM}  Remote: ${REMOTE}:${remote_ws}/mlflow.db${NC}"

    local remote_snap="/tmp/mlflow_snap_$$"
    ssh "$REMOTE" bash -s "$remote_ws" "$remote_snap" <<'RSCRIPT'
set -e
WS_DIR="$1"; SNAP="$2"
echo "Creating hot backup..."
sqlite3 "$WS_DIR/mlflow.db" ".backup '$SNAP.db'"
echo "Compressing..."
gzip -1 "$SNAP.db"
ls -lh "$SNAP.db.gz"
RSCRIPT

    echo -e "${BLUE}Downloading...${NC}"
    scp "${REMOTE}:${remote_snap}.db.gz" "${snap_path}.gz"

    echo -e "${BLUE}Decompressing...${NC}"
    gunzip -f "${snap_path}.gz"

    ssh "$REMOTE" "rm -f ${remote_snap}.db.gz" 2>/dev/null || true
    rm -f "${snap_path}-wal" "${snap_path}-shm"

    local size
    size=$(stat -c%s "$snap_path" 2>/dev/null || stat -f%z "$snap_path" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Snapshot saved: $snap_path ($(human_size "$size"))"

  else
    if [[ ! -f "$db" ]]; then
      echo -e "${RED}✗${NC} Database not found: $db"
      exit 1
    fi

    local timestamp
    timestamp=$(date +%d_%m)
    local ws_name
    ws_name=$(_ws_name "$ws")
    local snap_name="mlflow_snapshot_${ws_name}_${timestamp}.db"
    local snap_path="$ws_dir/$snap_name"

    echo -e "${BLUE}Creating local snapshot of workspace \"${ws}\"...${NC}"
    _sqlite3_backup "$db" "$snap_path"
    _sqlite3 "$snap_path" "PRAGMA wal_checkpoint(TRUNCATE);" 2>/dev/null || true
    rm -f "${snap_path}-wal" "${snap_path}-shm"

    local size
    size=$(stat -c%s "$snap_path" 2>/dev/null || stat -f%z "$snap_path" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Snapshot saved: $snap_path ($(human_size "$size"))"
  fi
}


# ══════════════════════════════════════════════════════════════════════
#  COMMAND: list
# ══════════════════════════════════════════════════════════════════════
cmd_list() {
  echo -e "${BOLD}═══ Available Snapshots ═══${NC}"
  echo ""

  local ws_root="$PROJECT_DIR/workspaces"
  if [[ ! -d "$ws_root" ]]; then
    echo -e "${YELLOW}No workspaces/ directory found.${NC}"
    return
  fi

  local found=false
  while IFS= read -r f; do
    found=true
    local fname size mtime
    fname=$(basename "$f")
    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
    mtime=$(stat -c%y "$f" 2>/dev/null | cut -d'.' -f1 || stat -f%Sm "$f" 2>/dev/null)

    local n_exp n_runs
    n_exp=$(_sqlite3_scalar "$f" "SELECT COUNT(*) FROM experiments WHERE lifecycle_stage='active'" 2>/dev/null || echo "?")
    n_runs=$(_sqlite3_scalar "$f" "SELECT COUNT(*) FROM runs WHERE status='FINISHED'" 2>/dev/null || echo "?")

    echo -e "  ${BOLD}$fname${NC}"
    echo -e "    Size: $(human_size "$size")  |  Experiments: $n_exp  |  Finished runs: $n_runs  |  Modified: $mtime"
  done < <(find "$ws_root" -name "mlflow_snapshot*.db" -not -name "*-wal" -not -name "*-shm" | sort -r)

  if [[ "$found" == false ]]; then
    echo -e "${YELLOW}  No snapshots found in workspaces/${NC}"
  fi
}


# ══════════════════════════════════════════════════════════════════════
#  COMMAND: prune
# ══════════════════════════════════════════════════════════════════════
cmd_prune() {
  local keep_n="$1"
  echo -e "${BOLD}═══ Pruning Snapshots (keeping $keep_n most recent) ═══${NC}"

  local snaps
  snaps=$(find "$PROJECT_DIR/workspaces" -name "mlflow_snapshot*.db" -not -name "*-wal" -not -name "*-shm" | sort -r)
  local count
  count=$(echo "$snaps" | grep -c . || true)

  if [[ $count -le $keep_n ]]; then
    echo -e "${GREEN}✓${NC} Only $count snapshots found, nothing to prune."
    return
  fi

  local to_delete
  to_delete=$(echo "$snaps" | tail -n +"$((keep_n + 1))")

  echo -e "${YELLOW}Will delete:${NC}"
  while IFS= read -r f; do
    local fname size
    fname=$(basename "$f")
    size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
    echo -e "  ${RED}✗${NC} $fname  ($(human_size "$size"))"
  done <<< "$to_delete"

  echo ""
  read -r -p "  Proceed? [y/N] " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    return
  fi

  while IFS= read -r f; do
    rm -f "$f" "${f}-wal" "${f}-shm"
    echo -e "  ${DIM}Deleted: $(basename "$f")${NC}"
  done <<< "$to_delete"

  echo -e "${GREEN}✓${NC} Pruned $((count - keep_n)) snapshot(s)."
}


# ══════════════════════════════════════════════════════════════════════
#  MAIN — Argument Parsing
# ══════════════════════════════════════════════════════════════════════

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
fi

COMMAND="${1:-}"
shift || true

REMOTE_MODE=true
LOCAL_MODE=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -w|--workspace) WORKSPACE="$2";          shift 2 ;;
    -p|--profile)   shift 2 ;;  # already handled in pre-parse
    --db)           DB_PATH="$2";            shift 2 ;;
    --keep)         KEEP_EXPERIMENTS="$2";   shift 2 ;;
    --keep-n)       KEEP_N="$2";             shift 2 ;;
    --remote)       REMOTE_MODE=true;        shift ;;
    --local)        LOCAL_MODE=true; REMOTE_MODE=false; shift ;;
    -h|--help)      usage ;;
    *)              echo -e "${RED}Unknown option: $1${NC}"; usage ;;
  esac
done

# Normalize workspace to origin/name format
WORKSPACE=$(_normalize_workspace "$WORKSPACE")

case "$COMMAND" in
  new)
    if [[ "$LOCAL_MODE" == "true" ]]; then
      cmd_new "$WORKSPACE" "false"
    else
      cmd_new "$WORKSPACE" "true"
    fi
    ;;
  reset)
    if [[ "$LOCAL_MODE" == "true" ]]; then
      cmd_reset "$WORKSPACE" "false"
    else
      cmd_reset "$WORKSPACE" "true"
    fi
    ;;
  compact)
    db="${DB_PATH:-$(ws_db_path "$WORKSPACE")}"
    if [[ ! -f "$db" ]]; then
      echo -e "${RED}✗${NC} No DB found for workspace \"$WORKSPACE\". Use --db to specify one."
      exit 1
    fi
    cmd_compact "$db" "$KEEP_EXPERIMENTS" "$WORKSPACE"
    ;;
  snapshot)
    db="${DB_PATH:-$(ws_db_path "$WORKSPACE")}"
    if [[ "$LOCAL_MODE" == "true" ]]; then
      if [[ ! -f "$db" ]]; then
        echo -e "${RED}✗${NC} No local DB for workspace \"$WORKSPACE\". Use --db to specify."
        exit 1
      fi
      cmd_snapshot "$db" "false" "$WORKSPACE"
    else
      cmd_snapshot "" "true" "$WORKSPACE"
    fi
    ;;
  info)
    db="${DB_PATH:-$(ws_db_path "$WORKSPACE")}"
    if [[ ! -f "$db" ]]; then
      echo -e "${RED}✗${NC} No DB found for workspace \"$WORKSPACE\". Use --db to specify one."
      exit 1
    fi
    cmd_info "$db" "$WORKSPACE"
    ;;
  list)
    cmd_list
    ;;
  prune)
    cmd_prune "$KEEP_N"
    ;;
  workspaces|ws)
    cmd_workspaces
    ;;
  ""|help)
    usage
    ;;
  *)
    echo -e "${RED}Unknown command: $COMMAND${NC}"
    usage
    ;;
esac
