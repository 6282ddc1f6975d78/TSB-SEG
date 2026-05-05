#!/bin/bash

# submit_mlflow.sh
# Lancer un serveur MLflow sur un nœud calcul via Slurm.
# Gère la soumission, l'attente du démarrage effectif et le tunnel SSH.

set -euo pipefail

# --- Configuration ---
# Resolve symlinks so that ~/.local/bin/submit_mlflow → scripts/mlflow/submit_mlflow.sh works
SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Pre-parse: extract --profile before loading .env ---
_prev=""
for _arg in "$@"; do
  if [[ "$_prev" == "-p" || "$_prev" == "--profile" ]]; then
    export CLUSTER_PROFILE="$_arg"; break
  fi
  _prev="$_arg"
done
unset _prev _arg

# --- Load .env ---
if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$PROJECT_DIR/.env"
  set +a
fi

REMOTE_USER="${CLUSTER_USER:-$USER}"
REMOTE_HOST="${CLUSTER_HOST:-cleps.inria.fr}"
BASE_DIR="${CLUSTER_WORKDIR:-/scratch/$USER/tsseg-exp}"
LOCAL_SBATCH="$SCRIPT_DIR/run_mlflow.sbatch"
BASE_MLFLOW_PORT=15050
SQUEUE_FORMAT="%i|%P|%j|%u|%T|%M|%D|%R"

# --- Couleurs & Styles ---
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m' # No Color


# --- Parsing des arguments ---
VERBOSE=false
HELP=false
# Par défaut, WORKSPACE="main" (et non "local/main")
WORKSPACE="${TSSEG_WORKSPACE:-main}"

# ── Workspace naming helpers ─────────────────────────────────────────
_normalize_workspace() {
  local ws="$1"; [[ "$ws" != */* ]] && ws="local/$ws"; echo "$ws"
}
_ws_name() { local ws; ws=$(_normalize_workspace "$1"); echo "${ws#*/}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v|--verbose)
      VERBOSE=true
      shift
      ;;
    -w|--workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    -p|--profile)
      shift 2  # already handled in pre-parse
      ;;
    -h|--help)
      HELP=true
      shift
      ;;
    *)
      echo "Option inconnue: $1"
      exit 1
      ;;
  esac
done

# Normalize workspace
WORKSPACE=$(_normalize_workspace "$WORKSPACE")
WS_NAME=$(_ws_name "$WORKSPACE")

# ── Derive per-workspace MLflow port ─────────────────────
# Hash includes username to avoid port conflicts when two users share a cluster
_port_key="${REMOTE_USER}:${WS_NAME}"
_hash=$(echo -n "$_port_key" | cksum | awk '{print $1}')
MLFLOW_PORT=$(( BASE_MLFLOW_PORT + (_hash % 200) + 1 ))

# Derived paths (remote always uses bare name)
MLFLOW_DIR="${BASE_DIR}/workspaces/${WS_NAME}"
REMOTE_SBATCH="${BASE_DIR}/run_mlflow.sbatch"

# --- Aide ---
if [[ "$HELP" == "true" ]]; then
  cat <<EOF
${BOLD}Usage:${NC} $(basename "$0") [options]

${BOLD}Description:${NC}
  Lance un serveur MLflow sur un nœud de calcul (via Slurm) et établit un tunnel SSH.
  Idéal pour éviter de surcharger le nœud de login avec des processus longue durée.

${BOLD}Options:${NC}
  -w, --workspace NAME  Workspace à utiliser (default: \$TSSEG_WORKSPACE ou "main").
  -v, --verbose   Mode verbeux (debug logs + tunnel interactif).
  -h, --help      Affiche cette aide.

EOF
  exit 0
fi

# --- Fonctions Utilitaires ---

log_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
  echo -e "${GREEN}[OK]${NC}   $1"
}

log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
  echo -e "${RED}[ERR]${NC}  $1" >&2
}

log_debug() {
  if [[ "$VERBOSE" == "true" ]]; then
    echo -e "${BOLD}[DBG]${NC}  $1"
  fi
}

fail() {
  log_error "$1"
  exit "${2:-1}"
}

# Conversion temps Slurm -> Secondes
parse_slurm_time_to_seconds() {
  local raw="$1"
  local days=0
  local rest="$raw"

  if [[ "$rest" == *-* ]]; then
    days=${rest%%-*}
    rest=${rest#*-}
  fi

  local parts
  IFS=':' read -r -a parts <<< "$rest"
  local count=${#parts[@]}
  local hh=0 mm=0 ss=0

  if (( count == 3 )); then
    hh=${parts[0]}; mm=${parts[1]}; ss=${parts[2]}
  elif (( count == 2 )); then
    if [[ "$raw" == *-* ]]; then hh=${parts[0]}; mm=${parts[1]}; else mm=${parts[0]}; ss=${parts[1]}; fi
  elif (( count == 1 )); then
    mm=${parts[0]}
  fi

  echo $(( (10#$days * 86400) + (10#${hh:-0} * 3600) + (10#${mm:-0} * 60) + (10#${ss:-0}) ))
}

format_runtime() {
  local seconds
  seconds=$(parse_slurm_time_to_seconds "$1")
  local d=$((seconds/86400)) h=$(( (seconds%86400)/3600 )) m=$(( (seconds%3600)/60 )) s=$((seconds%60))
  
  if (( d > 0 )); then echo "${d}j ${h}h ${m}m"; else echo "${h}h ${m}m ${s}s"; fi
}

# Vérifie si le port MLflow est ouvert sur le nœud distant via SSH
wait_for_remote_port() {
  local node="$1"
  local max_retries=30 # 30 * 2s = 60 secondes max
  local attempt=1

  log_info "Vérification du service MLflow sur ${node}:${MLFLOW_PORT}..."

  while [[ $attempt -le $max_retries ]]; do
    # On utilise /dev/tcp pour vérifier le port sans dépendre de nc/netcat
    if ssh -J "${REMOTE_USER}@${REMOTE_HOST}" "${REMOTE_USER}@${node}" \
       "timeout 1 bash -c 'true < /dev/tcp/127.0.0.1/${MLFLOW_PORT}' 2>/dev/null"; then
      log_success "Service MLflow actif et répondant sur le port ${MLFLOW_PORT}."
      return 0
    fi
    
    if [[ $((attempt % 5)) -eq 0 ]]; then
        log_debug "Service non prêt... (tentative $attempt/$max_retries)"
    fi
    sleep 2
    ((attempt++))
  done

  log_warn "Le service MLflow semble lent à démarrer (port ${MLFLOW_PORT} fermé après 60s)."
  log_warn "Le tunnel va être établi, mais vous devrez peut-être attendre quelques secondes avant d'accéder à l'UI."
  return 0 
}

open_tunnel() {
  local node="$1"
  if [[ -z "$node" || "$node" == "UNASSIGNED" ]]; then fail "Nœud invalide pour le tunnel."; fi

  # 1. Attente active du port
  wait_for_remote_port "$node"

  # 2. Nettoyage port local
  if lsof -i :${MLFLOW_PORT} -t >/dev/null 2>&1; then
    local pids
    pids=$(lsof -i :${MLFLOW_PORT} -t)
    log_info "Port local ${MLFLOW_PORT} occupé (PIDs: $(echo "$pids" | tr '\n' ' ')). Libération..."
    echo "$pids" | xargs kill -9 2>/dev/null || true
    
    # Petite boucle d'attente
    while lsof -i :${MLFLOW_PORT} -t >/dev/null 2>&1; do
      sleep 0.2
    done
  fi

  # 3. Ouverture navigateur
  (sleep 2 && xdg-open "http://localhost:${MLFLOW_PORT}" >/dev/null 2>&1 || true) &

  # 4. Lancement SSH
  if [[ "$VERBOSE" == "true" ]]; then
    log_info "Ouverture du tunnel interactif (Ctrl+C pour quitter)..."
    ssh -J "${REMOTE_USER}@${REMOTE_HOST}" -L ${MLFLOW_PORT}:127.0.0.1:${MLFLOW_PORT} "${REMOTE_USER}@${node}"
  else
    log_info "Ouverture du tunnel en arrière-plan..."
    ssh -f -N -J "${REMOTE_USER}@${REMOTE_HOST}" -L ${MLFLOW_PORT}:127.0.0.1:${MLFLOW_PORT} "${REMOTE_USER}@${node}"
    log_success "Tunnel établi ! MLflow accessible sur : ${BOLD}http://localhost:${MLFLOW_PORT}${NC}"
  fi
}

find_existing_job() {
  # Strategie 1 : Fichier jobid
  local jobid jobinfo=""
  jobid=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "head -n1 '$MLFLOW_DIR/mlflow_jobid.txt' 2>/dev/null" | tr -d '\r' || true)
  
  if [[ -n "$jobid" ]]; then
    jobinfo=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "squeue -j '$jobid' -o '$SQUEUE_FORMAT' --noheader" 2>/dev/null || true)
  fi

  # Strategie 2 : Recherche par nom
  if [[ -z "$jobinfo" ]]; then
    jobinfo=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "squeue -u '$REMOTE_USER' -o '$SQUEUE_FORMAT' --noheader" 2>/dev/null | awk -F'|' '$3=="mlflow" && ($5=="R" || $5=="CF" || $5=="PD") {print; exit}' || true)
  fi
  echo "$jobinfo"
}

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

echo -e "${BOLD}=== Gestionnaire Serveur MLflow (Cluster) ===${NC}"
log_debug "Remote: ${REMOTE_USER}@${REMOTE_HOST}"
log_debug "Workdir: ${MLFLOW_DIR}"
log_info "Workspace: ${BOLD}${WORKSPACE}${NC}  (remote: ${WS_NAME}, port ${MLFLOW_PORT})"

# --- 1. Vérifications Préliminaires ---
log_info "Vérification de la connexion SSH..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "${REMOTE_USER}@${REMOTE_HOST}" 'echo OK' 2>/dev/null | grep -q 'OK'; then
  fail "Connexion SSH impossible. Vérifiez vos clés ou le VPN."
fi

# --- 2. Vérification Job Existant ---
EXISTING_JOB=$(find_existing_job)

if [[ -n "$EXISTING_JOB" ]]; then
  IFS='|' read -r J_ID J_PART J_NAME J_USER J_STATE J_TIME J_NODES J_NODELIST <<< "$EXISTING_JOB"

  if [[ "$J_STATE" == "RUNNING" ]]; then
    log_success "Job MLflow retrouvé : ${BOLD}${J_ID}${NC} sur ${BOLD}${J_NODELIST}${NC} (depuis $(format_runtime "$J_TIME"))"
    open_tunnel "$J_NODELIST"
    exit 0
  fi

  # Job exists but is not yet RUNNING (PENDING, CONFIGURING, …)
  # %R contains the scheduling reason (e.g. "(Priority)"), NOT a hostname.
  log_success "Job MLflow retrouvé : ${BOLD}${J_ID}${NC}"
  log_info "État : ${BOLD}${J_STATE}${NC} — raison : ${J_NODELIST} (depuis $(format_runtime "$J_TIME"))"
  log_info "Attente du statut RUNNING (Ctrl+C pour annuler)..."

  local_node=""
  for i in $(seq 1 300); do
    line=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" \
      "squeue -j '${J_ID}' -o '${SQUEUE_FORMAT}' --noheader" 2>/dev/null || true)

    if [[ -z "$line" ]]; then
      fail "Le job ${J_ID} a disparu (annulé ou crash ?). Vérifiez avec 'sacct -j ${J_ID}'."
    fi

    IFS='|' read -r _ _ _ _ st _ _ nl <<< "$line"

    if [[ "$st" == "RUNNING" ]]; then
      log_success "Le job est en cours d'exécution sur ${BOLD}${nl}${NC}."
      local_node="$nl"
      # Update node file so run_experiments.sh / configure_mlflow() can auto-discover
      ssh "${REMOTE_USER}@${REMOTE_HOST}" "echo '${nl}' > '${MLFLOW_DIR}/mlflow_node.txt'" 2>/dev/null || true
      break
    fi

    if (( i % 10 == 0 )); then
      log_info "Toujours ${st}... ($((i*3))s écoulées)"
    fi
    sleep 3
  done

  if [[ -z "$local_node" ]]; then
    fail "Timeout : le job ${J_ID} n'est toujours pas RUNNING après ~15 min."
  fi

  open_tunnel "$local_node"
  exit 0
fi

log_info "Aucun serveur actif trouvé. Lancement d'une nouvelle instance..."

# --- 3. Déploiement Script ---
if [[ -f "$LOCAL_SBATCH" ]]; then
  log_debug "Synchronisation du script sbatch..."
  _local_md5=$(md5sum "$LOCAL_SBATCH" | awk '{print $1}')
  _remote_md5=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "md5sum '$REMOTE_SBATCH' 2>/dev/null | awk '{print \$1}'" || echo "")
  if [[ "$_local_md5" != "$_remote_md5" ]]; then
    log_info "Envoi du script sbatch (mis à jour)..."
    scp "$LOCAL_SBATCH" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_SBATCH}" >/dev/null
    log_success "Script sbatch synchronisé."
  else
    log_debug "Script sbatch déjà à jour."
  fi
else
  log_warn "Script local introuvable. On suppose qu'il existe sur le cluster."
fi

# --- 4. Soumission Job ---
log_info "Soumission du job Slurm..."

REMOTE_CMD=$(cat <<EOF
set -euo pipefail
cd "${BASE_DIR}"
mkdir -p "${MLFLOW_DIR}/mlartifacts" "${MLFLOW_DIR}/outputs" "${MLFLOW_DIR}/multirun"
JOBID=\$(sbatch --parsable --export=ALL,TSSEG_WORKSPACE=${WS_NAME},MLFLOW_PORT=${MLFLOW_PORT},CLUSTER_WORKDIR=${BASE_DIR},CONDA_ENV=${CONDA_ENV:-tsseg-env} "run_mlflow.sbatch")
echo "JOBID:\$JOBID"

# Attente allocation noeud (max 30s pour squeue)
NODELIST=""
for i in {1..30}; do
  NODELIST=\$(scontrol show job "\$JOBID" | awk -F= '/NodeList=/{print \$2}' | awk '{print \$1}' | grep -v 'Unassigned' | grep -v 'null' | head -n1 || true)
  if [[ -n "\$NODELIST" && "\$NODELIST" != "(null)" ]]; then break; fi
  sleep 1
done
echo "NODE:\${NODELIST:-UNASSIGNED}"

echo "\$JOBID" > "${MLFLOW_DIR}/mlflow_jobid.txt"
echo "\$NODELIST" > "${MLFLOW_DIR}/mlflow_node.txt"
EOF
)

OUTPUT=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" bash -s <<< "$REMOTE_CMD")
JOBID=$(echo "$OUTPUT" | grep "JOBID:" | cut -d: -f2 | tr -d '\r')
NODE=$(echo "$OUTPUT" | grep "NODE:" | cut -d: -f2 | tr -d '\r')

if [[ -z "$JOBID" ]]; then fail "Échec de la récupération du JOBID."; fi
log_success "Job soumis : ${BOLD}$JOBID${NC}"

if [[ -z "$NODE" || "$NODE" == "UNASSIGNED" ]]; then
  log_info "Job en file d'attente. Recherche du nœud alloué..."
  for i in {1..120}; do
    NODE=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "squeue -j $JOBID -h -o %N" 2>/dev/null | tr -d '\r' || true)
    if [[ -n "$NODE" && "$NODE" != "Unassigned" ]]; then break; fi
    sleep 2
    echo -n "."
  done
  echo ""
fi

if [[ -z "$NODE" || "$NODE" == "Unassigned" ]]; then
  fail "Impossible d'obtenir le nœud alloué (File d'attente trop longue ?). Vérifiez avec 'squeue'."
fi

log_success "Nœud alloué : ${BOLD}$NODE${NC}"

# Update node file for auto-discovery by run_experiments.sh / configure_mlflow()
ssh "${REMOTE_USER}@${REMOTE_HOST}" "echo '${NODE}' > '${MLFLOW_DIR}/mlflow_node.txt'" 2>/dev/null || true

# --- 5. Attente Démarrage ---
log_info "Attente du statut RUNNING..."
for i in {1..300}; do
  STATE=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "squeue -j $JOBID -h -o %T" 2>/dev/null | tr -d '\r' || echo "UNKNOWN")
  
  if [[ "$STATE" == "RUNNING" ]]; then
    log_success "Le job est en cours d'exécution."
    break
  elif [[ -z "$STATE" ]]; then
    fail "Le job semble avoir disparu (crash ?). Vérifiez les logs sur le cluster."
  fi
  sleep 3
done

# --- 6. Connexion ---
open_tunnel "$NODE"