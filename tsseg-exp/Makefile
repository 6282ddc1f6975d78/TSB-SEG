.PHONY: help install clean mlflow-server mlflow-stop mlflow-ui mlflow-status run-benchmark sync-fm-configs

# Load .env file if it exists and export the variables
ifneq (,$(wildcard .env))
	include .env
	export
endif

# If CONDA is not set in .env, try to find it in the PATH
ifeq ($(CONDA),)
	CONDA := $(shell which conda)
endif

# Variables
CONDA_ENV_NAME = tsseg-env

help:
	@echo "Makefile for tsseg-exp"
	@echo ""
	@echo "Usage:"
	@echo "  make install          Create/update conda environment and run setup."
	@echo "  make clean            Remove the conda environment."
	@echo ""
	@echo "  make run-benchmark    Submit the full benchmark to SLURM."
	@echo "                        Override: make run-benchmark ALGO=clasp DS=mocap"
	@echo ""
	@echo "  make sync-fm-configs    Symlink tsseg-fm algorithm configs (fm-*.yaml)."
	@echo ""
	@echo "  make mlflow-server    Start a local MLflow server (SQLite backend)."
	@echo "  make mlflow-stop      Stop the local MLflow server."
	@echo "  make mlflow-ui        Open the MLflow UI in the browser."
	@echo "  make mlflow-status    Show current MLflow tracking configuration."
	@echo ""
	@echo "Configuration:"
	@echo "  - Copy .env.example to .env and edit to match your setup."
	@echo "  - See scripts/README.md for cluster (SLURM) MLflow deployment."

install:
	@echo "--> Checking for conda..."
	@if [ -z "$(CONDA)" ] || ! [ -x "$(CONDA)" ]; then \
		echo "Error: conda executable not found or not executable at '$(CONDA)'"; \
		echo "Please ensure conda is in your PATH, or create a .env file with the correct CONDA=/path/to/conda"; \
		exit 1; \
	fi
	@echo "--> Using conda at: $(CONDA)"
	@echo "--> Updating conda environment $(CONDA_ENV_NAME) with tsseg-exp dependencies..."
	@"$(CONDA)" env update -f environment.yml --prune
	@echo "--> Installing local dependency: tsseg"
	@"$(CONDA)" run -n $(CONDA_ENV_NAME) pip install -e ../tsseg
	@echo "--> Installing tsseg-exp"
	@"$(CONDA)" run -n $(CONDA_ENV_NAME) pip install -e .[dev]
	@echo "--> Installation complete."
	@echo "--> To activate the environment, run: conda activate $(CONDA_ENV_NAME)"

clean:
	@echo "--> Checking for conda..."
	@if [ -z "$(CONDA)" ] || ! [ -x "$(CONDA)" ]; then \
		echo "Error: conda executable not found or not executable at '$(CONDA)'"; \
		echo "Please ensure conda is in your PATH, or create a .env file with the correct CONDA=/path/to/conda"; \
		exit 1; \
	fi
	@echo "--> Removing conda environment $(CONDA_ENV_NAME)..."
	@"$(CONDA)" env remove -n $(CONDA_ENV_NAME) --all -y
	@echo "--> Done."

# ── MLflow ────────────────────────────────────────────────────────────

# Workspace: use origin/name format, e.g. local/dev, cluster/main.
# A bare name like "main" is treated as "local/main".
TSSEG_WORKSPACE ?= local/main
WS              ?= $(TSSEG_WORKSPACE)
MLFLOW_PORT     ?= 15050
WS_DIR           = $(CURDIR)/workspaces/$(WS)
MLFLOW_DB        = $(WS_DIR)/mlflow.db
MLFLOW_ARTIFACTS = $(WS_DIR)/mlartifacts

mlflow-server: ## Start a local MLflow server (SQLite + WAL)
	@mkdir -p "$(WS_DIR)/mlartifacts" "$(WS_DIR)/outputs" "$(WS_DIR)/multirun"
	@echo "--> Starting MLflow server on http://localhost:$(MLFLOW_PORT)"
	@echo "    Workspace: $(WS)"
	@echo "    Backend : sqlite:///$(MLFLOW_DB)"
	@echo "    Artifacts: $(MLFLOW_ARTIFACTS)"
	@echo "    Stop with: make mlflow-stop"
	@MLFLOW_TRACKING_URI="sqlite:///$(MLFLOW_DB)" \
	mlflow server \
		--backend-store-uri "sqlite:///$(MLFLOW_DB)" \
		--default-artifact-root "$(MLFLOW_ARTIFACTS)" \
		--host 127.0.0.1 \
		--port $(MLFLOW_PORT)

mlflow-stop: ## Stop the local MLflow server
	@echo "--> Stopping MLflow server on port $(MLFLOW_PORT)..."
	@-lsof -ti :$(MLFLOW_PORT) | xargs kill -TERM 2>/dev/null || true
	@echo "    Done."

mlflow-ui: ## Open MLflow UI in the default browser
	@xdg-open "http://localhost:$(MLFLOW_PORT)" 2>/dev/null || \
	 open "http://localhost:$(MLFLOW_PORT)" 2>/dev/null || \
	 echo "Open http://localhost:$(MLFLOW_PORT) in your browser."

mlflow-status: ## Show current MLflow tracking configuration
	@echo "=== MLflow Configuration ==="
	@echo "  TSSEG_WORKSPACE     = $(WS)"
	@echo "  Workspace dir       = $(WS_DIR)"
	@if [ -n "$$MLFLOW_TRACKING_URI" ]; then \
		echo "  MLFLOW_TRACKING_URI = $$MLFLOW_TRACKING_URI"; \
	else \
		echo "  MLFLOW_TRACKING_URI = (not set)"; \
	fi
	@echo "============================"

# ── Benchmark ─────────────────────────────────────────────────────────

# Override on the command-line:
#   make run-benchmark ALGO=clasp,ticc DS=mocap,utsa EXP=unsupervised
ALGO ?=
DS   ?=
EXP  ?=

run-benchmark: ## Submit the full benchmark to SLURM
	@./scripts/run_experiments.sh \
		$(if $(ALGO),-a $(ALGO)) \
		$(if $(DS),-d $(DS)) \
		$(if $(EXP),-e $(EXP))

# ── tsseg-fm config sync ─────────────────────────────────────────

TSFM_CONFIG_DIR ?= ../tsseg-fm/configs/algorithm

sync-fm-configs: ## Symlink fm-*.yaml configs from tsseg-fm into tsseg-exp
	@echo "--> Syncing tsseg-fm algorithm configs..."
	@if [ ! -d "$(TSFM_CONFIG_DIR)" ]; then \
		echo "Error: tsseg-fm config directory not found at $(TSFM_CONFIG_DIR)"; \
		exit 1; \
	fi
	@for f in $(TSFM_CONFIG_DIR)/fm-*.yaml; do \
		ln -sf "$$(realpath $$f)" configs/algorithm/$$(basename $$f); \
	done
	@echo "    Linked $$(ls -1 configs/algorithm/fm-*.yaml 2>/dev/null | wc -l) fm-*.yaml configs."
