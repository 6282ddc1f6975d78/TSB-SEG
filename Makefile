.PHONY: help install clean

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
	@echo "  make install    Create/update conda environment and run setup."
	@echo "  make clean      Remove the conda environment."
	@echo ""
	@echo "Configuration:"
	@echo "  - The Makefile will automatically find 'conda' in your PATH."
	@echo "  - Alternatively, create a '.env' file with 'CONDA=/path/to/conda' to specify the path."

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
