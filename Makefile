SHELL := /bin/sh
.DEFAULT_GOAL := help
.NOTPARALLEL:

UV ?= uv
PYTHON_VERSION ?= 3.13.7

GUPPY_ENV := guppy/.venv
PYTKET_ENV := pytket/.venv

GUPPY_PYTHON := $(GUPPY_ENV)/bin/python
PYTKET_PYTHON := $(PYTKET_ENV)/bin/python

GUPPY_KERNEL_NAME := quops-guppy
PYTKET_KERNEL_NAME := quops-pytket

.PHONY: help setup sync sync-guppy sync-pytket kernels kernel-guppy kernel-pytket verify

help:
	@echo "Available targets:"
	@echo "  make setup          Sync both environments and register both Jupyter kernels"
	@echo "  make sync           Sync both environments"
	@echo "  make sync-guppy     Sync guppy/.venv from the guppy dependency group"
	@echo "  make sync-pytket    Sync pytket/.venv from the pytket dependency group"
	@echo "  make kernels        Sync and register both Jupyter kernels"
	@echo "  make kernel-guppy   Sync and register the Guppy kernel"
	@echo "  make kernel-pytket  Sync and register the Pytket kernel"
	@echo "  make verify         Print the Python executable used by each environment"

setup: kernels
	@echo "Setup complete."
	@echo "Select 'QUOPS Guppy (Python $(PYTHON_VERSION))' for notebooks under guppy/."
	@echo "Select 'QUOPS pytket (Python $(PYTHON_VERSION))' for notebooks under pytket/."

sync: sync-guppy sync-pytket

sync-guppy:
	@command -v $(UV) >/dev/null 2>&1 || { echo "Error: uv is not installed or not on PATH." >&2; exit 1; }
	UV_PROJECT_ENVIRONMENT=$(GUPPY_ENV) $(UV) sync \
		--no-active \
		--python $(PYTHON_VERSION) \
		--group guppy

sync-pytket:
	@command -v $(UV) >/dev/null 2>&1 || { echo "Error: uv is not installed or not on PATH." >&2; exit 1; }
	UV_PROJECT_ENVIRONMENT=$(PYTKET_ENV) $(UV) sync \
		--no-active \
		--python $(PYTHON_VERSION) \
		--group pytket

kernels: kernel-guppy kernel-pytket

kernel-guppy: sync-guppy
	$(GUPPY_PYTHON) -m ipykernel install \
		--user \
		--name $(GUPPY_KERNEL_NAME) \
		--display-name "QUOPS Guppy (Python $(PYTHON_VERSION))"

kernel-pytket: sync-pytket
	$(PYTKET_PYTHON) -m ipykernel install \
		--user \
		--name $(PYTKET_KERNEL_NAME) \
		--display-name "QUOPS pytket (Python $(PYTHON_VERSION))"

verify:
	@test -x $(GUPPY_PYTHON) || { echo "Missing $(GUPPY_PYTHON); run 'make sync-guppy'." >&2; exit 1; }
	@test -x $(PYTKET_PYTHON) || { echo "Missing $(PYTKET_PYTHON); run 'make sync-pytket'." >&2; exit 1; }
	@$(GUPPY_PYTHON) -c 'import sys; print("Guppy:", sys.executable, sys.version.split()[0])'
	@$(PYTKET_PYTHON) -c 'import sys; print("pytket:", sys.executable, sys.version.split()[0])'