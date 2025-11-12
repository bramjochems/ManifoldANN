PYTHON ?= python3
VENV_DIR := benchmarking/venv
VENV_PYTHON := $(VENV_DIR)/bin/python

.PHONY: test format benchmark benchmark-setup clean

test:
	julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'

format:
	julia --project=. -e 'using Pkg; Pkg.instantiate(); using JuliaFormatter; format(["src","test","docs/examples"])'

# Setup benchmarking environment (venv + dependencies + Julia + ann-benchmarks)
benchmark-setup:
	@cd benchmarking && bash setup.sh

# Run benchmarks (ensures venv exists, uses it if available)
benchmark:
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "Using virtual environment at $(VENV_DIR)"; \
		cd benchmarking && $(VENV_PYTHON) benchmark.py $(ARGS); \
	else \
		echo "Virtual environment not found. Run 'make benchmark-setup' first."; \
		echo "Falling back to system Python..."; \
		cd benchmarking && $(PYTHON) benchmark.py $(ARGS); \
	fi

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
