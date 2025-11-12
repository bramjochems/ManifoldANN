#!/bin/bash
# Comprehensive setup script for ManifoldANN benchmarking environment
#
# This script:
# 1. Creates a Python virtual environment
# 2. Installs the benchmarking package in editable mode
# 3. Fetches ann-benchmarks repository (if needed)
# 4. Sets up the Julia environment

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "ManifoldANN Benchmarking Environment Setup"
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.8 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "Found Python $PYTHON_VERSION"

# Check if Julia is available
if ! command -v julia &> /dev/null; then
    echo ""
    echo "Warning: Julia not found. Skipping Julia environment setup."
    echo "Install Julia from https://julialang.org/downloads/"
    JULIA_AVAILABLE=false
else
    JULIA_VERSION=$(julia --version | cut -d' ' -f3)
    echo "Found Julia $JULIA_VERSION"
    JULIA_AVAILABLE=true
fi

# ============================================================
# 1. Create Python virtual environment
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Python Virtual Environment"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "✓ Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# ============================================================
# 2. Install benchmarking package
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Install Benchmarking Package"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Installing benchmarking package with dependencies..."
pip install -e ".[all]"
echo "✓ Package installed in editable mode with all optional dependencies"

# ============================================================
# 3. Setup Julia environments
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Julia Environments"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ "$JULIA_AVAILABLE" = true ]; then
    echo "Setting up ManifoldANN environment..."
    cd "$PARENT_DIR"
    julia --project=. -e 'using Pkg; Pkg.instantiate()'
    echo "✓ ManifoldANN Julia environment ready"

    echo ""
    echo "Setting up Julia benchmarking environment..."
    cd "$SCRIPT_DIR/julia"

    # Add ManifoldANN as a local development dependency BEFORE instantiate
    echo "Adding ManifoldANN as local development dependency..."
    julia --project=. -e "using Pkg; Pkg.develop(path=\"$PARENT_DIR\")"

    # Now instantiate to install all dependencies
    julia --project=. -e 'using Pkg; Pkg.instantiate()'

    echo "✓ Julia benchmark environment ready (NearestNeighbors.jl, HNSW.jl, ManifoldANN)"
else
    echo "⊘ Julia not available - skipped"
    echo "  Install from: https://julialang.org/downloads/"
fi

# ============================================================
# Done!
# ============================================================
echo ""
echo "=================================================="
echo "Setup Complete! 🎉"
echo "=================================================="
echo ""
echo "Summary:"
echo "  ✓ Python venv at: $VENV_DIR"
echo "  ✓ Benchmarking package installed (editable)"
echo "  ✓ Optional ANN libraries installed (Annoy, HNSWlib, FAISS, SciPy)"
if [ "$JULIA_AVAILABLE" = true ]; then
    echo "  ✓ ManifoldANN Julia environment"
    echo "  ✓ Benchmark Julia environment (NearestNeighbors.jl, HNSW.jl)"
fi
echo ""
echo "Note: Datasets are downloaded automatically on first use from"
echo "      https://ann-benchmarks.com/ (fashion-mnist, sift, glove, etc.)"
echo ""
echo "To activate the environment in the future:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run benchmarks:"
echo "  make benchmark ARGS=\"fashion-mnist\""
echo "  make benchmark ARGS=\"--list-configs\""
echo ""
echo "Or directly:"
echo "  cd benchmarking && source venv/bin/activate"
echo "  python benchmark.py nytimes"
echo ""
