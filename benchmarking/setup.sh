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
VENV_DIR="$SCRIPT_DIR/.venv"
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
    # Prefer uv if available, fall back to python3 -m venv
    if command -v uv &> /dev/null; then
        echo "Using uv to create virtual environment..."
        uv venv "$VENV_DIR"
    else
        echo "Using python3 -m venv..."
        python3 -m venv "$VENV_DIR"
    fi
    echo "✓ Virtual environment created at $VENV_DIR"
else
    echo "✓ Virtual environment already exists at $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip (unless using uv)
if ! command -v uv &> /dev/null; then
    echo "Upgrading pip..."
    pip install --upgrade pip --quiet
fi

# ============================================================
# 2. Install benchmarking package
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Install Benchmarking Package"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Installing benchmarking package with dependencies..."
# Use uv if available for faster installation
if command -v uv &> /dev/null; then
    echo "Using uv for package installation..."
    uv pip install -e ".[all]"
else
    pip install -e ".[all]"
fi
echo "✓ Package installed in editable mode with all optional dependencies (ANN + ORC)"

# ============================================================
# 3. Clone orcml (optional ORC benchmark dependency)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Clone orcml (ORC reference implementation)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ORCML_DIR="$SCRIPT_DIR/external/orcml"

if [ ! -d "$ORCML_DIR" ]; then
    echo "Cloning orcml from GitHub..."
    mkdir -p "$SCRIPT_DIR/external"
    git clone https://github.com/TristanSaidi/orcml.git "$ORCML_DIR"
    echo "✓ orcml cloned to $ORCML_DIR"
else
    echo "✓ orcml already exists at $ORCML_DIR"
    # Optionally update it
    echo "Updating orcml..."
    cd "$ORCML_DIR" && git pull || echo "  (update failed - continuing with existing version)"
    cd "$SCRIPT_DIR"
fi

# Install minimal orcml dependencies (skip incompatible ones)
echo "Installing minimal orcml dependencies..."
# Only install what we actually need for ORC computation:
# - scikit-learn (for k-NN graph building)
# - tqdm (for progress bars)
# Other deps are already installed or not needed for basic ORC
if command -v uv &> /dev/null; then
    uv pip install scikit-learn tqdm || echo "  (Some orcml dependencies failed - orcml may not be available)"
else
    pip install scikit-learn tqdm || echo "  (Some orcml dependencies failed - orcml may not be available)"
fi
echo "✓ Minimal dependencies installed (orcml may require Python <3.11 for full functionality)"

# ============================================================
# 4. Setup Julia environments
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Julia Environments"
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
echo "  ⚠ orcml cloned (may not be functional on Python 3.11+ due to dependency constraints)"
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
echo "  make benchmark-orc"
echo ""
echo "Or directly:"
echo "  cd benchmarking && source .venv/bin/activate"
echo "  python benchmark.py nytimes"
echo ""
