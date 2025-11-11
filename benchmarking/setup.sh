#!/bin/bash
# Quick setup script for benchmarking environment

set -e

echo "=========================================="
echo "ManifoldANN Benchmarking Setup"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found"; exit 1; }

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Setup Julia environment
echo ""
echo "Setting up Julia environment..."
cd ..
julia --project=. -e 'using Pkg; Pkg.instantiate()'
cd benchmarking

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run benchmark:            python test_comparison.py"
echo "  2. Try different dataset:    python test_comparison.py --dataset glove-25"
echo ""
