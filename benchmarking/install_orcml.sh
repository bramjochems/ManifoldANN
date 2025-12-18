#!/bin/bash
# Install orcml for ORC benchmarking comparison
set -e

echo "Installing orcml (ICLR 2025 reference implementation)..."
echo ""

# Clone repository
if [ -d "/tmp/orcml" ]; then
    echo "✓ orcml already exists at /tmp/orcml"
else
    echo "Cloning orcml to /tmp/orcml..."
    git clone https://github.com/TristanSaidi/orcml /tmp/orcml
    echo "✓ Cloned orcml"
fi

# Install dependencies in venv
if [ -d "venv/bin" ]; then
    echo ""
    echo "Installing orcml dependencies in venv..."
    source venv/bin/activate
    pip install -r /tmp/orcml/requirements.txt
    echo "✓ Dependencies installed"
else
    echo ""
    echo "Warning: venv not found. Run 'make benchmark-setup' first."
    echo "You can manually install with:"
    echo "  source benchmarking/venv/bin/activate"
    echo "  pip install -r /tmp/orcml/requirements.txt"
fi

echo ""
echo "✓ orcml installed successfully!"
echo ""
echo "Test with: make benchmark-orc"
