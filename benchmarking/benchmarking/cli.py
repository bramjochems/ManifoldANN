"""Command-line interface for ManifoldANN benchmarks."""

import sys
from pathlib import Path

# Make benchmark.py's main() available
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark import main

if __name__ == "__main__":
    main()
