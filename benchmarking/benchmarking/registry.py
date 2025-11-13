"""Algorithm registry and factory for creating wrappers from config."""

from typing import Dict, Type, Optional
import warnings

from .wrappers.base import BaseANNWrapper
from .wrappers.manifoldann import (
    ManifoldANN_BruteForce,
    ManifoldANN_LSH,
    ManifoldANN_KDTree,
    ManifoldANN_HNSW,
    ManifoldANN_IVFHNSW,
    ManifoldANN_NNDescent,
)
from .wrappers.annoy import Annoy
from .wrappers.hnswlib import HNSWlib
from .wrappers.faiss import FAISS_IVF
from .wrappers.scipy import SciPy_KDTree
from .wrappers.pynndescent import PyNNDescent
from .wrappers.julia_external import NearestNeighbors_KDTree, HNSW_jl, NearestNeighborDescent_jl


# Registry mapping algorithm names to their wrapper classes
ALGORITHM_REGISTRY: Dict[str, Type[BaseANNWrapper]] = {
    # ManifoldANN algorithms (with MANN- prefix for cleaner output)
    "MANN-BruteForce": ManifoldANN_BruteForce,
    "MANN-LSH": ManifoldANN_LSH,
    "MANN-KDTree": ManifoldANN_KDTree,
    "MANN-HNSW": ManifoldANN_HNSW,
    "MANN-IVF-HNSW": ManifoldANN_IVFHNSW,
    "MANN-NNDescent": ManifoldANN_NNDescent,
    # External libraries
    "Annoy": Annoy,
    "HNSWlib": HNSWlib,
    "FAISS-IVF": FAISS_IVF,
    "SciPy-KDTree": SciPy_KDTree,
    "PyNNDescent": PyNNDescent,
    "NearestNeighbors-KDTree": NearestNeighbors_KDTree,
    "HNSW-jl": HNSW_jl,
    "NearestNeighborDescent-jl": NearestNeighborDescent_jl,
}


def create_algorithm(
    name: str, metric: str, params: Dict = None
) -> Optional[BaseANNWrapper]:
    """Create an algorithm instance from its name and parameters.

    Args:
        name: Algorithm name (must be in ALGORITHM_REGISTRY or a variant with suffix)
        metric: Distance metric ('angular' or 'euclidean')
        params: Algorithm-specific parameters (default: {})

    Returns:
        Algorithm instance, or None if library is not available

    Raises:
        ValueError: If algorithm name is unknown
    """
    # Try exact match first
    if name in ALGORITHM_REGISTRY:
        wrapper_class = ALGORITHM_REGISTRY[name]
    else:
        # Try prefix match for variants (e.g., "ManifoldANN-HNSW-heuristic" -> "ManifoldANN-HNSW")
        wrapper_class = None
        for base_name, cls in ALGORITHM_REGISTRY.items():
            if name.startswith(base_name + "-"):
                wrapper_class = cls
                break

        if wrapper_class is None:
            raise ValueError(
                f"Unknown algorithm: {name}. "
                f"Available algorithms: {list(ALGORITHM_REGISTRY.keys())}"
            )

    # Store original name for display
    wrapper_class = wrapper_class

    # Check if the library is available
    if not wrapper_class.is_available():
        warnings.warn(
            f"Skipping {name}: Required library not installed. "
            f"Install with: pip install manifoldann-benchmarks[{name.lower().split('-')[0]}]",
            UserWarning,
        )
        return None

    # Create instance with metric and parameters
    if params is None:
        params = {}

    try:
        instance = wrapper_class(metric=metric, **params)
        return instance
    except Exception as e:
        warnings.warn(f"Failed to create {name}: {e}", UserWarning)
        return None


def list_available_algorithms() -> Dict[str, bool]:
    """List all registered algorithms and their availability status.

    Returns:
        Dictionary mapping algorithm names to availability (True/False)
    """
    return {name: cls.is_available() for name, cls in ALGORITHM_REGISTRY.items()}
