"""
Graph Curvature Module

Provides Ollivier-Ricci curvature computation and curvature-based graph filtering
for improving geodesic distance estimates on kNN graphs.

# Core Functionality

- **Curvature Computation**: Calculate Ollivier-Ricci curvature on graph edges
- **Graph Filtering**: Remove low-curvature edges that cut across the manifold
- **Multiple Solvers**: Fast matching, Hungarian algorithm, LP solver, greedy heuristic

# Curvature Solvers

1. **FastMatchingSolver**: O(k³) via Hungarian algorithm for uniform measures
2. **GenericOTSolver**: Exact OT via LP or greedy heuristic for non-uniform measures
3. **BruteForceSolver**: O(k!) exhaustive search for verification (small k only)

# Example
```julia
# Build and filter graph
index = build_index(NNDescentIndex, data; k=20)
graph = build_knn_graph(index, data; k=20)

filtered = filter_graph(
    graph, data,
    curvature_threshold=0.0,
    min_neighbors=5
)

# Analyze curvatures
curvatures = compute_all_curvatures(graph, data)
stats = curvature_statistics(curvatures)
```
"""

include("types.jl")
include("solvers.jl")
include("filtering.jl")
