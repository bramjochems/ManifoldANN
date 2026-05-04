"""
Graph Curvature Module

Provides Ollivier-Ricci curvature computation and curvature-based graph filtering
for improving geodesic distance estimates on kNN graphs.

# Core Functionality

- **Curvature Computation**: Calculate Ollivier-Ricci curvature on graph edges
- **Graph Filtering**: Remove low-curvature edges that cut across the manifold
- **Multiple Solvers**: Fast matching, Hungarian algorithm, LP solver, greedy heuristic

# Optimal Transport Solvers

Available OT solvers for curvature computation:
1. **HungarianSolver**: O(k³) exact OT for uniform distributions (Hungarian.jl)
2. **SinkhornSolver**: O(k² × iter) approximate OT for any distribution (OptimalTransport.jl)
3. **ClpSolver**: Exact OT via Clp simplex (general LP, low per-call alloc)
4. **LPReferenceSolver**: O(k³) reference implementation (HiGHS)
5. **GreedySolver**: O(k³) fast approximate OT (could be O(k² log k) with a heap; not pursued — k small in practice)
6. **GenericOTSolver**: Convenience wrapper (maps to specific solvers)

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
include("effective_epsilon_policy.jl")
include("filtering.jl")
