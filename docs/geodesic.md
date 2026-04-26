# Geodesic Distance Estimation

Geodesic distance measures the shortest path along a manifold surface, as opposed to Euclidean distance which measures straight-line distance through ambient space. ManifoldANN provides tools to estimate geodesic distances using local tangent space approximations.

## Overview

The geodesic module provides:
1. **Weighted kNN graphs**: kNN graphs with tangent-space edge weights
2. **Edge weight modes**: Different strategies for computing edge weights
3. **Tangent sharing**: Memory optimization by sharing similar tangent planes
4. **Geodesic model**: Main interface for geodesic distance queries

## Why Geodesic Distance?

Consider the classic Swiss Roll example: two points may be close in 3D space (through the "air") but far apart along the rolled surface. Euclidean kNN would incorrectly consider these points neighbors.

```
Euclidean distance: short (through the roll)
Geodesic distance: long (along the surface)
```

Geodesic distance correctly captures the intrinsic geometry of the data.

## Weighted kNN Graph

### WeightedKNNGraph

Extends `KNNGraph` with local geometry and geodesic-aware edge weights.

```julia
struct WeightedKNNGraph{T,G}
    graph::KNNGraph           # Underlying neighbor structure
    geometries::Vector{G}     # Per-node local geometry
    edge_weights::Vector{Vector{T}}  # Geodesic-aware edge weights
end
```

**Building:**
```julia
using ManifoldANN

# From existing graph
graph = build_knn_graph(index, data; k=15)
method = PCAMethod(intrinsic_dim=2)
wg = build_weighted_graph(method, graph, data)

# Convenience: build both in one call
wg = build_weighted_graph(method, index, data; k=15)
```

**Accessors:**
```julia
# Basic properties
length(wg)                    # Number of nodes
configured_k(wg)              # Neighbors per node

# Node access
neighbors(wg, i)              # Neighbor indices for node i
neighbor_weights(wg, i)       # Edge weights for node i
neighbors_with_weights(wg, i) # Iterator of (neighbor, weight) pairs
node_geometry(wg, i)          # Local geometry at node i

# Statistics
mean_edge_weight(wg)
total_edge_weight(wg)
edge_weight_statistics(wg)    # (min, max, mean, total, n_edges)
```

## Edge Weight Modes

Edge weights can be computed using different strategies for handling the transition between adjacent tangent planes.

### SourceTangent (Default)

Use only the source node's tangent plane:

```
weight(i → j) = local_distance(geom_i, point_i, point_j)
```

```julia
wg = build_weighted_graph(method, index, data; k=15, edge_weight_mode=SourceTangent())
```

**Properties:**
- Fastest (one tangent plane evaluation per edge)
- Asymmetric: `weight(i → j) ≠ weight(j → i)` in general
- Good default for most cases

### SymmetricMean

Average distance from both tangent planes:

```
weight(i → j) = (local_distance(geom_i, p_i, p_j) + local_distance(geom_j, p_i, p_j)) / 2
```

```julia
wg = build_weighted_graph(method, index, data; k=15, edge_weight_mode=SymmetricMean())
```

**Properties:**
- Symmetric weights
- More robust in high-curvature regions
- Requires accessing both nodes' geometries

### SymmetricMax

Maximum (conservative) distance from both tangent planes:

```
weight(i → j) = max(local_distance(geom_i, ...), local_distance(geom_j, ...))
```

```julia
wg = build_weighted_graph(method, index, data; k=15, edge_weight_mode=SymmetricMax())
```

**Properties:**
- Symmetric weights
- Conservative: tends to overestimate distances
- Useful when one tangent plane may be poorly fit

## Tangent Plane Sharing

For large datasets, fitting a unique tangent plane at every node may be wasteful when nearby nodes have similar local geometry. Tangent sharing reduces memory and computation by reusing tangent plane bases.

### NoSharing (Default)

Each node gets its own tangent plane:

```julia
wg = build_weighted_graph(method, index, data; k=15, tangent_sharing=NoSharing())
```

### ShareSimilarTangents

Nodes with sufficiently similar tangent planes share the same basis:

```julia
# Share if tangent planes differ by less than 15 degrees
sharing = ShareSimilarTangents(
    SubspaceAngleCriterion(π/12),  # ~15 degrees
    max_graph_distance=2           # Consider sharing within 2 hops
)
wg = build_weighted_graph(method, index, data; k=15, tangent_sharing=sharing)

# Check sharing statistics
println("Unique tangent planes: $(unique_geometry_count(wg))")
println("Sharing ratio: $(round(geometry_sharing_ratio(wg) * 100, digits=1))%")
```

**Algorithm:**
1. Process nodes in order
2. For each new node, check if any nearby fitted node has a similar tangent plane
3. If similar (angle below threshold): reuse the existing basis with a new center
4. If not similar: fit a new tangent plane
5. Shared nodes can act as donors for subsequent nodes (chain propagation)

**Sharing criteria options:**
- `SubspaceAngleCriterion(max_angle)`: Compare subspace orientations
- `FitErrorCriterion(max_error)`: Compare reconstruction errors
- `DistortionCriterion(max_distortion)`: Compare distance distortions

**Note:** Sharing reuses the tangent **directions** (basis) but each node keeps its own **center**. This ensures correct local distance computation.

## Geodesic Distance Model

### GeodesicDistanceModel

Main interface for geodesic distance queries:

```julia
struct GeodesicDistanceModel{I,W,M}
    index::I           # ANN index for neighbor queries
    weighted_graph::W  # Weighted kNN graph
    method::M          # Geometry method for new points
end
```

**Building:**
```julia
using ManifoldANN

# Build model
data = randn(3, 1000)
index = build_index(BruteForceIndex, data)
method = PCAMethod(intrinsic_dim=2)
model = build_geodesic_model(method, index, data; k=15)

# With a non-default edge-weight rule
model = build_geodesic_model(method, index, data;
    k=15,
    edge_weight=TangentProjectedSymmetricMean(),
)
```

### Distance Queries

**Between graph nodes:**
```julia
# Distance from node 1 to node 100
d = geodesic_distance(model, data, 1, 100)
```

**From new point to graph node:**
```julia
# Distance from query point to node 100
query = randn(3)
d = geodesic_distance(model, data, query, 100; entry_k=5)
```

**Between two new points:**
```julia
# Distance between two query points
point_a = randn(3)
point_b = randn(3)
d = geodesic_distance(model, data, point_a, point_b; entry_k=5)
```

### Path Reconstruction

Get the shortest path along with distance:

```julia
result = shortest_path_with_path(model, data, 1, 100)
println("Distance: $(result.distance)")
println("Path: $(result.path)")  # Vector of node indices
```

### All-Pairs Distances

Compute full distance matrix (O(n²) - use only for small graphs):

```julia
D = all_pairs_geodesic_distances(model, data)
# D[i,j] = geodesic distance from node i to node j
```

## Geodesic Curve Refinement

The discrete shortest path from Dijkstra's algorithm is piecewise linear (straight lines between graph nodes). For smoother approximations of geodesic curves, ManifoldANN provides refinement methods.

### Refinement Interface

All refinement methods implement:

```julia
refine_path(method::AbstractGeodesicRefinement,
            model::GeodesicDistanceModel,
            data::AbstractMatrix,
            path::Vector{Int}) -> RefinedPath
```

Returns a `RefinedPath` containing:
- `points::Vector{Vector{T}}`: Dense sequence of curve points
- `distance::T`: Total arc length
- `original_path::Vector{Int}`: Original discrete path
- `segment_lengths::Vector{T}`: Per-segment arc lengths

### NoRefinement

Identity refinement that returns the original waypoints:

```julia
refinement = NoRefinement()
refined = refine_path(refinement, model, data, path)
# refined.points == [data[:, i] for i in path]
```

### SubdivisionSmoothing

Iterative smoothing via subdivision, averaging, and tangent projection:

**Algorithm:**
1. Subdivide each segment into multiple linear pieces
2. Iteratively move each interior point to the average of its neighbors
3. Project back onto the local tangent plane (interpolated from graph)
4. Repeat until convergence

```julia
refinement = SubdivisionSmoothing(
    subdivisions=10,      # Points per original segment
    max_iterations=30,    # Smoothing iterations
    tolerance=1e-4,       # Convergence threshold
    damping=0.5          # Update damping factor
)
refined = refine_path(refinement, model, data, path)

println("Original path: $(length(path)) waypoints")
println("Refined curve: $(length(refined.points)) points")
println("Arc length: $(refined.distance)")
```

**Properties:**
- Produces smooth curves that respect local tangent geometry
- Endpoints are preserved
- Converges to a curve with lower total curvature
- Not guaranteed to be a true geodesic, but a smooth approximation

**When to use:**
- Visualization of geodesic paths
- When you need dense sampling along the curve
- When smoothness is more important than exact geodesic optimality

### CurvatureCorrectedDistance

Improves distance estimates using second-order curvature information from PCA.

**Design Note:** Unlike SubdivisionSmoothing (which produces smooth curves), CurvatureCorrectedDistance
focuses on accurate **distance estimation**. It can work standalone on discrete paths OR be composed
with other refinements to correct their distances.

**Algorithm:**
1. For each edge in the discrete path, estimate local curvature from eigenvalue ratios
2. Apply Taylor expansion correction: `d_corrected ≈ d_tangent * (1 + (κ × d)² / 24)`
3. Sum corrected distances

The curvature κ is computed as `κ = κ_indicator / length_scale`, where κ_indicator comes from
PCA eigenvalue spread (dimensionless 0-1) and length_scale (default: mean edge weight) converts
it to physical curvature (units: 1/length).

```julia
# Use auto-detected length scale
refinement = CurvatureCorrectedDistance()
refined = refine_path(refinement, model, data, path)

# Specify length scale explicitly (e.g., if you know manifold radius)
refinement = CurvatureCorrectedDistance(length_scale=2.0)
refined = refine_path(refinement, model, data, path)

# Compose with smoothing
refinement = CurvatureCorrectedDistance(
    base_refinement=SubdivisionSmoothing(subdivisions=10)
)
refined = refine_path(refinement, model, data, path)
```

**Properties:**
- Returns the same waypoints (no curve densification) unless composed with base_refinement
- Provides more accurate distance estimates on curved manifolds
- O(path length) - very fast
- Correction is second-order: typically 0.1-5% for well-sampled manifolds

**When to use:**
- When you need accurate distance but not the full curve
- On manifolds with significant curvature
- As a quick improvement over tangent-space distances
- Composed with smoothing when you need both smooth curves and accurate distances

### Comparison

| Method | Output | Smoothness | Accuracy | Speed |
|--------|--------|------------|----------|-------|
| `NoRefinement` | Waypoints | Piecewise linear | Graph distance | Instant |
| `SubdivisionSmoothing` | Dense curve | Smooth | Approximate geodesic | Iterative (slow) |
| `CurvatureCorrectedDistance` | Waypoints | Piecewise linear | Second-order corrected | Fast |

### Example: Refining a Swiss Roll Path

```julia
using ManifoldANN

# Build model (Swiss Roll data)
index = build_index(BruteForceIndex, data)
method = PCAMethod(intrinsic_dim=2)
model = build_geodesic_model(method, index, data; k=15)

# Get discrete shortest path
result = shortest_path_with_path(model, data, start_idx, end_idx)
path = result.path

# Compare refinement methods
refinements = [
    NoRefinement(),
    SubdivisionSmoothing(subdivisions=10, max_iterations=20),
    CurvatureCorrectedDistance()
]

for method in refinements
    refined = refine_path(method, model, data, path)
    println("$(typeof(method).name):")
    println("  Points: $(length(refined.points))")
    println("  Distance: $(round(refined.distance, digits=3))")
end
```

### Future Extensions

The refinement interface is designed to accommodate additional methods:

- **Heat Method** (Crane et al.): Solve heat diffusion + Poisson equation for globally optimal geodesics
- **Variational Optimization**: Minimize arc length subject to tangent constraints
- **Hermite Splines**: Smooth interpolation respecting tangent velocities at waypoints

To implement a new refinement method:

```julia
struct MyRefinement <: AbstractGeodesicRefinement
    # parameters...
end

function ManifoldANN.refine_path(method::MyRefinement,
                                  model::GeodesicDistanceModel,
                                  data::AbstractMatrix,
                                  path::Vector{Int};
                                  kwargs...)
    # Your refinement logic
    points = ...
    distance = sum(segment_lengths)
    return RefinedPath(points, distance, path, segment_lengths)
end
```

## Algorithm Details

### Shortest Path Computation

Geodesic distance between graph nodes uses Dijkstra's algorithm on the weighted graph:

1. Initialize distances: `dist[source] = 0`, all others `Inf`
2. Use min-heap priority queue
3. Relax edges using tangent-space edge weights
4. Return `dist[target]` (or `Inf` if unreachable)

### New Point Handling

For points not in the graph:

1. **Find entry points**: Query ANN index for k nearest graph nodes
2. **Fit local geometry**: Create tangent plane at query point
3. **Compute multi-hop distance**:
   - For each entry node: `local_distance(query → entry) + graph_distance(entry → target)`
   - Return minimum total distance

For two new points:
```
min over (entry_a, entry_b):
    local_distance(a → entry_a) +
    graph_distance(entry_a → entry_b) +
    local_distance(entry_b → b)
```

## Practical Examples

### Swiss Roll Manifold

```julia
using ManifoldANN
using LinearAlgebra

# Generate Swiss Roll data
n = 500
t = 1.5π .+ 3π .* rand(n)
height = 10 .* rand(n)
data = vcat((t .* cos.(t))', height', (t .* sin.(t))')

# Build geodesic model
index = build_index(BruteForceIndex, data)
method = PCAMethod(intrinsic_dim=2)
model = build_geodesic_model(method, index, data; k=15)

# Find points close in Euclidean but far on manifold
i, j = 42, 317  # Example pair
euclidean = norm(data[:, i] - data[:, j])
geodesic = geodesic_distance(model, data, i, j)

println("Euclidean: $(round(euclidean, digits=2))")
println("Geodesic: $(round(geodesic, digits=2))")
println("Ratio: $(round(geodesic/euclidean, digits=1))x")
```

### With Adaptive Geometry

```julia
using ManifoldANN

# Adaptive neighborhood selection
strategy = AdaptiveNeighborhood(max_neighbors=30, min_neighbors=8)
method = PCAMethod(intrinsic_dim=2)
estimator = LocalGeometryEstimator(strategy, method)

# Build with more candidates for adaptive fitting
model = build_geodesic_model(estimator, index, data; k=15, candidate_k=30)
```

### With Tangent Sharing

```julia
using ManifoldANN

# Share similar tangent planes
sharing = ShareSimilarTangents(SubspaceAngleCriterion(π/6))

method = PCAMethod(intrinsic_dim=2)
wg = build_weighted_graph(method, index, data; k=15, tangent_sharing=sharing)

println("Nodes: $(length(wg))")
println("Unique tangent planes: $(unique_geometry_count(wg))")
println("Memory saved: $(round((1 - geometry_sharing_ratio(wg)) * 100))%")
```

### Query New Points

```julia
using ManifoldANN

# Build model
model = build_geodesic_model(method, index, data; k=15)

# Query with a new point on the manifold
t_query = 2.5π
query = [t_query * cos(t_query), 5.0, t_query * sin(t_query)]

# Find geodesic distance to several targets
for target in [1, 100, 250, 400]
    d = geodesic_distance(model, data, query, target)
    println("Distance to node $target: $(round(d, digits=2))")
end
```

## Performance Considerations

### Build Time

| Component | Complexity | Notes |
|-----------|------------|-------|
| kNN graph | O(n × k × query_cost) | Depends on ANN index |
| Geometry fitting | O(n × k × d²) | PCA uses SVD |
| Edge weights | O(n × k) | One tangent projection per edge |

### Query Time

| Query Type | Complexity | Notes |
|------------|------------|-------|
| Graph nodes | O((n + m) log n) | Dijkstra with early termination |
| New point | O(entry_k × Dijkstra) | Multiple entry points tried |
| Two new points | O(entry_k² × Dijkstra) | All pairs of entry points |

### Memory

| Component | Memory | Notes |
|-----------|--------|-------|
| Weighted graph | O(n × k) | Same as KNNGraph |
| Geometries | O(n × D × d) | Basis matrices (D=ambient, d=intrinsic) |
| With sharing | Reduced | Shared bases, unique centers |

### Optimization Tips

1. **Use sharing** for large datasets with smooth manifolds
2. **Smaller k** for faster queries (but lower accuracy)
3. **Early termination** in Dijkstra when target is reached
4. **HNSW index** for faster new-point neighbor queries
5. **Batch queries** to amortize geometry fitting

## Comparison with Related Methods

| Method | Approach | ManifoldANN Analogue |
|--------|----------|---------------------|
| ISOMAP | Graph shortest path on kNN | Same core idea |
| LTSA | Local tangent alignment | `PCAMethod` provides local tangent |
| Hessian LLE | Second-order local fitting | Could add `HessianMethod` |

**Key differences from ISOMAP:**
- ISOMAP uses raw Euclidean edge weights
- ManifoldANN uses tangent-space projected distances
- Better for non-isometric embeddings and varying local scale

## See Also

- `docs/geometry.md` - Local geometry estimation details
- `docs/graphs.md` - kNN graph construction
- `docs/examples/geodesic/01-geodesic-swiss-roll.jl` - Complete Swiss Roll example
- `docs/examples/geodesic/02-strategy-comparison.jl` - Comparing neighborhood strategies
- `docs/design/ADR-0010-geodesic-distance-estimation.md` - Architecture decisions
