# Local Geometry Estimation

Local geometry estimation provides the foundation for geodesic distance computation in ManifoldANN. By fitting tangent spaces at each point, we can measure distances along the manifold surface rather than through ambient space.

## Overview

The geometry module provides:
1. **Geometry methods**: Define how to fit local tangent spaces (e.g., PCA)
2. **Neighborhood strategies**: Control how neighbors are selected for fitting
3. **Selection criteria**: Evaluate geometry quality and similarity
4. **Estimators**: Compose strategies with methods for adaptive fitting

## Core Concepts

### Local Tangent Space

At each point on a manifold, we approximate the local surface with a tangent plane. This linearization allows us to:
- Project nearby points into a low-dimensional coordinate system
- Measure distances in the tangent space (approximating geodesic distances)
- Detect curvature by comparing adjacent tangent planes

### Distance Computation

Local distance between two points using a tangent plane:
1. Project both points onto the tangent space
2. Compute Euclidean distance in the projected coordinates

This approximates the true geodesic distance for points within the tangent plane's validity region.

## Geometry Methods

### AbstractLocalGeometryMethod

Base type for all geometry fitting methods:

```julia
abstract type AbstractLocalGeometryMethod end
abstract type AbstractLocalGeometry end

# Required interface
fit_geometry(method, data, center_idx, neighbor_indices) → AbstractLocalGeometry
fit_geometry(method, data, query_point, neighbor_indices) → AbstractLocalGeometry
local_distance(geom, from, to) → Real

# Optional interface
supports_projection(geom) → Bool
project(geom, point) → Vector
reconstruct(geom, local_coords) → Vector
```

### PCAMethod

Fits local geometry using Principal Component Analysis.

**Parameters:**
- `intrinsic_dim::Union{Int,Nothing}`: Fixed dimension, or `nothing` for auto-detection
- `min_variance_ratio::Float64`: Minimum cumulative variance to retain (default: 0.95)

**Example:**
```julia
# Fixed 2D tangent planes (for a surface in 3D)
method = PCAMethod(intrinsic_dim=2)

# Auto-detect dimension based on variance
method = PCAMethod(min_variance_ratio=0.95)

# Fit geometry at a node
geom = fit_geometry(method, data, center_idx, neighbor_indices)

# Compute local distance
d = local_distance(geom, point_a, point_b)
```

### PCAGeometry

The fitted geometry from PCAMethod:

**Fields:**
- `center::Vector{T}`: Center point (tangent plane origin)
- `basis::Matrix{T}`: Principal component basis (D × d), columns are PCs
- `eigenvalues::Vector{T}`: Variance along each principal direction

**Operations:**
```julia
# Project point onto tangent space
local_coords = project(geom, point)

# Reconstruct from tangent coordinates
reconstructed = reconstruct(geom, local_coords)

# Get intrinsic dimension
d = intrinsic_dimension(geom)

# Get variance explained
ratio = explained_variance_ratio(geom)

# Compute reconstruction error
error = fit_error(geom, point)
```

## Neighborhood Strategies

Neighborhood strategies control how neighbors are selected for geometry fitting. This is separate from the geometry method itself, enabling mix-and-match composition.

### FixedNeighborhood

Use all provided neighbors without filtering.

```julia
strategy = FixedNeighborhood()
```

**Use cases:**
- Simple baseline
- When neighbors are pre-filtered
- Predictable behavior for benchmarking

### AdaptiveNeighborhood

Start with all neighbors, iteratively remove outliers based on quality criterion.

**Parameters:**
- `max_neighbors::Int`: Maximum neighbors to use
- `min_neighbors::Int`: Stop shrinking at this count
- `criterion::AbstractSelectionCriterion`: Quality metric
- `max_iterations::Int`: Maximum refinement iterations

```julia
# Remove neighbors with high reconstruction error
strategy = AdaptiveNeighborhood(
    max_neighbors=30,
    min_neighbors=8,
    criterion=FitErrorCriterion(max_relative_error=0.1),
    max_iterations=10
)
```

**Algorithm:**
1. Fit geometry with all candidates
2. Evaluate each neighbor's quality
3. Remove worst neighbor if it exceeds threshold
4. Repeat until stable or min_neighbors reached

### ExpandingNeighborhood

Start small, grow until quality degrades.

**Parameters:**
- `initial_k::Int`: Starting neighborhood size
- `max_k::Int`: Maximum neighborhood size
- `criterion::AbstractSelectionCriterion`: Quality metric
- `step_size::Int`: Growth increment

```julia
# Grow until geometry becomes unstable
strategy = ExpandingNeighborhood(
    initial_k=5,
    max_k=50,
    criterion=SubspaceAngleCriterion(π/6),  # 30 degrees
    step_size=3
)
```

**Algorithm:**
1. Fit geometry with initial_k neighbors
2. Add step_size more neighbors
3. Refit and compare subspace angles
4. Stop when angle exceeds threshold or max_k reached

## Selection Criteria

Criteria evaluate geometry quality and are used by adaptive strategies.

### FitErrorCriterion

Evaluate based on reconstruction error (distance from point to tangent plane).

```julia
criterion = FitErrorCriterion(max_relative_error=0.1)

# Check if geometry fits a point well
passes = passes_threshold(criterion, geom, point)
```

**Use cases:**
- Detecting outliers in the neighborhood
- Removing points that don't lie on the local tangent plane

### DistortionCriterion

Evaluate based on distance distortion between tangent plane and ambient space.

```julia
criterion = DistortionCriterion(max_distortion=0.2)
```

**Distortion formula:**
```
distortion = |d_tangent - d_euclidean| / max(d_tangent, d_euclidean)
```

**Use cases:**
- Detecting non-isometric regions
- Comparing geometry similarity

### SubspaceAngleCriterion

Compare tangent planes using principal angles between subspaces.

```julia
criterion = SubspaceAngleCriterion(max_angle=π/6)  # 30 degrees

# Compare two geometries
angle = subspace_angle(geom1, geom2)
similar = angle <= criterion.max_angle
```

**Principal angles:**
- Computed via SVD of `basis1' * basis2`
- Maximum angle indicates largest discrepancy between subspaces
- Works for any dimension (not just 2D)

**Use cases:**
- Detecting curvature (angle between adjacent tangent planes)
- Tangent plane sharing (share if planes are similar)
- Stopping criterion for expanding neighborhoods

## LocalGeometryEstimator

Composes a neighborhood strategy with a geometry method.

```julia
# Create an adaptive PCA estimator
strategy = AdaptiveNeighborhood(max_neighbors=30, min_neighbors=8)
method = PCAMethod(intrinsic_dim=2)
estimator = LocalGeometryEstimator(strategy, method)

# Use like any geometry method
geom = fit_geometry(estimator, data, center_idx, candidate_indices)

# Result is wrapped with metadata
@assert geom isa EstimatedGeometry
println("Used $(used_neighbor_count(geom)) neighbors")
println("Refinement iterations: $(refinement_iterations(geom))")
println("Final error: $(max_reconstruction_error(geom))")

# Underlying geometry is accessible
inner = unwrap_geometry(geom)  # PCAGeometry
```

### EstimatedGeometry

Wrapper that preserves selection metadata:

```julia
struct EstimatedGeometry{G,T} <: AbstractLocalGeometry
    geometry::G           # Underlying geometry (e.g., PCAGeometry)
    used_neighbors::Vector{Int}
    selection_iterations::Int
    final_error::T
end

# Access metadata
used_neighbor_count(geom)      # How many neighbors were used
refinement_iterations(geom)    # How many strategy iterations
max_reconstruction_error(geom) # Final quality metric
unwrap_geometry(geom)          # Get underlying PCAGeometry
```

## Practical Examples

### Basic PCA Fitting

```julia
using ManifoldANN

# Generate Swiss Roll data (2D manifold in 3D)
n = 500
t = 1.5π .+ 3π .* rand(n)
data = vcat((t .* cos.(t))', 10 .* rand(1, n), (t .* sin.(t))')

# Fit tangent plane at point 1
method = PCAMethod(intrinsic_dim=2)
neighbors = 2:20  # Use points 2-20 as neighbors
geom = fit_geometry(method, data, 1, neighbors)

# Project and reconstruct
local_coords = project(geom, data[:, 5])
reconstructed = reconstruct(geom, local_coords)
```

### Adaptive Neighborhood Selection

```julia
using ManifoldANN

# Create adaptive estimator
strategy = AdaptiveNeighborhood(
    max_neighbors=30,
    min_neighbors=8,
    criterion=FitErrorCriterion(max_relative_error=0.15)
)
estimator = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=2))

# Fit with outlier removal
index = build_index(BruteForceIndex, data)
candidates = [n.id for n in query(index, data, data[:, 1], 31)][2:end]
geom = fit_geometry(estimator, data, 1, candidates)

println("Started with $(length(candidates)) candidates")
println("Used $(used_neighbor_count(geom)) after filtering")
```

### Expanding Until Curvature

```julia
using ManifoldANN

# Grow neighborhood until tangent plane becomes unstable
strategy = ExpandingNeighborhood(
    initial_k=5,
    max_k=40,
    criterion=SubspaceAngleCriterion(π/8),  # 22.5 degrees
    step_size=3
)
estimator = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=2))

# Build kNN graph for neighbor walking
graph = build_knn_graph(index, data; k=50)

# Fit with graph access for expanding beyond initial candidates
geom = fit_geometry(estimator, data, 1, graph[1]; graph=graph)
println("Expanded to $(used_neighbor_count(geom)) neighbors")
```

## Design Notes

### Why Separate Strategy from Method?

The separation enables orthogonal composition:
- Any strategy works with any geometry method
- Strategies can be swapped without changing method parameters
- New methods automatically work with existing strategies

### When to Use Each Strategy

| Strategy | Use When |
|----------|----------|
| `FixedNeighborhood` | Simple baseline, pre-filtered neighbors |
| `AdaptiveNeighborhood` | Noisy data, outlier removal needed |
| `ExpandingNeighborhood` | Varying local scale, curvature detection |

### Performance Considerations

- PCA uses SVD, O(k × d²) per fit where k=neighbors, d=dimension
- Adaptive strategies may fit multiple times per node
- ExpandingNeighborhood requires graph access for neighbor walks
- Consider caching fitted geometries when querying multiple times

## See Also

- `docs/geodesic.md` - Using local geometry for geodesic distance
- `docs/examples/geodesic/02-strategy-comparison.jl` - Strategy benchmarking
- `docs/design/ADR-0010-geodesic-distance-estimation.md` - Architecture decisions
