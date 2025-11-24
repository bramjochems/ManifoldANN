# ADR-0010: Geodesic Distance Estimation Architecture

## Context

The core goal of ManifoldANN is to support approximate nearest neighbor search on
data lying on or near manifolds, where geodesic distances (distances along the
manifold surface) are more meaningful than ambient Euclidean distances. This
requires:

1. Estimating local manifold geometry at each point in a kNN graph
2. Using local geometry to compute edge weights that approximate local geodesic
   distances
3. Finding shortest paths on the weighted graph to estimate global geodesics
4. Handling new query points not in the original graph
5. (Future) Refining discrete paths into smooth geodesic curves

The existing infrastructure provides ANN indices for fast neighbor lookup and
`KNNGraph` for graph structure, but no mechanism for manifold-aware distance
estimation. We need abstractions that are flexible enough to support multiple
local geometry methods (PCA-based tangent estimation, heat kernel methods, etc.)
while integrating cleanly with existing index and graph types.

## Decision

### 1. Pipeline Architecture

The geodesic estimation pipeline follows this flow:

```
Data (ambient space)
    │
    ▼
[Optional: AbstractTransform preprocessing]
    │
    ▼
AbstractANNIndex (for fast neighbor queries)
    │
    ▼
KNNGraph (captures local neighborhood structure via Euclidean distances)
    │
    ▼
Local Geometry Fitting (per-node tangent/metric estimation)
    │
    ▼
WeightedKNNGraph (graph + geometries + geodesic edge weights)
    │
    ▼
GeodesicDistanceModel (query interface for geodesic distances)
    │
    ▼
[Future: Path refinement into smooth geodesic curves]
```

**Rationale**: Euclidean distances are good approximations locally (nearby points
on a manifold), so building the kNN graph in ambient space captures the correct
neighborhood structure. Local geometry then refines the *distances* (edge weights)
without changing *who* the neighbors are. This separation also allows using
different metrics for neighbor finding vs. geodesic estimation.

### 2. Local Geometry Abstraction

Two abstract types separate the method (algorithm) from the fitted result:

```julia
abstract type AbstractLocalGeometryMethod end
abstract type AbstractLocalGeometry end
```

**`AbstractLocalGeometryMethod`**: Specifies how to fit local geometry from
neighborhood data. Implementations include `PCAMethod`, `HeatKernelMethod`, etc.

**`AbstractLocalGeometry`**: The fitted per-node result storing tangent space
information, local metric, etc. Implementations include `PCAGeometry`,
`HeatKernelGeometry`, etc.

Required interface:
- `fit_geometry(method, data, center_idx, neighbor_indices) -> AbstractLocalGeometry`
- `fit_geometry(method, data, query_point, neighbor_indices) -> AbstractLocalGeometry`
- `local_distance(geom, from_point, to_point) -> Real`

Optional interface (for future curve refinement):
- `project(geom, point) -> local_coords` - project to local coordinates
- `reconstruct(geom, local_coords) -> point` - reconstruct from local coordinates
- `supports_projection(geom) -> Bool` - capability introspection

**Rationale**: The same fitted geometry (e.g., PCA basis) serves both distance
estimation now and curve refinement later. Separating method from result follows
the `fit!/transform` pattern established by `AbstractTransform`. Capability
introspection follows the `supports_*` pattern used throughout the package.

### 3. WeightedKNNGraph

```julia
struct WeightedKNNGraph{T,G<:AbstractLocalGeometry}
    graph::KNNGraph
    geometries::Vector{G}
    edge_weights::Vector{Vector{T}}
end
```

Combines the adjacency structure with per-node geometry and precomputed edge
weights. Builder function:

```julia
build_weighted_graph(method, graph, data) -> WeightedKNNGraph
```

**Rationale**: Edge weights are computed once during construction rather than
on-demand, since shortest path algorithms will access them repeatedly. Storing
geometries alongside the graph keeps related data together and enables new-point
queries that need to reference nearby node geometries.

### 4. GeodesicDistanceModel

```julia
struct GeodesicDistanceModel{I,W,M}
    index::I                 # AbstractANNIndex for neighbor lookup
    weighted_graph::W        # WeightedKNNGraph
    method::M                # AbstractLocalGeometryMethod for new points
end
```

Primary query interface:
- `geodesic_distance(model, data, i::Int, j::Int)` - between graph nodes
- `geodesic_distance(model, data, point, j::Int)` - from new point to graph node
- `geodesic_distance(model, data, point_a, point_b)` - between two new points
- `shortest_paths(model, data, i, j, k)` - top-k shortest paths for later refinement

**Rationale**: The model combines all components needed for geodesic queries. The
index enables fast neighbor lookup for new points. The method is retained so new
points can have their local geometry fitted on-demand using the same algorithm
used for graph nodes.

### 5. New Point Handling

For a query point not in the graph:

1. Find k nearest graph nodes using the index (fast, approximate)
2. Fit local geometry for the query point using those neighbors
3. Estimate local geodesic distance from query to nearest graph node(s)
4. Use graph shortest paths for the remaining distance

**Rationale**: The index exists precisely for fast neighbor lookup; reusing it
avoids duplicate infrastructure. Fitting geometry on-demand for query points
avoids storing geometry for points that may never be queried.

### 6. File Organization

```
src/
├── geometry/
│   ├── local_geometry.jl      # Abstract types, interface functions
│   └── pca.jl                 # PCAMethod, PCAGeometry
├── graphs/
│   ├── knn_graph.jl           # (existing)
│   └── weighted_knn_graph.jl  # WeightedKNNGraph
└── geodesic/
    └── geodesic_model.jl      # GeodesicDistanceModel
```

**Rationale**: Follows existing package structure where related abstractions
are grouped (cf. `transforms/`, `indices/`). Local geometry is a new concept
distinct from transforms (which preprocess data) and indices (which enable
fast queries).

### 7. Deferred Decisions

The following are explicitly deferred:

- **Symmetry handling**: Local distances may be asymmetric
  (`local_distance(geom_a, a, b) != local_distance(geom_b, a, b)`). Symmetrization
  strategies (average, min, directed graph) will be addressed when needed.

- **Data ownership**: Whether `GeodesicDistanceModel` holds a reference to or
  copy of the data matrix. Current index design passes data at query time;
  geodesic model will likely follow the same pattern.

- **Shortest path algorithm**: Top-k shortest paths (e.g., Yen's algorithm)
  vs. approximate methods for large graphs. Will be decided during implementation.

- **Geometry refitting**: Whether inserting new points should update neighboring
  geometries. Initially, geometries are immutable after construction.

## Consequences

- Local geometry estimation becomes a first-class abstraction, enabling
  experimentation with PCA, heat kernels, and other methods through a common
  interface.

- The weighted graph cleanly extends `KNNGraph` without modifying it, following
  the composition-over-modification principle used elsewhere in the package.

- New points are handled uniformly through the same geometry-fitting mechanism
  used for graph construction, ensuring consistency.

- The pipeline stages (graph construction, geometry fitting, distance queries)
  remain independent, allowing users to mix components (e.g., HNSW index with
  PCA geometry, or NNDescent index with heat kernel geometry).

- Future curve refinement can reuse the per-node geometries already computed,
  since `project`/`reconstruct` operate on the same `AbstractLocalGeometry`
  that provides `local_distance`.

- The `supports_projection` capability introspection ensures code that needs
  curve refinement can check at runtime whether the geometry supports it,
  failing fast with clear errors otherwise.
