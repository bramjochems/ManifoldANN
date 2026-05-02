# Graph Utilities Overview

Graph-building utilities live under `src/graphs/` and support constructing both directed and undirected k-nearest-neighbor graphs from any `AbstractANNIndex`.

## KNNGraph
- **Structure**: stores per-vertex neighbor lists, requested `k`, an `include_self` flag, and optional node metadata. Supports standard container operations (`length`, indexing, iteration) so you can iterate directly over `graph`.
- **Construction**: use `build_knn_graph(index, data; k=..., include_self=false, directed=true, metadata=nothing)` to query every column of `data` through the supplied index.
- **Resolving `k`**:
  - If `k` is omitted, `build_knn_graph` falls back to `configured_k(index)`.
  - Passing an explicit `k` greater than `configured_k` raises an error to surface mismatched expectations early.
- **`include_self`**:
  - Defaults to `false`. When `false`, self-loops are stripped and construction errors if the index cannot deliver enough non-self neighbors.
  - When `true`, self-loops are preserved; use this if your index naturally returns the query id.
- **`directed`** (NEW):
  - Defaults to `true` for standard directed k-NN graphs.
  - When `false`, symmetrizes the graph by taking the union of i→j and j→i edges, creating an undirected graph.
  - After symmetrization, nodes may have more than k neighbors (typically 15-54 for k=15).
  - The `original_k` value is stored in `graph.metadata["original_k"]` for algorithms that need it.
  - Required for Ollivier-Ricci curvature computations (see orcml replication below).
- **Metadata**:
  - Pass a vector via `metadata=` whose length matches the dataset columns to attach per-node payloads (labels, tangent bases, etc.).
  - The graph also stores metadata about its construction in `graph.metadata`, including `"original_k"` and `"directed"`.
  - Use `has_metadata(graph)`, `graph_metadata(graph)`, or `node_metadata(graph, i)` to consume payloads safely.
- **Query kwargs**: additional keyword arguments are forwarded to `query` (e.g. `candidate_cap` for `LSHIndex`).

## Pattern of Use
1. Build an index (`BruteForceIndex`, `KDTreeIndex`, `LSHIndex`, …).
2. Call `build_knn_graph(index, data; k=..., include_self=...)`.
3. Iterate the resulting `KNNGraph` or pass it to downstream workflows (e.g. shortest paths, diffusion) once those modules are implemented.

Examples:
- `docs/examples/graphs/01-build-knn-graph.jl`: brute-force baseline.
- `docs/examples/graphs/02-lsh-knn-graph.jl`: approximate graph via LSH.
- `docs/examples/graphs/03-kdtree-knn-graph.jl`: exact graph using a KD-tree.

## Performance: threading and BLAS

`build_weighted_graph` parallelises per-node geometry fitting across Julia
threads when `tangent_sharing=NoSharing()` (the default). To benefit, start
Julia with `-t auto` or `JULIA_NUM_THREADS=N`.

If you see no speedup at higher ambient dimensions (d ≥ 128), suspect
**BLAS thread oversubscription**: by default OpenBLAS uses up to 8 threads
per call, which contend with Julia's outer threads on the small symmetric
eigenproblems used by local PCA. Setting BLAS to a single thread typically
helps:

```julia
using LinearAlgebra
BLAS.set_num_threads(1)
```

This is a session-level setting; the package does not change it for you.
Measured impact on a 16-logical-core box at d=128, n=5000, k=15: build
time 2082 ms with default BLAS=8 → 1464 ms with BLAS=1 (1.4× speedup).

`tangent_sharing=ShareSimilarTangents(...)` runs serially by design — each
node's decision to share or fit fresh depends on what was decided for
earlier nodes, so threading would either lose the sharing structure or
produce schedule-dependent output.

### Thread-safety contract for custom geometry methods

Threading the fit loop is a public-API contract: any user-defined
`AbstractLocalGeometryMethod` subclass with its own `fit_geometry` method
must be safe to call concurrently across nodes. Concretely, custom
implementations must be pure — read-only access to `data`, no mutation
of method-internal state (`mutable struct` fields used as caches, etc.),
no side effects on shared globals. The built-in `PCAMethod` and
`EstimatedGeometry` satisfy this trivially. Methods that violate the
contract will produce wrong or non-deterministic results under threading.

This contract follows the standard Julia ecosystem convention (see
DataFrames.jl, Distances.jl): threading is a performance default, the
contract is documented, and there is no opt-out kwarg.
