# TODO: Multi-Projection Ensemble Voting System

## Overview

This document outlines the implementation plan for **two independent but complementary contributions**:

1. **Multi-Projection Ensemble Voting** (Phase 1): Robust k-NN graph construction via voting across multiple random projections
2. **Grassmannian Averaging** (Phase 2): Robust tangent plane estimation by averaging across multiple projected spaces

### Independence and Modularity

**Both contributions use random projections, but they are modular**:

| Contribution | Requires Projections? | Works Standalone? | Can Reuse Projections? |
|--------------|----------------------|-------------------|------------------------|
| Ensemble Voting | ✅ Yes (for graph construction) | ✅ Yes, use standard PCA for tangents | ✅ Yes, from Grassmannian |
| Grassmannian Averaging | ✅ Yes (for tangent estimation) | ✅ Yes, use standard k-NN for graph | ✅ Yes, from Voting |

**Usage patterns**:
- **Config 1** ✅ **Core**: Ensemble voting generates M projections → build k-NN graph → **discard projections** → standard PCA on consensus neighbors in ambient space
- **Config 2** ⚠️ **Questionable**: Standard k-NN in ambient space → **generate M new projections** → Grassmannian averaging for tangents (weak contribution if graph is already poor)
- **Config 3** ✅ **Core**: Generate M projections once → use for voting (Phase 1) → **reuse** for Grassmannian (Phase 2) ← **Most efficient and strongest results!**
- **Config 4** ✅ **Core**: Ablation study with Baseline, Config 1, Config 3

**Recommended focus**: Configs 1, 3, and 4 provide the clearest thesis contributions.

## Core Algorithm

### Phase 1a: Candidate Set Construction (Early Exploration Projections)

**Goal**: Build a candidate set of potential neighbors for each point using a small number of full ANN searches.

```
Parameters:
- p_early = 3-5           # Number of early "exploration" projections
- l = 2*k or 3*k          # Larger neighborhood for exploration (l > k)
- M = 10-50               # Total number of projections

Algorithm:
1. For m = 1 to p_early:
   a. Generate JL projection: Φ_m: R^D → R^{d_m}
   b. Project data: Y_m = Φ_m * X
   c. Build l-NN graph on Y_m using full ANN index (configurable: HNSW, BruteForce, etc.)
   d. Store edges E_m

2. For each point i:
   C_i = ∪_{m=1}^{p_early} neighbors_m(i)
   # |C_i| ≈ O(p_early * l), but with overlap maybe ~2l to 3l in practice
```

**Computational cost**: Full ANN only p_early times (not M times!)

### Phase 1b: Efficient Voting on Candidate Sets

**Goal**: Count votes for edges within candidate sets using all M projections.

```
Algorithm:
1. Initialize vote_count matrix (sparse, n × n)

2. For m = 1 to M:
   a. Generate JL projection: Φ_m: R^D → R^{d_m}
   b. Project data: Y_m = Φ_m * X

   c. For each point i:
      # Only compute distances within candidate set C_i
      distances = {norm(Y_m[:, i] - Y_m[:, j]) : j ∈ C_i}

      # Find k-nearest within candidates
      knn_m[i] = top-k points from C_i by distance

      # Cast votes
      for j in knn_m[i]:
          vote_count[i, j] += 1
```

**Computational cost**: O(|C_i|) distances per point per projection, not O(n)

### Phase 1c: Top-k Selection by Vote Count

**Goal**: Select final k neighbors based on vote counts with tie-breaking.

```
Algorithm:
For each point i:
    1. Get candidates sorted by vote count:
       candidates = [(j, vote_count[i,j]) for j in C_i]
       sort!(candidates, by = x -> x[2], rev=true)  # Descending

    2. Tie-breaking for equal vote counts:
       For each vote_level:
           ties = [j for (j, v) in candidates if v == vote_level]

           if length(ties) > 1:
               # Option A: Ambient distance
               distances = [norm(X[:, i] - X[:, j]) for j in ties]

               # Option B: Average projected distance
               distances = [mean([norm(Y_m[:, i] - Y_m[:, j]) for m in 1:M])
                           for j in ties]

               # Sort ties by distance (ascending)
               sort ties by distance

    3. Take top k neighbors
    4. Add edges i → neighbors[i] to directed graph G_raw
```

### Phase 1d: Graph Symmetrization

**Goal**: Convert directed graph to undirected with chosen symmetrization strategy.

```
Options:

1. Mutual k-NN (intersection):
   G_consensus = intersection(G_raw, reverse(G_raw))
   # Keep edge only if both i→j AND j→i exist

2. Union:
   G_consensus = union(G_raw, reverse(G_raw))
   # Keep edge if i→j OR j→i exists

3. Weighted voting:
   For each potential edge (i,j):
       combined_votes = vote_count[i,j] + vote_count[j,i]
       if combined_votes >= threshold:
           Add (i,j) to G_consensus
   # Preserves asymmetric voting information

4. None:
   G_consensus = G_raw
   # Keep as directed graph
```

**Output**: Consensus k-NN graph G_consensus

---

### Phase 2: Grassmannian Averaging for Tangent Plane Estimation

**⚠️ INDEPENDENT CONTRIBUTION**: This phase can work with **any k-NN graph**, not just the ensemble-voted one from Phase 1.

**Goal**: Robustly estimate tangent planes by averaging across multiple projections.

**Input**: Any k-NN graph (from ensemble voting, standard HNSW, brute force, etc.)

#### Option A: Standard PCA (Simple, Baseline)
```
For each point i in graph:
    neighbors_i = neighbors from k-NN graph
    Fit PCA on X[:, neighbors_i] in ambient space R^D
    → Single tangent basis U_i ∈ R^{D×k_intrinsic}
```

**Use this if**: You want fast tangent estimation or are using Phase 1 results only

#### Option B: Grassmannian Averaging (Robust, Novel Contribution)

**Mathematical Background**:
- For point x_i, we estimate its tangent space using M different projections
- In each projected space, we fit a local tangent plane
- We map these back to ambient space and average them
- The average is computed via projection operators PP^T

```
Algorithm:
For each point i in G_consensus:
    neighbors_i = {j : (i,j) ∈ G_consensus}

    Initialize accumulator: A = zeros(D, D)

    For each projection m = 1 to M:
        1. Get projection: Y_m = Φ_m * X

        2. Fit PCA on neighbors in projected space:
           neighbor_data_m = Y_m[:, neighbors_i]
           center_m = mean(neighbor_data_m, dims=2)
           centered = neighbor_data_m .- center_m

           U_m, Σ_m, _ = svd(centered)
           P_m = U_m[:, 1:k_intrinsic]  # (d_m × k_intrinsic)

        3. Map projection basis back to ambient space:
           V_m = Φ_m^T * P_m  # (D × k_intrinsic)

        4. Accumulate projection operator:
           A += V_m * V_m^T  # (D × D), rank-k update

    5. Average:
       A = A / M

    6. Extract consensus tangent space:
       eigenvals, eigenvecs = eigen(A)
       idx = sortperm(eigenvals, rev=true)
       U_i = eigenvecs[:, idx[1:k_intrinsic]]  # (D × k_intrinsic)

    Store tangent basis U_i for node i
```

**Why this works**:
- In projected space R^{d_m}, P_m P_m^T is an orthogonal projection operator
- Eigenvalues: {1, 1, ..., 1, 0, 0, ..., 0} (k_intrinsic ones)
- V_m V_m^T is the same operator in ambient coordinates
- Averaging V_m V_m^T gives a robust estimate (extrinsic Grassmannian mean)
- Top k eigenvectors of average converge to true tangent space

### Phase 3: Edge Weights from Tangent Distances

**Goal**: Assign geodesic-aware edge weights using tangent planes.

```
For each edge (i,j) in G_consensus:
    Compute geodesic distance using tangent planes
    (Use existing WeightedKNNGraph logic:
     - SourceTangent: project using U_i
     - SymmetricMean: average of U_i and U_j projections
     - SymmetricMax: conservative max)
```

**Output**: Weighted k-NN graph ready for geodesic distance queries

---

## Modular Usage Patterns

The two main contributions can be mixed and matched based on research questions:

### Configuration 1: Ensemble Voting with Standard Tangent Estimation
**Use case**: Study robustness of k-NN graph topology to projection noise

**Projections used**: For graph construction only

```julia
# Phase 1: Build consensus k-NN via ensemble voting (M projections)
config = EnsembleConfig(
    k=10,
    total_projections=20,      # M=20 projections
    target_dims=50
)
graph = build_ensemble_knn_graph(data, config)

# Phase 2A: Standard PCA for tangent planes on the consensus graph
# Fit PCA on neighbors from consensus graph using original ambient data X
method = PCAMethod(intrinsic_dim=2)
tangents = Vector{PCAGeometry}(undef, n)
for i in 1:n
    neighbor_indices = graph[i]  # Get neighbors from consensus graph
    tangents[i] = fit_geometry(method, data, i, neighbor_indices)
end

# Phase 3: Weighted graph with tangent-aware edge weights
weighted_graph = WeightedKNNGraph(data, graph, tangents, mode=:SourceTangent)
```

**What happens here**:
- Generate M=20 random projections → build candidate sets → vote → consensus k-NN graph
- Projections are discarded after Phase 1
- **Tangent planes ARE estimated** via standard single PCA on neighbors from the consensus graph
- Each node's tangent plane is fit once in ambient space using the (hopefully better) consensus neighbors

**Contributions tested**:
- Novel ensemble voting algorithm with candidate filtering
- Improved k-NN graph recall vs single projection or ambient HNSW
- Shows value of better graph topology even without Grassmannian averaging

### Configuration 2: Grassmannian Averaging Only (⚠️ Questionable Standalone Value)
**Use case**: Study robustness of tangent plane estimation independently

**Projections used**: For tangent estimation only

**⚠️ Note**: This configuration may have **limited thesis value** because:
- If the ambient k-NN graph has poor topology (wrong neighbors), better tangent estimation won't fix it
- The whole point of ensemble voting is to get better neighbors first
- Better tangents on wrong neighbors likely doesn't help much for geodesic distances

**Possible use**: Could show that Grassmannian averaging helps even with standard graphs, but this is a weaker contribution

```julia
# Phase 1: Standard k-NN in AMBIENT SPACE (no projections)
# Use HNSW, brute force, or any standard ANN method on original data X
index = HNSWIndex(data, M=16, ef_construction=200)
build!(index, data)
graph = build_knn_graph(index, data, k=10)

# Phase 2B: Generate NEW projections ONLY for Grassmannian averaging
M = 20
projections = generate_multiple_projections(data, M, target_dim=50)

# Fit tangent planes via Grassmannian averaging
# Uses neighbors from ambient k-NN graph, but estimates tangents in M projected spaces
tangents = fit_grassmann_tangents(data, graph, projections, k_intrinsic=2)

# Phase 3: Weighted graph with robust tangent estimation
weighted_graph = WeightedKNNGraph(data, graph, tangents, mode=:SourceTangent)
```

**What happens here**:
- k-NN graph built in ambient space R^D (potentially poor topology for manifold data)
- Generate M=20 projections ONLY for tangent estimation
- For each node i with neighbors from ambient k-NN:
  - Fit PCA on those neighbors in EACH of the M projected spaces
  - Map back to ambient: V_m = Φ_m^T * P_m
  - Average: A = (1/M) Σ V_m V_m^T
  - Extract consensus tangent space from eigendecomposition of A

**Potential contributions** (but weak):
- Novel Grassmannian averaging via projection operators
- Shows tangent improvement is orthogonal to graph construction
- Could be useful for applying Grassmannian averaging to other methods

**Recommendation**: Consider **skipping this configuration** and focusing on Configs 1, 3, and 4 (ablation) instead

### Configuration 3: Full Pipeline (Both Contributions)
**Use case**: Maximum robustness for both graph topology and tangent estimation

**Projections used**: Once, reused for both phases (efficient!)

```julia
# Phase 1: Build consensus k-NN via ensemble voting (M projections)
config = EnsembleConfig(
    k=10,
    total_projections=20,
    target_dims=50,
    store_projections=true  # IMPORTANT: Keep projections for Phase 2
)
graph, projections = build_ensemble_knn_graph(data, config, return_projections=true)

# Phase 2B: Reuse SAME projections for Grassmannian averaging
# No new projection generation needed!
tangents = fit_grassmann_tangents(data, graph, projections, k_intrinsic=2)

# Phase 3: Weighted graph with both contributions
weighted_graph = WeightedKNNGraph(data, graph, tangents, mode=:SourceTangent)
```

**What happens here**:
- Generate M=20 projections once
- Phase 1: Use them for ensemble voting → consensus k-NN graph
- Phase 2: Reuse them for Grassmannian averaging → robust tangent planes
- Both graph topology AND tangent estimation benefit from projections

**Contributions tested**:
- Combined: robust graph + robust tangents
- Computational efficiency: projections computed once, reused twice
- This is the "full method" for thesis

### Configuration 4: Ablation Studies (Core Thesis Evaluation)
**Use case**: Isolate contributions for thesis evaluation

**Projections**: Each method uses projections consistently for fair comparison

**Recommended ablations**: Baseline, Config 1, Config 3 (skip Config 2 per discussion above)

```julia
M = 20  # Same number of projections for all methods
target_dim = 50
k = 10
k_intrinsic = 2

# --- Baseline: No projections ---
# Standard k-NN in ambient space + standard PCA in ambient space
index_baseline = HNSWIndex(data)
build!(index_baseline, data)
graph_baseline = build_knn_graph(index_baseline, data, k=k)

method = PCAMethod(intrinsic_dim=k_intrinsic)
tangents_baseline = [fit_geometry(method, data, i, graph_baseline[i]) for i in 1:n]
weighted_baseline = WeightedKNNGraph(data, graph_baseline, tangents_baseline)

# --- Ablation A: Ensemble voting + standard tangent estimation (Config 1) ---
# M projections for graph construction, then discard projections
# Standard PCA on consensus neighbors in ambient space
config_a = EnsembleConfig(k=k, total_projections=M, target_dims=target_dim)
graph_a = build_ensemble_knn_graph(data, config_a)  # Projections discarded

tangents_a = [fit_geometry(method, data, i, graph_a[i]) for i in 1:n]  # Ambient PCA
weighted_a = WeightedKNNGraph(data, graph_a, tangents_a)

# --- [OPTIONAL] Ablation B: Grassmannian averaging only (Config 2) ---
# Only include if you want to show Grassmannian averaging helps even with poor graphs
# NO projections for graph, M NEW projections for tangents
# graph_b = build_knn_graph(index_baseline, data, k=k)  # Same as baseline graph
# projections_b = generate_multiple_projections(data, M, target_dim)
# tangents_b = fit_grassmann_tangents(data, graph_b, projections_b, k_intrinsic)
# weighted_b = WeightedKNNGraph(data, graph_b, tangents_b)

# --- Full: Both contributions (Config 3) ---
# M projections for BOTH graph AND tangents (reuse)
config_full = EnsembleConfig(
    k=k,
    total_projections=M,
    target_dims=target_dim,
    store_projections=true
)
graph_full, projections_full = build_ensemble_knn_graph(data, config_full, return_projections=true)
tangents_full = fit_grassmann_tangents(data, graph_full, projections_full, k_intrinsic)
weighted_full = WeightedKNNGraph(data, graph_full, tangents_full)

# --- Evaluation ---
# Compare geodesic distance quality on three main methods:
# - baseline: ambient k-NN + ambient PCA (no projections)
# - weighted_a: ensemble k-NN + ambient PCA (voting contribution only)
# - weighted_full: ensemble k-NN + Grassmannian PCA (both contributions)
```

**Projection usage summary (Core ablations)**:
| Method | Projections for Graph? | Projections for Tangents? | Total Projections |
|--------|------------------------|---------------------------|-------------------|
| Baseline | ❌ No | ❌ No | 0 |
| Ablation A (Config 1) | ✅ Yes (M) | ❌ No (standard PCA) | M |
| Full (Config 3) | ✅ Yes (M) | ✅ Yes (same M, reused) | M (reused) |

**Optional**: Include Ablation B (Config 2) if you want to test Grassmannian averaging independently, but expect weak results

**Thesis value**:
- **Baseline vs A**: Quantify ensemble voting contribution (better graph topology)
- **A vs Full**: Quantify Grassmannian averaging contribution (better tangents on good graph)
- **Baseline vs Full**: Show combined improvement
- **Computational efficiency**: Full uses M projections once, not 2M (A+B would use 2M)

---

## Design Decisions & Open Questions

### 1. Projection Specifications

**Question**: Should all M projections have the same target dimension?

**Options**:
- A. Fixed dimension: All projections to d = 50 or 100 (simpler)
- B. Varying dimensions: Random in [30, 100] for diversity
- C. Adaptive: Use suggested_dimension() with varying ε

**Current thinking**: Start with option A for simplicity, make B available via config

### 2. Candidate Set Symmetry

**Question**: Should candidate sets C_i be symmetric?

**Status**: Not necessary. Symmetrization happens later in Phase 1d.

**Note**: This allows directed graphs if needed (e.g., for hubness studies)

### 3. Re-evaluation of Early Projections

**Question**: In Phase 1b, should we re-evaluate the p_early projections?

**Options**:
- A. Yes, recompute k-NN within C_i for consistency
- B. No, just use their original l-NN results restricted to top-k
- C. Recompute using stored Y_m (avoid regenerating random matrices)

**Recommendation**: Option C - store Y_m from Phase 1a, recompute distances within C_i

### 4. Tie-Breaking Strategy

**Question**: For equal vote counts, how to break ties?

**Options**:
- A. Ambient distance: norm(X[:, i] - X[:, j])
- B. Average projected distance: mean([norm(Y_m[:, i] - Y_m[:, j]) for m in 1:M])
- C. Random (for reproducibility, use seeded RNG)

**Recommendation**: Start with A (simpler), expose B as config option

### 5. Candidate Set Size Limits

**Question**: Should we cap |C_i| to prevent outliers from huge candidate sets?

**Options**:
- A. No limit (trust p_early * l is reasonable)
- B. Hard cap: max |C_i| = 5*k or 10*k
- C. Adaptive: Remove candidates beyond distance threshold

**Recommendation**: Start with A, add B as safety parameter if needed

### 6. Voting Threshold Tuning

**Question**: How to set the threshold for edge inclusion in Phase 1c?

**Options**:
- A. Top-k by vote count (no explicit threshold, always get k neighbors)
- B. Absolute threshold: vote_count >= M/2 (majority)
- C. Relative threshold: vote_count >= α*M for tunable α ∈ [0,1]
- D. Adaptive per point: Keep edges until k reached or vote count drops below threshold

**Current design**: Using option A (top-k), makes threshold implicit

### 7. Projection Reuse for Phase 2

**Question**: Should we reuse projections from Phase 1b for Grassmannian averaging in Phase 2?

**Options**:
- A. Yes, reuse Y_m and Φ_m (memory efficient)
- B. No, generate fresh projections (independent noise)
- C. Configurable

**Recommendation**: Option A - projections are expensive, reuse them

### 8. Tangent Sharing Integration

**Question**: Should the existing ShareSimilarTangents mechanism from weighted_knn_graph.jl be integrated?

**Status**: Yes, but after Grassmannian averaging
- First compute consensus tangent U_i via averaging
- Then apply sharing policy to reuse similar tangent bases
- This reduces memory/computation for dense regions

### 9. Index Configuration for Early Projections

**Question**: What should be the default ANN index for Phase 1a?

**Options**:
- A. BruteForceIndex (exact, simple, works for small n)
- B. HNSWIndex (fast, approximate, scales better)
- C. NNDescentIndex (graph-based, good recall)
- D. Configurable (expose index_builder function)

**Recommendation**: Option D with HNSWIndex as default

### 10. Vote Count Storage

**Question**: How to store and expose vote counts?

**Options**:
- A. Store in KNNGraph.metadata as Vector{Vector{Tuple{Int,Int}}} (neighbor_id, vote_count)
- B. Create wrapper struct EnsembleKNNGraph with separate vote_counts field
- C. Optional return: (graph, vote_matrix) if return_votes=true
- D. Don't store (recompute if needed)

**Recommendation**: Option C - clean API, optional for analysis

---

## Proposed Module Structure

### File Organization

```
src/ensemble/
├── MultiProjectionGraph.jl     # Main implementation
├── ProjectionSpecs.jl           # Projection configuration types
├── VotingStrategies.jl          # Phase 1c logic (top-k, tie-breaking)
├── Symmetrization.jl            # Phase 1d (mutual, union, etc.)
└── GrassmannianAveraging.jl    # Phase 2 (optional enhancement)
```

### API Design

```julia
# Configuration type
struct EnsembleConfig{T<:AbstractFloat}
    # Phase 1a: Candidate generation
    n_early_projections::Int
    candidate_neighbors::Int            # l > k
    early_index_builder::Function       # (data::Matrix{T}) -> AbstractANNIndex

    # Phase 1b: Voting
    total_projections::Int             # M
    target_dims::Vector{Int}           # Length M, varying dimensions allowed
    projection_type::Symbol            # :gaussian or :sparse
    sparse_density::Float64            # For sparse projections

    # Phase 1c: Top-k selection
    k::Int
    tie_breaking::Symbol               # :ambient or :average_projected

    # Phase 1d: Symmetrization
    symmetrization::Symbol             # :mutual, :union, :weighted, :none
    symmetrization_threshold::Float64  # For :weighted mode

    # Optional
    random_seed::Union{Int, Nothing}
    store_projections::Bool            # For Phase 2 reuse
end

# Constructor with defaults
function EnsembleConfig(;
    n_early_projections::Int = 3,
    candidate_neighbors::Int = 30,
    early_index_builder::Function = data -> HNSWIndex(data, M=16, ef_construction=200),
    total_projections::Int = 20,
    target_dims::Union{Int, Vector{Int}} = 50,  # Broadcast if scalar
    projection_type::Symbol = :gaussian,
    sparse_density::Float64 = 1/3,
    k::Int,  # Required
    tie_breaking::Symbol = :ambient,
    symmetrization::Symbol = :mutual,
    symmetrization_threshold::Float64 = 0.5,
    random_seed::Union{Int, Nothing} = nothing,
    store_projections::Bool = false
)
    # Convert scalar target_dims to vector
    dims = target_dims isa Int ? fill(target_dims, total_projections) : target_dims
    length(dims) == total_projections || throw(ArgumentError("target_dims length must match total_projections"))

    # Validation
    k > 0 || throw(ArgumentError("k must be positive"))
    candidate_neighbors >= k || throw(ArgumentError("candidate_neighbors must be >= k"))
    n_early_projections <= total_projections || throw(ArgumentError("n_early_projections must be <= total_projections"))

    EnsembleConfig{Float64}(
        n_early_projections, candidate_neighbors, early_index_builder,
        total_projections, dims, projection_type, sparse_density,
        k, tie_breaking, symmetrization, symmetrization_threshold,
        random_seed, store_projections
    )
end

# Main API
"""
    build_ensemble_knn_graph(data::AbstractMatrix, config::EnsembleConfig;
                            return_votes=false) -> KNNGraph

Build consensus k-NN graph via multi-projection voting.

# Algorithm
1. Phase 1a: Generate candidate sets using n_early_projections full ANN searches
2. Phase 1b: Vote on edges within candidate sets across all M projections
3. Phase 1c: Select top-k neighbors by vote count with tie-breaking
4. Phase 1d: Symmetrize graph using specified strategy

# Returns
- If return_votes=false: KNNGraph (compatible with existing code)
- If return_votes=true: (KNNGraph, vote_counts::SparseMatrix{Int})

# Example
```julia
config = EnsembleConfig(
    k = 10,
    n_early_projections = 3,
    candidate_neighbors = 30,
    total_projections = 20,
    target_dims = 50
)

graph = build_ensemble_knn_graph(data, config)
```
"""
function build_ensemble_knn_graph(
    data::AbstractMatrix{T},
    config::EnsembleConfig;
    return_votes::Bool = false
) where {T<:AbstractFloat}
    # Implementation here
end

# Phase 2 API (Grassmannian averaging)
"""
    fit_grassmann_tangents(data::AbstractMatrix, graph::KNNGraph, projections, k_intrinsic) -> Vector{PCAGeometry}

Fit tangent planes using Grassmannian averaging across multiple projections.

Uses the consensus k-NN graph from Phase 1 to determine neighborhoods, then
averages tangent plane estimates across all projections.

# Arguments
- data: Original data matrix (D × n)
- graph: Consensus k-NN graph from build_ensemble_knn_graph
- projections: Projection data (Y_m matrices and Φ_m matrices) from Phase 1
- k_intrinsic: Intrinsic manifold dimension

# Returns
Vector of PCAGeometry objects (one per node), compatible with WeightedKNNGraph
"""
function fit_grassmann_tangents(
    data::AbstractMatrix{T},
    graph::KNNGraph,
    projections::Vector{ProjectionData},
    k_intrinsic::Int
) where {T<:AbstractFloat}
    # Implementation here
end
```

---

## Implementation Plan

**Note**: Priorities 1-2 (ensemble voting) and Priority 3 (Grassmannian averaging) are **independent tracks** and can be implemented in parallel or separately.

### Priority 1: Core Voting System (Independent Contribution #1)
- [ ] Implement Phase 1a: Candidate set construction
- [ ] Implement Phase 1b: Vote counting
- [ ] Implement Phase 1c: Top-k by vote count
- [ ] Implement Phase 1d: Symmetrization strategies
- [ ] Basic tests with synthetic data

**Deliverable**: Standalone ensemble voting module for k-NN graph construction

### Priority 2: Voting System Integration
- [ ] Ensure output is compatible with existing KNNGraph type
- [ ] Integration with WeightedKNNGraph for edge weights (using standard PCA tangents)
- [ ] Tests with existing geometry code (PCAMethod)
- [ ] Benchmarks vs standard k-NN construction

**Deliverable**: Production-ready ensemble voting ready for thesis experiments

### Priority 3: Grassmannian Averaging (Independent Contribution #2)
- [ ] Implement projection operator averaging (PP^T accumulation)
- [ ] Implement eigen-decomposition for consensus tangent
- [ ] API that accepts any k-NN graph (not just ensemble-voted)
- [ ] Integration with LocalGeometryEstimator interface
- [ ] Tests comparing to standard PCA on same k-NN graphs
- [ ] Benchmarks on synthetic manifolds (Swiss roll, S-curve)

**Deliverable**: Standalone Grassmannian averaging module for tangent estimation

### Priority 4: Combined Pipeline (Optional)
- [ ] Projection reuse: ensemble voting → Grassmannian averaging
- [ ] Combined benchmarks (ablation study: voting only, Grassmannian only, both)
- [ ] Compare to baseline (standard k-NN + standard PCA)

**Deliverable**: Full pipeline demonstrating synergy between contributions

### Priority 5: Optimization & Polish
- [ ] Memory optimization (store Y_m efficiently, sparse matrices)
- [ ] Parallel projection generation (multi-threading)
- [ ] Progress monitoring for long runs
- [ ] Documentation and examples for each configuration
- [ ] Thesis-ready plotting code for results

---

## Testing Strategy

### Unit Tests
- Candidate set construction (coverage of all neighbors)
- Vote counting (correctness, symmetry)
- Tie-breaking (determinism with fixed seed)
- Symmetrization (mutual, union, weighted correctness)

### Integration Tests
- Small synthetic manifolds (Swiss roll, S-curve)
- Compare to standard k-NN (recall, precision)
- Verify output works with WeightedKNNGraph

### Manifold Tests
- Generate data from known manifold
- Check that consensus graph has higher recall than single projection
- Verify Grassmannian averaging improves tangent estimates

### Benchmark Tests
- Timing: Phase 1a vs naive M full ANN builds
- Memory: Sparse vote storage
- Scalability: n = 1K, 10K, 100K

---

## Related Code Locations

### Existing Code to Integrate With
- `src/graphs/knn_graph.jl` - KNNGraph type and build_knn_graph()
- `src/graphs/weighted_knn_graph.jl` - WeightedKNNGraph for geodesic edge weights
- `src/geometry/pca.jl` - PCAMethod and PCAGeometry for tangent planes
- `src/preprocessing/RandomProjectionTransform.jl` - JL projections

### New Code to Create
- `src/ensemble/MultiProjectionGraph.jl` - Main implementation
- `src/ensemble/GrassmannianAveraging.jl` - Phase 2
- `test/unit/ensemble/` - Test suite
- `docs/examples/ensemble/` - Usage examples

---

## References & Background

### Johnson-Lindenstrauss (JL) Projections
- Standard: d = O(log n / ε²) for n points
- Preserves pairwise distances with high probability

### Oblivious Subspace Embeddings (OSE)
- Stronger: d = O(k / ε²) for k-dimensional subspaces
- Better for manifold data where intrinsic dimension k << log n
- Gaussian random matrices satisfy OSE properties

### Grassmann Manifold Gr(k, D)
- Space of k-dimensional linear subspaces of R^D
- Each tangent space is a point on Gr(k, D)
- Extrinsic average: mean of projection operators P P^T
- Geodesic average: Fréchet mean (more expensive)

### Ensemble Methods
- Majority voting improves robustness to noise
- Multiple projections capture different geometric aspects
- Similar to random forest idea but for graphs

---

## Future Extensions (Out of Scope for Initial Implementation)

- [ ] Weighted voting (higher confidence for some projections)
- [ ] Adaptive projection dimensions per point (data-dependent)
- [ ] Hierarchical voting (coarse → fine)
- [ ] Integration with other edge pruning (ORC, TSP-based)
- [ ] GPU acceleration for projection generation
- [ ] Distributed computation for large datasets
