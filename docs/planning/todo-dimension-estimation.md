# TODO: Graph-Based Dimension Estimation

This document tracks the implementation of various graph-based intrinsic dimensionality estimators for manifold learning.

## Motivation

Current capabilities:
- Local PCA-based dimension estimation (via eigenvalue gaps)
- Used in `LocalGeometryEstimator` for tangent plane fitting

**Goal:** Implement multiple dimension estimation methods to:
1. Compare different estimators on real datasets
2. Provide robust ensemble estimates
3. Validate against ground truth (synthetic manifolds)
4. Use for adaptive preprocessing and index tuning

---

## User's Priority Methods

These are the specific dimension estimators to implement:
1. **MLE (Levina & Bickel)** - Maximum likelihood from k-NN distances
2. **PCA Eigenvalue Gap** - Local tangent space analysis (from our experiments!)
3. **First/Second Neighbor Ratio** - Expansion dimension estimator
4. **Sum of Edge Weight Powers** - Graph-based scaling estimator
5. **MST Scaling** - Minimum spanning tree growth rate
6. **Correlation Dimension (Grassberger & Procaccia)** - Nice addition for comparison

---

## Priority: HIGH

### 1. Maximum Likelihood Estimator (MLE) - Levina & Bickel

**Reference:** Levina & Bickel (2005), "Maximum likelihood estimation of intrinsic dimension"
**Paper:** `/home/bram/Levina_Bickel_2005_MLE_Intrinsic_Dimension.pdf`

**Method:**
- For each point, get k nearest neighbors
- Estimate local dimension from k-NN distance distribution
- Assume neighbors uniformly distributed on m-dimensional manifold
- ML estimate: `m̂ = -k / Σᵢ log(rᵢ/rₖ)` where rᵢ is distance to i-th neighbor

**Tasks:**

#### 1.1 Implement Core MLE Estimator
- [ ] Create new file: `src/dimension/mle_estimator.jl`
- [ ] Implement `mle_dimension` function:
  ```julia
  """
      mle_dimension(index, data, point_idx, k; method=:averaging)

  Estimate intrinsic dimension at a point using Levina-Bickel MLE.

  # Arguments
  - `index`: ANN index for neighbor queries
  - `data`: data matrix (d × n)
  - `point_idx`: index of point to estimate dimension for
  - `k`: number of neighbors to use (typically 10-50)
  - `method`: :averaging (smooth over neighbors) or :single (just this point)

  # Returns
  - Estimated intrinsic dimension (Float64)
  """
  function mle_dimension(index, data, point_idx::Int, k::Int; method=:averaging)
      # Get k+1 nearest neighbors (includes self)
      neighbors = query(index, data, data[:, point_idx], k + 1)

      # Extract distances (skip self at distance 0)
      distances = [n.distance for n in neighbors[2:end]]

      # Avoid log(0) by adding small epsilon
      r_k = distances[end]
      ε = 1e-10

      # MLE formula: m̂ = -(k-1) / Σ log(rᵢ/rₖ)
      log_ratios = [log(max(r / r_k, ε)) for r in distances[1:end-1]]
      dimension = -(k - 1) / sum(log_ratios)

      return dimension
  end
  ```

#### 1.2 Implement Neighborhood Averaging
- [ ] Estimate dimension at multiple points, average locally
- [ ] Handle singularities (all neighbors at same distance)
- [ ] Add bias correction (Levina & Bickel discuss k-dependent bias)

#### 1.3 Implement Global Estimator
- [ ] Sample random points or use all points
- [ ] Aggregate local estimates (mean, median, trimmed mean)
- [ ] Return confidence intervals

**New file:** `src/dimension/mle_estimator.jl`

**Tests:**
- [ ] Test on Swiss roll (should estimate ~2)
- [ ] Test on sphere (should estimate ~2 on surface)
- [ ] Test on cube (should estimate ambient dimension)
- [ ] Compare vs ground truth on synthetic manifolds

**Test file:** `test/unit/dimension/mle_tests.jl`

---

## Priority: MEDIUM (Good Addition for Comparison)

### 7. Correlation Dimension - Grassberger & Procaccia

**Reference:** Grassberger & Procaccia (1983), "Measuring the strangeness of strange attractors"

**Method:**
- Count pairs of points within distance r
- Correlation sum: `C(r) = (1/n²) Σᵢⱼ I(‖xᵢ - xⱼ‖ < r)`
- For manifold: `C(r) ~ r^m` where m is dimension
- Estimate: `m̂ = d log C(r) / d log r`

**Tasks:**

#### 2.1 Implement Correlation Dimension
- [ ] Create file: `src/dimension/correlation_dimension.jl`
- [ ] Compute correlation sum for multiple scales r
- [ ] Estimate dimension from log-log slope
- [ ] Use linear regression in scaling region

```julia
"""
    correlation_dimension(data; r_min=0.1, r_max=1.0, n_scales=20)

Estimate intrinsic dimension using correlation dimension.
"""
function correlation_dimension(data::AbstractMatrix;
                                r_min=0.1, r_max=1.0, n_scales=20)
    n = size(data, 2)
    scales = 10 .^ range(log10(r_min), log10(r_max), length=n_scales)

    correlation_sums = zeros(n_scales)

    for (i, r) in enumerate(scales)
        count = 0
        for j in 1:n, k in (j+1):n
            dist = norm(data[:, j] - data[:, k])
            if dist < r
                count += 1
            end
        end
        correlation_sums[i] = 2 * count / (n * (n - 1))
    end

    # Fit line in log-log space
    log_r = log10.(scales)
    log_C = log10.(correlation_sums .+ 1e-10)

    # Find scaling region (where log-log is approximately linear)
    dimension = estimate_slope(log_r, log_C)

    return dimension
end
```

#### 2.2 Optimize for Large Datasets
- [ ] Use spatial indexing (KDTree) for range queries
- [ ] Sample subset of points if n is large
- [ ] Parallelize pairwise distance computation

#### 2.3 Automatic Scaling Region Detection
- [ ] Identify linear region in log-log plot automatically
- [ ] Avoid boundary effects (too small/large r)
- [ ] Return quality metric for fit

**New file:** `src/dimension/correlation_dimension.jl`

---

## Priority: HIGH

### 3. First/Second Neighbor Ratio Estimator

**Reference:** Related to expansion dimension / doubling dimension estimators
**Idea:** On an m-dimensional manifold, the ratio of second to first neighbor distance follows a specific distribution that depends on m.

**Method:**
- For each point, compute distances to 1st and 2nd nearest neighbors: r₁, r₂
- The ratio r₂/r₁ has expected value that scales with dimension
- For uniform distribution on m-dim manifold: E[r₂/r₁] ≈ 2^(1/m)
- Invert to estimate m: m̂ ≈ log(2) / log(E[r₂/r₁])

**Tasks:**

#### 3.1 Implement Ratio Estimator
- [ ] Create file: `src/dimension/ratio_estimator.jl`
- [ ] For each sampled point, get r₁ and r₂
- [ ] Compute ratio r₂/r₁
- [ ] Estimate dimension from mean ratio

```julia
"""
    ratio_dimension(index, data; sample_size=500)

Estimate intrinsic dimension using first/second neighbor distance ratio.

For a uniform distribution on an m-dimensional manifold:
    E[r₂/r₁] ≈ 2^(1/m)

Therefore: m̂ = log(2) / log(mean(r₂/r₁))

# Arguments
- `index`: ANN index for neighbor queries
- `data`: data matrix (d × n)
- `sample_size`: number of points to sample for estimation

# Returns
- Estimated intrinsic dimension
"""
function ratio_dimension(index, data; sample_size=500)
    n = size(data, 2)
    sample_indices = shuffle(1:n)[1:min(sample_size, n)]

    ratios = Float64[]

    for idx in sample_indices
        # Get 3 nearest neighbors (includes self)
        neighbors = query(index, data, data[:, idx], 3)

        # Extract distances to 1st and 2nd neighbors (skip self)
        r1 = neighbors[2].distance
        r2 = neighbors[3].distance

        # Avoid division by zero
        if r1 > 1e-10
            push!(ratios, r2 / r1)
        end
    end

    # Estimate dimension
    mean_ratio = mean(ratios)

    if mean_ratio > 1.0
        dimension = log(2) / log(mean_ratio)
    else
        dimension = Inf  # Invalid case
    end

    return dimension
end
```

#### 3.2 Improve with Multiple Neighbor Ratios
- [ ] Use not just r₂/r₁ but also r₃/r₂, r₄/r₃, etc.
- [ ] Average over multiple ratios for robustness
- [ ] Weight by reliability (closer neighbors more reliable)

#### 3.3 Theoretical Refinement
- [ ] Account for non-uniform density on manifold
- [ ] Add correction factors for boundary effects
- [ ] Use median instead of mean for robustness to outliers

**New file:** `src/dimension/ratio_estimator.jl`

**Benefits:**
- Very fast (only need 2-3 nearest neighbors!)
- Simple implementation
- Good for quick dimension estimates

**Tests:**
- [ ] Swiss roll should give ~2
- [ ] Check sensitivity to noise
- [ ] Compare to MLE on same data

---

## Priority: HIGH

### 4. Sum of Edge Weight Powers (Graph-Based)

**Reference:** Geometric measure theory / fractal dimension
**Idea:** For a graph on an m-dimensional manifold, the sum of k-NN edge weights raised to power m should scale in a specific way with n.

**Method:**
- Build k-NN graph with n points on m-dim manifold
- Sum of edge lengths to power m: S(m, n) = Σᵢⱼ w(i,j)^m
- For true dimension m: S(m, n) ~ n^((m-m)/m) = constant
- For wrong dimension: S grows or shrinks with n
- Estimate m by finding power where S(m, n) stabilizes

**Alternative formulation:**
- Try multiple values of m
- For each m, compute: log(Σ wᵢⱼᵐ)
- True dimension minimizes the rate of change with respect to n

**Tasks:**

#### 4.1 Implement Edge Weight Power Sum
- [ ] Create file: `src/dimension/edge_power_sum.jl`
- [ ] Build k-NN graph
- [ ] For candidate dimensions m ∈ [1, 2, ..., d_max], compute sum of w^m
- [ ] Find m where sum is most stable (or use optimization)

```julia
"""
    edge_power_dimension(index, data, k;
                         max_dim=10, sample_fraction=1.0)

Estimate dimension using sum of edge weight powers on k-NN graph.

The idea: For true dimension m, the sum Σ wᵢⱼᵐ over k-NN edges
should have specific scaling properties.

# Arguments
- `index`: ANN index
- `data`: data matrix
- `k`: number of neighbors for graph
- `max_dim`: maximum dimension to test
- `sample_fraction`: fraction of points to use (for large datasets)

# Returns
- Estimated dimension (minimizes score function)
"""
function edge_power_dimension(index, data, k;
                              max_dim=10, sample_fraction=1.0)
    n = size(data, 2)
    n_sample = round(Int, n * sample_fraction)
    sample_indices = shuffle(1:n)[1:n_sample]

    # Collect all k-NN edge weights for sampled points
    edge_weights = Float64[]

    for idx in sample_indices
        neighbors = query(index, data, data[:, idx], k + 1)
        for neighbor in neighbors[2:end]  # Skip self
            push!(edge_weights, neighbor.distance)
        end
    end

    # Test different dimensions
    scores = Float64[]

    for m in 1:max_dim
        # Compute sum of w^m
        power_sum = sum(w^m for w in edge_weights)

        # Normalize by number of edges
        normalized_sum = power_sum / length(edge_weights)

        # Score: how much does this vary? (heuristic)
        # Better formulation: compare across different subsamples
        score = normalized_sum
        push!(scores, score)
    end

    # Find dimension with "elbow" in scores
    # Simple heuristic: maximum second derivative
    dimension = find_elbow(scores)

    return dimension
end

function find_elbow(values)
    # Compute second derivative (discrete)
    second_deriv = diff(diff(values))
    # Find maximum curvature
    return argmax(abs.(second_deriv)) + 1
end
```

#### 4.2 Improve with Subsample Comparison
- [ ] Instead of single sample, use multiple subsamples of different sizes
- [ ] For each m, check how S(m, n) scales with n
- [ ] True m: S(m, n) / n should be approximately constant
- [ ] Plot log(S(m, n)) vs log(n) for different m, find linear region

#### 4.3 Weighted Version
- [ ] Weight edges by local density (inverse of local density)
- [ ] Account for non-uniform sampling on manifold
- [ ] Use geodesic edge weights instead of Euclidean

**New file:** `src/dimension/edge_power_sum.jl`

**Tests:**
- [ ] Swiss roll with varying n (1k, 5k, 10k, 50k)
- [ ] Check scaling behavior
- [ ] Compare to other methods

---

## Priority: HIGH

### 5. MST Scaling Estimator

**Reference:** Fractal dimension / minimal spanning tree growth
**Idea:** The total weight of a minimum spanning tree (MST) on a manifold scales with dimension.

**Method:**
- For n points uniformly distributed on an m-dimensional manifold with diameter D:
  - E[MST weight] ~ D · n^((m-1)/m)
- Taking logs: log(W_MST) ≈ log(D) + ((m-1)/m) · log(n)
- Estimate m from slope of log(W_MST) vs log(n)
- Solve: slope = (m-1)/m → m = 1 / (1 - slope)

**Tasks:**

#### 5.1 Implement MST Scaling
- [ ] Create file: `src/dimension/mst_scaling.jl`
- [ ] Build MST using Kruskal's or Prim's algorithm
- [ ] Compute total MST weight
- [ ] Estimate dimension from scaling

```julia
"""
    mst_dimension(data; subsamples=[100, 200, 500, 1000, 2000])

Estimate intrinsic dimension using MST weight scaling.

Builds MSTs on subsamples of increasing size and estimates dimension
from the scaling of total MST weight with sample size.

For m-dimensional manifold: W_MST ~ n^((m-1)/m)
Therefore: m = 1 / (1 - slope) where slope = d(log W_MST)/d(log n)

# Arguments
- `data`: data matrix (d × n)
- `subsamples`: vector of subsample sizes to test

# Returns
- Estimated dimension from MST scaling
- (Optional) scaling plot data for visualization
"""
function mst_dimension(data; subsamples=[100, 200, 500, 1000, 2000])
    n = size(data, 2)

    log_n = Float64[]
    log_weights = Float64[]

    for n_sub in subsamples
        n_sub > n && break

        # Random subsample
        sample_idx = shuffle(1:n)[1:n_sub]
        data_sub = data[:, sample_idx]

        # Build MST
        mst_weight = compute_mst_weight(data_sub)

        push!(log_n, log(n_sub))
        push!(log_weights, log(mst_weight))
    end

    # Linear regression: log(W) = a + b·log(n)
    # Slope b ≈ (m-1)/m
    slope = estimate_slope(log_n, log_weights)

    # Solve for m
    if slope < 1.0 && slope > 0.0
        dimension = 1 / (1 - slope)
    else
        dimension = NaN  # Invalid slope
    end

    return dimension, (log_n=log_n, log_weights=log_weights, slope=slope)
end

"""
    compute_mst_weight(data)

Compute total weight of minimum spanning tree.
"""
function compute_mst_weight(data::AbstractMatrix)
    n = size(data, 2)

    # Build complete graph with Euclidean distances
    # Use Kruskal's algorithm
    edges = []
    for i in 1:n
        for j in (i+1):n
            dist = norm(data[:, i] - data[:, j])
            push!(edges, (i, j, dist))
        end
    end

    # Sort by weight
    sort!(edges, by = e -> e[3])

    # Kruskal's algorithm with union-find
    parent = collect(1:n)

    function find(x)
        if parent[x] != x
            parent[x] = find(parent[x])
        end
        return parent[x]
    end

    function union!(x, y)
        px, py = find(x), find(y)
        if px != py
            parent[px] = py
            return true
        end
        return false
    end

    mst_weight = 0.0
    edges_added = 0

    for (i, j, w) in edges
        if union!(i, j)
            mst_weight += w
            edges_added += 1
            edges_added == n - 1 && break
        end
    end

    return mst_weight
end

function estimate_slope(x, y)
    # Simple linear regression
    n = length(x)
    x_mean = mean(x)
    y_mean = mean(y)

    slope = sum((x[i] - x_mean) * (y[i] - y_mean) for i in 1:n) /
            sum((x[i] - x_mean)^2 for i in 1:n)

    return slope
end
```

#### 5.2 Optimize for Large Datasets
- [ ] For large n, computing full MST is expensive (O(n² log n))
- [ ] Use approximate MST from k-NN graph
- [ ] Or use well-separated pair decomposition for faster MST

#### 5.3 Robustness Improvements
- [ ] Multiple random subsamples at each size (average results)
- [ ] Use robust regression (RANSAC or Theil-Sen) for slope estimation
- [ ] Check R² of fit to validate linear scaling

**New file:** `src/dimension/mst_scaling.jl`

**Benefits:**
- Theoretically well-founded
- Works well for uniformly distributed data
- Less sensitive to local density variations than k-NN methods

**Tests:**
- [ ] Swiss roll with varying n
- [ ] Check if slope is consistent
- [ ] Compare to other methods

---

## Priority: HIGH (moved up from MEDIUM)

### 6. Local PCA / Eigenvalue Gap Method

**Reference:** Fan et al. (2009), "Intrinsic dimension estimation of data by principal component analysis"

**Method:**
- Same as our experiments! (`scripts/tangent_plane_eigenvalue_*.jl`)
- Fit local PCA, measure eigenvalue gap λₘ / λₘ₊₁
- Large gap indicates m-dimensional manifold

**Tasks:**

#### 3.1 Formalize as Dimension Estimator
- [ ] Create file: `src/dimension/pca_gap_estimator.jl`
- [ ] Extract code from our experiments
- [ ] Implement automatic dimension detection:
  - Threshold on eigenvalue gap (e.g., gap > 2.0)
  - Threshold on explained variance (e.g., 90%)
  - Elbow detection in scree plot

```julia
"""
    pca_gap_dimension(index, data, point_idx, k_max;
                      gap_threshold=1.5, var_threshold=0.90)

Estimate dimension by detecting eigenvalue gap in local PCA.
"""
function pca_gap_dimension(index, data, point_idx, k_max;
                           gap_threshold=1.5, var_threshold=0.90)
    # Get neighbors
    neighbors = query(index, data, data[:, point_idx], k_max + 1)
    neighbor_indices = [n.id for n in neighbors[2:end]]

    # Local PCA
    neighborhood = data[:, neighbor_indices]
    centered = neighborhood .- mean(neighborhood, dims=2)
    cov = (centered * centered') / (k_max - 1)

    eigenvalues = sort(eigvals(Symmetric(cov)), rev=true)

    # Method 1: Find first large gap
    gaps = eigenvalues[1:end-1] ./ eigenvalues[2:end]
    for (i, gap) in enumerate(gaps)
        if gap > gap_threshold
            return i
        end
    end

    # Method 2: Explained variance threshold
    total_var = sum(eigenvalues)
    cumulative_var = cumsum(eigenvalues) / total_var
    for (i, var) in enumerate(cumulative_var)
        if var > var_threshold
            return i
        end
    end

    return length(eigenvalues)  # fallback
end
```

#### 3.2 Adaptive k Selection
- [ ] Try multiple k values (e.g., k ∈ [5, 10, 20, 50])
- [ ] Use consistency across scales to improve estimate
- [ ] Return confidence based on agreement across k

#### 3.3 Integration with Random Projection Analysis
- [ ] Use insights from our experiments (k vs ambient dimension)
- [ ] Adjust k based on ambient dimension: `k ~ 0.5 * d to 1.0 * d`
- [ ] Warn if eigenvalue gap is weak (ambiguous dimension)

**New file:** `src/dimension/pca_gap_estimator.jl`

---

## Priority: LOW (Advanced, Optional)

### 8. Multiscale SVD - Little et al.

**Reference:** Little et al. (2009), "Estimation of intrinsic dimensionality of samples from noisy low-dimensional manifolds in high dimensions with multiscale SVD"
**Paper:** `/home/bram/Little_et_al_2009_Multiscale_SVD.pdf`

**Method:**
- Analyze singular values at multiple neighborhood scales
- More robust to noise than single-scale PCA
- Detect dimension where singular values stabilize across scales

**Tasks:**

#### 4.1 Implement Multiscale SVD
- [ ] Create file: `src/dimension/multiscale_svd.jl`
- [ ] Compute SVD for neighborhoods at scales k₁, k₂, ..., kₘ
- [ ] Track how eigenvalues change with scale
- [ ] Intrinsic dimension: where eigenvalues plateau

```julia
"""
    multiscale_svd_dimension(index, data, point_idx;
                             scales=[5, 10, 20, 50, 100])

Estimate dimension using multiscale SVD analysis.
"""
function multiscale_svd_dimension(index, data, point_idx;
                                  scales=[5, 10, 20, 50, 100])
    eigenvalue_matrix = []  # Each row = eigenvalues at one scale

    for k in scales
        neighbors = query(index, data, data[:, point_idx], k + 1)
        neighbor_indices = [n.id for n in neighbors[2:end]]

        neighborhood = data[:, neighbor_indices]
        centered = neighborhood .- mean(neighborhood, dims=2)

        # SVD (more stable than eigenvalue decomposition)
        U, S, V = svd(centered)
        eigenvalues = (S .^ 2) / (k - 1)

        push!(eigenvalue_matrix, eigenvalues)
    end

    # Analyze stability across scales
    # Dimension = index where eigenvalues start varying significantly
    dimension = detect_stable_subspace(eigenvalue_matrix, scales)

    return dimension
end
```

#### 4.2 Noise Robustness
- [ ] Implement noise variance estimation (from smallest eigenvalues)
- [ ] Subtract noise floor from signal eigenvalues
- [ ] More accurate for high-dimensional noisy data

**New file:** `src/dimension/multiscale_svd.jl`

---

## Priority: LOW (Future Extensions)

### 9. Additional Graph-Based Methods

#### 5.1 Geodesic Distance-Based Method
- [ ] Use our geodesic distance infrastructure!
- [ ] Idea: Intrinsic dimension relates to graph path length vs Euclidean distance
- [ ] For m-dim manifold: `geodesic_dist / euclidean_dist ~ O(1)` for nearby points
- [ ] For high curvature: ratio increases

**New file:** `src/dimension/geodesic_based.jl`

#### 5.2 Graph Laplacian Spectrum
- [ ] Compute graph Laplacian from k-NN graph
- [ ] Eigenvalue decay rate indicates dimension
- [ ] Reference: Belkin & Niyogi (2003) Laplacian Eigenmaps

#### 5.3 Heat Kernel Methods
- [ ] Estimate dimension from heat kernel trace
- [ ] Reference: Bérard et al. (1994)
- [ ] More advanced, lower priority

---

## Priority: HIGH

### 2. Ensemble Estimator

**Updated to use your specific methods!**

**Motivation:** Different methods have different strengths/weaknesses

**Tasks:**

#### 6.1 Implement Ensemble
- [ ] Create file: `src/dimension/ensemble_estimator.jl`
- [ ] Run multiple estimators (MLE, PCA gap, correlation dim, etc.)
- [ ] Aggregate results:
  - Median (robust to outliers)
  - Weighted average (weight by confidence)
  - Voting (round to integer)
- [ ] Return uncertainty estimate (std dev across methods)

```julia
"""
    ensemble_dimension(index, data;
                       methods=[:mle, :pca_gap, :ratio, :edge_power, :mst],
                       aggregation=:median,
                       k=20)

Estimate dimension using ensemble of your specified methods.

# Methods
- :mle - Levina & Bickel maximum likelihood
- :pca_gap - Eigenvalue gap analysis
- :ratio - First/second neighbor distance ratio
- :edge_power - Sum of edge weight powers
- :mst - MST scaling
- :correlation - Grassberger & Procaccia (optional)

# Returns
- (estimate, individual_estimates, confidence)
"""
function ensemble_dimension(index, data;
                            methods=[:mle, :pca_gap, :ratio, :edge_power, :mst],
                            aggregation=:median,
                            k=20)
    estimates = Dict{Symbol, Float64}()

    # Run each method
    :mle in methods && (estimates[:mle] = mle_dimension(index, data; k=k))
    :pca_gap in methods && (estimates[:pca_gap] = pca_gap_dimension(index, data; k=k))
    :ratio in methods && (estimates[:ratio] = ratio_dimension(index, data))
    :edge_power in methods && (estimates[:edge_power] = edge_power_dimension(index, data, k))
    :mst in methods && (estimates[:mst] = mst_dimension(data)[1])
    :correlation in methods && (estimates[:correlation] = correlation_dimension(data))

    # Filter out invalid estimates (NaN, Inf)
    valid_estimates = filter(x -> isfinite(x.second), estimates)

    if isempty(valid_estimates)
        return (estimate=NaN, individual=estimates, confidence=0.0)
    end

    # Aggregate
    vals = collect(values(valid_estimates))
    estimate = if aggregation == :median
        median(vals)
    elseif aggregation == :mean
        mean(vals)
    else
        mode(round.(Int, vals))
    end

    # Confidence: inverse of standard deviation (normalized)
    confidence = length(vals) > 1 ? 1.0 / (1.0 + std(vals)) : 0.5

    return (estimate=estimate, individual=estimates, confidence=confidence)
end
```

#### 6.2 Confidence Scoring
- [ ] Compute agreement across methods
- [ ] High confidence: all methods agree
- [ ] Low confidence: methods disagree (flag for investigation)

**New file:** `src/dimension/ensemble_estimator.jl`

---

## Integration with Existing Code

### Where to Use Dimension Estimation

1. **Adaptive Preprocessing**
   - [ ] Auto-select PCA target dimension based on estimated intrinsic dim
   - [ ] In `src/preprocessing/pca_transform.jl`: use `m̂` instead of hardcoded value

2. **Index Parameter Tuning**
   - [ ] For LSH: number of hash functions depends on dimension
   - [ ] For HNSW: M parameter could adapt to intrinsic dimension
   - [ ] For IVF: number of clusters ~ n^(1/m̂)

3. **Graph Construction**
   - [ ] Adaptive k for k-NN graph: `k ~ 2 * m̂` to `5 * m̂`
   - [ ] In `build_weighted_graph`: auto-select k if not specified

4. **Validation**
   - [ ] Report estimated dimension in examples
   - [ ] Warning if dimension estimate is unstable
   - [ ] Use in benchmark metadata

### API Design

```julia
# Top-level interface in src/dimension/estimators.jl

abstract type DimensionEstimator end

struct MLEEstimator <: DimensionEstimator
    k::Int
end

struct PCAGapEstimator <: DimensionEstimator
    k::Int
    gap_threshold::Float64
end

# ... etc

"""
    estimate_dimension(estimator, index, data, point_idx)

Estimate intrinsic dimension at a specific point.
"""
function estimate_dimension(estimator::DimensionEstimator,
                            index, data, point_idx::Int)
    # Dispatch to specific method
end

"""
    estimate_global_dimension(estimator, index, data;
                              sample_size=500)

Estimate global intrinsic dimension by sampling.
"""
function estimate_global_dimension(estimator::DimensionEstimator,
                                   index, data; sample_size=500)
    n = size(data, 2)
    sample = shuffle(1:n)[1:min(sample_size, n)]

    dims = [estimate_dimension(estimator, index, data, i) for i in sample]

    return (
        mean = mean(dims),
        median = median(dims),
        std = std(dims),
        estimates = dims
    )
end
```

---

## Validation & Benchmarking

### Synthetic Manifolds for Testing

- [ ] Swiss roll (m=2 in 3D)
- [ ] Sphere (m=2 in 3D)
- [ ] Torus (m=2 in 3D)
- [ ] Random manifold (varying m in high-D)
- [ ] Noisy manifolds (signal + Gaussian noise)

### Metrics

- [ ] Absolute error: `|m̂ - m_true|`
- [ ] Relative error
- [ ] Consistency across multiple runs
- [ ] Runtime
- [ ] Robustness to noise

### Benchmark Suite

**New file:** `benchmarks/dimension_estimation/benchmark.jl`

```julia
using ManifoldANN
using BenchmarkTools

# Test on Swiss roll
n_points = 5000
data, params = generate_swiss_roll(n_points)

index = build_index(BruteForceIndex, data)

# MLE
@btime estimate_global_dimension(MLEEstimator(k=20), $index, $data)

# PCA Gap
@btime estimate_global_dimension(PCAGapEstimator(k=50, gap_threshold=1.5), $index, $data)

# etc.
```

---

## Documentation

### Examples

**New file:** `docs/examples/dimension/01-estimate-dimension.jl`

```julia
#=
Example: Estimating Intrinsic Dimension

This example shows how to estimate the intrinsic dimension of a manifold
using various methods.
=#

using ManifoldANN

# Generate Swiss roll (2D manifold in 3D)
data, params = generate_swiss_roll(1000)

# Build index for neighbor queries
index = build_index(BruteForceIndex, data)

# Method 1: MLE (Levina-Bickel)
mle_est = estimate_global_dimension(MLEEstimator(k=20), index, data)
println("MLE estimate: $(round(mle_est.mean, digits=2)) ± $(round(mle_est.std, digits=2))")

# Method 2: PCA Gap
pca_est = estimate_global_dimension(PCAGapEstimator(k=50), index, data)
println("PCA Gap estimate: $(round(pca_est.median, digits=2))")

# Method 3: Ensemble
ensemble_est = estimate_global_dimension(
    EnsembleEstimator(methods=[:mle, :pca_gap, :correlation]),
    index, data
)
println("Ensemble estimate: $(round(ensemble_est.mean, digits=2))")
```

### API Documentation

- [ ] Add docstrings to all estimator types
- [ ] Document when to use each method
- [ ] Add references to papers
- [ ] Include complexity analysis (time/space)

---

## Testing

### Unit Tests

For each estimator:
- [ ] Test on known manifolds (Swiss roll, sphere)
- [ ] Test edge cases (d=1, d=ambient, empty data)
- [ ] Test numerical stability (nearly collinear points)
- [ ] Test with different k values

### Integration Tests

- [ ] Test interaction with preprocessing
- [ ] Test with different index types
- [ ] Test on real datasets (MNIST, etc.)

---

## Timeline

### Phase 1: Your Core Methods (3 weeks)
Week 1:
- [ ] MLE estimator (Levina & Bickel)
- [ ] PCA gap estimator (from our experiments)
- [ ] Basic tests on Swiss roll

Week 2:
- [ ] Ratio estimator (first/second neighbor)
- [ ] Edge power sum estimator
- [ ] MST scaling estimator

Week 3:
- [ ] Ensemble estimator combining all 5 methods
- [ ] Comprehensive testing
- [ ] Parameter tuning (k selection, etc.)

### Phase 2: Optional Extensions (1-2 weeks)
- [ ] Correlation dimension (Grassberger & Procaccia)
- [ ] Multiscale SVD (if needed for noise robustness)
- [ ] Additional graph-based methods

### Phase 3: Integration (1 week)
- [ ] Connect to preprocessing
- [ ] Adaptive parameter selection
- [ ] Documentation

### Phase 4: Validation (1 week)
- [ ] Comprehensive benchmarks
- [ ] Paper reproduction (validate against published results)
- [ ] Real dataset evaluation

---

## Success Criteria

- [ ] All 5 core estimators implemented (MLE, PCA gap, ratio, edge power, MST)
- [ ] Ensemble method combining all estimators
- [ ] Accurate on Swiss roll (error < 0.5 dimensions)
- [ ] Tested on multiple manifolds (sphere, torus, etc.)
- [ ] Robust to noise (tested with σ = 0.1, 0.5, 1.0)
- [ ] Robust to varying ambient dimensions (from our experiments: 5D, 10D, 20D, 50D)
- [ ] Fast enough for interactive use:
  - Ratio: < 0.1s for n=1000
  - MLE, PCA gap, Edge power: < 1s for n=1000
  - MST: < 5s for n=1000
- [ ] Well-documented with examples
- [ ] Integrated with existing workflow
- [ ] Can auto-select dimension for PCA preprocessing

---

## References

1. Levina & Bickel (2005) - MLE method
   - PDF: `/home/bram/Levina_Bickel_2005_MLE_Intrinsic_Dimension.pdf`

2. Little et al. (2009) - Multiscale SVD
   - PDF: `/home/bram/Little_et_al_2009_Multiscale_SVD.pdf`

3. Carter et al. (2010) - Survey of methods
   - PDF: `/home/bram/Carter_et_al_2010_Local_Intrinsic_Dimension.pdf`

4. Fan et al. (2009) - PCA-based methods
   - arXiv: https://arxiv.org/abs/0908.3417

5. Grassberger & Procaccia (1983) - Correlation dimension
   - Classic reference

6. Our experiments:
   - `scripts/tangent_plane_eigenvalue_study.jl`
   - `scripts/tangent_plane_eigenvalue_repeated.jl`
   - `scripts/dimensionality_effect_on_k.jl`

---

**Last Updated:** 2025-12-13
**Status:** Planning phase
**Priority:** HIGH (core functionality for manifold learning)
