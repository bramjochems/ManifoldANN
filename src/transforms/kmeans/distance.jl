"""
Distance computation utilities for KMeans clustering.
"""

using Distances
using LinearAlgebra

"""
    pairwise_distances!(D::Matrix, X::Matrix, centroids::Matrix, distance::SemiMetric)

Compute pairwise distances between data points (columns of X) and centroids.

# Arguments
- `D`: Pre-allocated distance matrix of size (k, n) where k = number of centroids, n = number of points
- `X`: Data matrix (d × n) where each column is a data point
- `centroids`: Centroid matrix (d × k) where each column is a centroid
- `distance`: Distance function from Distances.jl

# Notes
- D[i, j] = distance(centroids[:, i], X[:, j])
- Uses BLAS-optimized vectorization for Euclidean distances
- Falls back to multi-threaded loops for other distance metrics

# Performance
- Euclidean: ~50-100x faster than naive loops (BLAS GEMM)
- Other metrics: ~10x faster (multi-threaded)
"""
function pairwise_distances!(D::Matrix, X::Matrix, centroids::Matrix, distance::SemiMetric)
    k, n = size(D)
    d = size(X, 1)

    @assert size(centroids) == (d, k) "Centroids must be d × k"
    @assert size(X) == (d, n) "Data must be d × n"

    # Fast path: BLAS-optimized for Euclidean distance
    if distance isa Euclidean
        pairwise_euclidean!(D, X, centroids)
    elseif distance isa SqEuclidean
        pairwise_sqeuclidean!(D, X, centroids)
    else
        # Fallback: multi-threaded for other distance metrics
        pairwise_generic!(D, X, centroids, distance)
    end

    return D
end

"""
    pairwise_euclidean!(D::Matrix, X::Matrix, centroids::Matrix)

BLAS-optimized Euclidean distance computation using the identity:
    ||a - b||² = ||a||² + ||b||² - 2(a·b)

This uses highly optimized GEMM (matrix multiplication) from BLAS, which is
50-100x faster than naive loops and already multi-threaded.

# Arguments
- `D`: Pre-allocated distance matrix of size (k, n)
- `X`: Data matrix (d × n)
- `centroids`: Centroid matrix (d × k)

# Implementation
1. Compute ||centroid_i||² for all i (k operations)
2. Compute ||x_j||² for all j (n operations)
3. Compute centroids' * X using BLAS GEMM (the expensive part, but highly optimized)
4. Combine: D[i,j] = sqrt(||c_i||² + ||x_j||² - 2*dot(c_i, x_j))
"""
function pairwise_euclidean!(D::Matrix{T}, X::Matrix{T}, centroids::Matrix{T}) where {T}
    k, n = size(D)
    d = size(X, 1)

    # Compute squared norms of centroids (k values)
    centroid_sq_norms = vec(sum(abs2, centroids; dims=1))  # 1 × k -> k

    # Compute squared norms of data points (n values)
    data_sq_norms = vec(sum(abs2, X; dims=1))  # 1 × n -> n

    # Compute dot products: centroids' * X using BLAS GEMM
    # Result is k × n matrix where result[i,j] = dot(centroids[:,i], X[:,j])
    mul!(D, centroids', X)  # BLAS GEMM - highly optimized!

    # Combine: ||a-b||² = ||a||² + ||b||² - 2(a·b)
    # Then take sqrt for Euclidean distance
    @inbounds for j in 1:n
        data_norm_sq = data_sq_norms[j]
        for i in 1:k
            sq_dist = centroid_sq_norms[i] + data_norm_sq - 2 * D[i, j]
            # Clamp to avoid sqrt of negative due to floating point errors
            D[i, j] = sqrt(max(sq_dist, zero(T)))
        end
    end

    return D
end

"""
    pairwise_sqeuclidean!(D::Matrix, X::Matrix, centroids::Matrix)

BLAS-optimized squared Euclidean distance (same as pairwise_euclidean! but without sqrt).
"""
function pairwise_sqeuclidean!(D::Matrix{T}, X::Matrix{T}, centroids::Matrix{T}) where {T}
    k, n = size(D)
    d = size(X, 1)

    # Compute squared norms
    centroid_sq_norms = vec(sum(abs2, centroids; dims=1))
    data_sq_norms = vec(sum(abs2, X; dims=1))

    # Compute dot products using BLAS GEMM
    mul!(D, centroids', X)

    # Combine: ||a-b||² = ||a||² + ||b||² - 2(a·b)
    @inbounds for j in 1:n
        data_norm_sq = data_sq_norms[j]
        for i in 1:k
            sq_dist = centroid_sq_norms[i] + data_norm_sq - 2 * D[i, j]
            D[i, j] = max(sq_dist, zero(T))  # Clamp to avoid negative
        end
    end

    return D
end

"""
    pairwise_generic!(D::Matrix, X::Matrix, centroids::Matrix, distance::SemiMetric)

Multi-threaded distance computation for non-Euclidean metrics.

Parallelizes over centroids for good load balancing and cache locality.
"""
function pairwise_generic!(D::Matrix, X::Matrix, centroids::Matrix, distance::SemiMetric)
    k, n = size(D)

    # Parallel over centroids (outer loop)
    Threads.@threads for i in 1:k
        centroid = view(centroids, :, i)
        for j in 1:n
            D[i, j] = evaluate(distance, centroid, view(X, :, j))
        end
    end

    return D
end

"""
    compute_distances(x::AbstractVector, centroids::Matrix, distance::SemiMetric)::Vector

Compute distances from a single point to all centroids.

# Arguments
- `x`: Data point (d-dimensional vector)
- `centroids`: Centroid matrix (d × k) where each column is a centroid
- `distance`: Distance function from Distances.jl

# Returns
- Vector of length k containing distances to each centroid

# Performance
- Uses BLAS-optimized GEMV for Euclidean distances (query-time critical path)
- Falls back to loop for other metrics
"""
function compute_distances(
    x::AbstractVector,
    centroids::Matrix,
    distance::SemiMetric,
    centroid_norms::Union{Nothing, AbstractVector}=nothing,
)
    # Fast path for Euclidean distance
    if distance isa Euclidean
        return compute_distances_euclidean(x, centroids, centroid_norms)
    elseif distance isa SqEuclidean
        return compute_distances_sqeuclidean(x, centroids, centroid_norms)
    end

    # Generic fallback
    k = size(centroids, 2)
    distances = Vector{eltype(centroids)}(undef, k)

    for i in 1:k
        distances[i] = evaluate(distance, view(centroids, :, i), x)
    end

    return distances
end

"""
    compute_distances_euclidean(x::AbstractVector, centroids::Matrix)

BLAS-optimized Euclidean distance for a single query point to all centroids.
Uses GEMV (matrix-vector multiply) for optimal performance.
"""
function compute_distances_euclidean(
    x::AbstractVector{T},
    centroids::Matrix{T},
    centroid_norms::Union{Nothing, AbstractVector{T}}=nothing,
) where {T}
    k = size(centroids, 2)
    distances = Vector{T}(undef, k)

    # Compute ||x||²
    x_norm_sq = sum(abs2, x)

    # Compute centroids' * x using BLAS GEMV
    # This computes all dot products at once: [c1·x, c2·x, ..., ck·x]
    dots = centroids' * x  # BLAS GEMV - very fast!

    # Compute distances: ||ci - x||² = ||ci||² + ||x||² - 2(ci·x)
    @inbounds for i in 1:k
        centroid_norm_sq = centroid_norms === nothing ? sum(abs2, view(centroids, :, i)) : centroid_norms[i]
        sq_dist = centroid_norm_sq + x_norm_sq - 2 * dots[i]
        distances[i] = sqrt(max(sq_dist, zero(T)))
    end

    return distances
end

"""
    compute_distances_sqeuclidean(x::AbstractVector, centroids::Matrix)

BLAS-optimized squared Euclidean distance for a single query point.
"""
function compute_distances_sqeuclidean(
    x::AbstractVector{T},
    centroids::Matrix{T},
    centroid_norms::Union{Nothing, AbstractVector{T}}=nothing,
) where {T}
    k = size(centroids, 2)
    distances = Vector{T}(undef, k)

    x_norm_sq = sum(abs2, x)
    dots = centroids' * x  # BLAS GEMV

    @inbounds for i in 1:k
        centroid_norm_sq = centroid_norms === nothing ? sum(abs2, view(centroids, :, i)) : centroid_norms[i]
        sq_dist = centroid_norm_sq + x_norm_sq - 2 * dots[i]
        distances[i] = max(sq_dist, zero(T))
    end

    return distances
end

"""
    assign_clusters!(assignments::Vector{Int}, D::Matrix)

Assign each point to its nearest centroid based on distance matrix.

# Arguments
- `assignments`: Pre-allocated vector of length n for cluster assignments
- `D`: Distance matrix (k × n) where D[i, j] = distance from centroid i to point j

# Modifies
- `assignments[j]` is set to argmin_i D[i, j]
"""
function assign_clusters!(assignments::Vector{Int}, D::Matrix)
    k, n = size(D)

    for j in 1:n
        min_dist = D[1, j]
        min_idx = 1

        for i in 2:k
            if D[i, j] < min_dist
                min_dist = D[i, j]
                min_idx = i
            end
        end

        assignments[j] = min_idx
    end

    return assignments
end
