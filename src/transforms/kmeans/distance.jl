"""
Distance computation utilities for KMeans clustering.
"""

using Distances

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
- Vectorized for performance
"""
function pairwise_distances!(D::Matrix, X::Matrix, centroids::Matrix, distance::SemiMetric)
    k, n = size(D)
    d = size(X, 1)

    @assert size(centroids) == (d, k) "Centroids must be d × k"
    @assert size(X) == (d, n) "Data must be d × n"

    # Compute distances for each centroid
    for i in 1:k
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
"""
function compute_distances(x::AbstractVector, centroids::Matrix, distance::SemiMetric)
    k = size(centroids, 2)
    distances = Vector{eltype(centroids)}(undef, k)

    for i in 1:k
        distances[i] = evaluate(distance, view(centroids, :, i), x)
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
