"""
Initialization strategies for KMeans clustering.
"""

using Random
using Distances

"""
    init_random(X::Matrix, k::Int; rng::AbstractRNG=Random.GLOBAL_RNG)

Initialize k centroids by randomly selecting k points from X.

# Arguments
- `X`: Data matrix (d × n) where each column is a data point
- `k`: Number of clusters
- `rng`: Random number generator (default: GLOBAL_RNG)

# Returns
- Matrix (d × k) where each column is an initial centroid
"""
function init_random(X::Matrix, k::Int; rng::AbstractRNG=Random.GLOBAL_RNG)
    d, n = size(X)
    @assert k <= n "Cannot initialize $k clusters with only $n points"

    # Randomly select k distinct indices
    indices = randperm(rng, n)[1:k]

    # Return selected points as centroids
    return X[:, indices]
end

"""
    init_kmeans_plus_plus(X::Matrix, k::Int, distance::SemiMetric; rng::AbstractRNG=Random.GLOBAL_RNG)

Initialize k centroids using KMeans++ algorithm.

The KMeans++ algorithm:
1. Choose first centroid uniformly at random
2. For each subsequent centroid:
   - Compute D(x)² for each point x (squared distance to nearest existing centroid)
   - Choose next centroid with probability proportional to D(x)²

# Arguments
- `X`: Data matrix (d × n) where each column is a data point
- `k`: Number of clusters
- `distance`: Distance function from Distances.jl
- `rng`: Random number generator (default: GLOBAL_RNG)

# Returns
- Matrix (d × k) where each column is an initial centroid

# Reference
- Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding.
"""
function init_kmeans_plus_plus(
    X::Matrix,
    k::Int,
    distance::SemiMetric;
    rng::AbstractRNG=Random.GLOBAL_RNG
)
    d, n = size(X)
    @assert k <= n "Cannot initialize $k clusters with only $n points"

    # Allocate centroid matrix
    centroids = similar(X, d, k)

    # Choose first centroid uniformly at random
    first_idx = rand(rng, 1:n)
    centroids[:, 1] = X[:, first_idx]

    # Track minimum squared distances to nearest centroid
    min_sq_distances = fill(Inf, n)

    # Choose remaining k-1 centroids
    for i in 2:k
        # Update minimum squared distances with new centroid
        new_centroid = view(centroids, :, i-1)
        for j in 1:n
            dist = evaluate(distance, new_centroid, view(X, :, j))
            sq_dist = dist * dist
            if sq_dist < min_sq_distances[j]
                min_sq_distances[j] = sq_dist
            end
        end

        # Compute cumulative distribution
        cumsum_sq_dist = cumsum(min_sq_distances)
        total_sq_dist = cumsum_sq_dist[end]

        # Sample proportional to squared distance
        threshold = rand(rng) * total_sq_dist
        next_idx = searchsortedfirst(cumsum_sq_dist, threshold)

        # Handle edge case where threshold == total_sq_dist
        next_idx = min(next_idx, n)

        centroids[:, i] = X[:, next_idx]
    end

    return centroids
end
