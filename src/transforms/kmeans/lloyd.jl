"""
Lloyd's algorithm for KMeans clustering.
"""

using Distances

"""
    lloyd!(
        centroids::Matrix,
        X::Matrix,
        distance::SemiMetric;
        max_iters::Int=100,
        tol::Float64=1e-6
    )

Run Lloyd's algorithm to refine KMeans centroids.

# Arguments
- `centroids`: Initial centroid matrix (d × k) - will be modified in place
- `X`: Data matrix (d × n) where each column is a data point
- `distance`: Distance function from Distances.jl
- `max_iters`: Maximum number of iterations (default: 100)
- `tol`: Convergence tolerance for centroid movement (default: 1e-6)

# Returns
- `centroids`: Refined centroid matrix (same reference as input)
- `assignments`: Final cluster assignments (vector of length n)
- `n_iters`: Number of iterations performed

# Algorithm
1. Assign each point to nearest centroid
2. Update centroids as mean of assigned points
3. Handle empty clusters by reassigning to farthest point
4. Repeat until convergence or max_iters reached
"""
function lloyd!(
    centroids::Matrix,
    X::Matrix,
    distance::SemiMetric;
    max_iters::Int=KMEANS_DEFAULT_MAX_ITERATIONS,
    tol::Float64=KMEANS_DEFAULT_TOLERANCE
)
    d, n = size(X)
    d_c, k = size(centroids)

    @assert d == d_c "Data and centroids must have same dimensionality"

    # Allocate workspace
    D = Matrix{eltype(X)}(undef, k, n)  # Distance matrix
    assignments = Vector{Int}(undef, n)  # Cluster assignments
    new_centroids = similar(centroids)
    cluster_sizes = zeros(Int, k)

    converged = false
    iter = 0

    for it in 1:max_iters
        iter = it
        # E-step: Assign points to nearest centroids
        pairwise_distances!(D, X, centroids, distance)
        assign_clusters!(assignments, D)

        # M-step: Update centroids as mean of assigned points
        fill!(new_centroids, 0)
        fill!(cluster_sizes, 0)

        for j in 1:n
            cluster = assignments[j]
            new_centroids[:, cluster] .+= X[:, j]
            cluster_sizes[cluster] += 1
        end

        # Handle empty clusters by assigning them to the farthest point
        for i in 1:k
            if cluster_sizes[i] == 0
                # Find point farthest from its assigned centroid
                max_dist = -Inf
                farthest_idx = 1

                for j in 1:n
                    dist = D[assignments[j], j]
                    if dist > max_dist
                        max_dist = dist
                        farthest_idx = j
                    end
                end

                # Reassign this point to the empty cluster
                new_centroids[:, i] = X[:, farthest_idx]
                cluster_sizes[i] = 1
            else
                # Normalize by cluster size to get mean
                new_centroids[:, i] ./= cluster_sizes[i]
            end
        end

        # Check convergence
        max_movement = 0.0
        for i in 1:k
            movement = evaluate(distance, view(centroids, :, i), view(new_centroids, :, i))
            max_movement = max(max_movement, movement)
        end

        # Update centroids
        centroids .= new_centroids

        if max_movement < tol
            converged = true
            break
        end
    end

    # Final assignment with converged centroids
    pairwise_distances!(D, X, centroids, distance)
    assign_clusters!(assignments, D)

    return centroids, assignments, iter
end
