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

        # M-step: parallel accumulation across n with per-chunk buffers,
        # then a serial reduction. The chunked-tasks pattern uses
        # Threads.@spawn (stable since 1.7) so we do not rely on the
        # discouraged Threads.threadid() partitioning idiom.
        _m_step_parallel!(new_centroids, cluster_sizes, X, assignments, k)

        # Empty-cluster handling: only enter the slow farthest-point scan
        # if any cluster came back empty. With non-empty clusters this is
        # the common case at moderate nlist and lets us skip an O(n*k_empty)
        # scan over the distance matrix.
        any_empty = false
        @inbounds for i in 1:k
            if cluster_sizes[i] == 0
                any_empty = true
                break
            end
        end

        if any_empty
            # When ≥2 clusters become empty in the same iteration we must
            # guard against picking the same data point twice; mark the
            # chosen point's distance column with -Inf so the next farthest
            # scan skips it.
            for i in 1:k
                if cluster_sizes[i] == 0
                    max_dist = -Inf
                    farthest_idx = 1
                    @inbounds for j in 1:n
                        dist = D[assignments[j], j]
                        if dist > max_dist
                            max_dist = dist
                            farthest_idx = j
                        end
                    end

                    @inbounds new_centroids[:, i] = X[:, farthest_idx]
                    cluster_sizes[i] = 1
                    @inbounds D[assignments[farthest_idx], farthest_idx] = -Inf
                else
                    @inbounds new_centroids[:, i] ./= cluster_sizes[i]
                end
            end
        else
            # Common path: every cluster received at least one point;
            # normalise without an empty-cluster scan.
            @inbounds for i in 1:k
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

"""
    _m_step_parallel!(new_centroids, cluster_sizes, X, assignments, k)

Threaded M-step: accumulate per-cluster sums of X into `new_centroids` and
per-cluster counts into `cluster_sizes`. Each worker task gets its own
`(centroid_buf, sizes_buf)` so writes to the per-cluster columns never
race; a serial reduction folds the per-task buffers back into the
caller-provided `new_centroids` / `cluster_sizes`.

The previous serial loop `new_centroids[:, cluster] .+= X[:, j]` was the
single largest cost at small d / large n / large k (e.g. glove-100 with
nlist=4096), where the M-step dominated each Lloyd iteration.
"""
function _m_step_parallel!(
    new_centroids::Matrix,
    cluster_sizes::Vector{Int},
    X::Matrix,
    assignments::Vector{Int},
    k::Int,
)
    d, n = size(X)
    nthreads = Threads.nthreads()

    if nthreads <= 1 || n < 4_000
        # Below the threading threshold the spawn/fetch overhead beats the
        # per-thread reduction win. Same scalar inner loop as before but
        # @inbounds-annotated; this is the codepath for tests / small data.
        fill!(new_centroids, 0)
        fill!(cluster_sizes, 0)
        @inbounds for j in 1:n
            cluster = assignments[j]
            for r in 1:d
                new_centroids[r, cluster] += X[r, j]
            end
            cluster_sizes[cluster] += 1
        end
        return
    end

    # Chunk the n points across `nthreads` tasks. Each task owns its own
    # buffers, so accumulation is race-free.
    chunk_size = max(1, cld(n, nthreads))
    n_chunks = cld(n, chunk_size)

    local_centroids = Vector{Matrix{eltype(X)}}(undef, n_chunks)
    local_sizes = Vector{Vector{Int}}(undef, n_chunks)
    tasks = Vector{Task}(undef, n_chunks)

    @inbounds for c in 1:n_chunks
        lo = (c - 1) * chunk_size + 1
        hi = min(c * chunk_size, n)
        local_centroids[c] = zeros(eltype(X), d, k)
        local_sizes[c] = zeros(Int, k)
        lc = local_centroids[c]
        ls = local_sizes[c]
        tasks[c] = Threads.@spawn begin
            @inbounds for j in lo:hi
                cluster = assignments[j]
                for r in 1:d
                    lc[r, cluster] += X[r, j]
                end
                ls[cluster] += 1
            end
        end
    end

    foreach(wait, tasks)

    # Serial reduction: sum the per-task buffers back into the shared output.
    fill!(new_centroids, 0)
    fill!(cluster_sizes, 0)
    @inbounds for c in 1:n_chunks
        new_centroids .+= local_centroids[c]
        cluster_sizes .+= local_sizes[c]
    end
    return
end
