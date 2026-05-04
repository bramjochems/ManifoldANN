"""
    KMeansTransform <: AbstractTransform

KMeans clustering transform for multi-level indexing (IVF-style bucketing).

This transform:
1. During `fit!`: Learns k cluster centroids from training data
2. During `transform`: Returns original data + distances to all centroids

The distance vector enables top-K routing strategies for multi-probe search.

# Fields
- `k`: Number of clusters
- `distance`: Distance metric from Distances.jl
- `init`: Initialization strategy (`:random` or `:kmeans_plus_plus`)
- `max_iters`: Maximum Lloyd iterations (default: 100)
- `tol`: Convergence tolerance (default: 1e-6)
- `centroid_type`: Storage precision for learned centroids (configurable via keyword, default `Float32`)
- `centroids`: Learned centroids (d × k matrix, set by `fit!`)

# Examples
```julia
using Distances
using Random

# Create transform with kmeans++ initialization
t = KMeansTransform(k=100, distance=Euclidean(), init=:kmeans_plus_plus)

# Fit to training data
X = rand(Float32, 128, 10000)
fit!(t, X)

# Transform a query point
q = rand(Float32, 128)
result = transform(t, q)

# result.data === q (original vector)
# result.assignment.distances = [d1, d2, ..., d100] (distances to all centroids)
```
"""
mutable struct KMeansTransform{D<:SemiMetric,TC<:AbstractFloat} <: AbstractTransform
    k::Int
    distance::D
    init::Symbol
    max_iters::Int
    tol::Float64
    # When set, Lloyd iterations run on a uniform random subsample of size
    # `subsample_size`; the final assignment pass over the full dataset still
    # produces n point→cluster assignments. `nothing` = use all data each
    # iteration. Useful at scale where Lloyd convergence on a few hundred
    # points per centroid is essentially the same as on the full dataset.
    subsample_size::Union{Nothing, Int}
    centroids::Union{Nothing, Matrix{TC}}
    centroid_norms::Union{Nothing, Vector{TC}}
    pending_assignments::Union{Nothing, Vector{Int}}

    function KMeansTransform(;
        k::Int,
        distance::D,
        init::Symbol=:kmeans_plus_plus,
        max_iters::Int=KMEANS_DEFAULT_MAX_ITERATIONS,
        tol::Float64=KMEANS_DEFAULT_TOLERANCE,
        subsample_size::Union{Nothing,Int}=nothing,
        centroid_type::Type{<:AbstractFloat}=Float32,
    ) where {D<:SemiMetric}
        @assert k > 0 "Number of clusters must be positive"
        @assert init in (:random, :kmeans_plus_plus) "init must be :random or :kmeans_plus_plus"
        @assert max_iters > 0 "max_iters must be positive"
        @assert tol > 0 "tol must be positive"
        subsample_size === nothing || subsample_size > 0 ||
            throw(ArgumentError("subsample_size must be positive or nothing"))

        new{D, centroid_type}(k, distance, init, max_iters, tol, subsample_size, nothing, nothing, nothing)
    end
end

"""
    KMeansAssignment

Routing information for KMeans transform.

# Fields
- `distances`: Vector of distances to all k centroids
"""
struct KMeansAssignment{T<:Real}
    distances::Vector{T}
end

# Include KMeans implementation modules
include("kmeans/distance.jl")
include("kmeans/init.jl")
include("kmeans/lloyd.jl")

"""
    fit!(t::KMeansTransform, X::Matrix)

Learn k cluster centroids from training data using Lloyd's algorithm.

# Arguments
- `t`: KMeansTransform to fit (will be modified in place)
- `X`: Training data matrix (d × n) where each column is a data point

# Effects
- Sets `t.centroids` to learned centroid matrix (d × k)
"""
function fit!(t::KMeansTransform{D,TC}, X::Matrix{T};
              rng::AbstractRNG=Random.default_rng()) where {D<:SemiMetric,TC<:AbstractFloat,T<:Real}
    d, n = size(X)

    @assert t.k <= n "Cannot fit $(t.k) clusters with only $n points"

    # If subsample_size is set and < n, fit Lloyd on a uniform random
    # subsample of the data and finish with a full-dataset assignment pass.
    # Otherwise X_train = X (no copy).
    use_subsample = t.subsample_size !== nothing && t.subsample_size < n
    X_train = if use_subsample
        sample_n = max(t.subsample_size, t.k)  # need at least k points to fit k clusters
        idx = randperm(rng, n)[1:sample_n]
        X[:, idx]
    else
        X
    end

    # Initialize centroids
    if t.init == :random
        centroids = init_random(X_train, t.k; rng=rng)
    elseif t.init == :kmeans_plus_plus
        centroids = init_kmeans_plus_plus(X_train, t.k, t.distance; rng=rng)
    else
        throw(ArgumentError("Unknown initialization strategy: $(t.init)"))
    end

    # Run Lloyd's algorithm (on subsample if applicable)
    centroids, assignments_train, _n_iters = lloyd!(
        centroids,
        X_train,
        t.distance;
        max_iters=t.max_iters,
        tol=t.tol
    )

    # If we fit on a subsample, redo assignments over the full dataset so
    # downstream partitioning sees every point. Lloyd's centroids are stable
    # enough on a subsample that a single E-step over the full data gives
    # the IVF cell membership we need.
    assignments = if use_subsample
        D_full = Matrix{eltype(X)}(undef, t.k, n)
        pairwise_distances!(D_full, X, centroids, t.distance)
        assign_full = Vector{Int}(undef, n)
        assign_clusters!(assign_full, D_full)
        assign_full
    else
        assignments_train
    end

    # Store fitted centroids using the configured storage precision
    t.centroids = centroids isa Matrix{TC} ? centroids : Matrix{TC}(centroids)
    t.centroid_norms = vec(sum(abs2, t.centroids; dims=1))
    t.pending_assignments = assignments

    return t
end

"""
    transform(t::KMeansTransform, x::AbstractVector)::TransformResult

Transform a query point by computing distances to all centroids.

# Arguments
- `t`: Fitted KMeansTransform
- `x`: Query point (d-dimensional vector)

# Returns
- `TransformResult` where:
  - `data = x` (original vector unchanged)
  - `assignment = KMeansAssignment(distances)` (distances to all k centroids)

# Throws
- Error if transform has not been fitted
"""
function transform(t::KMeansTransform, x::AbstractVector)
    if isnothing(t.centroids)
        throw(ArgumentError("KMeansTransform must be fitted before transforming"))
    end

    # Compute distances to all centroids using cached centroid norms when possible
    distances = compute_distances(x, t.centroids, t.distance, t.centroid_norms)

    return TransformResult(x, KMeansAssignment(distances))
end

function take_pending_assignments!(t::KMeansTransform)
    assignments = t.pending_assignments
    t.pending_assignments = nothing
    return assignments
end

preserves_data(::KMeansTransform) = true
