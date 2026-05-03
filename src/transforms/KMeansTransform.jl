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
    centroids::Union{Nothing, Matrix{TC}}
    centroid_norms::Union{Nothing, Vector{TC}}
    pending_assignments::Union{Nothing, Vector{Int}}

    function KMeansTransform(;
        k::Int,
        distance::D,
        init::Symbol=:kmeans_plus_plus,
        max_iters::Int=KMEANS_DEFAULT_MAX_ITERATIONS,
        tol::Float64=KMEANS_DEFAULT_TOLERANCE,
        centroid_type::Type{<:AbstractFloat}=Float32,
    ) where {D<:SemiMetric}
        @assert k > 0 "Number of clusters must be positive"
        @assert init in (:random, :kmeans_plus_plus) "init must be :random or :kmeans_plus_plus"
        @assert max_iters > 0 "max_iters must be positive"
        @assert tol > 0 "tol must be positive"

        new{D, centroid_type}(k, distance, init, max_iters, tol, nothing, nothing, nothing)
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
function fit!(t::KMeansTransform{D,TC}, X::Matrix{T}) where {D<:SemiMetric,TC<:AbstractFloat,T<:Real}
    d, n = size(X)

    @assert t.k <= n "Cannot fit $(t.k) clusters with only $n points"

    # Initialize centroids
    if t.init == :random
        centroids = init_random(X, t.k)
    elseif t.init == :kmeans_plus_plus
        centroids = init_kmeans_plus_plus(X, t.k, t.distance)
    else
        throw(ArgumentError("Unknown initialization strategy: $(t.init)"))
    end

    # Run Lloyd's algorithm
    centroids, assignments, n_iters = lloyd!(
        centroids,
        X,
        t.distance;
        max_iters=t.max_iters,
        tol=t.tol
    )

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
