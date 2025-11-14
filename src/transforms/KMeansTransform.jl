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
mutable struct KMeansTransform{D<:SemiMetric} <: AbstractTransform
    k::Int
    distance::D
    init::Symbol
    max_iters::Int
    tol::Float64
    centroids::Union{Nothing, Matrix{Float32}}

    function KMeansTransform(;
        k::Int,
        distance::D,
        init::Symbol=:kmeans_plus_plus,
        max_iters::Int=100,
        tol::Float64=1e-6
    ) where {D<:SemiMetric}
        @assert k > 0 "Number of clusters must be positive"
        @assert init in (:random, :kmeans_plus_plus) "init must be :random or :kmeans_plus_plus"
        @assert max_iters > 0 "max_iters must be positive"
        @assert tol > 0 "tol must be positive"

        new{D}(k, distance, init, max_iters, tol, nothing)
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
function fit!(t::KMeansTransform, X::Matrix{T}) where {T<:Real}
    d, n = size(X)

    @assert t.k <= n "Cannot fit $(t.k) clusters with only $n points"

    # Initialize centroids
    if t.init == :random
        centroids = init_random(X, t.k)
    elseif t.init == :kmeans_plus_plus
        centroids = init_kmeans_plus_plus(X, t.k, t.distance)
    else
        error("Unknown initialization strategy: $(t.init)")
    end

    # Run Lloyd's algorithm
    centroids, assignments, n_iters = lloyd!(
        centroids,
        X,
        t.distance;
        max_iters=t.max_iters,
        tol=t.tol
    )

    # Store fitted centroids
    t.centroids = centroids

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
        error("KMeansTransform must be fitted before transforming")
    end

    # Compute distances to all centroids
    distances = compute_distances(x, t.centroids, t.distance)

    return TransformResult(x, KMeansAssignment(distances))
end

preserves_data(::KMeansTransform) = true
