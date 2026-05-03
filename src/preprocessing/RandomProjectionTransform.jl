"""
    RandomProjectionTransform{T} <: AbstractTransform

Dimensionality reduction via random projection (Johnson-Lindenstrauss).

Supports Gaussian and sparse random matrices for fast dimensionality reduction
while approximately preserving pairwise distances.

# Fields
- `target_dim`: Target dimension for projection
- `projection_type`: `:gaussian` or `:sparse`
- `density`: Density parameter for sparse matrices (default 1/3)

# Methods
- `fit!(transform, X)` - Generate random projection matrix
- `transform(transform, x)` - Apply random projection to vector x
- `suggested_dimension(n_samples; epsilon=0.1)` - Compute JL-optimal dimension
"""
mutable struct RandomProjectionTransform{T<:AbstractFloat} <: AbstractTransform
    target_dim::Int
    projection_type::Symbol
    density::Float64
    fitted::Bool
    projection::Union{Matrix{T},Nothing}

    function RandomProjectionTransform{T}(;
        target_dim::Int,
        projection_type::Symbol=:gaussian,
        density::Float64=1/3
    ) where {T<:AbstractFloat}
        target_dim >= 1 || throw(ArgumentError("target_dim must be >= 1, got $target_dim"))
        projection_type ∈ (:gaussian, :sparse) || throw(ArgumentError("projection_type must be :gaussian or :sparse"))
        0.0 < density <= 1.0 || throw(ArgumentError("density must be in (0, 1]"))
        new{T}(target_dim, projection_type, density, false, nothing)
    end
end

RandomProjectionTransform(; kwargs...) = RandomProjectionTransform{Float64}(; kwargs...)

function fit!(transform::RandomProjectionTransform{T}, X::AbstractMatrix) where {T}
    ambient_dim = size(X, 1)

    if transform.target_dim > ambient_dim
        @warn "target_dim ($(transform.target_dim)) > ambient_dim ($ambient_dim); projection will not reduce dimensionality"
    end

    if transform.projection_type == :gaussian
        transform.projection = _generate_gaussian_projection(T, transform.target_dim, ambient_dim)
    elseif transform.projection_type == :sparse
        transform.projection = _generate_sparse_projection(T, transform.target_dim, ambient_dim, transform.density)
    end

    transform.fitted = true
    nothing
end

function transform(transform::RandomProjectionTransform, x::AbstractVector)
    transform.fitted || throw(ArgumentError("RandomProjectionTransform must be fitted before transforming data"))
    projected = transform.projection * x
    TransformResult(projected, nothing)
end

target_dimension(transform::RandomProjectionTransform) = transform.target_dim
preserves_data(::RandomProjectionTransform) = false

function _generate_gaussian_projection(::Type{T}, target_dim::Int, ambient_dim::Int) where {T}
    scale = one(T) / sqrt(T(target_dim))
    randn(T, target_dim, ambient_dim) .* scale
end

function _generate_sparse_projection(::Type{T}, target_dim::Int, ambient_dim::Int, density::Float64) where {T}
    projection = zeros(T, target_dim, ambient_dim)
    scale = sqrt(one(T) / (T(density) * T(target_dim)))

    for i in 1:target_dim
        for j in 1:ambient_dim
            r = rand()
            if r < density / 2
                projection[i, j] = scale
            elseif r < density
                projection[i, j] = -scale
            end
        end
    end
    projection
end

"""
    suggested_dimension(n_samples::Int; epsilon::Float64=0.1) -> Int

Compute suggested target dimension for random projection based on JL lemma.

For n samples, returns d = O(log(n)/ε²) which guarantees all pairwise
distances are preserved within (1±ε) with high probability.
"""
function suggested_dimension(n_samples::Int; epsilon::Float64=0.1)
    n_samples >= 1 || throw(ArgumentError("n_samples must be positive"))
    epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
    max(1, ceil(Int, 8 * log(n_samples) / (epsilon^2)))
end
