"""
    PCATransform{T} <: AbstractTransform

Dimensionality reduction via Principal Component Analysis.

Supports three modes for determining the target dimension:
1. Fixed dimension: `PCATransform(target_dim=50)`
2. Variance threshold: `PCATransform(variance_threshold=0.95)`
3. Automatic: `PCATransform()` - retains 95% variance

# Methods
- `fit!(transform, X)` - Learn PCA from data matrix X (dimensions × points)
- `transform(transform, x)` - Project vector x to reduced space
- `inverse_transform(transform, x_reduced)` - Reconstruct from reduced space
- `explained_variance_ratio(transform)` - Get variance explained by each component
- `target_dimension(transform)` - Get number of retained components
"""
mutable struct PCATransform{T<:AbstractFloat} <: AbstractTransform
    target_dim::Union{Int,Nothing}
    variance_threshold::Float64
    fitted::Bool
    mean::Union{Vector{T},Nothing}
    projection::Union{Matrix{T},Nothing}
    explained_variance::Union{Vector{T},Nothing}

    function PCATransform{T}(;
        target_dim::Union{Int,Nothing}=nothing,
        variance_threshold::Float64=0.95
    ) where {T<:AbstractFloat}
        if !isnothing(target_dim) && target_dim < 1
            throw(ArgumentError("target_dim must be >= 1, got $target_dim"))
        end
        if !(0.0 < variance_threshold <= 1.0)
            throw(ArgumentError("variance_threshold must be in (0, 1], got $variance_threshold"))
        end
        new{T}(target_dim, variance_threshold, false, nothing, nothing, nothing)
    end
end

PCATransform(; kwargs...) = PCATransform{Float64}(; kwargs...)

function fit!(transform::PCATransform{T}, X::AbstractMatrix) where {T}
    ambient_dim, n_points = size(X)
    n_points >= 2 || throw(ArgumentError("Need at least 2 points to fit PCA, got $n_points"))

    data = T == eltype(X) ? X : convert(Matrix{T}, X)
    mean_vec = vec(sum(data, dims=2) ./ n_points)
    centered = data .- mean_vec

    F = svd(centered')
    eigenvalues = F.S .^ 2

    d = _determine_pca_dimension(
        transform.target_dim,
        transform.variance_threshold,
        eigenvalues,
        ambient_dim
    )

    transform.mean = mean_vec
    transform.projection = Matrix{T}(F.V[:, 1:d])
    transform.explained_variance = Vector{T}(eigenvalues[1:d])
    transform.fitted = true
    nothing
end

function transform(transform::PCATransform, x::AbstractVector)
    transform.fitted || throw(ArgumentError("PCATransform must be fitted before transforming data"))
    centered = x - transform.mean
    projected = transform.projection' * centered
    TransformResult(projected, nothing)
end

function inverse_transform(transform::PCATransform, x_reduced::AbstractVector)
    transform.fitted || throw(ArgumentError("PCATransform must be fitted before inverse transform"))
    transform.mean + transform.projection * x_reduced
end

function explained_variance_ratio(transform::PCATransform)
    transform.fitted || throw(ArgumentError("PCATransform must be fitted first"))
    total = sum(transform.explained_variance)
    total ≈ 0 ? fill(1.0 / length(transform.explained_variance), length(transform.explained_variance)) :
                transform.explained_variance ./ total
end

function target_dimension(transform::PCATransform)
    transform.fitted || throw(ArgumentError("PCATransform must be fitted first"))
    size(transform.projection, 2)
end

preserves_data(::PCATransform) = false

function _determine_pca_dimension(
    target_dim::Union{Int,Nothing},
    variance_threshold::Float64,
    eigenvalues::AbstractVector,
    ambient_dim::Int
)
    max_d = min(length(eigenvalues), ambient_dim)

    if !isnothing(target_dim)
        return min(target_dim, max_d)
    end

    if isempty(eigenvalues) || all(iszero, eigenvalues)
        return 1
    end

    total_var = sum(eigenvalues)
    if total_var ≈ 0
        return 1
    end

    cumvar = cumsum(eigenvalues) ./ total_var
    d = findfirst(>=(variance_threshold), cumvar)
    return isnothing(d) ? max_d : d
end
