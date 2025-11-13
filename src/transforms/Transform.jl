"""
    AbstractTransform

Abstract base type for all data transformations in multi-level indices.

All transforms must implement:
- `fit!(t::AbstractTransform, X::Matrix)` - Fit the transform to training data
- `transform(t::AbstractTransform, x::AbstractVector)::TransformResult` - Transform a single vector

Transforms may modify the data representation (e.g., quantization, dimensionality reduction)
and/or provide routing information (e.g., cluster assignments).
"""
abstract type AbstractTransform end

"""
    TransformResult{T,B}

Result of applying a transform to a data point.

# Fields
- `data::T` - Transformed representation of the input (may be same as input)
- `assignment::B` - Routing information (e.g., cluster distances), or `nothing` if no bucketing

# Examples
```julia
# Identity transform (no bucketing)
TransformResult(x, nothing)

# KMeans transform (with bucketing)
TransformResult(x, KMeansAssignment([0.5, 1.2, 0.8, ...]))

# PQ transform (encoding without bucketing)
TransformResult(pq_codes, nothing)
```
"""
struct TransformResult{T,B}
    data::T
    assignment::B
end

"""
    fit!(transform::AbstractTransform, X::Matrix)

Fit the transform to training data `X` (each column is a data point).

This method learns any parameters needed by the transform (e.g., cluster centroids,
quantization codebooks, projection matrices).
"""
function fit!(::AbstractTransform, ::Matrix)
    error("fit! not implemented for this transform type")
end

"""
    transform(transform::AbstractTransform, x::AbstractVector)::TransformResult

Apply the fitted transform to a single vector `x`.

Returns a `TransformResult` containing:
- `data`: The transformed representation
- `assignment`: Routing information (or `nothing` if no bucketing)
"""
function transform(::AbstractTransform, ::AbstractVector)
    error("transform not implemented for this transform type")
end
