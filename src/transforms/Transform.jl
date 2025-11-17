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
    take_pending_assignments!(transform::AbstractTransform) -> Union{Nothing,Vector{Int}}

Retrieve any cached bucket assignments produced during `fit!` and clear them
from the transform. The default implementation returns `nothing`; transforms
that can reuse training assignments (e.g., `KMeansTransform`) should override
this to provide a freshly allocated vector for one-time consumption.
"""
take_pending_assignments!(::AbstractTransform) = nothing

"""
    preserves_data(transform::AbstractTransform) -> Bool

Return true when `transform` leaves the underlying data representation unchanged.
Transforms that simply attach routing metadata (e.g., `KMeansTransform`,
`IdentityTransform`) should override this to return `true` so that multi-level
indices can avoid caching redundant copies of the dataset. The default is
`false`, meaning callers must assume the transform produces a new representation.
"""
preserves_data(::AbstractTransform) = false

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
