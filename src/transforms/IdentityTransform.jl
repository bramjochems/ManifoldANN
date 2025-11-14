"""
    IdentityTransform <: AbstractTransform

A no-op transform that returns the input data unchanged with no bucketing.

This is useful as a pass-through layer in multi-level index structures, or for
composing with routing strategies that operate on the original data.

# Examples
```julia
t = IdentityTransform()
X = rand(10, 100)
fit!(t, X)  # No-op

x = rand(10)
result = transform(t, x)
@assert result.data === x
@assert result.assignment === nothing
```
"""
struct IdentityTransform <: AbstractTransform end

"""
    fit!(::IdentityTransform, X::Matrix)

No-op for identity transform (nothing to learn).
"""
fit!(::IdentityTransform, ::Matrix) = nothing

"""
    transform(::IdentityTransform, x::AbstractVector)::TransformResult

Return the input unchanged with no bucketing information.
"""
transform(::IdentityTransform, x::AbstractVector) = TransformResult(x, nothing)

preserves_data(::IdentityTransform) = true
