"""
Utility functions for working with transforms in multi-level indices.
"""

"""
    has_bucketing(assignment)::Bool

Check if a transform produces bucketing information.

Returns `true` if the assignment is non-nothing, indicating that the transform
partitions data into buckets/clusters. Returns `false` for transforms that only
encode data without bucketing (e.g., PQ) or pass-through transforms (e.g., Identity).

# Examples
```julia
# Identity transform (no bucketing)
result = transform(IdentityTransform(), x)
@assert !has_bucketing(result.assignment)

# KMeans transform (with bucketing)
result = transform(kmeans_transform, x)
@assert has_bucketing(result.assignment)
```
"""
has_bucketing(assignment) = !isnothing(assignment)

"""
    get_bucket_assignment(assignment)::Int

Extract the primary bucket assignment from transform assignment info.

For KMeans, returns the index of the nearest centroid.
For other bucketing transforms, returns the appropriate bucket index.

# Examples
```julia
result = transform(kmeans_transform, x)
bucket = get_bucket_assignment(result.assignment)  # Index of nearest centroid
```
"""
function get_bucket_assignment(assignment::KMeansAssignment)
    return argmin(assignment.distances)
end

"""
    partition_by_transform(X::AbstractMatrix, transform::AbstractTransform; capture_data::Bool = true)

Partition data matrix `X` into buckets according to the transform's bucketing.

Each point is transformed and assigned to a bucket based on the assignment info.
Returns optional child datasets plus ID mappings from local to global IDs.

# Arguments
- `X`: Data matrix (d × n) where each column is a data point
- `transform`: Fitted transform with bucketing (e.g., `KMeansTransform`)
- `capture_data`: When `true`, materialize transformed data for each bucket so downstream
  indices can operate in the transformed space. When `false`, only the ID mappings are
  produced, which is appropriate for transforms that leave the data unchanged.

# Returns
- Tuple of `(partitions, id_mappings)` where:
  - `partitions`: `nothing` if `capture_data == false`, otherwise a vector of matrices
    containing transformed data grouped per bucket
  - `id_mappings[i]` maps local IDs in partition `i` to global IDs in `X`
"""
function partition_by_transform(
    X::AbstractMatrix,
    trans::AbstractTransform;
    capture_data::Bool=true,
)
    d, n = size(X)
    n > 0 || throw(ArgumentError("Cannot partition empty dataset"))

    # Peek at first point to determine output type when capturing data
    if capture_data
        first_result = ManifoldANN.transform(trans, X[:, 1])
        if !has_bucketing(first_result.assignment)
            error("Transform does not produce bucketing information")
        end

        first_data = first_result.data
        first_data isa AbstractVector ||
            throw(ArgumentError("Transform data must be a vector, got $(typeof(first_data))"))

        DataVec = typeof(first_data)
        stored_data = Vector{DataVec}(undef, n)
        stored_data[1] = first_data
        start_idx = 2
    else
        stored_data = nothing
        start_idx = 1
    end

    # First pass: determine bucket assignments
    assignments = Vector{Int}(undef, n)
    if capture_data
        assignments[1] = get_bucket_assignment(first_result.assignment)
    end

    for j in start_idx:n
        x = @view X[:, j]
        result = ManifoldANN.transform(trans, x)
        if !has_bucketing(result.assignment)
            error("Transform does not produce bucketing information")
        end
        assignments[j] = get_bucket_assignment(result.assignment)
        if capture_data
            stored_data[j] = result.data
        end
    end

    # Determine number of buckets and count points per bucket
    num_buckets = maximum(assignments)
    bucket_sizes = zeros(Int, num_buckets)
    for assignment in assignments
        bucket_sizes[assignment] += 1
    end

    # Allocate partition matrices and ID mappings
    partitions =
        capture_data ? _allocate_partitions(stored_data, bucket_sizes, num_buckets) : nothing
    id_mappings = [Vector{Int}(undef, bucket_sizes[i]) for i in 1:num_buckets]

    # Fill partitions and ID mappings
    bucket_indices = ones(Int, num_buckets)
    for j in 1:n
        bucket = assignments[j]
        local_idx = bucket_indices[bucket]
        if capture_data
            partitions[bucket][:, local_idx] = stored_data[j]
        end
        id_mappings[bucket][local_idx] = j  # Store global ID
        bucket_indices[bucket] += 1
    end

    return partitions, id_mappings
end

function _allocate_partitions(
    stored_data::Vector{V},
    bucket_sizes::Vector{Int},
    num_buckets::Int,
) where {V<:AbstractVector}
    # stored_data is now type-stable (Vector{SomeVectorType}), so we can directly use it
    isempty(stored_data) && throw(ArgumentError("Cannot partition empty dataset"))

    sample = stored_data[1]
    data_dim = length(sample)
    T = eltype(sample)

    return [Matrix{T}(undef, data_dim, bucket_sizes[i]) for i in 1:num_buckets]
end

"""
    apply_transform_batch(transform::AbstractTransform, X::AbstractMatrix)::Matrix

Apply transform to all columns of X and return transformed data matrix.

This is used for transforms that modify the data representation but don't
produce bucketing (e.g., PQ encoding, dimensionality reduction).

# Arguments
- `transform`: Fitted transform
- `X`: Data matrix (d × n) where each column is a data point

# Returns
- Transformed data matrix

# Note
Generic implementation uses loop over columns. Specialized methods exist for
specific transform types (e.g., IdentityTransform returns input directly).
"""
function apply_transform_batch(trans::AbstractTransform, X::AbstractMatrix)
    results = [ManifoldANN.transform(trans, x) for x in eachcol(X)]

    # Extract data from transform results
    transformed_data = [result.data for result in results]

    # Stack into matrix (assuming all results have same type and size)
    return hcat(transformed_data...)
end

"""
    apply_transform_batch(::IdentityTransform, X::AbstractMatrix)::AbstractMatrix

Optimized batch transform for IdentityTransform - returns input directly.

This avoids unnecessary allocation and copying since IdentityTransform
returns data unchanged.
"""
apply_transform_batch(::IdentityTransform, X::AbstractMatrix) = X
