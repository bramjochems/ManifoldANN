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
    partition_by_transform(X::AbstractMatrix, transform::AbstractTransform)::Tuple{Vector{Matrix}, Vector{Vector{Int}}}

Partition data matrix X into buckets according to the transform's bucketing.

Each point is transformed and assigned to a bucket based on the assignment info.
Returns partitions and ID mappings from local to global IDs.

# Arguments
- `X`: Data matrix (d × n) where each column is a data point
- `transform`: Fitted transform with bucketing (e.g., KMeansTransform)

# Returns
- Tuple of (partitions, id_mappings) where:
  - partitions[i] contains all points assigned to bucket i
  - id_mappings[i] maps local IDs in partition i to global IDs in X

# Examples
```julia
kmeans = KMeansTransform(k=10, distance=Euclidean())
fit!(kmeans, X)
partitions, id_mappings = partition_by_transform(X, kmeans)
# partitions is a vector of 10 matrices, one per cluster
# id_mappings[i][j] gives the global ID for local ID j in partition i
```
"""
function partition_by_transform(X::AbstractMatrix, trans::AbstractTransform)
    d, n = size(X)

    # First pass: determine bucket assignments
    assignments = Vector{Int}(undef, n)
    for (j, x) in enumerate(eachcol(X))
        result = ManifoldANN.transform(trans, x)
        if !has_bucketing(result.assignment)
            error("Transform does not produce bucketing information")
        end
        assignments[j] = get_bucket_assignment(result.assignment)
    end

    # Determine number of buckets and count points per bucket
    num_buckets = maximum(assignments)
    bucket_sizes = zeros(Int, num_buckets)
    for assignment in assignments
        bucket_sizes[assignment] += 1
    end

    # Allocate partition matrices and ID mappings
    partitions = [Matrix{eltype(X)}(undef, d, bucket_sizes[i]) for i in 1:num_buckets]
    id_mappings = [Vector{Int}(undef, bucket_sizes[i]) for i in 1:num_buckets]

    # Fill partitions and ID mappings
    bucket_indices = ones(Int, num_buckets)
    for (j, x) in enumerate(eachcol(X))
        bucket = assignments[j]
        local_idx = bucket_indices[bucket]
        partitions[bucket][:, local_idx] = x
        id_mappings[bucket][local_idx] = j  # Store global ID
        bucket_indices[bucket] += 1
    end

    return partitions, id_mappings
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
