using Random

struct LSHTable{H<:AbstractLSHHash}
    hash_function::H
    buckets::Dict{UInt64, Vector{Int}}
end

"""
    LSHIndex

Locality-sensitive hashing index composed of multiple independent tables. The
index never stores the dataset itself—callers pass `data` when querying so they
may use denoised or projected representations.
"""
mutable struct LSHIndex{T<:BlasFloat,H<:AbstractLSHHash} <: AbstractANNIndex
    tables::Vector{LSHTable{H}}
    dimension::Int
    n_points::Int
end

configured_k(::LSHIndex) = nothing
supports_mutation(::LSHIndex) = true

function build_index(
    ::Type{LSHIndex},
    data::AbstractMatrix{T};
    n_tables::Integer = LSH_DEFAULT_N_TABLES,
    hash_length::Integer = LSH_DEFAULT_HASH_LENGTH,
    hash_factory::Function = make_random_hyperplane_hash,
    rng::AbstractRNG = Random.default_rng(),
    hash_kwargs...,
) where {T<:BlasFloat}
    d, n = size(data)
    d > 0 || throw(ArgumentError("Dataset must have positive dimension"))
    n > 0 || throw(ArgumentError("Dataset must contain at least one point"))
    n_tables > 0 || throw(ArgumentError("n_tables must be positive"))
    hash_length > 0 || throw(ArgumentError("hash_length must be positive"))

    tables = Vector{LSHTable}(undef, n_tables)
    table_rngs = spawn_child_rngs(rng, n_tables)

    for (idx, trng) in enumerate(table_rngs)
        hash_fn = hash_factory(d, hash_length; rng = trng, hash_kwargs...)
        hashes = hash_batch(hash_fn, data)
        buckets = Dict{UInt64, Vector{Int}}()
        @inbounds for col in 1:n
            h = hashes[col]
            push!(get!(buckets, h, Int[]), col)
        end
        tables[idx] = LSHTable(hash_fn, buckets)
    end

    return LSHIndex{T, typeof(tables[1].hash_function)}(tables, d, n)
end

function query(
    index::LSHIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    candidate_cap::Union{Nothing,Int} = nothing,
) where {T<:BlasFloat}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]

    candidates = _collect_candidates(index, q)
    isempty(candidates) && return Neighbor{S}[]

    if candidate_cap !== nothing && length(candidates) > candidate_cap
        resize!(candidates, candidate_cap)
    end

    dist_fn = distance_function(index.tables[1].hash_function)
    distances = Vector{S}(undef, length(candidates))
    @inbounds for (i, id) in enumerate(candidates)
        distances[i] = dist_fn(@view(data[:, id]), q)
    end

    actual_k = min(k, length(candidates))
    perm = partialsortperm(distances, 1:actual_k)
    results = Vector{Neighbor{S}}(undef, actual_k)
    @inbounds for (pos, idx) in enumerate(perm)
        id = candidates[idx]
        results[pos] = Neighbor{S}(id, distances[idx])
    end
    return results
end

"""
    query(index::LSHIndex, data::Matrix, queries::Matrix, k::Integer; kwargs...)

Batch query interface for LSH: process multiple queries efficiently.
"""
function query(
    index::LSHIndex{T},
    data::AbstractMatrix{T},
    queries::AbstractMatrix{T},
    k::Integer;
    kwargs...
) where {T<:BlasFloat}
    size(queries, 1) == index.dimension ||
        throw(DimensionMismatch("Expected queries with $(index.dimension) rows"))

    n_queries = size(queries, 2)
    results = Vector{Vector{Neighbor{float(T)}}}(undef, n_queries)

    @inbounds for i in 1:n_queries
        q = @view queries[:, i]
        results[i] = query(index, data, q, k; kwargs...)
    end

    return results
end

"""
    query(index::LSHIndex, data::Matrix, queries::Vector{<:Vector}, k::Integer; kwargs...)

Convenience batch query interface using a vector of query vectors.
"""
function query(
    index::LSHIndex{T},
    data::AbstractMatrix{T},
    queries::Vector{<:AbstractVector{T}},
    k::Integer;
    kwargs...
) where {T<:BlasFloat}
    isempty(queries) && return Vector{Vector{Neighbor{float(T)}}}()
    queries_mat = reduce(hcat, queries)
    return query(index, data, queries_mat, k; kwargs...)
end

function _collect_candidates(index::LSHIndex, q::AbstractVector)
    candidates = Int[]
    n_tables = length(index.tables)
    n_tables == 0 && return candidates
    # Pre-allocate based on expected candidates per table
    sizehint!(candidates, n_tables * LSH_EXPECTED_CANDIDATES_PER_TABLE)
    for table in index.tables
        h = hash_point(table.hash_function, q)
        bucket = get(table.buckets, h, nothing)
        if bucket !== nothing
            append!(candidates, bucket)
        end
    end
    if !isempty(candidates)
        sort!(candidates)
        unique!(candidates)
    end
    return candidates
end

function insert!(index::LSHIndex{T}, point::AbstractVector{T}) where {T<:BlasFloat}
    length(point) == index.dimension ||
        throw(DimensionMismatch("Expected point dimension $(index.dimension)"))
    new_id = index.n_points + 1
    index.n_points = new_id
    for table in index.tables
        h = hash_point(table.hash_function, point)
        push!(get!(table.buckets, h, Int[]), new_id)
    end
    return index
end

function insert!(index::LSHIndex{T}, points::AbstractMatrix{T}) where {T<:BlasFloat}
    size(points, 1) == index.dimension ||
        throw(DimensionMismatch("Expected point dimension $(index.dimension)"))
    n_new = size(points, 2)
    n_new == 0 && return index
    start_id = index.n_points
    index.n_points += n_new
    for table in index.tables
        hashes = hash_batch(table.hash_function, points)
        for (offset, h) in enumerate(hashes)
            point_id = start_id + offset
            push!(get!(table.buckets, h, Int[]), point_id)
        end
    end
    return index
end

const _ = nothing # placeholder to keep file ending clean
