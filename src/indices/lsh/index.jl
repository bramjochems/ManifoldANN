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
    # Expected per-bucket size = n_points / 2^hash_length. Used to size the
    # candidates / seen scratch buffers in _collect_candidates! so they don't
    # grow during the table sweep.
    mean_bucket_size::Float64
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

    # Default the hash element type to match the data eltype so hash_batch
    # dispatches without a Float32/Float64 mismatch. Caller can override via
    # hash_kwargs (e.g. `T=Float64` to force higher-precision projections).
    factory_kwargs = haskey(hash_kwargs, :T) ? hash_kwargs : (; hash_kwargs..., T = T)

    # Threaded: each table is independent (own RNG, hash fn, Dict) and writes
    # only to its pre-allocated `tables[idx]` slot. Per-table seeds are derived
    # serially above so determinism is independent of thread scheduling.
    Threads.@threads for idx in 1:n_tables
        trng = table_rngs[idx]
        hash_fn = hash_factory(d, hash_length; rng = trng, factory_kwargs...)
        hashes = hash_batch(hash_fn, data)
        buckets = Dict{UInt64, Vector{Int}}()
        @inbounds for col in 1:n
            h = hashes[col]
            push!(get!(buckets, h, Int[]), col)
        end
        tables[idx] = LSHTable(hash_fn, buckets)
    end

    mean_bucket = n / max(2.0^hash_length, 1.0)
    return LSHIndex{T, typeof(tables[1].hash_function)}(tables, d, n, mean_bucket)
end

"""
    query(index::LSHIndex, data, q, k; candidate_cap=nothing, rng=Random.default_rng())

Query the LSH index for the `k` nearest neighbours of `q`. When `candidate_cap`
is set and the candidate set exceeds it, a uniform sample is drawn using `rng`.

Thread-safety: `rng` must be thread-local. The default `Random.default_rng()`
is task-local on Julia ≥ 1.7 and safe under the threaded matrix-input batch
`query`. A caller-supplied shared RNG (e.g. one `MersenneTwister` reused across
queries) will race when used through the batch path; pass `nothing` or a
freshly-constructed RNG per task in that case.
"""
function query(
    index::LSHIndex{T},
    data::AbstractMatrix{T},
    q::AbstractVector{T},
    k::Integer;
    candidate_cap::Union{Nothing,Int} = nothing,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:BlasFloat}
    validate_index_dimensions(index, data, q)
    S = float(T)
    k <= 0 && return Neighbor{S}[]

    scratch = _lsh_task_scratch(S)
    candidates = _collect_candidates!(scratch, index, q)
    isempty(candidates) && return Neighbor{S}[]

    if candidate_cap !== nothing && length(candidates) > candidate_cap
        shuffle!(rng, candidates)
        resize!(candidates, candidate_cap)
    end

    dist_fn = distance_function(index.tables[1].hash_function)
    distances = scratch.distances
    resize!(distances, length(candidates))
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

# Per-task scratch buffers reused across queries on the same task. Eliminates
# the per-call alloc of `candidates::Vector{Int}`, `distances::Vector{S}`, and
# the BitSet visited set. Same task-local-storage pattern as the KDTree query
# path. Each query starts by resetting the buffers (empty! / fill! generation
# bump) — capacity is preserved.
mutable struct LSHQueryScratch{S<:AbstractFloat}
    candidates::Vector{Int}
    distances::Vector{S}
    seen::BitSet
end

LSHQueryScratch{S}() where {S<:AbstractFloat} = LSHQueryScratch{S}(Int[], S[], BitSet())

const _LSH_TASK_SCRATCH_KEY = :ManifoldANN_LSH_task_scratch

@inline function _lsh_task_scratch(::Type{S}) where {S<:AbstractFloat}
    tls = task_local_storage()
    stored = get(tls, _LSH_TASK_SCRATCH_KEY, nothing)
    if stored isa LSHQueryScratch{S}
        empty!(stored.candidates)
        empty!(stored.seen)
        # `distances` is resize!'d at use site, no need to clear here.
        return stored::LSHQueryScratch{S}
    end
    scratch = LSHQueryScratch{S}()
    tls[_LSH_TASK_SCRATCH_KEY] = scratch
    return scratch
end

function _collect_candidates!(scratch::LSHQueryScratch, index::LSHIndex, q::AbstractVector)
    candidates = scratch.candidates
    seen = scratch.seen
    n_tables = length(index.tables)
    n_tables == 0 && return candidates
    # Pre-dedup expected candidate count = n_tables * mean_bucket_size, capped
    # at n_points (can't dedup to more unique ids than exist). The constant
    # LSH_EXPECTED_CANDIDATES_PER_TABLE was a stand-in that systematically
    # under-sized at typical hash_length; the build-time-derived estimate
    # eliminates the per-call _growend! churn.
    expected = min(index.n_points, ceil(Int, n_tables * index.mean_bucket_size))
    sizehint!(candidates, expected)
    sizehint!(seen, expected)
    for table in index.tables
        h = hash_point(table.hash_function, q)
        bucket = get(table.buckets, h, nothing)
        bucket === nothing && continue
        @inbounds for id in bucket
            if !(id in seen)
                push!(seen, id)
                push!(candidates, id)
            end
        end
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
