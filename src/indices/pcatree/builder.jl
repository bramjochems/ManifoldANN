using LinearAlgebra
using Random

"""
    PCABuildContext{T}

Opaque context plumbed through the binary-partition-tree recursion.
Holds the data matrix and the per-tree RNG. Per-tree RNG (NOT
`Random.default_rng`) so a future PCA forest is a thin wrapper.
"""
struct PCABuildContext{T<:AbstractFloat}
    data::Matrix{T}
    rng::AbstractRNG
end

# `bpt_split!` overload: the splitter walks the four-trait pipeline and
# emits either a `BPTLeaf` or a `BPTInternal{PCANodePayload{T}}`.
function bpt_split!(
    splitter::PCASplitter,
    ctx::PCABuildContext{T},
    indices::Vector{Int},
    depth::Int,
) where {T<:AbstractFloat}
    n_node = length(indices)
    # Trivial leaf: 0 or 1 point can't be split.
    n_node <= 1 && return BPTLeaf()

    # Cheap-stop: if no criterion needs the spectrum and the size criterion
    # already says stop, skip the SVD.
    needs_spec = needs_spectrum(splitter.stopping)
    if !needs_spec && should_stop(splitter.stopping, n_node, nothing)
        return BPTLeaf()
    end

    # Build centered submatrix (d × n_node) once. We use a fresh allocation
    # so the splitter doesn't retain a reference to the parent vectors.
    d = size(ctx.data, 1)
    Xsub = Matrix{T}(undef, d, n_node)
    @inbounds for j in 1:n_node
        idx = indices[j]
        for i in 1:d
            Xsub[i, j] = ctx.data[i, idx]
        end
    end
    center = vec(sum(Xsub; dims = 2)) ./ T(n_node)
    @inbounds for j in 1:n_node, i in 1:d
        Xsub[i, j] -= center[i]
    end

    # Spectrum: one call serves both stopping and direction.
    spectrum = estimate_spectrum(splitter.estimator, Xsub, ctx.rng)

    # Stopping criterion now has access to the spectrum.
    if should_stop(splitter.stopping, n_node, spectrum)
        return BPTLeaf()
    end

    # Pick split direction (unit-normalise defensively — `pick_direction`
    # promises unit-norm but the random-combo policy may renormalise to
    # paper over zero-norm pathology).
    direction = pick_direction(splitter.direction, spectrum, ctx.rng)
    nrm = norm(direction)
    if nrm == 0
        return BPTLeaf()
    end
    direction ./= nrm

    # Project the centered submatrix on the direction.
    projections = Vector{T}(undef, n_node)
    @inbounds for j in 1:n_node
        s = zero(T)
        @simd for i in 1:d
            s += direction[i] * Xsub[i, j]
        end
        projections[j] = s
    end

    threshold = T(pick_split_value(splitter.split_value, projections, ctx.rng))

    # Partition. If degenerate (all on one side), fall back to a leaf.
    left_idx = Int[]
    right_idx = Int[]
    sizehint!(left_idx, n_node >>> 1)
    sizehint!(right_idx, n_node >>> 1)
    @inbounds for j in 1:n_node
        if projections[j] <= threshold
            push!(left_idx, indices[j])
        else
            push!(right_idx, indices[j])
        end
    end
    if isempty(left_idx) || isempty(right_idx)
        return BPTLeaf()
    end

    payload = PCANodePayload{T}(direction, threshold, center)
    return BPTInternal{PCANodePayload{T}}(left_idx, right_idx, payload)
end

"""
    build_index(PCATreeIndex, data; splitter=PCASplitter(), distance=default_distance, rng=Random.default_rng())

Build a single PCA tree over `data` (a `d × n` matrix). `splitter`
composes the four extension-point traits — see [`PCASplitter`](@ref).
The per-tree `rng` is plumbed through every randomised call site so a
future PCA forest is a thin wrapper.

Routes via the binary-partition-tree helper at
`src/utils/binary_partition_tree.jl`.
"""
function build_index(
    ::Type{PCATreeIndex},
    data::AbstractMatrix{T};
    splitter::PCASplitter = PCASplitter(),
    distance::D = default_distance,
    rng::AbstractRNG = Random.default_rng(),
) where {T<:AbstractFloat,D}
    d, n = size(data)
    d > 0 || throw(ArgumentError("data must have positive dimension"))
    n > 0 || throw(ArgumentError("data must contain at least one point"))

    # Promote to a concrete Matrix{T} once so the build context holds a
    # stable, strided buffer (matches the kmeans-strided rationale in
    # TODO_cleanup.md).
    data_mat = data isa Matrix{T} ? data : Matrix{T}(data)
    ctx = PCABuildContext{T}(data_mat, rng)
    indices = collect(1:n)
    nodes, leaf_members, root = bpt_build!(
        splitter, ctx, indices;
        payload_type = PCANodePayload{T},
    )
    return PCATreeIndex{T,D,typeof(splitter)}(
        nodes, leaf_members, root, d, n, distance, splitter,
    )
end
