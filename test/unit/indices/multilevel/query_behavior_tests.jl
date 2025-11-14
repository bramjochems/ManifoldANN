using Test
using Random
using Distances: Euclidean
using ManifoldANN
using ManifoldANN: TransformResult, KMeansAssignment

struct RecordingIndex <: ManifoldANN.AbstractANNIndex
    last_kwargs::Ref{NamedTuple}
end

function ManifoldANN.build_index(::Type{RecordingIndex}, data::AbstractMatrix; kwargs...)
    return RecordingIndex(Ref{NamedTuple}(NamedTuple()))
end

function ManifoldANN.query(
    index::RecordingIndex,
    data::AbstractMatrix,
    q::AbstractVector,
    k::Integer;
    kwargs...
)
    index.last_kwargs[] = NamedTuple(kwargs)
    max_id = min(k, size(data, 2))
    neighbors = Vector{Neighbor{Float32}}(undef, max_id)
    @inbounds for i in 1:max_id
        neighbors[i] = Neighbor{Float32}(i, 0f0)
    end
    return neighbors
end

struct ShiftBucketTransform <: ManifoldANN.AbstractTransform
    shift::Float32
end

ManifoldANN.fit!(::ShiftBucketTransform, ::Matrix) = nothing

function ManifoldANN.transform(t::ShiftBucketTransform, x::AbstractVector)
    shifted = x .+ t.shift
    bucket = x[1] >= 0 ? 1 : 2
    distances = bucket == 1 ? Float32[0.0, 1.0] : Float32[1.0, 0.0]
    return TransformResult(shifted, KMeansAssignment(distances))
end

ManifoldANN.preserves_data(::ShiftBucketTransform) = false

@testset "MultiLevelIndex forwards kwargs" begin
    data = rand(Float32, 4, 8)
    config = TransformedConfig(
        IdentityTransform(),
        ExhaustiveRouting(),
        TerminalConfig(RecordingIndex, ()),
    )

    index = build_index(MultiLevelIndex, data, config)
    q = rand(Float32, size(data, 1))
    neighbors = query(index, data, q, 3; test_kw = 42)

    @test Set(neighbor_ids(neighbors)) == Set([1, 2, 3])
    recorder = index.root.indices[1]
    @test recorder.last_kwargs[] == (test_kw = 42,)
end

@testset "Bucketing transforms use transformed child data" begin
    rng = MersenneTwister(11)
    data = randn(rng, Float32, 3, 40)
    transform = ShiftBucketTransform(0.5f0)
    config = TransformedConfig(
        transform,
        TopKRouting(2),
        TerminalConfig(BruteForceIndex, ()),
    )

    index = build_index(MultiLevelIndex, data, config)
    root = index.root

    @test root.child_data !== nothing
    @test length(root.child_data) == length(root.id_mappings)

    for (bucket, ids) in enumerate(root.id_mappings)
        stored = root.child_data[bucket]
        expected = data[:, ids] .+ transform.shift
        @test stored ≈ expected
    end

    brute = build_index(BruteForceIndex, data)
    for _ in 1:3
        q = randn(rng, Float32, size(data, 1))
        result = query(index, data, q, 5)
        expected = query(brute, data, q, 5)
        @test length(result) == length(expected)
        for i in eachindex(result)
            @test result[i].id == expected[i].id
            @test result[i].dist ≈ expected[i].dist
        end
    end
end

@testset "Transforms that preserve data avoid caching partitions" begin
    rng = MersenneTwister(7)
    data = rand(rng, Float32, 5, 20)
    config = TransformedConfig(
        KMeansTransform(k = 3, distance = Euclidean(), init = :random),
        TopKRouting(2),
        TerminalConfig(BruteForceIndex, ()),
    )

    index = build_index(MultiLevelIndex, data, config)
    @test index.root.child_data === nothing
end
