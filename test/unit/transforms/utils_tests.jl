using Test
using Random
using ManifoldANN
using Distances

@testset "partition_by_transform accepts AbstractMatrix" begin
    rng = MersenneTwister(321)
    data = rand(rng, Float32, 4, 24)
    kmeans = KMeansTransform(k = 3, distance = Euclidean(), init = :kmeans_plus_plus)
    fit!(kmeans, data)

    data_view = @view data[:, 5:24]
    partitions, id_mappings = partition_by_transform(data_view, kmeans)

    @test sum(size(partition, 2) for partition in partitions) == size(data_view, 2)
    for (bucket, mapping) in enumerate(id_mappings)
        for (local_idx, global_idx) in enumerate(mapping)
            @test partitions[bucket][:, local_idx] == data_view[:, global_idx]
        end
    end
end

struct DoubleTransform <: AbstractTransform end

ManifoldANN.fit!(::DoubleTransform, ::AbstractMatrix) = nothing
ManifoldANN.transform(::DoubleTransform, x::AbstractVector) = TransformResult(2 .* x, nothing)

@testset "apply_transform_batch handles views" begin
    data = rand(Float32, 3, 10)
    data_view = @view data[:, 1:6]

    identity = IdentityTransform()
    @test apply_transform_batch(identity, data_view) === data_view

    double = DoubleTransform()
    expected = 2 .* data_view
    result = apply_transform_batch(double, data_view)
    @test result == expected
end
