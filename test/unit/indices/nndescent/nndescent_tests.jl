using ManifoldANN
using Random
using Test

@testset "NN-Descent index construction" begin
    Random.seed!(42)
    data = randn(Float32, 6, 48)
    k = 6
    index = build_index(
        NNDescentIndex,
        data;
        k = k,
        max_iterations = 20,
        convergence_threshold = 0.0,
        sampling_policy = :uniform,
        rng = Random.MersenneTwister(0xBEEF),
        distance = default_squared_distance,
    )

    @test index.k == k
    @test index.dimension == size(data, 1)
    @test length(index.neighbors) == size(data, 2)
    for (i, neighs) in enumerate(index.neighbors)
        @test !(i in neighs)
        @test length(unique(neighs)) == length(neighs)
        # Verify graph symmetry: if i has nb as neighbor, nb must have i as neighbor
        for nb in neighs
            @test i in index.neighbors[nb]
        end
    end

    graph = materialize_graph(index)
    @test graph.k == k
    @test length(graph.neighbors) == size(data, 2)
end

@testset "NN-Descent queries reach high recall" begin
    Random.seed!(7)
    data = randn(Float32, 8, 64)
    brute = build_index(BruteForceIndex, data)
    index = build_index(
        NNDescentIndex,
        data;
        k = 12,
        max_iterations = 25,
        convergence_threshold = 0.0,
        sampling_policy = :uniform,
        rng = Random.MersenneTwister(1234),
        distance = default_squared_distance,
    )

    for trial in 1:8
        q = randn(Float32, size(data, 1))
        nn_result = query(
            index,
            data,
            q,
            8;
            ef_search = 24,
            rng = Random.MersenneTwister(1000 + trial),
        )
        brute_result = query(brute, data, q, 8)
        recall =
            length(intersect(Set(nn_result), Set(brute_result))) / length(brute_result)
        @test recall >= 0.75
    end
end
