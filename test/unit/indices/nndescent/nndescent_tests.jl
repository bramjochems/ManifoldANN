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
    # With FullSymmetry (default), realized k can be > configured k due to reverse edges
    @test graph.k >= k
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
        nn_neighbors = query(
            index,
            data,
            q,
            8;
            ef_search = 24,
            rng = Random.MersenneTwister(1000 + trial),
        )
        brute_neighbors = query(brute, data, q, 8)
        nn_ids = neighbor_ids(nn_neighbors)
        brute_ids = neighbor_ids(brute_neighbors)
        recall = length(intersect(Set(nn_ids), Set(brute_ids))) / length(brute_ids)
        @test recall >= 0.75
    end
end

# Stronger tests added before items 2 + 3 (insert-O(k) + reverse-neighbor
# sampling). The 75% recall threshold above is too loose to detect subtle
# graph-quality regressions; these pin tighter numbers on a larger, fixed-seed
# build so any drop in graph quality from those refactors will surface.

@testset "NN-Descent build is reproducible under fixed seed" begin
    data = randn(Random.MersenneTwister(42), Float32, 8, 200)
    function build()
        return build_index(
            NNDescentIndex,
            data;
            k = 10,
            max_iterations = 25,
            convergence_threshold = 0.0,
            sampling_policy = :uniform,
            rng = Random.MersenneTwister(0xC0FFEE),
            distance = default_squared_distance,
        )
    end
    a = build()
    b = build()
    @test length(a.neighbors) == length(b.neighbors)
    for i in eachindex(a.neighbors)
        @test a.neighbors[i] == b.neighbors[i]
    end
end

@testset "NN-Descent graph-quality on n=500 (recall + MRR)" begin
    data = randn(Random.MersenneTwister(11), Float32, 16, 500)
    brute = build_index(BruteForceIndex, data)
    index = build_index(
        NNDescentIndex,
        data;
        k = 15,
        max_iterations = 30,
        convergence_threshold = 0.0,
        sampling_policy = :uniform,
        rng = Random.MersenneTwister(0xDEAD),
        distance = default_squared_distance,
    )

    n_queries = 50
    recalls = Float64[]
    mrrs = Float64[]
    for trial in 1:n_queries
        q = randn(Random.MersenneTwister(2000 + trial), Float32, size(data, 1))
        nn = query(
            index, data, q, 10;
            ef_search = 30,
            rng = Random.MersenneTwister(3000 + trial),
        )
        bf = query(brute, data, q, 10)
        nn_ids = neighbor_ids(nn)
        bf_ids = neighbor_ids(bf)
        push!(recalls, length(intersect(Set(nn_ids), Set(bf_ids))) / length(bf_ids))
        # MRR of the brute-force top-1 inside the NN-Descent top-10
        rank = findfirst(==(bf_ids[1]), nn_ids)
        push!(mrrs, rank === nothing ? 0.0 : 1.0 / rank)
    end
    mean_recall = sum(recalls) / length(recalls)
    mean_mrr = sum(mrrs) / length(mrrs)
    # Pinned thresholds: current implementation comfortably exceeds these
    # (typically ~0.95 recall, ~0.97 MRR on this config). Set ~5% below
    # observed to allow noise but catch real regressions.
    @test mean_recall >= 0.90
    @test mean_mrr >= 0.92
end
