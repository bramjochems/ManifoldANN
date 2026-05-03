using Test
using Random
using ManifoldANN

@testset "RPTreeForestIndex build smoke test" begin
    rng = MersenneTwister(0xF1)
    data = randn(rng, Float32, 8, 64)
    idx = build_index(RPTreeForestIndex, data;
        n_trees = 4, leaf_cap = 16, rng = MersenneTwister(1))
    @test idx isa RPTreeForestIndex
    @test idx.n_points == 64
    @test idx.dimension == 8
    @test length(idx.trees) == 4
    @test idx.leaf_cap == 16
end

@testset "RPTreeForestIndex single-query path" begin
    rng = MersenneTwister(0xF2)
    data = randn(rng, Float32, 6, 80)
    idx = build_index(RPTreeForestIndex, data;
        n_trees = 5, leaf_cap = 16, rng = MersenneTwister(7))
    q = randn(MersenneTwister(11), Float32, 6)
    res = query(idx, data, q, 5)
    @test length(res) <= 5
    @test all(r -> 1 <= r.id <= 80, res)
    dists = [r.dist for r in res]
    @test dists == sort(dists)
end

@testset "RPTreeForestIndex batch-query path" begin
    rng = MersenneTwister(0xF3)
    data = randn(rng, Float32, 5, 60)
    idx = build_index(RPTreeForestIndex, data;
        n_trees = 3, leaf_cap = 12, rng = MersenneTwister(13))
    queries = randn(MersenneTwister(17), Float32, 5, 8)
    batch = query(idx, data, queries, 4)
    @test length(batch) == 8
    for r in batch
        @test length(r) <= 4
    end
end

@testset "RPTreeForestIndex deterministic build under fixed RNG" begin
    rng = MersenneTwister(0xF4)
    data = randn(rng, Float32, 8, 100)
    a = build_index(RPTreeForestIndex, data;
        n_trees = 6, leaf_cap = 20, rng = MersenneTwister(42))
    b = build_index(RPTreeForestIndex, data;
        n_trees = 6, leaf_cap = 20, rng = MersenneTwister(42))
    @test length(a.trees) == length(b.trees)
    for i in eachindex(a.trees)
        @test a.trees[i].leaf_members == b.trees[i].leaf_members
        @test length(a.trees[i].nodes) == length(b.trees[i].nodes)
    end
    q = randn(MersenneTwister(999), Float32, 8)
    @test query(a, data, q, 5) == query(b, data, q, 5)
end

@testset "RPTreeForestIndex k larger than n_points" begin
    rng = MersenneTwister(0xF5)
    data = randn(rng, Float32, 4, 10)
    idx = build_index(RPTreeForestIndex, data;
        n_trees = 4, leaf_cap = 16, rng = MersenneTwister(3))
    q = randn(MersenneTwister(5), Float32, 4)
    res = query(idx, data, q, 100)
    @test length(res) <= 10
end

@testset "RPTreeForestIndex dimension validation" begin
    data = randn(Float64, 3, 5)
    idx = build_index(RPTreeForestIndex, data; n_trees = 2)
    @test_throws DimensionMismatch query(idx, data, randn(4), 2)
end

@testset "RPTreeForestIndex argument validation" begin
    data = randn(Float32, 4, 20)
    @test_throws ArgumentError build_index(RPTreeForestIndex, data; n_trees = 0)
    @test_throws ArgumentError build_index(RPTreeForestIndex, data; leaf_cap = 0)
end

@testset "RPTreeForestIndex recall floor on random data" begin
    rng = MersenneTwister(0xBABE)
    n, d, k = 1000, 32, 10
    data = randn(rng, Float32, d, n)

    idx = build_index(RPTreeForestIndex, data;
        n_trees = 8, leaf_cap = 32, rng = MersenneTwister(2024))
    brute = build_index(BruteForceIndex, data)

    recalls = Float64[]
    for trial in 1:30
        q = randn(MersenneTwister(5000 + trial), Float32, d)
        ann = neighbor_ids(query(idx, data, q, k))
        bf = neighbor_ids(query(brute, data, q, k))
        push!(recalls, length(intersect(Set(ann), Set(bf))) / length(bf))
    end
    avg_recall = sum(recalls) / length(recalls)
    # Forest should beat single-tree (>=0.10) by a healthy margin. Empirically
    # ~0.49 with these settings; floor is set to leave headroom for run-to-run
    # variation under different thread counts / RNG draws.
    @test avg_recall >= 0.35
end

@testset "RPTreeForestIndex forest beats single tree on average recall" begin
    rng = MersenneTwister(0xCAFE)
    n, d, k = 600, 16, 10
    data = randn(rng, Float32, d, n)

    forest = build_index(RPTreeForestIndex, data;
        n_trees = 8, leaf_cap = 24, rng = MersenneTwister(1234))
    single = build_index(RPTreeIndex, data;
        leaf_cap = 24, rng = MersenneTwister(1234))
    brute = build_index(BruteForceIndex, data)

    forest_recalls = Float64[]
    single_recalls = Float64[]
    for trial in 1:25
        q = randn(MersenneTwister(7000 + trial), Float32, d)
        bf = Set(neighbor_ids(query(brute, data, q, k)))
        push!(forest_recalls,
            length(intersect(Set(neighbor_ids(query(forest, data, q, k))), bf)) / length(bf))
        push!(single_recalls,
            length(intersect(Set(neighbor_ids(query(single, data, q, k))), bf)) / length(bf))
    end
    @test sum(forest_recalls) / length(forest_recalls) >
          sum(single_recalls) / length(single_recalls)
end
