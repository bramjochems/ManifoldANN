using Test
using Random
using ManifoldANN

@testset "RPTreeIndex build smoke test" begin
    rng = MersenneTwister(0xA1)
    data = randn(rng, Float32, 8, 64)
    idx = build_index(RPTreeIndex, data; rng = MersenneTwister(1))
    @test idx isa RPTreeIndex
    @test idx.n_points == 64
    @test idx.dimension == 8
end

@testset "RPTreeIndex single-query path" begin
    rng = MersenneTwister(0xA2)
    data = randn(rng, Float32, 6, 80)
    idx = build_index(RPTreeIndex, data; leaf_cap = 16, rng = MersenneTwister(7))
    q = randn(MersenneTwister(11), Float32, 6)
    res = query(idx, data, q, 5)
    @test length(res) <= 5
    @test all(r -> 1 <= r.id <= 80, res)
    dists = [r.dist for r in res]
    @test dists == sort(dists)
end

@testset "RPTreeIndex batch-query path" begin
    rng = MersenneTwister(0xA3)
    data = randn(rng, Float32, 5, 60)
    idx = build_index(RPTreeIndex, data; leaf_cap = 12, rng = MersenneTwister(13))
    queries = randn(MersenneTwister(17), Float32, 5, 8)
    batch = query(idx, data, queries, 4)
    @test length(batch) == 8
    for r in batch
        @test length(r) <= 4
    end
end

@testset "RPTreeIndex deterministic build under fixed RNG" begin
    rng = MersenneTwister(0xA4)
    data = randn(rng, Float32, 8, 100)
    a = build_index(RPTreeIndex, data; leaf_cap = 20, rng = MersenneTwister(42))
    b = build_index(RPTreeIndex, data; leaf_cap = 20, rng = MersenneTwister(42))
    @test a.tree.leaf_members == b.tree.leaf_members
    @test length(a.tree.nodes) == length(b.tree.nodes)
    q = randn(MersenneTwister(999), Float32, 8)
    @test query(a, data, q, 5) == query(b, data, q, 5)
end

@testset "RPTreeIndex k larger than n_points" begin
    rng = MersenneTwister(0xA5)
    data = randn(rng, Float32, 4, 10)
    idx = build_index(RPTreeIndex, data; leaf_cap = 16, rng = MersenneTwister(3))
    q = randn(MersenneTwister(5), Float32, 4)
    res = query(idx, data, q, 100)
    @test length(res) <= 10
end

@testset "RPTreeIndex dimension validation" begin
    data = randn(Float64, 3, 5)
    idx = build_index(RPTreeIndex, data)
    @test_throws DimensionMismatch query(idx, data, randn(4), 2)
end

@testset "RPTreeIndex recall floor on random data" begin
    rng = MersenneTwister(0xBEEF)
    n, d, k = 1000, 32, 10
    data = randn(rng, Float32, d, n)

    idx = build_index(RPTreeIndex, data; leaf_cap = 64, rng = MersenneTwister(2024))
    brute = build_index(BruteForceIndex, data)

    recalls = Float64[]
    for trial in 1:30
        q = randn(MersenneTwister(5000 + trial), Float32, d)
        ann = neighbor_ids(query(idx, data, q, k))
        bf = neighbor_ids(query(brute, data, q, k))
        push!(recalls, length(intersect(Set(ann), Set(bf))) / length(bf))
    end
    avg_recall = sum(recalls) / length(recalls)
    @test avg_recall >= 0.10
end
