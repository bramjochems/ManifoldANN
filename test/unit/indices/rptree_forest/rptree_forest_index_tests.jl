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

@testset "RPTreeForestIndex re-entrancy of public query API" begin
    # The forest's `query` claims concurrent-safety. Two threads calling
    # it on the same index with different queries must produce results
    # identical to a serial loop. Mirrors the HNSW pattern.
    if Threads.nthreads() >= 2
        rng = MersenneTwister(0xF6)
        data = randn(rng, Float32, 12, 600)
        idx = build_index(RPTreeForestIndex, data;
            n_trees = 6, leaf_cap = 24, rng = MersenneTwister(0xF6))
        queries = [randn(MersenneTwister(0x100 + i), Float32, 12) for i in 1:32]

        ref = [query(idx, data, q, 5) for q in queries]
        got = Vector{Vector{ManifoldANN.Neighbor{Float32}}}(undef, 32)
        Threads.@threads for i in 1:32
            got[i] = query(idx, data, queries[i], 5)
        end
        for i in 1:32
            @test [n.id for n in got[i]] == [n.id for n in ref[i]]
        end
    else
        @warn "RPTreeForestIndex re-entrancy test requires Threads.nthreads() ≥ 2; skipping"
    end
end
