using Test
using Random
using ManifoldANN

@testset "HNSW batch query: matches single-query results elementwise" begin
    # The batch path uses a worker-pool with per-task scratch reused across
    # queries. Results MUST match what the single-query API produces — the
    # only difference is allocation strategy, not algorithm.
    rng = MersenneTwister(0xBA7C4)
    data = randn(rng, Float32, 16, 800)
    idx = build_index(HNSWIndex, data; M=8, ef_construction=80, rng=rng)

    queries = randn(MersenneTwister(0xC0DE), Float32, 16, 30)

    # Reference: call single-query API in a serial loop.
    ref = [query(idx, data, @view(queries[:, i]), 5; ef_search=40) for i in 1:30]

    # Batch path.
    got = query(idx, data, queries, 5; ef_search=40)

    @test length(got) == 30
    for i in 1:30
        @test length(got[i]) == length(ref[i])
        for j in eachindex(ref[i])
            # Exact id match — single-query and batch take the same code
            # path through _query_with_scratch and produce identical results
            # for identical (q, ef) inputs. Tie-breaks line up because the
            # heap sort is MergeSort-pinned.
            @test got[i][j].id == ref[i][j].id
            @test got[i][j].dist ≈ ref[i][j].dist
        end
    end
end

@testset "HNSW batch query: re-entrancy of public single-query API" begin
    # The single-query API must remain re-entrant. Two threads calling it
    # concurrently with the same (index, data) but different queries must
    # not interfere. (The internal _query_with_scratch is NOT re-entrant —
    # this asserts only the public boundary.)
    if Threads.nthreads() >= 2
        rng = MersenneTwister(0x5A1F)
        data = randn(rng, Float32, 12, 600)
        idx = build_index(HNSWIndex, data; M=8, ef_construction=60, rng=rng)
        queries = [randn(MersenneTwister(0x100 + i), Float32, 12) for i in 1:32]

        ref = [query(idx, data, q, 5; ef_search=30) for q in queries]
        got = Vector{Vector{ManifoldANN.Neighbor{Float32}}}(undef, 32)
        Threads.@threads for i in 1:32
            got[i] = query(idx, data, queries[i], 5; ef_search=30)
        end
        for i in 1:32
            @test [n.id for n in got[i]] == [n.id for n in ref[i]]
        end
    end
end

@testset "HNSW batch query: empty input edge cases" begin
    rng = MersenneTwister(1)
    data = randn(rng, Float32, 8, 100)
    idx = build_index(HNSWIndex, data; M=4, ef_construction=20, rng=rng)

    # Zero queries
    empty_q = randn(rng, Float32, 8, 0)
    @test query(idx, data, empty_q, 3) == Vector{Vector{ManifoldANN.Neighbor{Float32}}}()

    # k=0
    qs = randn(rng, Float32, 8, 5)
    res0 = query(idx, data, qs, 0)
    @test length(res0) == 5
    for r in res0; @test isempty(r); end
end

@testset "HNSW batch query: nested @threads over batch query is safe" begin
    # Smoke test: a caller wraps the batch query in their own @threads loop.
    # The inner batch query spawns its own workers — same-pool tasks should
    # cooperate without deadlock, even if total task count exceeds nthreads.
    if Threads.nthreads() >= 2
        rng = MersenneTwister(0x33)
        data = randn(rng, Float32, 12, 600)
        idx = build_index(HNSWIndex, data; M=8, ef_construction=60, rng=rng)

        # 3 outer batches, each with 20 queries.
        all_qs = [randn(MersenneTwister(0x40 + i), Float32, 12, 20) for i in 1:3]
        all_results = Vector{Any}(undef, 3)
        Threads.@threads for i in 1:3
            all_results[i] = query(idx, data, all_qs[i], 5; ef_search=30)
        end
        for i in 1:3
            @test length(all_results[i]) == 20
            for r in all_results[i]
                @test length(r) == 5
                @test all(1 <= n.id <= 600 for n in r)
            end
        end
    end
end

@testset "HNSW batch query: correctness under stress (≥ 2 threads)" begin
    if Threads.nthreads() >= 2
        rng = MersenneTwister(0x11)
        data = randn(rng, Float32, 16, 1500)
        idx = build_index(HNSWIndex, data; M=8, ef_construction=80, rng=rng)

        queries = randn(MersenneTwister(0x22), Float32, 16, 200)

        # Reference from serial single-query path.
        ref_ids = [[n.id for n in query(idx, data, @view(queries[:, i]), 10; ef_search=40)] for i in 1:200]

        # 5 batch runs, all must match the reference exactly. The worker-pool
        # scratch reuse and Channel scheduling make insertion order
        # nondeterministic for build, but query is deterministic given a
        # fixed graph.
        for trial in 1:5
            got = query(idx, data, queries, 10; ef_search=40)
            for i in 1:200
                @test [n.id for n in got[i]] == ref_ids[i]
            end
        end
    end
end
