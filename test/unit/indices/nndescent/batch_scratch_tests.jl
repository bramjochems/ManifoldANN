using Test
using Random
using ManifoldANN

# Tests for NN-Descent per-task batch scratch pooling.

@testset "NN-Descent batch scratch: matches single-query loop (serial)" begin
    rng_seed = MersenneTwister(0x5C8A7C4)
    data = randn(rng_seed, Float32, 12, 600)
    n_queries = 80   # > BATCH_THREAD_THRESHOLD so threaded path triggers under -t > 1
    queries = randn(rng_seed, Float32, 12, n_queries)
    idx = build_index(NNDescentIndex, data; k = 10, max_iterations = 4,
                      threaded = false, rng = MersenneTwister(0xA))

    seed_rng = MersenneTwister(123)
    child_rngs = ManifoldANN.spawn_child_rngs(seed_rng, n_queries)
    ref = [query(idx, data, view(queries, :, i), 5; ef_search = 20, rng = child_rngs[i])
           for i in 1:n_queries]

    got = query(idx, data, queries, 5; ef_search = 20, rng = MersenneTwister(123))
    @test length(got) == n_queries
    for i in 1:n_queries
        @test [n.id for n in got[i]] == [n.id for n in ref[i]]
        @test [n.dist for n in got[i]] == [n.dist for n in ref[i]]
    end
end

@testset "NN-Descent batch scratch: empty/zero-k edge cases" begin
    data = randn(MersenneTwister(1), Float32, 8, 200)
    idx = build_index(NNDescentIndex, data; k = 8, max_iterations = 3,
                      threaded = false, rng = MersenneTwister(2))
    empty_q = Matrix{Float32}(undef, 8, 0)
    got = query(idx, data, empty_q, 5; ef_search = 20, rng = MersenneTwister(3))
    @test got isa Vector
    @test isempty(got)

    qs = randn(MersenneTwister(4), Float32, 8, 70)
    got_zero = query(idx, data, qs, 0; ef_search = 20, rng = MersenneTwister(5))
    @test length(got_zero) == 70
    @test all(isempty, got_zero)
end

@testset "NN-Descent batch scratch: dimension mismatch surfaces" begin
    data = randn(MersenneTwister(6), Float32, 8, 200)
    idx = build_index(NNDescentIndex, data; k = 6, max_iterations = 2,
                      threaded = false, rng = MersenneTwister(7))
    bad_qs = randn(MersenneTwister(8), Float32, 99, 70)
    @test_throws DimensionMismatch query(idx, data, bad_qs, 3)
end

@testset "NN-Descent batch scratch: threaded equivalence (≥ 2 threads)" begin
    if Threads.nthreads() >= 2
        data = randn(MersenneTwister(0x33), Float32, 16, 800)
        n_queries = 200
        queries = randn(MersenneTwister(0x44), Float32, 16, n_queries)
        idx = build_index(NNDescentIndex, data; k = 12, max_iterations = 4,
                          threaded = false, rng = MersenneTwister(0x55))

        # Same seed must yield identical results regardless of thread count
        # because child RNGs are spawned deterministically from the parent.
        a = query(idx, data, queries, 5; ef_search = 25, rng = MersenneTwister(99))
        b = query(idx, data, queries, 5; ef_search = 25, rng = MersenneTwister(99))
        @test length(a) == length(b) == n_queries
        for i in 1:n_queries
            @test [n.id for n in a[i]] == [n.id for n in b[i]]
        end

        # Cross-thread-count determinism: batch result must equal the
        # serial single-query loop driven by the same child-RNG sequence,
        # regardless of how many workers process the batch.
        seed_rng = MersenneTwister(99)
        child_rngs = ManifoldANN.spawn_child_rngs(seed_rng, n_queries)
        ref = [query(idx, data, view(queries, :, i), 5; ef_search = 25,
                     rng = child_rngs[i]) for i in 1:n_queries]
        for i in 1:n_queries
            @test [n.id for n in a[i]] == [n.id for n in ref[i]]
            @test [n.dist for n in a[i]] == [n.dist for n in ref[i]]
        end
    end
end

@testset "NN-Descent single-query API still allocates fresh (re-entrant)" begin
    # Single-query path must remain re-entrant: must not mutate any shared
    # state on `index`. Concurrent calls from multiple tasks against the
    # same index/data must produce the SAME results as serial calls.
    data = randn(MersenneTwister(0x77), Float32, 10, 400)
    idx = build_index(NNDescentIndex, data; k = 10, max_iterations = 3,
                      threaded = false, rng = MersenneTwister(0x88))
    n_queries = 64
    queries = [randn(MersenneTwister(UInt64(0x900 + i)), Float32, 10) for i in 1:n_queries]

    # Serial reference (each call gets its OWN rng so result is determined
    # purely by the rng).
    serial = Vector{Vector{Neighbor{Float32}}}(undef, n_queries)
    for i in 1:n_queries
        serial[i] = query(idx, data, queries[i], 5; ef_search = 20,
                          rng = MersenneTwister(UInt64(0xBEEF + i)))
    end

    # Concurrent — same per-query RNG, must match.
    parallel = Vector{Vector{Neighbor{Float32}}}(undef, n_queries)
    Threads.@threads for i in 1:n_queries
        parallel[i] = query(idx, data, queries[i], 5; ef_search = 20,
                            rng = MersenneTwister(UInt64(0xBEEF + i)))
    end
    for i in 1:n_queries
        @test [n.id for n in parallel[i]] == [n.id for n in serial[i]]
    end
end

@testset "NN-Descent batch scratch: per-query allocation budget reduced" begin
    # Sanity check on the pool savings — a deliberately loose bound that will
    # still detect a regression to per-query falses(n_points) + Set + heap
    # allocation. Sized to be insensitive to GC heap noise.
    data = randn(MersenneTwister(0x1A), Float32, 16, 4000)
    n_queries = 200
    queries = randn(MersenneTwister(0x2B), Float32, 16, n_queries)
    idx = build_index(NNDescentIndex, data; k = 12, max_iterations = 4,
                      threaded = false, rng = MersenneTwister(0x3C))

    # Warm
    query(idx, data, queries, 5; ef_search = 20, rng = MersenneTwister(0x4D))
    GC.gc()

    bytes = @allocated query(idx, data, queries, 5; ef_search = 20,
                              rng = MersenneTwister(0x5E))
    # Pooled path measures ~36 KB/query at n=20k under the production bench
    # (commit message). At n=4000 with smaller beam the figure is comparable.
    # Pin 45 KB/query — tight enough to catch a regression that re-introduces
    # a per-query Set or BitVector (each ~500 B) without false-failing on GC
    # heap noise.
    @test bytes < 45 * 1024 * n_queries
end
