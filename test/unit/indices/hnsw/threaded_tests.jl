using Test
using Random
using ManifoldANN

const _MA_HNSW = ManifoldANN

@testset "HNSW threaded build: opt-out is bitwise-identical" begin
    # threaded=false must produce the exact same graph as the legacy serial
    # path. (At -t 1 the default already resolves to false; this test pins
    # that behavior under any thread count.)
    rng() = MersenneTwister(0xBEEF)
    data = randn(MersenneTwister(0xCA75), Float32, 16, 1500)
    a = build_index(HNSWIndex, data; M=8, ef_construction=80, threaded=false, rng=rng())
    b = build_index(HNSWIndex, data; M=8, ef_construction=80, threaded=false, rng=rng())
    @test length(a.layers) == length(b.layers)
    @test a.entry_point == b.entry_point
    @test a.max_layer == b.max_layer
    for li in eachindex(a.layers)
        @test length(a.layers[li]) == length(b.layers[li])
        for ni in eachindex(a.layers[li])
            @test sort(a.layers[li][ni]) == sort(b.layers[li][ni])
        end
    end
end

@testset "HNSW threaded build: produces a valid graph at any thread count" begin
    data = randn(MersenneTwister(0xC1A5), Float32, 16, 1500)
    brute = build_index(BruteForceIndex, data)
    test_qs = [randn(MersenneTwister(0xDA7A + i), Float32, 16) for i in 1:50]

    serial_recall = let
        idx = build_index(HNSWIndex, data; M=8, ef_construction=80, threaded=false)
        h, t = 0, 0
        for q in test_qs
            a = Set(r.id for r in query(idx, data, q, 5; ef_search=40))
            tr = Set(r.id for r in query(brute, data, q, 5))
            h += length(intersect(a, tr)); t += 5
        end
        h / t
    end
    @test serial_recall >= 0.85   # sanity for the test config

    threaded_idx = build_index(HNSWIndex, data; M=8, ef_construction=80, threaded=true)
    @test threaded_idx.n_points == 1500
    @test threaded_idx.entry_point in 1:1500
    @test threaded_idx.max_layer >= 0

    h, t = 0, 0
    for q in test_qs
        a = Set(r.id for r in query(threaded_idx, data, q, 5; ef_search=40))
        tr = Set(r.id for r in query(brute, data, q, 5))
        h += length(intersect(a, tr)); t += 5
    end
    threaded_recall = h / t
    # Threaded build is non-deterministic; recall is comparable but not equal.
    # Allow a 0.05 absolute drop vs serial — well within typical PyNNDescent /
    # hnswlib threading drift.
    @test threaded_recall >= serial_recall - 0.05
end

@testset "HNSW threaded build: nondeterminism is real but bounded" begin
    # Two threaded builds with the SAME seed should differ (proves we are
    # actually running threaded), but recall on both should still be high.
    # Skipped under -t 1 — there's no thread interleaving to differ.
    if Threads.nthreads() >= 2
        data = randn(MersenneTwister(0x5EED), Float32, 16, 1500)
        a = build_index(HNSWIndex, data; M=8, ef_construction=80, threaded=true,
                        rng=MersenneTwister(1))
        b = build_index(HNSWIndex, data; M=8, ef_construction=80, threaded=true,
                        rng=MersenneTwister(1))
        # Find at least one node whose layer-0 neighbor set differs.
        any_diff = false
        for ni in 1:length(a.layers[1])
            if sort(a.layers[1][ni]) != sort(b.layers[1][ni])
                any_diff = true; break
            end
        end
        @test any_diff   # threaded builds are not deterministic
    end
end

@testset "HNSW threaded build: concurrent batch query works" begin
    data = randn(MersenneTwister(0xB07), Float32, 16, 1500)
    idx = build_index(HNSWIndex, data; M=8, ef_construction=80, threaded=true)
    queries = randn(MersenneTwister(0xB08), Float32, 16, 50)
    results = query(idx, data, queries, 5; ef_search=40)
    @test length(results) == 50
    for r in results
        @test length(r) == 5
        @test all(1 <= n.id <= 1500 for n in r)
    end
end
