using Test
using Random
using ManifoldANN

# Tests for the threaded NN-Descent local-join. These are most meaningful at
# `-t >= 2` — the bugs they target (read-during-mutate races on heaps,
# threadid-indexed buffers under task migration) cannot fire at -t 1.

if Threads.nthreads() < 2
    @warn "NN-Descent threaded tests run at Threads.nthreads()=$(Threads.nthreads()); concurrency bugs cannot fire below -t 2"
end

# Structural invariants for an NN-Descent index built over `n` points with
# requested capacity `k`. Catches corruption (wrong-buffer writes, torn struct
# reads, duplicate insertion) that a recall-only test might miss.
function _nnd_check_invariants(idx, n::Int, k::Int)
    @test idx.n_points == n
    @test idx.k == k
    @test length(idx.neighbors) == n
    for i in 1:n
        adj = idx.neighbors[i]
        # No self-loop
        @test i ∉ adj
        # No duplicates
        @test allunique(adj)
        # All ids in valid range
        @test all(1 <= id <= n for id in adj)
        @test all(id != 0 for id in adj)
        # Length within configured cap. FullSymmetry (default) may push past
        # k via reverse edges, but the upper bound is still all other points.
        @test length(adj) <= n - 1
    end
end

@testset "NN-Descent threaded build: structural invariants (single build)" begin
    n, k = 2000, 20
    data = randn(MersenneTwister(0x11), Float32, 16, n)
    idx = build_index(NNDescentIndex, data; k=k, threaded=true,
                      rng=MersenneTwister(0xA1))
    _nnd_check_invariants(idx, n, k)
end

@testset "NN-Descent threaded build: stress (10 builds, structural + recall floor)" begin
    # Repeated builds at the same config catch races that fire only
    # occasionally. 10 trials is a compromise between coverage and CI time.
    n, k = 2000, 20
    data = randn(MersenneTwister(0x21), Float32, 16, n)
    brute = build_index(BruteForceIndex, data)
    test_qs = [randn(MersenneTwister(0x100 + i), Float32, 16) for i in 1:30]

    recalls = Float64[]
    for trial in 1:10
        idx = build_index(NNDescentIndex, data; k=k, threaded=true,
                          rng=MersenneTwister(UInt64(0x500 + trial)))
        _nnd_check_invariants(idx, n, k)
        h, t = 0, 0
        for q in test_qs
            a = Set(r.id for r in query(idx, data, q, 5))
            tr = Set(r.id for r in query(brute, data, q, 5))
            h += length(intersect(a, tr)); t += 5
        end
        push!(recalls, h / t)
    end
    # Floor exists to catch corruption-induced collapse, not to assert
    # quality. Typical recall here is ~0.95+.
    @test all(r >= 0.70 for r in recalls)
end

@testset "NN-Descent serial build: bitwise determinism with same seed" begin
    # Mirrors the HNSW serial determinism check. The serial path
    # (`threaded=false`) must be reproducible across two builds with the
    # same rng. Threaded path is intentionally non-deterministic — see
    # build_index docstring.
    data = randn(MersenneTwister(0xBEEF), Float32, 12, 800)
    a = build_index(NNDescentIndex, data; k=10, threaded=false,
                    rng=MersenneTwister(0xCAFE))
    b = build_index(NNDescentIndex, data; k=10, threaded=false,
                    rng=MersenneTwister(0xCAFE))
    @test length(a.neighbors) == length(b.neighbors)
    for i in eachindex(a.neighbors)
        # Sorted-id equality: serial path is deterministic up to graph
        # signature.
        @test sort(a.neighbors[i]) == sort(b.neighbors[i])
    end
end

@testset "NN-Descent threaded build: nondeterminism is real (≥ 2 threads)" begin
    # Two threaded builds with the SAME seed should differ — proves the
    # threaded path is actually parallel. Skipped under -t 1.
    if Threads.nthreads() >= 2
        n, k = 1500, 12
        data = randn(MersenneTwister(0x5EED), Float32, 16, n)
        a = build_index(NNDescentIndex, data; k=k, threaded=true,
                        rng=MersenneTwister(1))
        b = build_index(NNDescentIndex, data; k=k, threaded=true,
                        rng=MersenneTwister(1))
        any_diff = false
        for i in eachindex(a.neighbors)
            if sort(a.neighbors[i]) != sort(b.neighbors[i])
                any_diff = true; break
            end
        end
        @test any_diff
    end
end
