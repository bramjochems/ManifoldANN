using Test
using Random
using LinearAlgebra
using ManifoldANN

# HNSW regression: single-threaded build is deterministic for a fixed seed,
# and recall@10 vs brute force stays above a floor.
#
# Determinism is asserted by comparing per-node sorted neighbour lists across
# two independent builds with the same RNG seed — a refactor that perturbs
# the build order silently fails this. The recall floor catches the "still
# deterministic, but converged to a worse graph" failure mode.
#
# Sized to run in ~seconds; the perf-tracking version (build wall-clock,
# alloc bytes, larger N) lives in scripts/perf/ as a developer tool.

const _HNSW_REG_N    = 2_000
const _HNSW_REG_D    = 16
const _HNSW_REG_M    = 16
const _HNSW_REG_EFC  = 100
const _HNSW_REG_EFS  = 50
const _HNSW_REG_K    = 10
const _HNSW_REG_NQ   = 50
const _HNSW_REG_SEED = 0xC0FFEE
const _HNSW_REG_RECALL_FLOOR = 0.90

function _hnsw_reg_build()
    rng_data = MersenneTwister(_HNSW_REG_SEED)
    data = randn(rng_data, Float32, _HNSW_REG_D, _HNSW_REG_N)
    rng_build = MersenneTwister(UInt(_HNSW_REG_SEED) ⊻ 0xDEADBEEF)
    idx = build_index(HNSWIndex, data;
                      M=_HNSW_REG_M, ef_construction=_HNSW_REG_EFC,
                      ef_search=_HNSW_REG_EFS, rng=rng_build)
    return idx, data
end

function _hnsw_reg_signature(idx)
    sig = Vector{Vector{Vector{Int}}}(undef, length(idx.layers))
    n = idx.n_points
    for (li, layer) in enumerate(idx.layers)
        per_layer = Vector{Vector{Int}}(undef, n)
        for ni in 1:n
            per_layer[ni] = sort!(collect(ManifoldANN.layer_neighbors(layer, ni)))
        end
        sig[li] = per_layer
    end
    return (n_layers=length(idx.layers), entry=idx.entry_point,
            max_layer=idx.max_layer, layers=sig)
end

@testset "HNSW regression" begin
    idx_a, data_a = _hnsw_reg_build()

    # HNSW build is intentionally only deterministic single-threaded — the
    # threaded path uses non-deterministic task scheduling. Skip the
    # determinism gate when the test session has multiple threads; preserve
    # it for anyone running `julia -t 1 ... runtests.jl`.
    if Threads.nthreads() == 1
        @testset "single-threaded build is deterministic" begin
            idx_b, _ = _hnsw_reg_build()
            sig_a = _hnsw_reg_signature(idx_a)
            sig_b = _hnsw_reg_signature(idx_b)
            @test sig_a.n_layers == sig_b.n_layers
            @test sig_a.entry == sig_b.entry
            @test sig_a.max_layer == sig_b.max_layer
            @test sig_a.layers == sig_b.layers
        end
    end

    @testset "recall@$(_HNSW_REG_K) ≥ $(_HNSW_REG_RECALL_FLOOR)" begin
        rng_q = MersenneTwister(UInt(_HNSW_REG_SEED) ⊻ 0xBEEF)
        queries = randn(rng_q, Float32, _HNSW_REG_D, _HNSW_REG_NQ)
        brute = build_index(BruteForceIndex, data_a)
        hits = 0
        total = 0
        for j in 1:_HNSW_REG_NQ
            q = @view queries[:, j]
            approx = query(idx_a, data_a, q, _HNSW_REG_K; ef_search=_HNSW_REG_EFS)
            truth = query(brute, data_a, q, _HNSW_REG_K)
            hits += length(intersect(Set(ManifoldANN.neighbor_ids(approx)),
                                     Set(ManifoldANN.neighbor_ids(truth))))
            total += _HNSW_REG_K
        end
        @test hits / total ≥ _HNSW_REG_RECALL_FLOOR
    end
end
