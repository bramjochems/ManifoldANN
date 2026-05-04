using Test
using Random
using LinearAlgebra
using ManifoldANN

# NN-Descent regression: serial build is deterministic for a fixed seed,
# and recall@10 vs brute force stays above a floor (serial and threaded).
#
# Sized to run in ~seconds. The perf-tracking version (build wall-clock,
# alloc bytes, larger N) lives in scripts/perf/ as a developer tool.

const _NND_REG_N    = 2_000
const _NND_REG_D    = 16
const _NND_REG_K    = 15
const _NND_REG_NQ   = 50
const _NND_REG_KQ   = 10
const _NND_REG_SEED = 0xC0FFEE
const _NND_REG_RECALL_FLOOR = 0.85

function _nnd_reg_build(threaded::Bool)
    rng_data = MersenneTwister(_NND_REG_SEED)
    data = randn(rng_data, Float32, _NND_REG_D, _NND_REG_N)
    rng_build = MersenneTwister(UInt(_NND_REG_SEED) ⊻ 0xDEADBEEF)
    idx = build_index(NNDescentIndex, data; k=_NND_REG_K,
                      rng=rng_build, threaded=threaded)
    return idx, data
end

_nnd_reg_signature(idx) = [sort(nbrs) for nbrs in idx.neighbors]

function _nnd_reg_recall(idx, data)
    rng_q = MersenneTwister(UInt(_NND_REG_SEED) ⊻ 0xBEEF)
    queries = randn(rng_q, Float32, _NND_REG_D, _NND_REG_NQ)
    brute = build_index(BruteForceIndex, data)
    hits = 0
    total = 0
    for j in 1:_NND_REG_NQ
        q = @view queries[:, j]
        approx = query(idx, data, q, _NND_REG_KQ)
        truth = query(brute, data, q, _NND_REG_KQ)
        hits += length(intersect(Set(ManifoldANN.neighbor_ids(approx)),
                                 Set(ManifoldANN.neighbor_ids(truth))))
        total += _NND_REG_KQ
    end
    return hits / total
end

@testset "NN-Descent regression" begin
    @testset "serial build is deterministic" begin
        idx_a, _ = _nnd_reg_build(false)
        idx_b, _ = _nnd_reg_build(false)
        @test _nnd_reg_signature(idx_a) == _nnd_reg_signature(idx_b)
    end

    @testset "serial recall@$(_NND_REG_KQ) ≥ $(_NND_REG_RECALL_FLOOR)" begin
        idx, data = _nnd_reg_build(false)
        @test _nnd_reg_recall(idx, data) ≥ _NND_REG_RECALL_FLOOR
    end

    @testset "threaded recall@$(_NND_REG_KQ) ≥ $(_NND_REG_RECALL_FLOOR)" begin
        idx, data = _nnd_reg_build(true)
        @test _nnd_reg_recall(idx, data) ≥ _NND_REG_RECALL_FLOOR
    end
end
