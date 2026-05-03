#!/usr/bin/env julia
# Regression harness for HNSW perf work. Captures:
#   - build wall-clock + alloc bytes
#   - graph signature (per-node sorted neighbor lists, hashed) — must be
#     bitwise-identical across single-threaded refactors
#   - recall@10 vs brute force at fixed ef_search
#
# Usage:
#   julia --project=. -t 1 scripts/hnsw_regression_check.jl save  baseline.jls
#   julia --project=. -t 1 scripts/hnsw_regression_check.jl check baseline.jls
#
# The graph-signature check guarantees single-threaded refactors are exact.

using Random, Printf, LinearAlgebra, Serialization, SHA
using ManifoldANN
const MA = ManifoldANN

const N    = parse(Int, get(ENV, "HNSW_N", "20000"))
const D    = 32
const M    = 16
const EFC  = 200
const EFS  = 80
const K    = 10
const NQ   = 200
const SEED = 0xC0FFEE

BLAS.set_num_threads(1)

function build_with_seed()
    rng = MersenneTwister(SEED)
    data = randn(rng, Float32, D, N)
    GC.gc()
    rng_build = MersenneTwister(UInt(SEED) ⊻ 0xDEADBEEF)
    t0 = time_ns(); a0 = Base.gc_bytes()
    idx = MA.build_index(MA.HNSWIndex, data;
                         M=M, ef_construction=EFC, ef_search=EFS, rng=rng_build)
    t1 = time_ns(); a1 = Base.gc_bytes()
    return idx, data, (t1-t0)/1e9, (a1-a0)/(1024^2)
end

function graph_signature(idx)
    ctx = SHA2_256_CTX()
    update!(ctx, reinterpret(UInt8, [Int64(length(idx.layers))]))
    update!(ctx, reinterpret(UInt8, [Int64(idx.entry_point), Int64(idx.max_layer)]))
    for (li, layer) in enumerate(idx.layers)
        update!(ctx, reinterpret(UInt8, [Int64(li), Int64(length(layer))]))
        for (ni, nbrs) in enumerate(layer)
            sorted = sort(nbrs)
            update!(ctx, reinterpret(UInt8, [Int64(ni), Int64(length(sorted))]))
            isempty(sorted) || update!(ctx, reinterpret(UInt8, Int64.(sorted)))
        end
    end
    return bytes2hex(digest!(ctx))
end

function eval_recall(idx, data)
    rng = MersenneTwister(UInt(SEED) ⊻ 0xBEEF)
    queries = randn(rng, Float32, D, NQ)
    brute = MA.build_index(MA.BruteForceIndex, data)
    hits = 0
    total = 0
    for j in 1:NQ
        q = @view queries[:, j]
        a = MA.query(idx, data, q, K; ef_search=EFS)
        t = MA.query(brute, data, q, K)
        a_ids = Set(MA.neighbor_ids(a))
        t_ids = Set(MA.neighbor_ids(t))
        hits += length(intersect(a_ids, t_ids))
        total += K
    end
    return hits / total
end

function run()
    println("Threads.nthreads() = $(Threads.nthreads())   BLAS = $(BLAS.get_num_threads())")
    # warm
    let small = randn(Float32, D, 1_000)
        MA.build_index(MA.HNSWIndex, small; M=M, ef_construction=EFC)
    end
    GC.gc()
    idx, data, dt, mb = build_with_seed()
    sig = graph_signature(idx)
    rec = eval_recall(idx, data)
    return (build_s=dt, alloc_mb=mb, sig=sig, recall=rec)
end

function main(args)
    r = run()
    @printf "build = %.3fs   alloc = %.1f MB   recall@%d = %.4f   sig = %s\n" r.build_s r.alloc_mb K r.recall r.sig[1:16]
    if length(args) >= 2 && args[1] == "save"
        open(args[2], "w") do io; serialize(io, r); end
        println("saved → $(args[2])")
    elseif length(args) >= 2 && args[1] == "check"
        baseline = open(deserialize, args[2])
        @printf "baseline:  build = %.3fs   alloc = %.1f MB   recall@%d = %.4f   sig = %s\n" baseline.build_s baseline.alloc_mb K baseline.recall baseline.sig[1:16]
        ok = r.sig == baseline.sig
        @printf "graph signature match: %s\n" (ok ? "YES (bitwise-identical)" : "NO")
        if !ok
            # also report whether recall matches even if graph differs
            @printf "recall delta: %+.4f\n" (r.recall - baseline.recall)
        end
        @printf "speedup: %.2f×   alloc reduction: %.2f×\n" (baseline.build_s/r.build_s) (baseline.alloc_mb/r.alloc_mb)
        ok || exit(2)
    end
end

main(ARGS)
