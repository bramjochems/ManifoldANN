#!/usr/bin/env julia
# Regression harness for NN-Descent perf work. Captures:
#   - threaded build wall-clock + alloc
#   - serial build wall-clock + alloc + bitwise graph signature (for determinism)
#   - recall@10 vs brute force (serial + threaded)
#
# Usage:
#   julia --project=. -t 4 scripts/nndescent_regression_check.jl save  baseline.jls
#   julia --project=. -t 4 scripts/nndescent_regression_check.jl check baseline.jls

using Random, Printf, LinearAlgebra, Serialization, SHA
using ManifoldANN
const MA = ManifoldANN

const N    = parse(Int, get(ENV, "NND_N", "20000"))
const D    = 32
const K    = 15
const NQ   = 200
const SEED = 0xC0FFEE

BLAS.set_num_threads(1)

function build(threaded)
    rng = MersenneTwister(SEED)
    data = randn(rng, Float32, D, N)
    GC.gc()
    rng_build = MersenneTwister(UInt(SEED) ⊻ 0xDEADBEEF)
    t0 = time_ns(); a0 = Base.gc_bytes()
    idx = MA.build_index(MA.NNDescentIndex, data; k=K, rng=rng_build, threaded=threaded)
    t1 = time_ns(); a1 = Base.gc_bytes()
    return idx, data, (t1-t0)/1e9, (a1-a0)/(1024^2)
end

function graph_signature(idx)
    ctx = SHA2_256_CTX()
    update!(ctx, reinterpret(UInt8, [Int64(length(idx.neighbors))]))
    for (ni, nbrs) in enumerate(idx.neighbors)
        sorted = sort(nbrs)
        update!(ctx, reinterpret(UInt8, [Int64(ni), Int64(length(sorted))]))
        isempty(sorted) || update!(ctx, reinterpret(UInt8, Int64.(sorted)))
    end
    return bytes2hex(digest!(ctx))
end

function eval_recall(idx, data)
    rng = MersenneTwister(UInt(SEED) ⊻ 0xBEEF)
    queries = randn(rng, Float32, D, NQ)
    brute = MA.build_index(MA.BruteForceIndex, data)
    hits = 0; total = 0
    for j in 1:NQ
        q = @view queries[:, j]
        a = MA.query(idx, data, q, 10)
        t = MA.query(brute, data, q, 10)
        a_ids = Set(MA.neighbor_ids(a))
        t_ids = Set(MA.neighbor_ids(t))
        hits += length(intersect(a_ids, t_ids)); total += 10
    end
    return hits / total
end

function run()
    println("Threads.nthreads() = $(Threads.nthreads())   BLAS = $(BLAS.get_num_threads())")
    let small = randn(Float32, D, 1_000)
        MA.build_index(MA.NNDescentIndex, small; k=K)
    end
    GC.gc()
    idx_s, data_s, dt_s, mb_s = build(false)
    sig_s = graph_signature(idx_s)
    rec_s = eval_recall(idx_s, data_s)
    GC.gc()
    idx_t, data_t, dt_t, mb_t = build(true)
    rec_t = eval_recall(idx_t, data_t)
    return (build_serial_s=dt_s, alloc_serial_mb=mb_s, sig_serial=sig_s, recall_serial=rec_s,
            build_threaded_s=dt_t, alloc_threaded_mb=mb_t, recall_threaded=rec_t)
end

function main(args)
    r = run()
    @printf "serial:   build = %.3fs   alloc = %.1f MB   recall@10 = %.4f   sig = %s\n" r.build_serial_s r.alloc_serial_mb r.recall_serial r.sig_serial[1:16]
    @printf "threaded: build = %.3fs   alloc = %.1f MB   recall@10 = %.4f\n" r.build_threaded_s r.alloc_threaded_mb r.recall_threaded
    if length(args) >= 2 && args[1] == "save"
        open(args[2], "w") do io; serialize(io, r); end
        println("saved → $(args[2])")
    elseif length(args) >= 2 && args[1] == "check"
        b = open(deserialize, args[2])
        @printf "baseline serial:   build = %.3fs   recall = %.4f   sig = %s\n" b.build_serial_s b.recall_serial b.sig_serial[1:16]
        @printf "baseline threaded: build = %.3fs   recall = %.4f\n" b.build_threaded_s b.recall_threaded
        @printf "serial sig match: %s\n" (r.sig_serial == b.sig_serial ? "YES (bitwise-identical)" : "NO")
        @printf "serial speedup: %.2f×   threaded speedup: %.2f×\n" (b.build_serial_s/r.build_serial_s) (b.build_threaded_s/r.build_threaded_s)
        @printf "serial recall delta: %+.4f   threaded recall delta: %+.4f\n" (r.recall_serial - b.recall_serial) (r.recall_threaded - b.recall_threaded)
    end
end

main(ARGS)
