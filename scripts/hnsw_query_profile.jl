#!/usr/bin/env julia
# Profile the HNSW QUERY path. Mirrors hnsw_profile.jl but for query.
# Single-threaded by design.
#
# Run: julia --project=. -t 1 scripts/hnsw_query_profile.jl

using Random, Printf, Profile, LinearAlgebra
using ManifoldANN
const MA = ManifoldANN

const N    = parse(Int, get(ENV, "HNSW_N", "50000"))
const D    = parse(Int, get(ENV, "HNSW_D", "32"))
const M    = 16
const EFC  = 200
const EFS  = 80
const K    = 10
const NQ   = 5_000
const SEED = 0xC0FFEE

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads(), "   BLAS = ", BLAS.get_num_threads())

# ---- Data ------------------------------------------------------------------
Random.seed!(SEED)
data = randn(Float32, D, N)
queries = randn(Float32, D, NQ)

# ---- Build (warm + real, but only the build is shared) ---------------------
let small = randn(Float32, D, 1_000)
    MA.build_index(MA.HNSWIndex, small; M=M, ef_construction=EFC)
end
GC.gc()

println("\n=== Build (warmed) ===")
@time idx = MA.build_index(MA.HNSWIndex, data; M=M, ef_construction=EFC)

# ---- Warmup query path -----------------------------------------------------
let
    MA.query(idx, data, @view(queries[:, 1]), K; ef_search=EFS)
end
GC.gc()

# ---- Single query timings + alloc ------------------------------------------
println("\n=== Per-query timing & alloc (single-thread) ===")
let
    GC.gc(); a0 = Base.gc_bytes()
    t = @elapsed for j in 1:NQ
        MA.query(idx, data, @view(queries[:, j]), K; ef_search=EFS)
    end
    a1 = Base.gc_bytes()
    @printf "  %d single queries: %.3fs total → %.1f µs/query, qps %.0f\n" NQ t (t*1e6/NQ) (NQ/t)
    @printf "  alloc: %.1f MB total → %.2f KB/query\n" ((a1-a0)/(1024^2)) ((a1-a0)/NQ/1024)
end

# ---- @profile sample on the per-query loop ---------------------------------
println("\n=== @profile (CPU sampling, $NQ queries) ===")
GC.gc()
Profile.clear()
Profile.init(n = 10^7, delay = 0.001)
@profile for j in 1:NQ
    MA.query(idx, data, @view(queries[:, j]), K; ef_search=EFS)
end

println("\n--- flat (top 30 self-counts, mincount=20) ---")
Profile.print(format=:flat, sortedby=:count, mincount=20, maxdepth=20)

println("\n--- tree (combined, threshold 2%) ---")
let nsamp = length(Profile.fetch())
    Profile.print(format=:tree, mincount=Int(round(0.02 * nsamp)), maxdepth=25, C=false)
end

println("\nDone.")
