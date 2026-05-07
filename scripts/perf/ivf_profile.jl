#!/usr/bin/env julia
# Localised performance probe for IVFFlatIndex.
#
# Goal: get a baseline measurement of build + query costs, and a profile
# breakdown of the query hot path, so we can verify (or falsify) the
# hypothesis that the per-list scan is dominated by scalar per-pair distance
# calls and would benefit from a precomputed-norms + GEMV rewrite.
#
# Self-contained, deps-free (uses stdlib Profile + @time). Designed to run
# in well under a minute on a laptop. NOT a fair head-to-head against FAISS;
# that comparison goes through the benchmarking/ harness.
#
# Run:
#   julia --project=. -t 1 scripts/perf/ivf_profile.jl
#
# Env knobs:
#   IVF_N=20000        training points
#   IVF_NQ=2000        test queries
#   IVF_D=64           ambient dimension
#   IVF_NLIST=128      number of inverted lists (k-means clusters)
#   IVF_NPROBE=8       lists scanned per query
#   IVF_K=10           neighbours per query
#   IVF_REPS=3         best-of timed reps for query throughput

using Random, Printf, LinearAlgebra, Statistics
using Profile
using ManifoldANN
const MA = ManifoldANN

const N      = parse(Int, get(ENV, "IVF_N", "20000"))
const NQ     = parse(Int, get(ENV, "IVF_NQ", "2000"))
const D      = parse(Int, get(ENV, "IVF_D", "64"))
const NLIST  = parse(Int, get(ENV, "IVF_NLIST", "128"))
const NPROBE = parse(Int, get(ENV, "IVF_NPROBE", "8"))
const K      = parse(Int, get(ENV, "IVF_K", "10"))
const REPS   = parse(Int, get(ENV, "IVF_REPS", "3"))
const SEED   = 0xBEEF

# Single-threaded so the per-query wall time is interpretable. The list scan
# is currently sequential anyway; a multithreaded variant is a separate axis.
BLAS.set_num_threads(1)

println("nthreads = $(Threads.nthreads()), BLAS = $(BLAS.get_num_threads())")
println("config: n=$N nq=$NQ d=$D nlist=$NLIST nprobe=$NPROBE k=$K reps=$REPS")
println()

Random.seed!(SEED)
data = randn(Float32, D, N)
queries = randn(Float32, D, NQ)

# Warmup: compile every path on a tiny dataset before any timed run.
let
    warm = randn(Float32, D, 500)
    warm_q = randn(Float32, D, 4)
    idx = MA.build_index(MA.IVFFlatIndex, warm; nlist=8, nprobe=2)
    MA.query(idx, warm, @view(warm_q[:, 1]), 5)
end

# ---------------------------------------------------------------------------
# Build timing
# ---------------------------------------------------------------------------
println("=== Build ===")
build_times = Float64[]
index = MA.build_index(MA.IVFFlatIndex, data; nlist=NLIST, nprobe=NPROBE)  # placeholder, overwritten below
for r in 1:REPS
    GC.gc()
    local t
    t = @elapsed (global index = MA.build_index(
        MA.IVFFlatIndex, data;
        nlist=NLIST, nprobe=NPROBE,
    ))
    push!(build_times, t)
    @printf "  rep %d: %.3f s\n" r t
end
@printf "  best build: %.3f s   (median %.3f s)\n\n" minimum(build_times) median(build_times)

# ---------------------------------------------------------------------------
# Query timing — wall time per query
# ---------------------------------------------------------------------------
println("=== Query (single-threaded) ===")
function run_queries(index, data, queries, k)
    s = 0
    @inbounds for i in 1:size(queries, 2)
        nbrs = MA.query(index, data, @view(queries[:, i]), k)
        s += length(nbrs)
    end
    return s
end

# Warm the query path on the real index.
run_queries(index, data, queries, K)

query_times = Float64[]
for r in 1:REPS
    GC.gc()
    t = @elapsed run_queries(index, data, queries, K)
    push!(query_times, t)
    @printf "  rep %d: %.3f s   (%.1f µs/query, %.0f QPS)\n" r t (1e6 * t / NQ) (NQ / t)
end
best_qt = minimum(query_times)
@printf "  best total: %.3f s   (%.1f µs/query, %.0f QPS)\n\n" best_qt (1e6 * best_qt / NQ) (NQ / best_qt)

# ---------------------------------------------------------------------------
# Allocation profile — does the hot path allocate?
# ---------------------------------------------------------------------------
println("=== Allocation snapshot for run_queries ===")
GC.gc()
allocs_before = Base.gc_num()
t_alloc = @elapsed run_queries(index, data, queries, K)
allocs_after = Base.gc_num()
gc_diff = Base.GC_Diff(allocs_after, allocs_before)
@printf "  total time:           %.3f s\n" t_alloc
@printf "  total allocations:    %d  (%.2f MiB)\n" gc_diff.malloc + gc_diff.realloc + gc_diff.poolalloc + gc_diff.bigalloc (gc_diff.allocd / 2^20)
@printf "  per-query allocs:     %.1f\n" ((gc_diff.malloc + gc_diff.realloc + gc_diff.poolalloc + gc_diff.bigalloc) / NQ)
@printf "  per-query bytes:      %.1f KiB\n\n" (gc_diff.allocd / NQ / 1024)

# ---------------------------------------------------------------------------
# Sampling profile of run_queries — where does query time go?
# ---------------------------------------------------------------------------
println("=== Profile: hot frames in run_queries ===")
Profile.clear()
# Run multiple times to accumulate enough samples on the inner frames.
Profile.@profile begin
    for _ in 1:5
        run_queries(index, data, queries, K)
    end
end

io = IOBuffer()
Profile.print(io;
    format = :flat,
    sortedby = :count,
    mincount = 20,
    maxdepth = 40,
)
output = String(take!(io))

# Filter out only obviously-uninteresting Base/runtime frames so the user
# sees the work the algorithm is actually doing. Keep everything else.
SKIP_PATTERNS = ["./client.jl", "./Base.jl", "ScopedValues",
                 "task.jl", "include(", "Profile.jl"]
println("(showing frames with count >= 20; trivial Base/runtime frames hidden)")
println()
for line in split(output, '\n')
    if isempty(line)
        println(line); continue
    end
    if !any(p -> occursin(p, line), SKIP_PATTERNS)
        println(line)
    end
end

println()
println("Tip: re-run with IVF_N=100000 IVF_NQ=5000 for a heavier profile.")
