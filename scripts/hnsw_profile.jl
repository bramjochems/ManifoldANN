#!/usr/bin/env julia
# HNSW build-time profiling. Single-threaded by design: we want to falsify
# hypotheses H1–H4 (per-call distance, _prune_list!, _search_layer alloc/sort,
# heap mechanics) before considering threading.
#
# Run: julia --project=. -t 1 scripts/hnsw_profile.jl

using Random, Printf, Profile, LinearAlgebra
using ManifoldANN
const MA = ManifoldANN

# ---- Config ----------------------------------------------------------------
const N    = 50_000
const D    = 32
const M    = 16
const EFC  = 200
const SEED = 0xC0FFEE

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads(), "   BLAS = ", BLAS.get_num_threads())

# ---- Data ------------------------------------------------------------------
Random.seed!(SEED)
data = randn(Float32, D, N)

# ---- Warm-up (compile) -----------------------------------------------------
let small = randn(Float32, D, 1_000)
    MA.build_index(MA.HNSWIndex, small; M=M, ef_construction=EFC)
end
GC.gc()

# ---- Wall-clock + alloc baseline ------------------------------------------
println("\n=== Build wall-clock + allocs (n=$N d=$D M=$M ef_c=$EFC) ===")
GC.gc(); t0 = time_ns(); a0 = Base.gc_bytes()
idx = MA.build_index(MA.HNSWIndex, data; M=M, ef_construction=EFC)
t1 = time_ns(); a1 = Base.gc_bytes()
build_s = (t1 - t0) / 1e9
build_mb = (a1 - a0) / (1024^2)
@printf "  build: %.3f s   alloc: %.1f MB   (%.1f µs / point)\n" build_s build_mb (build_s*1e6/N)

# ---- Profile sample on a fresh build --------------------------------------
println("\n=== @profile (CPU sampling) ===")
GC.gc()
Profile.clear()
Profile.init(n = 10^7, delay = 0.001)
@profile MA.build_index(MA.HNSWIndex, data; M=M, ef_construction=EFC)

# Flat profile, top 30 self-time entries
println("\n--- flat (top 30 by self-count) ---")
Profile.print(format=:flat, sortedby=:count, mincount=20, maxdepth=20)

println("\n--- tree (combined, threshold 2%) ---")
Profile.print(format=:tree, mincount=Int(round(0.02 * length(Profile.fetch()))), maxdepth=25, C=false)

# ---- Targeted micro-benchmarks --------------------------------------------
# Build a "mid-graph" index (half the points) so insert!/prune are exercised
# against a populated structure, then time each hot path on a single point.
println("\n=== Micro: mid-graph state (n_built = $(N÷2)) ===")
mid = MA.build_index(MA.HNSWIndex, @view(data[:, 1:N÷2]); M=M, ef_construction=EFC)
println("  layers        = ", length(mid.layers))
println("  max_layer     = ", mid.max_layer)
deg0 = [mid.layers[1].degree[i] for i in 1:mid.n_points]
@printf "  layer-0 deg   mean=%.1f  max=%d  p99=%d\n" (sum(deg0)/length(deg0)) maximum(deg0) sort(deg0)[max(1, end - end÷100)]

# H1: per-call distance overhead. Time a single _greedy_descent + _search_layer
# at layer 0 with an out-of-sample query.
println("\n=== H1: distance / search_layer cost on one query ===")
q = @view data[:, N÷2 + 1]
let
    # warm
    MA._greedy_descent(mid, 0, mid.entry_point, q, data)
    GC.gc()
    t = @elapsed for _ in 1:1000
        MA._greedy_descent(mid, 0, mid.entry_point, q, data)
    end
    @printf "  _greedy_descent layer 0 : %.2f µs/call (1000×)\n" (t*1e6/1000)
end
let
    entry = MA.NeighborCandidate(mid.entry_point,
            MA.default_distance(@view(data[:, mid.entry_point]), q))
    # warm
    MA._search_layer(mid, 0, entry, q, data, EFC)
    GC.gc(); a0 = Base.gc_bytes()
    t = @elapsed for _ in 1:200
        MA._search_layer(mid, 0, entry, q, data, EFC)
    end
    a1 = Base.gc_bytes()
    @printf "  _search_layer ef=%d     : %.2f µs/call   alloc %.1f KB/call (200×)\n" EFC (t*1e6/200) ((a1-a0)/200/1024)
end

# H2: _prune_slot! cost on a saturated neighbor list.
println("\n=== H2: _prune_slot! on a saturated layer-0 list ===")
let
    # find a node whose layer-0 slab column is at-capacity, copy it into a
    # synthetic 1-column HNSWLayer with one extra appended id so the prune
    # actually does work each iteration.
    layer0 = mid.layers[1]
    saturated = findfirst(i -> layer0.degree[i] >= M, 1:mid.n_points)
    @assert saturated !== nothing
    base_deg = layer0.degree[saturated]
    base_ids = collect(MA.layer_neighbors(layer0, saturated))
    # build a single-column synthetic layer reusing `mid`'s neighbor_policy
    cap = base_deg + 1
    function make_synth()
        synth = MA.HNSWLayer(zeros(Int, cap, 1), zeros(Int, 1), cap)
        for i in 1:base_deg
            synth.neighbors[i, 1] = base_ids[i]
        end
        synth.neighbors[cap, 1] = N÷2 + 1
        synth.degree[1] = cap
        return synth
    end
    # warm
    let synth = make_synth()
        MA._prune_slot!(mid, synth, 1, data, M)
    end
    GC.gc(); a0 = Base.gc_bytes()
    iters = 5_000
    t = @elapsed for _ in 1:iters
        synth = make_synth()
        MA._prune_slot!(mid, synth, 1, data, M)
    end
    a1 = Base.gc_bytes()
    @printf "  _prune_slot! (M=%d, list=%d→%d) : %.2f µs/call   alloc %.2f KB/call\n" M cap M (t*1e6/iters) ((a1-a0)/iters/1024)
end

# H3: _search_layer alloc breakdown — how much is the final `sort(...)` copy?
println("\n=== H3: _search_layer terminal sort allocation ===")
let
    entry = MA.NeighborCandidate(mid.entry_point,
            MA.default_distance(@view(data[:, mid.entry_point]), q))
    res = MA._search_layer(mid, 0, entry, q, data, EFC)
    @printf "  result length = %d   eltype = %s\n" length(res) eltype(res)
    GC.gc(); a0 = Base.gc_bytes()
    for _ in 1:1000
        sort(res, by = c -> c.dist)
    end
    a1 = Base.gc_bytes()
    @printf "  sort copy  : %.2f KB/call (1000×)\n" ((a1-a0)/1000/1024)
end

# H4: heap mechanics on its own
println("\n=== H4: BestCandidatesHeap push throughput ===")
let
    heap = MA.BestCandidatesHeap{Float32}(MA.NeighborCandidate{Float32}[], EFC)
    rng = MersenneTwister(1)
    cands = [MA.NeighborCandidate(i, rand(rng, Float32)) for i in 1:10_000]
    # warm
    for c in cands; push!(heap, c); end
    iters = 50
    GC.gc()
    t = @elapsed for _ in 1:iters
        h = MA.BestCandidatesHeap{Float32}(MA.NeighborCandidate{Float32}[], EFC)
        for c in cands; push!(h, c); end
    end
    @printf "  10k pushes into ef=%d heap : %.2f µs (%.1f ns/push)\n" EFC (t*1e6/iters) (t*1e9/iters/10_000)
end

println("\nDone.")
