#!/usr/bin/env julia
# Per-phase timing breakdown: MANN-NNDescent vs NND.jl.
# Times spent in distance evaluation, heap operations, and other inner-loop
# overhead, plus count of distance evaluations per query.
#
# Run: julia --project=benchmarking/julia -t N scripts/nndescent_phase_breakdown.jl

using Random, Printf, LinearAlgebra
using ManifoldANN
using NearestNeighborDescent
using Distances
using NearestNeighborDescent.DataStructures: BinaryMaxHeap
const MA = ManifoldANN
const NND = NearestNeighborDescent

const N      = parse(Int, get(ENV, "ND_N",   "20000"))
const D      = parse(Int, get(ENV, "ND_D",   "32"))
const K      = parse(Int, get(ENV, "ND_K",   "20"))
const EFS    = parse(Int, get(ENV, "ND_EFS", "60"))
const NQ     = parse(Int, get(ENV, "ND_NQ",  "5000"))
const SEED   = 0xC0FFEE
const MAX_IT = 8

BLAS.set_num_threads(1)
println("Threads.nthreads() = ", Threads.nthreads())
println("config: n=$N d=$D k=$K ef_s=$EFS n_queries=$NQ")

Random.seed!(SEED)
data_mat     = randn(Float32, D, N)
queries_mat  = randn(Float32, D, NQ)
data_cols    = [collect(data_mat[:, i])    for i in 1:N]
queries_cols = [collect(queries_mat[:, i]) for i in 1:NQ]

#==============================================================================
# Instrumented MANN inner loop
==============================================================================#
mutable struct MannCounters
    t_init::Float64
    t_distance::Float64
    t_heappush::Float64
    t_heappop::Float64
    t_other::Float64
    n_distance_evals::Int
end
MannCounters() = MannCounters(0.0, 0.0, 0.0, 0.0, 0.0, 0)

function instrumented_mann_query(idx, data, q, k; ef_search, rng, ctr)
    S = Float32
    actual_k = min(Int(k), idx.n_points)
    beam = max(actual_k, Int(ef_search))

    t0 = time_ns()
    start_ids = Int[]
    seen = Set{Int}()
    while length(start_ids) < min(idx.k, beam)
        c = rand(rng, 1:idx.n_points)
        if !(c in seen); push!(seen, c); push!(start_ids, c); end
    end
    visited = falses(idx.n_points)
    candidates = Tuple{Float32,Int}[]   # (dist, id) min-heap (kept as sorted insertion for simplicity)
    sizehint!(candidates, beam * 2)
    best = Tuple{Float32,Int}[]
    sizehint!(best, beam)

    for id in start_ids
        visited[id] = true
        td0 = time_ns()
        dist = Float32(idx.distance(view(data, :, id), q))
        ctr.t_distance += time_ns() - td0
        ctr.n_distance_evals += 1
        th0 = time_ns()
        # binary insert into sorted candidates (smallest first)
        pos = searchsortedfirst(candidates, (dist, id))
        insert!(candidates, pos, (dist, id))
        ctr.t_heappush += time_ns() - th0
    end
    ctr.t_init += time_ns() - t0

    while !isempty(candidates)
        tp0 = time_ns()
        current = popfirst!(candidates)
        ctr.t_heappop += time_ns() - tp0

        # stopping criterion
        if length(best) >= beam && current[1] > best[end][1]
            break
        end
        # insert current into best (sorted)
        to0 = time_ns()
        bpos = searchsortedfirst(best, current)
        insert!(best, bpos, current)
        if length(best) > beam
            pop!(best)
        end
        ctr.t_other += time_ns() - to0

        for nid in idx.neighbors[current[2]]
            visited[nid] && continue
            visited[nid] = true
            td0 = time_ns()
            d = Float32(idx.distance(view(data, :, nid), q))
            ctr.t_distance += time_ns() - td0
            ctr.n_distance_evals += 1
            th0 = time_ns()
            pos = searchsortedfirst(candidates, (d, nid))
            insert!(candidates, pos, (d, nid))
            ctr.t_heappush += time_ns() - th0
        end
    end
    return best[1:min(actual_k, length(best))]
end

#==============================================================================
# Instrumented NND.jl inner loop (mirrors search.jl serial path)
==============================================================================#
mutable struct NndCounters
    t_init::Float64
    t_distance::Float64
    t_heappush::Float64
    t_heappop::Float64
    t_other::Float64
    n_distance_evals::Int
end
NndCounters() = NndCounters(0.0, 0.0, 0.0, 0.0, 0.0, 0)

function instrumented_nnd_query(g, q, k; max_candidates, ctr)
    data = g.data
    metric = g.metric
    n = length(data)

    t0 = time_ns()
    candidates = BinaryMaxHeap{Tuple{Float32,Int,Bool}}()
    seen = falses(n)

    # init_candidates! body inlined
    sample = NND.KNNGraphs.sample_neighbors(n, max_candidates)
    for v in sample
        td0 = time_ns()
        d = Float32(Distances.evaluate(metric, q, data[v]))
        ctr.t_distance += time_ns() - td0
        ctr.n_distance_evals += 1
        th0 = time_ns()
        push!(candidates, (d, v, false))
        ctr.t_heappush += time_ns() - th0
        seen[v] = true
    end
    ctr.t_init += time_ns() - t0

    # main loop — mirror NND.search.jl
    while true
        tp0 = time_ns()
        # get_next_candidate! linear scan of unvisited
        min_idx = -1
        min_dist = typemax(Float32)
        for (i, t) in enumerate(candidates.valtree)
            if t[1] < min_dist && !t[3]
                min_idx = i
                min_dist = t[1]
            end
        end
        ctr.t_heappop += time_ns() - tp0
        if min_idx == -1; break; end

        to0 = time_ns()
        dist_v, node_v, _ = candidates.valtree[min_idx]
        candidates.valtree[min_idx] = (dist_v, node_v, true)  # mark visited
        ctr.t_other += time_ns() - to0

        for v in NND.KNNGraphs.outneighbors(g, node_v)
            if !seen[v]
                td0 = time_ns()
                d = Float32(Distances.evaluate(metric, q, data[v]))
                ctr.t_distance += time_ns() - td0
                ctr.n_distance_evals += 1
                tt0 = time_ns()
                topv = first(candidates.valtree)[1]   # max element of max-heap
                if d <= topv
                    pop!(candidates)
                    push!(candidates, (d, v, false))
                end
                ctr.t_heappush += time_ns() - tt0
                seen[v] = true
            end
        end
    end
    return nothing
end

#==============================================================================
# Build both indexes
==============================================================================#
println("\nBuilding indexes...")
idx_mann = MA.build_index(MA.NNDescentIndex, data_mat;
    k=K, max_iterations=MAX_IT, threaded=true, rng=MersenneTwister(SEED))
g_nnd = NND.nndescent(data_cols, K, NND.Euclidean(); max_iters=MAX_IT)

# Warm
for i in 1:8
    instrumented_mann_query(idx_mann, data_mat, view(queries_mat, :, i), K;
        ef_search=EFS, rng=MersenneTwister(i), ctr=MannCounters())
    instrumented_nnd_query(g_nnd, queries_cols[i], K; max_candidates=EFS,
        ctr=NndCounters())
end
GC.gc()

#==============================================================================
# Measure
==============================================================================#
mann_ctr = MannCounters()
t_mann_total = @elapsed begin
    for i in 1:NQ
        instrumented_mann_query(idx_mann, data_mat, view(queries_mat, :, i), K;
            ef_search=EFS, rng=MersenneTwister(i), ctr=mann_ctr)
    end
end

nnd_ctr = NndCounters()
t_nnd_total = @elapsed begin
    for i in 1:NQ
        instrumented_nnd_query(g_nnd, queries_cols[i], K; max_candidates=EFS,
            ctr=nnd_ctr)
    end
end

#==============================================================================
# Report
==============================================================================#
ms(ns) = ns / 1e6
function report(label, ctr, total_s, n_queries)
    @printf "\n[%s]  total wallclock: %.0f ms (%.0f qps)\n" label (total_s*1000) (n_queries/total_s)
    sum_phases = ctr.t_init + ctr.t_distance + ctr.t_heappush + ctr.t_heappop + ctr.t_other
    @printf "  init:           %7.0f ms (%4.1f%%)\n" ms(ctr.t_init)     (100*ctr.t_init/sum_phases)
    @printf "  distance:       %7.0f ms (%4.1f%%)   [%d evals total = %.1f/query]\n" ms(ctr.t_distance) (100*ctr.t_distance/sum_phases) ctr.n_distance_evals (ctr.n_distance_evals/n_queries)
    @printf "  heap-push:      %7.0f ms (%4.1f%%)\n" ms(ctr.t_heappush) (100*ctr.t_heappush/sum_phases)
    @printf "  heap-pop:       %7.0f ms (%4.1f%%)\n" ms(ctr.t_heappop)  (100*ctr.t_heappop/sum_phases)
    @printf "  other:          %7.0f ms (%4.1f%%)\n" ms(ctr.t_other)    (100*ctr.t_other/sum_phases)
    @printf "  ns/distance call: %.1f\n" (ctr.t_distance/ctr.n_distance_evals)
end

report("MANN (instrumented serial)", mann_ctr, t_mann_total, NQ)
report("NND.jl (instrumented serial)", nnd_ctr, t_nnd_total, NQ)

@printf "\n[ratio MANN/NND.jl]\n"
@printf "  total wallclock:  %.2fx\n" (t_mann_total/t_nnd_total)
@printf "  distance evals:   %.2fx\n" (mann_ctr.n_distance_evals/nnd_ctr.n_distance_evals)
@printf "  ns per dist call: %.2fx\n" ((mann_ctr.t_distance/mann_ctr.n_distance_evals) / (nnd_ctr.t_distance/nnd_ctr.n_distance_evals))

println("\nDone.")
