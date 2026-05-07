#!/usr/bin/env julia
# Localised perf probe for SimpleMerge.merge_results.
#
# Generates synthetic per-child result lists matching the shape produced by
# IVF-style queries (nprobe children, ~k_per_child results each), then times
# the merge step in isolation.
#
# Run:
#   julia --project=. -t 1 scripts/perf/simple_merge_profile.jl
#
# Env knobs:
#   SM_NPROBE=8        children probed
#   SM_KPC=10          results per child
#   SM_K=10            final top-k
#   SM_NCALLS=20000    merge calls per timed rep (one per query in a workload)
#   SM_REPS=3          best-of timed reps
#   SM_DUP_FRAC=0.0    fraction of ids duplicated across children (0 = IVF-like)

using Random, Printf, LinearAlgebra, Statistics
using ManifoldANN
using ManifoldANN: Neighbor, SimpleMerge, DisjointMerge, merge_results

const NPROBE   = parse(Int, get(ENV, "SM_NPROBE", "8"))
const KPC      = parse(Int, get(ENV, "SM_KPC", "10"))
const K        = parse(Int, get(ENV, "SM_K", "10"))
const NCALLS   = parse(Int, get(ENV, "SM_NCALLS", "20000"))
const REPS     = parse(Int, get(ENV, "SM_REPS", "3"))
const DUP_FRAC = parse(Float64, get(ENV, "SM_DUP_FRAC", "0.0"))
const SEED     = 0xBEEF

BLAS.set_num_threads(1)
println("nthreads = $(Threads.nthreads()), BLAS = $(BLAS.get_num_threads())")
println("config: nprobe=$NPROBE kpc=$KPC k=$K ncalls=$NCALLS reps=$REPS dup_frac=$DUP_FRAC")
println()

# Build a list of NCALLS distinct result-list inputs. Distinct inputs prevent
# the JIT from caching anything across iterations and matches the real
# workload (one merge per query).
function gen_inputs(rng, ncalls, nprobe, kpc, dup_frac)
    inputs = Vector{Vector{Vector{Neighbor{Float32}}}}(undef, ncalls)
    pool_size = nprobe * kpc * 4   # ample id space; collisions only when forced
    for c in 1:ncalls
        per_child = Vector{Vector{Neighbor{Float32}}}(undef, nprobe)
        for p in 1:nprobe
            ids = if dup_frac > 0 && p > 1 && rand(rng) < dup_frac
                # share some ids with child 1 to force dedup work
                shared = per_child[1]
                vcat([n.id for n in shared[1:min(2, length(shared))]],
                     rand(rng, 1:pool_size, kpc - min(2, length(shared))))
            else
                rand(rng, 1:pool_size, kpc)
            end
            dists = sort!(rand(rng, Float32, kpc))
            per_child[p] = [Neighbor{Float32}(ids[i], dists[i]) for i in 1:kpc]
        end
        inputs[c] = per_child
    end
    return inputs
end

rng = MersenneTwister(SEED)
inputs = gen_inputs(rng, NCALLS, NPROBE, KPC, DUP_FRAC)

function bench(label, strategy, inputs, K, NCALLS, REPS)
    println("=== $label ===")
    # Warm
    for c in 1:min(50, NCALLS)
        merge_results(strategy, inputs[c], K)
    end

    times = Float64[]
    s_last = 0
    for r in 1:REPS
        GC.gc()
        s = 0
        t = @elapsed begin
            @inbounds for c in 1:NCALLS
                r2 = merge_results(strategy, inputs[c], K)
                s += length(r2)
            end
        end
        push!(times, t)
        s_last = s
        @printf "  rep %d: %.3f s   (%.2f µs/merge, %.0f merges/s)   total kept=%d\n" r t (1e6 * t / NCALLS) (NCALLS / t) s
    end
    best = minimum(times)
    @printf "  best: %.3f s   (%.2f µs/merge, %.0f merges/s)\n" best (1e6 * best / NCALLS) (NCALLS / best)

    GC.gc()
    gc0 = Base.gc_num()
    t_alloc = @elapsed begin
        @inbounds for c in 1:NCALLS
            merge_results(strategy, inputs[c], K)
        end
    end
    gc1 = Base.gc_num()
    diff = Base.GC_Diff(gc1, gc0)
    n_allocs = diff.malloc + diff.realloc + diff.poolalloc + diff.bigalloc
    @printf "  per-merge allocs:     %.2f   per-merge bytes: %.1f B\n\n" (n_allocs / NCALLS) (diff.allocd / NCALLS)
    return (best=best, total_kept=s_last)
end

s_simple   = bench("SimpleMerge",   SimpleMerge(),   inputs, K, NCALLS, REPS)
s_disjoint = bench("DisjointMerge", DisjointMerge(), inputs, K, NCALLS, REPS)

@printf "Speedup (DisjointMerge / SimpleMerge): %.2fx\n" (s_simple.best / s_disjoint.best)
if s_simple.total_kept != s_disjoint.total_kept
    @printf "NOTE: kept counts differ — Simple=%d Disjoint=%d (expected when dup_frac > 0; Disjoint will return duplicates)\n" s_simple.total_kept s_disjoint.total_kept
end
