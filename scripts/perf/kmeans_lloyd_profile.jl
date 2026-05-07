#!/usr/bin/env julia
# Localised perf probe for KMeansTransform's Lloyd iterations.
#
# Sized to match the regime where the M-step dominates: low d, large n,
# moderate-to-large k. glove-100 with nlist=4096 is the worst case in the
# IVF cloud sweep; this script reproduces a smaller version of that shape
# locally so we can measure the M-step parallelism win without a full
# 1.18M-point dataset.
#
# Run:
#   julia --project=. -t 16 scripts/perf/kmeans_lloyd_profile.jl
#
# Env knobs:
#   KMEANS_N=200000         training points
#   KMEANS_D=100            ambient dim (matches glove-100)
#   KMEANS_K=1024           number of centroids
#   KMEANS_ITERS=10         Lloyd iterations
#   KMEANS_REPS=2           best-of timed reps

using Random, Printf, LinearAlgebra, Statistics
using ManifoldANN
using ManifoldANN: KMeansTransform, fit!
using Distances

# Defaults sized so the full fit + per-step harness fit comfortably in
# 8-16 GB of working memory while still putting Lloyd's iterations in the
# regime where the combine-pass and argmin become visible (k*n = 200M
# entries, D = ~800 MB). Bigger configs (e.g. KMEANS_N=500000 K=4096)
# match the real glove-100 cloud workload but need ~16 GB and risk OOM
# on a workstation with anything else running.
const N      = parse(Int, get(ENV, "KMEANS_N", "100000"))
const D      = parse(Int, get(ENV, "KMEANS_D", "100"))
const K      = parse(Int, get(ENV, "KMEANS_K", "2048"))
const ITERS  = parse(Int, get(ENV, "KMEANS_ITERS", "5"))
const REPS   = parse(Int, get(ENV, "KMEANS_REPS", "3"))
const SEED   = 0xBEEF

println("nthreads=$(Threads.nthreads()) BLAS=$(BLAS.get_num_threads())")
println("config: n=$N d=$D k=$K iters=$ITERS reps=$REPS")
println()

Random.seed!(SEED)
X = randn(Float32, D, N)

# Warmup so the first-time JIT cost doesn't poison the timed reps.
let
    warm_X = randn(Float32, D, 2_000)
    t = KMeansTransform(k=8, distance=Euclidean(), init=:kmeans_plus_plus, max_iters=2)
    fit!(t, warm_X; rng=MersenneTwister(0))
end

println("=== fit! timing ===")
times = Float64[]
for r in 1:REPS
    GC.gc()
    t = KMeansTransform(
        k=K,
        distance=Euclidean(),
        init=:kmeans_plus_plus,
        max_iters=ITERS,
        tol=1e-4,
        subsample_size=nothing,  # use full X
    )
    elapsed = @elapsed fit!(t, X; rng=MersenneTwister(SEED + r))
    push!(times, elapsed)
    @printf "  rep %d: %.3f s\n" r elapsed
end
@printf "  median: %.3f s   best: %.3f s   (%.1f s / iter, best)\n\n" median(times) minimum(times) (minimum(times) / ITERS)

# ---------------------------------------------------------------------------
# Per-step breakdown. Allocates its own D / X-subset to avoid carrying
# the full lloyd!() workspace alive (the full-fit run above already peaks
# at ~K*N*4 bytes = several GB; doubling that for a side-by-side D would
# OOM on machines with <16 GB of working memory).
#
# Uses N_PROFILE = N (or smaller via env) so the user can dial back if
# the full-fit phase already pushed memory close to the edge.
# ---------------------------------------------------------------------------
const N_PROFILE = parse(Int, get(ENV, "KMEANS_N_PROFILE", string(N)))
@printf "=== per-step breakdown (single iter, n=%d) ===\n" N_PROFILE

# Free everything we don't need from the fit phase before allocating.
X = nothing  # release the full training set
GC.gc(true)

X_p = randn(Float32, D, N_PROFILE)
using ManifoldANN: pairwise_distances!, assign_clusters!
const _MANN = ManifoldANN

centroids0 = X_p[:, 1:K]
D_buf = Matrix{Float32}(undef, K, N_PROFILE)
assignments = Vector{Int}(undef, N_PROFILE)
new_centroids = similar(centroids0)
cluster_sizes = zeros(Int, K)

# Warm
pairwise_distances!(D_buf, X_p, centroids0, Euclidean())
assign_clusters!(assignments, D_buf)

GC.gc()
t1 = @elapsed pairwise_distances!(D_buf, X_p, centroids0, Euclidean())
@printf "  pairwise_distances! : %.3f s\n" t1

GC.gc()
t2 = @elapsed assign_clusters!(assignments, D_buf)
@printf "  assign_clusters!    : %.3f s\n" t2

GC.gc()
t3 = @elapsed _MANN._m_step_parallel!(new_centroids, cluster_sizes, X_p, assignments, K)
@printf "  _m_step_parallel!   : %.3f s\n" t3

@printf "  sum                 : %.3f s   (×%d iters ≈ %.1f s/fit)\n" (t1+t2+t3) ITERS (ITERS*(t1+t2+t3))

println("=== allocations (single fit, fresh X) ===")
GC.gc(true)
X_alloc = randn(Float32, D, N_PROFILE)
GC.gc()
gc0 = Base.gc_num()
let
    t = KMeansTransform(k=K, distance=Euclidean(), init=:kmeans_plus_plus,
                       max_iters=ITERS, tol=1e-4, subsample_size=nothing)
    fit!(t, X_alloc; rng=MersenneTwister(SEED))
end
gc1 = Base.gc_num()
diff = Base.GC_Diff(gc1, gc0)
@printf "  allocations: %d  (%.2f MiB)\n" (diff.malloc + diff.realloc + diff.poolalloc + diff.bigalloc) (diff.allocd / 2^20)
