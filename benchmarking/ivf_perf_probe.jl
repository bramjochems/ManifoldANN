#!/usr/bin/env julia
# Quick IVF+HNSW profiling helper. Run from repo root:
#   julia --project=. benchmarking/ivf_perf_probe.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using ManifoldANN
using Distances
using Random

function parse_overrides()
    opts = Dict{Symbol,Any}()
    for arg in ARGS
        if occursin('=', arg)
            key, val = split(arg, '='; limit=2)
            sym = Symbol(key)
            if sym in (:n, :d, :nlist, :routing_k)
                opts[sym] = parse(Int, val)
            end
        end
    end
    return opts
end

function measure_ivf(; n=10_000, d=100, nlist=128, routing_k=10)
    println("=== IVF profiling ===")
    println("n = $n, d = $d, nlist = $nlist, routing_k = $routing_k")
    rng = MersenneTwister(42)
    X = rand(rng, Float32, d, n)

    # Fit KMeans and capture pending assignments
    kmeans = KMeansTransform(k=nlist, distance=Euclidean(), init=:kmeans_plus_plus)
    t_fit = @elapsed fit!(kmeans, X)
    assignments = ManifoldANN.take_pending_assignments!(kmeans)
    println("fit!                : $(round(t_fit, digits=4)) s, pending assignments: $(assignments === nothing ? "none" : length(assignments))")

    # Partition using cached assignments (no transforms)
    t_partition_cached = @elapsed partition_by_transform(
        X,
        kmeans;
        capture_data=false,
        precomputed_assignments=assignments,
    )
    println("partition (cached)  : $(round(t_partition_cached, digits=4)) s")

    # Partition without cached assignments (forces transform per point)
    t_partition_fresh = @elapsed partition_by_transform(
        X,
        kmeans;
        capture_data=false,
        precomputed_assignments=nothing,
    )
    println("partition (fresh)   : $(round(t_partition_fresh, digits=4)) s")

    # Full IVF+HNSW build
    t_build = @elapsed begin
        build_ivf_hnsw_index(
            X;
            nlist=nlist,
            routing_k=routing_k,
            kmeans_distance=Euclidean(),
            kmeans_init=:kmeans_plus_plus,
            kmeans_max_iters=5,
            kmeans_tol=1e-4,
            hnsw_M=16,
            hnsw_ef_construction=200,
            hnsw_ef_search=64,
            hnsw_neighbor_policy=:diversified,
            distance=ManifoldANN.default_distance,
        )
    end
    println("build_ivf_hnsw_index: $(round(t_build, digits=4)) s")
end

overrides = parse_overrides()
measure_ivf(; n=get(overrides, :n, 10_000),
             d=get(overrides, :d, 100),
             nlist=get(overrides, :nlist, 128),
             routing_k=get(overrides, :routing_k, 10))
