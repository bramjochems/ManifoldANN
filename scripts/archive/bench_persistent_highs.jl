#!/usr/bin/env julia
# Prototype: PersistentHiGHSSolver - drop-in replacement for NetworkSimplexSolver
# that uses HiGHS directly via MOI (skip JuMP DSL) and reuses one optimizer
# instance per thread across edges.
#
# The LP shape varies per edge (n_x + n_y constraints, n_x*n_y variables) so
# the "fixed-shape build-once" pattern doesn't apply — but skipping JuMP's
# CachingOptimizer and DSL macros is itself a big win, and HiGHS's simplex
# warm-start should kick in on shape-matching consecutive edges.
#
# Run: julia --project=. -t 8 scripts/bench_persistent_highs.jl

using Random, Logging, Statistics, Printf, Base.Threads
include(joinpath(@__DIR__, "..", "docs", "examples", "geodesic", "swiss_roll_utils.jl"))
using ManifoldANN
using HiGHS
using JuMP
const MOI = JuMP.MOI
const MA = ManifoldANN

# Per-thread persistent HiGHS optimizer storage
const _highs_pool = Vector{Union{Nothing, HiGHS.Optimizer}}(nothing, Threads.maxthreadid())

function _get_highs()
    tid = Threads.threadid()
    opt = _highs_pool[tid]
    if opt === nothing
        opt = HiGHS.Optimizer()
        MOI.set(opt, MOI.Silent(), true)
        _highs_pool[tid] = opt
    else
        # Reset for reuse (clears variables, constraints; keeps internal buffers)
        MOI.empty!(opt)
        MOI.set(opt, MOI.Silent(), true)
    end
    return opt
end

struct PersistentHiGHSSolver <: MA.AbstractOTSolver end

function MA.compute_curvature(::PersistentHiGHSSolver,
                              edge_view::MA.EdgeNeighborhoodView{T},
                              distance_fn::Function) where {T<:AbstractFloat}
    # Greedy shared-mass step (matching NetworkSimplexSolver's structure)
    shared_cost = zero(T)
    x_residual = copy(edge_view.x_probs)
    y_residual = copy(edge_view.y_probs)
    for z in edge_view.shared
        common_mass = min(edge_view.x_probs[z], edge_view.y_probs[z])
        x_residual[z] -= common_mass
        y_residual[z] -= common_mass
        x_residual[z] ≈ 0 && delete!(x_residual, z)
        y_residual[z] ≈ 0 && delete!(y_residual, z)
    end

    residual_cost = if isempty(x_residual) || isempty(y_residual)
        zero(T)
    else
        _solve_persistent_highs(x_residual, y_residual, distance_fn, T)
    end
    wasserstein = shared_cost + residual_cost
    curvature = one(T) - wasserstein / edge_view.edge_distance
    MA.CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein,
                          edge_view.edge_distance, :persistent_highs)
end

function _solve_persistent_highs(x_probs::Dict{Int,T}, y_probs::Dict{Int,T},
                                  distance_fn::Function, ::Type{T}) where {T}
    x_nodes = collect(keys(x_probs))
    y_nodes = collect(keys(y_probs))
    n_x, n_y = length(x_nodes), length(y_nodes)
    (n_x == 0 || n_y == 0) && return zero(T)

    opt = _get_highs()

    # Variables: γ[i,j] >= 0, flat-indexed as i + (j-1)*n_x
    n_vars = n_x * n_y
    vars = MOI.add_variables(opt, n_vars)
    for v in vars
        MOI.add_constraint(opt, v, MOI.GreaterThan(0.0))
    end

    # Objective: min Σ cost[i,j] * γ[i,j]
    obj_terms = Vector{MOI.ScalarAffineTerm{Float64}}(undef, n_vars)
    @inbounds for j in 1:n_y, i in 1:n_x
        idx = i + (j-1) * n_x
        c = Float64(distance_fn(x_nodes[i], y_nodes[j]))
        obj_terms[idx] = MOI.ScalarAffineTerm(c, vars[idx])
    end
    MOI.set(opt, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            MOI.ScalarAffineFunction(obj_terms, 0.0))
    MOI.set(opt, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # Row marginals: Σ_j γ[i,j] = x_probs[i]   for i = 1..n_x
    @inbounds for i in 1:n_x
        terms = [MOI.ScalarAffineTerm(1.0, vars[i + (j-1)*n_x]) for j in 1:n_y]
        MOI.add_constraint(opt, MOI.ScalarAffineFunction(terms, 0.0),
                           MOI.EqualTo(Float64(x_probs[x_nodes[i]])))
    end
    # Col marginals: Σ_i γ[i,j] = y_probs[j]   for j = 1..n_y
    @inbounds for j in 1:n_y
        terms = [MOI.ScalarAffineTerm(1.0, vars[i + (j-1)*n_x]) for i in 1:n_x]
        MOI.add_constraint(opt, MOI.ScalarAffineFunction(terms, 0.0),
                           MOI.EqualTo(Float64(y_probs[y_nodes[j]])))
    end

    MOI.optimize!(opt)

    status = MOI.get(opt, MOI.TerminationStatus())
    if status != MOI.OPTIMAL
        return typemax(T)
    end
    return T(MOI.get(opt, MOI.ObjectiveValue()))
end

# ---- Bench ----------------------------------------------------------------
global_logger(NullLogger())
data, _ = generate_swiss_roll(1000; rng=MersenneTwister(42))
idx = MA.build_index(MA.BruteForceIndex, data)
g = MA.build_knn_graph(idx, data; k=15, directed=false)

println("Threads.nthreads() = ", Threads.nthreads())
println("Warmup...")
for s in (MA.NetworkSimplexSolver(), PersistentHiGHSSolver())
    MA.compute_all_curvatures(g, data; variant=MA.ORCManL(),
        solver=s, fallback_solver=MA.NetworkSimplexSolver(),
        use_threading=true, verbose=false)
end
GC.gc()

function bench_main(g, data)
println("\nORCManL on swiss roll n=1000 k=15:")
ref_curvs = nothing
for (name, primary) in [
    ("NetworkSimplex (Tulip+OT.jl, current default)",  MA.NetworkSimplexSolver()),
    ("PersistentHiGHS (MOI direct, no JuMP)",          PersistentHiGHSSolver()),
]
    times = Float64[]; allocs = Float64[]
    local res
    for _ in 1:3
        GC.gc()
        a0 = Base.gc_bytes(); t0 = time_ns()
        res = MA.compute_all_curvatures(g, data;
            variant=MA.ORCManL(), solver=primary,
            fallback_solver=MA.NetworkSimplexSolver(),
            use_threading=true, verbose=false)
        t1 = time_ns(); a1 = Base.gc_bytes()
        push!(times, (t1-t0)/1e9)
        push!(allocs, (a1-a0)/1024^2)
    end
    curvs = [r.curvature for r in values(res)]
    n_nan = sum(isnan, curvs)
    finite = filter(isfinite, curvs)
    if ref_curvs === nothing
        ref_curvs = curvs
        @printf("  %-50s  wall=%5.2fs  alloc=%6.0f MB  nan=%d  μ=%.4f\n",
                name, median(times), median(allocs), n_nan, mean(finite))
    else
        # quality vs reference (NetworkSimplex)
        diffs = abs.(curvs .- ref_curvs)
        @printf("  %-50s  wall=%5.2fs  alloc=%6.0f MB  nan=%d  μ=%.4f  max-diff=%.4f  median-diff=%.5f\n",
                name, median(times), median(allocs), n_nan, mean(finite),
                maximum(diffs), median(diffs))
    end
end
end

bench_main(g, data)
