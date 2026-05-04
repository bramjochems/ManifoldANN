#!/usr/bin/env julia
# Bench: prototype ClpDirectSolver that skips MOI/CachingOptimizer and calls
# Clp_loadProblem directly via the Clp.jl C-API binding.
#
# Baseline: integrated MA.ClpSolver() (MOI CachingOptimizer + Clp).
#
# Run: julia --project=. -t 8 scripts/bench_clp_no_caching.jl
#
# OT LP layout (column-major CSC required by Clp_loadProblem):
#   Variables : γ[i,j], i∈1:n_x, j∈1:n_y; column index = (j-1)*n_x + i.
#   Rows      : n_x row-marginal eqs (sum over j) + n_y col-marginal eqs.
#   Each variable γ[i,j] is in exactly two rows (row i and row n_x+j),
#   so every column has 2 nonzeros — clean fixed-pattern CSC.

using Random, Printf, LinearAlgebra, Statistics, Logging
using Clp
using ManifoldANN
const MA = ManifoldANN

include(joinpath(@__DIR__, "..", "docs", "examples", "geodesic", "swiss_roll_utils.jl"))

# ---- Prototype solver: direct Clp C-API, no MOI ---------------------------

struct ClpDirectSolver <: MA.AbstractOTSolver end

function _solve_clp_direct_ot(x_probs::Dict{Int,T}, y_probs::Dict{Int,T},
                              distance_fn::Function, ::Type{T}) where {T}
    x_nodes = collect(keys(x_probs))
    y_nodes = collect(keys(y_probs))
    n_x, n_y = length(x_nodes), length(y_nodes)
    (n_x == 0 || n_y == 0) && return zero(T)

    n_vars = n_x * n_y
    n_rows = n_x + n_y
    nnz    = 2 * n_vars

    # CSC: column j*n_x+i corresponds to γ[i,j], rows = (i-1, n_x+j-1) (0-indexed Cint)
    col_start = Vector{Clp.CoinBigIndex}(undef, n_vars + 1)
    row_idx   = Vector{Cint}(undef, nnz)
    values    = Vector{Cdouble}(undef, nnz)
    obj       = Vector{Cdouble}(undef, n_vars)
    col_lb    = zeros(Cdouble, n_vars)
    col_ub    = fill(Cdouble(Inf), n_vars)
    row_lb    = Vector{Cdouble}(undef, n_rows)
    row_ub    = Vector{Cdouble}(undef, n_rows)

    @inbounds for j in 1:n_y, i in 1:n_x
        col = (j - 1) * n_x + i
        col_start[col] = 2 * (col - 1)
        row_idx[2*col - 1] = Cint(i - 1)         # row-marginal i
        row_idx[2*col    ] = Cint(n_x + j - 1)   # col-marginal j
        values[2*col - 1] = 1.0
        values[2*col    ] = 1.0
        obj[col] = Float64(distance_fn(x_nodes[i], y_nodes[j]))
    end
    col_start[n_vars + 1] = nnz

    @inbounds for i in 1:n_x
        m = Float64(x_probs[x_nodes[i]])
        row_lb[i] = m; row_ub[i] = m
    end
    @inbounds for j in 1:n_y
        m = Float64(y_probs[y_nodes[j]])
        row_lb[n_x + j] = m; row_ub[n_x + j] = m
    end

    model = Clp.Clp_newModel()
    try
        Clp.Clp_setLogLevel(model, Cint(0))
        Clp.Clp_loadProblem(model, Cint(n_vars), Cint(n_rows),
                            col_start, row_idx, values,
                            col_lb, col_ub, obj, row_lb, row_ub)
        Clp.Clp_setObjSense(model, 1.0)  # minimize
        Clp.Clp_initialSolve(model)
        if Clp.Clp_isProvenOptimal(model) == 0
            return typemax(T)
        end
        return T(Clp.Clp_getObjValue(model))
    finally
        Clp.Clp_deleteModel(model)
    end
end

function MA.compute_curvature(::ClpDirectSolver,
                              edge_view::MA.EdgeNeighborhoodView{T},
                              distance_fn::Function) where {T<:AbstractFloat}
    x_residual = copy(edge_view.x_probs)
    y_residual = copy(edge_view.y_probs)
    for z in edge_view.shared
        cm = min(edge_view.x_probs[z], edge_view.y_probs[z])
        x_residual[z] -= cm
        y_residual[z] -= cm
        x_residual[z] ≈ 0 && delete!(x_residual, z)
        y_residual[z] ≈ 0 && delete!(y_residual, z)
    end
    residual_cost = (isempty(x_residual) || isempty(y_residual)) ? zero(T) :
                    _solve_clp_direct_ot(x_residual, y_residual, distance_fn, T)
    wasserstein = residual_cost
    curvature = one(T) - wasserstein / edge_view.edge_distance
    MA.CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein,
                          edge_view.edge_distance, :clp_direct)
end

# ---- Pooled variant: one Clp_Simplex* per thread, reused via Clp_loadProblem ----

struct ClpDirectPooledSolver <: MA.AbstractOTSolver end

const _CLP_DIRECT_POOL = Dict{Int, Ptr{Cvoid}}()
const _CLP_DIRECT_LOCK = ReentrantLock()

function _get_pooled_clp_model()
    tid = Threads.threadid()
    m = get(_CLP_DIRECT_POOL, tid, C_NULL)
    m != C_NULL && return m
    return lock(_CLP_DIRECT_LOCK) do
        get!(_CLP_DIRECT_POOL, tid) do
            mdl = Clp.Clp_newModel()
            Clp.Clp_setLogLevel(mdl, Cint(0))
            mdl
        end
    end
end

function _solve_clp_direct_ot_pooled(x_probs::Dict{Int,T}, y_probs::Dict{Int,T},
                                     distance_fn::Function, ::Type{T}) where {T}
    x_nodes = collect(keys(x_probs))
    y_nodes = collect(keys(y_probs))
    n_x, n_y = length(x_nodes), length(y_nodes)
    (n_x == 0 || n_y == 0) && return zero(T)

    n_vars = n_x * n_y
    n_rows = n_x + n_y
    nnz    = 2 * n_vars

    col_start = Vector{Clp.CoinBigIndex}(undef, n_vars + 1)
    row_idx   = Vector{Cint}(undef, nnz)
    values    = Vector{Cdouble}(undef, nnz)
    obj       = Vector{Cdouble}(undef, n_vars)
    col_lb    = zeros(Cdouble, n_vars)
    col_ub    = fill(Cdouble(Inf), n_vars)
    row_lb    = Vector{Cdouble}(undef, n_rows)
    row_ub    = Vector{Cdouble}(undef, n_rows)

    @inbounds for j in 1:n_y, i in 1:n_x
        col = (j - 1) * n_x + i
        col_start[col] = 2 * (col - 1)
        row_idx[2*col - 1] = Cint(i - 1)
        row_idx[2*col    ] = Cint(n_x + j - 1)
        values[2*col - 1] = 1.0
        values[2*col    ] = 1.0
        obj[col] = Float64(distance_fn(x_nodes[i], y_nodes[j]))
    end
    col_start[n_vars + 1] = nnz

    @inbounds for i in 1:n_x
        m = Float64(x_probs[x_nodes[i]])
        row_lb[i] = m; row_ub[i] = m
    end
    @inbounds for j in 1:n_y
        m = Float64(y_probs[y_nodes[j]])
        row_lb[n_x + j] = m; row_ub[n_x + j] = m
    end

    model = _get_pooled_clp_model()
    Clp.Clp_loadProblem(model, Cint(n_vars), Cint(n_rows),
                        col_start, row_idx, values,
                        col_lb, col_ub, obj, row_lb, row_ub)
    Clp.Clp_setObjSense(model, 1.0)
    Clp.Clp_initialSolve(model)
    if Clp.Clp_isProvenOptimal(model) == 0
        return typemax(T)
    end
    return T(Clp.Clp_getObjValue(model))
end

function MA.compute_curvature(::ClpDirectPooledSolver,
                              edge_view::MA.EdgeNeighborhoodView{T},
                              distance_fn::Function) where {T<:AbstractFloat}
    x_residual = copy(edge_view.x_probs)
    y_residual = copy(edge_view.y_probs)
    for z in edge_view.shared
        cm = min(edge_view.x_probs[z], edge_view.y_probs[z])
        x_residual[z] -= cm
        y_residual[z] -= cm
        x_residual[z] ≈ 0 && delete!(x_residual, z)
        y_residual[z] ≈ 0 && delete!(y_residual, z)
    end
    residual_cost = (isempty(x_residual) || isempty(y_residual)) ? zero(T) :
                    _solve_clp_direct_ot_pooled(x_residual, y_residual, distance_fn, T)
    wasserstein = residual_cost
    curvature = one(T) - wasserstein / edge_view.edge_distance
    MA.CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein,
                          edge_view.edge_distance, :clp_direct_pooled)
end

# ---- Bench harness --------------------------------------------------------

const N, K, SEED = 1_000, 15, 42

println("Threads.nthreads() = ", Threads.nthreads(), "   BLAS = ", BLAS.get_num_threads())
Random.seed!(SEED)
data, _   = generate_swiss_roll(N; rng=MersenneTwister(SEED))
index     = MA.build_index(MA.BruteForceIndex, data)
graph_dir = MA.build_knn_graph(index, data; k=K, directed=true)
graph_und = MA.build_knn_graph(index, data; k=K, directed=false)

const _orig_logger = global_logger(NullLogger())

function run_once(graph, variant, solver)
    GC.gc()
    a0 = Base.gc_bytes(); t0 = time_ns()
    res = MA.compute_all_curvatures(graph, data;
        variant=variant, solver=solver,
        fallback_solver=MA.ClpSolver(),
        use_threading=true, verbose=false)
    t1 = time_ns(); a1 = Base.gc_bytes()
    return res, (t1 - t0) / 1e9, (a1 - a0) / 1024^2
end

function bench(label, graph, variant, solver, ref)
    # warmup
    run_once(graph, variant, solver)
    times, allocs = Float64[], Float64[]
    local res
    for _ in 1:3
        res, w, a = run_once(graph, variant, solver)
        push!(times, w); push!(allocs, a)
    end
    n_nan = sum(isnan(r.curvature) for r in values(res))
    if ref === nothing
        return res, label, median(times), median(allocs), 0.0, 0.0, n_nan
    end
    diffs = Float64[]
    for (k, v) in res
        haskey(ref, k) || continue
        a, b = v.curvature, ref[k].curvature
        (isnan(a) || isnan(b)) && continue
        push!(diffs, abs(a - b))
    end
    return res, label, median(times), median(allocs),
           isempty(diffs) ? 0.0 : maximum(diffs),
           isempty(diffs) ? 0.0 : median(diffs), n_nan
end

cases = [
    ("StandardORC", MA.StandardORC(), graph_dir),
    ("ORCManL",     MA.ORCManL(),     graph_und),
]

println("\nBenchmark (3 reps, median; warmup excluded). swiss-roll n=$N k=$K\n")
@printf("%-12s %-14s %10s %12s %12s %12s %6s\n",
        "case", "solver", "wall_s", "alloc_MB", "max_Δ", "med_Δ", "NaN")
println(repeat('-', 86))

for (cname, variant, graph) in cases
    ref_res, _, w_ref, a_ref, _, _, nan_ref =
        bench("ClpSolver", graph, variant, MA.ClpSolver(), nothing)
    @printf("%-12s %-14s %10.3f %12.1f %12s %12s %6d\n",
            cname, "ClpSolver", w_ref, a_ref, "—", "—", nan_ref)

    _, _, w_p, a_p, mx, md, nn =
        bench("ClpDirect", graph, variant, ClpDirectSolver(), ref_res)
    @printf("%-12s %-14s %10.3f %12.1f %12.2e %12.2e %6d\n",
            cname, "ClpDirect", w_p, a_p, mx, md, nn)

    _, _, w_pp, a_pp, mxp, mdp, nnp =
        bench("ClpDirectPool", graph, variant, ClpDirectPooledSolver(), ref_res)
    @printf("%-12s %-14s %10.3f %12.1f %12.2e %12.2e %6d\n",
            cname, "ClpDirectPool", w_pp, a_pp, mxp, mdp, nnp)

    @printf("%-12s %-14s direct/base wall=%.2fx alloc=%.2fx | pooled/base wall=%.2fx alloc=%.2fx\n\n",
            cname, "(ratios)", w_p / w_ref, a_p / a_ref, w_pp / w_ref, a_pp / a_ref)
end

global_logger(_orig_logger)
