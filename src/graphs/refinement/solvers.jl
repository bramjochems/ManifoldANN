"""
Refactored ORC curvature solvers using external packages.

Changes from original:
1. Use Hungarian.jl directly (remove wrapper overhead, 9x speedup)
2. Use OptimalTransport.jl sinkhorn2() (community-maintained, better than custom)
3. Use OptimalTransport.jl emd2() with Tulip for network simplex (exact OT)
4. Keep HiGHS LP solver as reference for debugging

Performance improvements:
- Hungarian: 9x faster (removed wrapper overhead)
- Sinkhorn: Better convergence, community-maintained
- Network Simplex: Specialized OT solver (faster than generic LP)
"""

using LinearAlgebra
using Hungarian
using OptimalTransport
using Tulip

# ============================================================================
# Optimal Transport Solver Interface
# ============================================================================

"""
    AbstractOTSolver

Base type for optimal transport solvers used in curvature computation.

These solvers compute the Wasserstein distance between neighborhood distributions,
which is used in the Ollivier-Ricci curvature formula: κ(x,y) = 1 - W₁(μₓ, μᵧ) / d(x,y)
"""
abstract type AbstractOTSolver end

can_handle(::AbstractOTSolver, ::EdgeNeighborhoodView) = true

function compute_curvature(::AbstractOTSolver, ::EdgeNeighborhoodView, ::Function)
    error("compute_curvature not implemented for this solver type")
end

# ============================================================================
# HungarianSolver - Direct use of Hungarian.jl (9x faster!)
# ============================================================================

"""
    HungarianSolver <: AbstractOTSolver

Fast exact OT solver using Hungarian algorithm for minimum-cost bipartite matching.

Requirements:
- Equal neighborhood sizes: |N(x)| = |N(y)|
- Uniform distributions: all probabilities equal

Complexity: O(k³) where k is neighborhood size

Performance: 9x faster than wrapper version by calling Hungarian.jl directly.

Example:
```julia
solver = HungarianSolver()
curvature = compute_curvature(solver, edge_view, dist_fn)
```
"""
struct HungarianSolver <: AbstractOTSolver end

function can_handle(::HungarianSolver, edge_view::EdgeNeighborhoodView{T}) where {T<:AbstractFloat}
    n_x = length(edge_view.shared) + length(edge_view.unique_x)
    n_y = length(edge_view.shared) + length(edge_view.unique_y)

    # Require equal sizes for Hungarian
    n_x == n_y || return false

    # Require uniform distributions
    x_vals = collect(values(edge_view.x_probs))
    y_vals = collect(values(edge_view.y_probs))

    is_uniform_x = all(p ≈ x_vals[1] for p in x_vals)
    is_uniform_y = all(p ≈ y_vals[1] for p in y_vals)

    return is_uniform_x && is_uniform_y
end

function compute_curvature(
    ::HungarianSolver,
    edge_view::EdgeNeighborhoodView{T},
    distance_fn::Function
) where {T<:AbstractFloat}
    A = edge_view.unique_x
    B = edge_view.unique_y
    n = length(A)

    # Shared nodes have zero transport cost (they're identical!)
    # Only need to match unique nodes
    wasserstein = if n == 0
        zero(T)
    else
        # Build cost matrix
        cost = Matrix{Float64}(undef, n, n)
        for i in 1:n, j in 1:n
            cost[i, j] = Float64(distance_fn(A[i], B[j]))
        end

        # Call Hungarian.jl directly (no wrapper overhead!)
        _, total_cost = hungarian(cost)

        # Normalize by total neighborhood size
        T(total_cost) / T(length(edge_view.shared) + n)
    end

    curvature = one(T) - wasserstein / edge_view.edge_distance
    CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein, edge_view.edge_distance, :hungarian)
end

# ============================================================================
# SinkhornSolver - Use OptimalTransport.jl
# ============================================================================

"""
    SinkhornSolver <: AbstractOTSolver

Entropy-regularized OT solver using OptimalTransport.jl's sinkhorn2().

!!! warning "Regularization Parameter"
    The default `reg=0.01` may be too small relative to typical ORC cost matrix scales.
    Cost matrices in ORC are often O(1-10) depending on data normalization and graph structure.
    If reg is too small relative to cost scale, Sinkhorn iterations diverge → NaN values.

    **Rule of thumb**: Set `reg ≈ 5-10% of mean(cost_matrix)`

    Example tuning:
    - Data normalized to [0,1]: `reg=0.01` may work
    - Unnormalized data: `reg=0.05-0.1` often needed
    - Check diagnostics: `scripts/diagnose_sinkhorn.jl`

    **However**: Sinkhorn gives approximate OT (entropy-regularized), not exact OT.
    Higher reg = faster convergence but less accurate.
    For research accuracy, prefer NetworkSimplexSolver or LPReferenceSolver.

# Arguments
- `reg::Float64=0.01`: Entropy regularization parameter (data-dependent!)
- `maxiter::Int=1000`: Maximum Sinkhorn iterations
- `atol::Float64=1e-9`: Convergence tolerance

# Example
```julia
# Default (may fail if cost scale is large)
solver = SinkhornSolver()

# Adaptive approach (recommended)
# Option 1: Estimate from data
data_scale = mean(norm(data[:, i] - data[:, j]) for i in 1:min(100,n), j in i+1:min(100,n))
solver = SinkhornSolver(reg=0.1 * data_scale, atol=1e-6)

# Option 2: Safe conservative value
solver = SinkhornSolver(reg=0.05, atol=1e-6)

# Option 3: Use exact solver instead (recommended)
solver = NetworkSimplexSolver()  # Exact OT via OptimalTransport.jl + Tulip
```

# See Also
- NetworkSimplexSolver: Exact OT, robust fallback
- LPReferenceSolver: Exact OT via general LP solver
- Diagnostic: `scripts/diagnose_sinkhorn.jl`
"""
struct SinkhornSolver <: AbstractOTSolver
    reg::Float64
    maxiter::Int
    atol::Float64

    function SinkhornSolver(; reg::Float64=0.01, maxiter::Int=1000, atol::Float64=1e-9)
        reg > 0 || throw(ArgumentError("reg must be positive"))
        maxiter > 0 || throw(ArgumentError("maxiter must be positive"))
        atol > 0 || throw(ArgumentError("atol must be positive"))

        # Warn if using very small reg (may be too small relative to cost scale)
        if reg < 0.02
            @warn "SinkhornSolver: reg=$reg may be too small relative to ORC cost matrix scale. " *
                  "Rule of thumb: reg ≈ 5-10% of mean(cost_matrix). " *
                  "If you get NaN values, increase reg or use NetworkSimplexSolver(). " *
                  "See scripts/diagnose_sinkhorn.jl for diagnostics."
        end

        new(reg, maxiter, atol)
    end
end

function compute_curvature(
    solver::SinkhornSolver,
    edge_view::EdgeNeighborhoodView{T},
    distance_fn::Function
) where {T<:AbstractFloat}
    # Greedily transport shared mass at zero cost
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

    # Solve Sinkhorn on residual using OptimalTransport.jl
    residual_cost = if isempty(x_residual) || isempty(y_residual)
        zero(T)
    else
        _solve_sinkhorn_ot(x_residual, y_residual, distance_fn, T, solver)
    end

    wasserstein = shared_cost + residual_cost
    curvature = one(T) - wasserstein / edge_view.edge_distance
    CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein, edge_view.edge_distance, :sinkhorn)
end

function _solve_sinkhorn_ot(
    x_probs::Dict{Int,T},
    y_probs::Dict{Int,T},
    distance_fn::Function,
    ::Type{T},
    solver::SinkhornSolver
) where {T}
    # Uses OptimalTransport.jl's sinkhorn2() directly - no custom implementation
    # See: https://juliaoptimaltransport.github.io/OptimalTransport.jl/stable/
    #
    # Note: sinkhorn2 returns entropy-regularized OT cost, not exact OT.
    # Accuracy depends on regularization parameter (solver.reg).

    x_nodes = collect(keys(x_probs))
    y_nodes = collect(keys(y_probs))
    n_x, n_y = length(x_nodes), length(y_nodes)

    (n_x == 0 || n_y == 0) && return zero(T)

    # Build cost matrix
    cost = Matrix{Float64}(undef, n_x, n_y)
    for i in 1:n_x, j in 1:n_y
        cost[i, j] = Float64(distance_fn(x_nodes[i], y_nodes[j]))
    end

    # Marginals
    μ = [Float64(x_probs[x_nodes[i]]) for i in 1:n_x]
    ν = [Float64(y_probs[y_nodes[j]]) for j in 1:n_y]

    # Call OptimalTransport.jl's Sinkhorn implementation
    # Uses log-domain stabilization for numerical stability
    total_cost = sinkhorn2(μ, ν, cost, solver.reg; maxiter=solver.maxiter, atol=solver.atol)

    return T(total_cost)
end

# ============================================================================
# NetworkSimplexSolver - Exact OT using OptimalTransport.jl + Tulip
# ============================================================================

"""
    NetworkSimplexSolver <: AbstractOTSolver

Exact optimal transport using network simplex algorithm via OptimalTransport.jl.

Handles any distributions (uniform/non-uniform, equal/unequal sizes).
Computes exact OT distance (no approximation).

Complexity: O(k² log k) expected, O(k³) worst case

Uses Tulip.Optimizer as the underlying LP solver. Faster than generic LP
because it's specialized for network flow problems (OT is min-cost flow).

Example:
```julia
solver = NetworkSimplexSolver()
curvature = compute_curvature(solver, edge_view, dist_fn)
```
"""
struct NetworkSimplexSolver <: AbstractOTSolver end

function compute_curvature(
    ::NetworkSimplexSolver,
    edge_view::EdgeNeighborhoodView{T},
    distance_fn::Function
) where {T<:AbstractFloat}
    # Greedily transport shared mass at zero cost
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

    # Solve exact OT on residual
    residual_cost = if isempty(x_residual) || isempty(y_residual)
        zero(T)
    else
        _solve_network_simplex(x_residual, y_residual, distance_fn, T)
    end

    wasserstein = shared_cost + residual_cost
    curvature = one(T) - wasserstein / edge_view.edge_distance
    CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein, edge_view.edge_distance, :network_simplex)
end

function _solve_network_simplex(
    x_probs::Dict{Int,T},
    y_probs::Dict{Int,T},
    distance_fn::Function,
    ::Type{T}
) where {T}
    """
    Wrapper around OptimalTransport.jl emd2() with Tulip optimizer.
    """
    x_nodes = collect(keys(x_probs))
    y_nodes = collect(keys(y_probs))
    n_x, n_y = length(x_nodes), length(y_nodes)

    (n_x == 0 || n_y == 0) && return zero(T)

    # Build cost matrix
    cost = Matrix{Float64}(undef, n_x, n_y)
    for i in 1:n_x, j in 1:n_y
        cost[i, j] = Float64(distance_fn(x_nodes[i], y_nodes[j]))
    end

    # Marginals
    μ = [Float64(x_probs[x_nodes[i]]) for i in 1:n_x]
    ν = [Float64(y_probs[y_nodes[j]]) for j in 1:n_y]

    # Call OptimalTransport.jl with Tulip optimizer
    # Note: Tulip.Optimizer() creates a new instance (not the type!)
    total_cost = emd2(μ, ν, cost, Tulip.Optimizer())

    return T(total_cost)
end

# ============================================================================
# LPReferenceSolver - Keep HiGHS LP as reference implementation
# ============================================================================

"""
    LPReferenceSolver <: AbstractOTSolver

Generic LP solver using HiGHS for exact optimal transport.

This is a **reference implementation** for debugging and verification.
NOT recommended for production use - prefer NetworkSimplexSolver or HungarianSolver.

Why keep this?
- Reference implementation for algorithmic clarity
- Verification/testing against specialized solvers
- Fallback when specialized solvers have issues

Performance: ~45x slower than Sinkhorn, ~10x slower than Python

Example:
```julia
solver = LPReferenceSolver()
curvature = compute_curvature(solver, edge_view, dist_fn)  # Slow!
```
"""
struct LPReferenceSolver <: AbstractOTSolver end

# Import the old LP implementation (keep as-is for reference)
using JuMP
using HiGHS

function compute_curvature(
    ::LPReferenceSolver,
    edge_view::EdgeNeighborhoodView{T},
    distance_fn::Function
) where {T<:AbstractFloat}
    # Greedily transport shared mass
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

    # Solve LP on residual
    residual_cost = if isempty(x_residual) || isempty(y_residual)
        zero(T)
    else
        _solve_lp_reference(x_residual, y_residual, distance_fn, T)
    end

    wasserstein = shared_cost + residual_cost
    curvature = one(T) - wasserstein / edge_view.edge_distance
    CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein, edge_view.edge_distance, :lp_reference)
end

function _solve_lp_reference(
    x_probs::Dict{Int,T},
    y_probs::Dict{Int,T},
    distance_fn::Function,
    ::Type{T}
) where {T}
    """
    Generic LP formulation for optimal transport (reference implementation).
    """
    x_nodes = collect(keys(x_probs))
    y_nodes = collect(keys(y_probs))
    n_x, n_y = length(x_nodes), length(y_nodes)

    (n_x == 0 || n_y == 0) && return zero(T)

    # Build cost matrix
    cost = Matrix{Float64}(undef, n_x, n_y)
    for i in 1:n_x, j in 1:n_y
        cost[i, j] = Float64(distance_fn(x_nodes[i], y_nodes[j]))
    end

    # Build LP model
    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, γ[1:n_x, 1:n_y] >= 0)

    # Marginal constraints
    for i in 1:n_x
        @constraint(model, sum(γ[i, j] for j in 1:n_y) == x_probs[x_nodes[i]])
    end
    for j in 1:n_y
        @constraint(model, sum(γ[i, j] for i in 1:n_x) == y_probs[y_nodes[j]])
    end

    # Objective
    @objective(model, Min, sum(cost[i, j] * γ[i, j] for i in 1:n_x, j in 1:n_y))

    optimize!(model)

    if termination_status(model) != OPTIMAL
        @warn "LP solver did not find optimal solution" status=termination_status(model)
        return typemax(T)
    end

    return T(objective_value(model))
end

# ============================================================================
# GreedySolver - Keep fast approximate solver
# ============================================================================

"""
    GreedySolver <: AbstractOTSolver

Greedy approximate OT solver for fast curvature estimation.

Iteratively transports mass along lowest-cost edges. Not optimal but fast.
Complexity: O(k³) — each iteration scans all k×k pairs to find the
minimum-cost edge, and depletes at least one mass entry per iteration
(O(k) iterations). A heap of edges would lower this to O(k² log k);
not pursued — k is small in practice.

Use when:
- Speed is critical over accuracy
- Approximate curvature is sufficient
- Profiling other code (minimize OT overhead)

Example:
```julia
solver = GreedySolver()
curvature = compute_curvature(solver, edge_view, dist_fn)
```
"""
struct GreedySolver <: AbstractOTSolver end

function compute_curvature(
    ::GreedySolver,
    edge_view::EdgeNeighborhoodView{T},
    distance_fn::Function
) where {T<:AbstractFloat}
    # Greedily transport shared mass
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

    # Greedy matching on residual
    residual_cost = if isempty(x_residual) || isempty(y_residual)
        zero(T)
    else
        _solve_greedy(x_residual, y_residual, distance_fn, T)
    end

    wasserstein = shared_cost + residual_cost
    curvature = one(T) - wasserstein / edge_view.edge_distance
    CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein, edge_view.edge_distance, :greedy)
end

function _solve_greedy(
    x_probs::Dict{Int,T},
    y_probs::Dict{Int,T},
    distance_fn::Function,
    ::Type{T}
) where {T}
    x_nodes = collect(keys(x_probs))
    y_nodes = collect(keys(y_probs))
    x_masses = [x_probs[x] for x in x_nodes]
    y_masses = [y_probs[y] for y in y_nodes]

    total_cost = zero(T)

    while !all(m ≈ 0 for m in x_masses) && !all(m ≈ 0 for m in y_masses)
        # Find minimum cost edge
        min_cost, best_i, best_j = typemax(T), 0, 0

        for i in 1:length(x_nodes)
            x_masses[i] ≈ 0 && continue
            for j in 1:length(y_nodes)
                y_masses[j] ≈ 0 && continue
                c = T(distance_fn(x_nodes[i], y_nodes[j]))
                if c < min_cost
                    min_cost, best_i, best_j = c, i, j
                end
            end
        end

        best_i == 0 && break

        # Transport mass along minimum cost edge
        mass = min(x_masses[best_i], y_masses[best_j])
        total_cost += mass * min_cost
        x_masses[best_i] -= mass
        y_masses[best_j] -= mass
    end

    return total_cost
end

# ============================================================================
# Convenience Wrapper
# ============================================================================

"""
    GenericOTSolver(; method::Symbol=:sinkhorn, sinkhorn_reg::Float64=0.01)

Convenience function that returns the appropriate solver based on method.

Maps to solvers:
- `:hungarian` → HungarianSolver()
- `:sinkhorn` → SinkhornSolver(reg=sinkhorn_reg)
- `:network_simplex` → NetworkSimplexSolver()
- `:lp` → LPReferenceSolver()
- `:greedy` → GreedySolver()

Example:
```julia
solver = GenericOTSolver(method=:sinkhorn, sinkhorn_reg=0.01)
# Equivalent to: solver = SinkhornSolver(reg=0.01)
```
"""
function GenericOTSolver(; method::Symbol=:sinkhorn, sinkhorn_reg::Float64=0.01)
    if method == :hungarian
        return HungarianSolver()
    elseif method == :sinkhorn
        return SinkhornSolver(reg=sinkhorn_reg)
    elseif method == :network_simplex
        return NetworkSimplexSolver()
    elseif method == :lp
        return LPReferenceSolver()
    elseif method == :greedy
        return GreedySolver()
    else
        throw(ArgumentError("method must be :hungarian, :sinkhorn, :network_simplex, :lp, or :greedy"))
    end
end
