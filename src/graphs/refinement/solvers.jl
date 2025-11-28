using LinearAlgebra
using Hungarian
using JuMP
using HiGHS

# ============================================================================
# Curvature Solver Interface
# ============================================================================

"""
    AbstractCurvatureSolver

Base type for curvature computation strategies.
"""
abstract type AbstractCurvatureSolver end

can_handle(::AbstractCurvatureSolver, ::EdgeNeighborhoodView) = true

function compute_curvature(::AbstractCurvatureSolver, ::EdgeNeighborhoodView, ::Function)
    error("compute_curvature not implemented for this solver type")
end

# ============================================================================
# FastMatchingSolver - Hungarian Algorithm
# ============================================================================

"""
    FastMatchingSolver <: AbstractCurvatureSolver

Fast curvature solver using Hungarian algorithm for minimum-cost bipartite matching.

Applicable when neighborhoods have equal size and uniform measures. Complexity O(k³).
"""
struct FastMatchingSolver <: AbstractCurvatureSolver
    use_hungarian::Bool
    FastMatchingSolver(; use_hungarian::Bool=true) = new(use_hungarian)
end

function can_handle(::FastMatchingSolver, edge_view::EdgeNeighborhoodView{T}) where {T<:AbstractFloat}
    n_x = length(edge_view.shared) + length(edge_view.unique_x)
    n_y = length(edge_view.shared) + length(edge_view.unique_y)

    n_x == n_y || return false

    x_vals = collect(values(edge_view.x_probs))
    y_vals = collect(values(edge_view.y_probs))

    is_uniform_x = all(p ≈ x_vals[1] for p in x_vals)
    is_uniform_y = all(p ≈ y_vals[1] for p in y_vals)

    return is_uniform_x && is_uniform_y
end

function compute_curvature(
    solver::FastMatchingSolver,
    edge_view::EdgeNeighborhoodView{T},
    distance_fn::Function
) where {T<:AbstractFloat}
    can_handle(solver, edge_view) || error("FastMatchingSolver cannot handle this edge")

    S = edge_view.shared
    A = edge_view.unique_x
    B = edge_view.unique_y

    wasserstein = if isempty(A)
        zero(T)
    else
        if solver.use_hungarian && length(A) > 8
            _solve_hungarian_matching(A, B, distance_fn, T)
        else
            _solve_brute_matching(A, B, distance_fn, T)
        end / T(length(edge_view.shared) + length(A))
    end

    curvature = one(T) - wasserstein / edge_view.edge_distance
    CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein, edge_view.edge_distance, :fast_matching)
end

function _solve_hungarian_matching(A::Vector{Int}, B::Vector{Int}, distance_fn::Function, ::Type{T}) where {T}
    n = length(A)
    @assert n == length(B)
    n == 0 && return zero(T)

    cost = Matrix{Float64}(undef, n, n)
    for i in 1:n, j in 1:n
        cost[i, j] = Float64(distance_fn(A[i], B[j]))
    end

    assignment, total_cost = hungarian(cost)
    return T(total_cost)
end

function _solve_brute_matching(A::Vector{Int}, B::Vector{Int}, distance_fn::Function, ::Type{T}) where {T}
    n = length(A)
    @assert n == length(B)
    n == 0 && return zero(T)

    cost = Matrix{T}(undef, n, n)
    for i in 1:n, j in 1:n
        cost[i, j] = T(distance_fn(A[i], B[j]))
    end

    best_cost = typemax(T)
    perm = collect(1:n)

    function permute!(k::Int)
        if k == 1
            total = sum(cost[i, perm[i]] for i in 1:n)
            total < best_cost && (best_cost = total)
        else
            for i in 1:k
                permute!(k - 1)
                perm[k % 2 == 0 ? i : 1], perm[k] = perm[k], perm[k % 2 == 0 ? i : 1]
            end
        end
    end

    permute!(n)
    return best_cost
end

# ============================================================================
# GenericOTSolver - LP or Greedy
# ============================================================================

"""
    GenericOTSolver <: AbstractCurvatureSolver

Generic optimal transport solver for arbitrary measures.

Supports exact LP solution or fast greedy heuristic. Handles non-uniform measures
and unequal neighborhood sizes.
"""
struct GenericOTSolver <: AbstractCurvatureSolver
    method::Symbol
    sinkhorn_reg::Float64

    function GenericOTSolver(; method::Symbol=:lp, sinkhorn_reg::Float64=0.01)
        method ∈ (:greedy, :sinkhorn, :lp) || throw(ArgumentError("method must be :greedy, :sinkhorn, or :lp"))
        new(method, sinkhorn_reg)
    end
end

function compute_curvature(
    solver::GenericOTSolver,
    edge_view::EdgeNeighborhoodView{T},
    distance_fn::Function
) where {T<:AbstractFloat}
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
    elseif solver.method == :greedy
        _solve_greedy_ot(x_residual, y_residual, distance_fn, T)
    elseif solver.method == :lp
        _solve_lp_ot(x_residual, y_residual, distance_fn, T)
    elseif solver.method == :sinkhorn
        _solve_sinkhorn_ot(x_residual, y_residual, distance_fn, T, solver.sinkhorn_reg)
    else
        _solve_greedy_ot(x_residual, y_residual, distance_fn, T)
    end

    wasserstein = shared_cost + residual_cost
    curvature = one(T) - wasserstein / edge_view.edge_distance
    CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein, edge_view.edge_distance, :generic_ot)
end

function _solve_lp_ot(x_probs::Dict{Int,T}, y_probs::Dict{Int,T}, distance_fn::Function, ::Type{T}) where {T}
    x_support = collect(keys(x_probs))
    y_support = collect(keys(y_probs))

    n_x, n_y = length(x_support), length(y_support)
    (n_x == 0 || n_y == 0) && return zero(T)

    cost = Matrix{Float64}(undef, n_x, n_y)
    for i in 1:n_x, j in 1:n_y
        cost[i, j] = Float64(distance_fn(x_support[i], y_support[j]))
    end

    model = Model(HiGHS.Optimizer)
    set_silent(model)

    @variable(model, γ[1:n_x, 1:n_y] >= 0)
    @objective(model, Min, sum(cost[i,j] * γ[i,j] for i in 1:n_x, j in 1:n_y))

    for i in 1:n_x
        @constraint(model, sum(γ[i,j] for j in 1:n_y) == x_probs[x_support[i]])
    end

    for j in 1:n_y
        @constraint(model, sum(γ[i,j] for i in 1:n_x) == y_probs[y_support[j]])
    end

    optimize!(model)

    if termination_status(model) != OPTIMAL
        @warn "LP solver did not find optimal solution, falling back to greedy" maxlog=1
        return _solve_greedy_ot(x_probs, y_probs, distance_fn, T)
    end

    return T(objective_value(model))
end

function _solve_greedy_ot(x_probs::Dict{Int,T}, y_probs::Dict{Int,T}, distance_fn::Function, ::Type{T}) where {T}
    x_nodes = collect(keys(x_probs))
    y_nodes = collect(keys(y_probs))
    x_masses = [x_probs[x] for x in x_nodes]
    y_masses = [y_probs[y] for y in y_nodes]

    total_cost = zero(T)

    while !all(m ≈ 0 for m in x_masses) && !all(m ≈ 0 for m in y_masses)
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

        mass = min(x_masses[best_i], y_masses[best_j])
        total_cost += mass * min_cost
        x_masses[best_i] -= mass
        y_masses[best_j] -= mass
    end

    return total_cost
end

function _solve_sinkhorn_ot(x_probs::Dict{Int,T}, y_probs::Dict{Int,T}, distance_fn::Function, ::Type{T}, reg::Float64) where {T}
    """
    Sinkhorn algorithm for entropy-regularized optimal transport.

    Solves: min_{γ} ⟨C, γ⟩ + ε·KL(γ || r⊗c)
    where C is cost matrix, r and c are marginals, ε is regularization.

    Complexity: O(k² iterations), typically converges in 10-100 iterations.
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
    r = [Float64(x_probs[x_nodes[i]]) for i in 1:n_x]
    c = [Float64(y_probs[y_nodes[j]]) for j in 1:n_y]

    # Kernel: K[i,j] = exp(-C[i,j] / ε)
    K = exp.(-cost ./ reg)

    # Sinkhorn iterations
    u = ones(n_x)
    v = ones(n_y)
    max_iter = 100
    tol = 1e-6

    for iter in 1:max_iter
        u_new = r ./ (K * v)
        v_new = c ./ (K' * u_new)

        # Check convergence
        if maximum(abs.(u_new - u)) < tol && maximum(abs.(v_new - v)) < tol
            break
        end

        u = u_new
        v = v_new
    end

    # Compute transport plan and cost
    γ = u .* K .* v'
    total_cost = sum(cost .* γ)

    return T(total_cost)
end

# ============================================================================
# BruteForceSolver - For Verification
# ============================================================================

"""
    BruteForceSolver <: AbstractCurvatureSolver

Exhaustive search solver for correctness verification. O(k!) complexity.
Use only for small k (≤ 6) and debugging.
"""
struct BruteForceSolver <: AbstractCurvatureSolver end

function compute_curvature(::BruteForceSolver, edge_view::EdgeNeighborhoodView{T}, distance_fn::Function) where {T<:AbstractFloat}
    x_support = sort(collect(keys(edge_view.x_probs)))
    y_support = sort(collect(keys(edge_view.y_probs)))

    n_x, n_y = length(x_support), length(y_support)

    cost = Matrix{T}(undef, n_x, n_y)
    for i in 1:n_x, j in 1:n_y
        cost[i, j] = T(distance_fn(x_support[i], y_support[j]))
    end

    wasserstein = if n_x == n_y && all(edge_view.x_probs[x_support[i]] ≈ edge_view.y_probs[y_support[i]]
                                       for i in 1:n_x if x_support[i] == y_support[i])
        _brute_force_uniform(x_support, y_support, cost, edge_view.x_probs, T)
    else
        _brute_force_general(x_support, y_support, cost, edge_view.x_probs, edge_view.y_probs, T)
    end

    curvature = one(T) - wasserstein / edge_view.edge_distance
    CurvatureResult{T}(edge_view.x_id, edge_view.y_id, curvature, wasserstein, edge_view.edge_distance, :brute_force)
end

function _brute_force_uniform(x_support::Vector{Int}, y_support::Vector{Int}, cost::Matrix{T}, x_probs::Dict{Int,T}, ::Type{T}) where {T}
    n = length(x_support)
    best_cost = typemax(T)
    perm = collect(1:n)

    function permute!(k::Int)
        if k == 1
            total = sum(cost[i, perm[i]] * x_probs[x_support[i]] for i in 1:n)
            total < best_cost && (best_cost = total)
        else
            for i in 1:k
                permute!(k - 1)
                perm[k % 2 == 0 ? i : 1], perm[k] = perm[k], perm[k % 2 == 0 ? i : 1]
            end
        end
    end

    permute!(n)
    return best_cost
end

function _brute_force_general(x_support::Vector{Int}, y_support::Vector{Int}, cost::Matrix{T},
                               x_probs::Dict{Int,T}, y_probs::Dict{Int,T}, ::Type{T}) where {T}
    @warn "General OT brute force using greedy approximation" maxlog=1
    x_masses = [x_probs[x] for x in x_support]
    y_masses = [y_probs[y] for y in y_support]
    total_cost = zero(T)

    for i in 1:length(x_support), j in 1:length(y_support)
        mass = min(x_masses[i], y_masses[j])
        total_cost += mass * cost[i, j]
        x_masses[i] -= mass
        y_masses[j] -= mass
    end

    return total_cost
end
