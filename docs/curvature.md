# Implementing and Comparing Curvature Solvers

This document outlines what's needed to fully implement and compare the FastMatchingSolver (Hungarian algorithm) with an LP-based solver for Ollivier-Ricci curvature computation.

## Current Implementation Status

### ✅ Fully Implemented
1. **FastMatchingSolver (Brute Force variant)**: O(k!) for small k ≤ 8
2. **GenericOTSolver (Greedy variant)**: Fast approximation via greedy coupling
3. **BruteForceSolver**: Exhaustive search for verification

### ⚠️ Placeholders
1. **FastMatchingSolver (Hungarian variant)**: O(k³) - needs Hungarian.jl integration
2. **GenericOTSolver (LP variant)**: Exact OT via linear programming - needs JuMP.jl
3. **GenericOTSolver (Sinkhorn variant)**: Entropic regularization - needs custom implementation

---

## 1. Hungarian Algorithm Implementation

The Hungarian algorithm solves the assignment problem (minimum-cost bipartite matching) in O(k³) time, making it practical for larger k values.

### Dependencies Required

Add to `Project.toml`:
```toml
[deps]
Hungarian = "e91730f6-4275-51fb-a7a0-7064cfbd3b39"
```

### Implementation Steps

**File**: `src/graphs/refinement/solvers.jl`

Replace the placeholder `_solve_hungarian_matching` function:

```julia
using Hungarian

function _solve_hungarian_matching(
    A::Vector{Int},
    B::Vector{Int},
    distance_fn::Function,
    ::Type{T}
) where {T}
    n = length(A)
    @assert n == length(B)

    if n == 0
        return zero(T)
    end

    # Build cost matrix
    cost = Matrix{T}(undef, n, n)
    for i in 1:n
        for j in 1:n
            cost[i, j] = T(distance_fn(A[i], B[j]))
        end
    end

    # Solve assignment problem using Hungarian algorithm
    assignment, total_cost = hungarian(cost)

    return T(total_cost)
end
```

### Testing

Add test in `test/unit/graphs/refinement_tests.jl`:

```julia
@testset "Hungarian vs Brute Force" begin
    # Test that Hungarian gives same result as brute force
    distance_fn = (i, j) -> abs(Float64(i - j))

    for k in 2:6
        A = collect(1:k)
        B = collect(k+1:2k)

        cost_hungarian = _solve_hungarian_matching(A, B, distance_fn, Float64)
        cost_brute = _solve_brute_matching(A, B, distance_fn, Float64)

        @test cost_hungarian ≈ cost_brute
    end
end
```

### Performance Comparison

| k | Brute Force O(k!) | Hungarian O(k³) | Speedup |
|---|-------------------|-----------------|---------|
| 3 | ~6 perms | 27 ops | 0.2x |
| 5 | ~120 perms | 125 ops | 0.96x |
| 8 | ~40,320 perms | 512 ops | **78x** |
| 10 | ~3.6M perms | 1,000 ops | **3,600x** |
| 20 | ~2.4×10¹⁸ perms | 8,000 ops | **~10¹⁵x** |

---

## 2. LP-Based Exact OT Solver

For non-uniform measures or unequal support sizes, we need a proper optimal transport solver. Linear programming provides the exact solution.

### Dependencies Required

Add to `Project.toml`:
```toml
[deps]
JuMP = "4076af6c-e467-56ae-b986-b466b2749572"
HiGHS = "87dc4568-4c63-4d18-b0c0-bb2238e4078b"  # Fast open-source LP solver
```

### Implementation

**File**: `src/graphs/refinement/solvers.jl`

Replace the placeholder `_solve_lp_ot` function:

```julia
using JuMP
using HiGHS

function _solve_lp_ot(
    x_probs::Dict{Int,T},
    y_probs::Dict{Int,T},
    distance_fn::Function,
    ::Type{T}
) where {T}
    x_support = collect(keys(x_probs))
    y_support = collect(keys(y_probs))

    n_x = length(x_support)
    n_y = length(y_support)

    if n_x == 0 || n_y == 0
        return zero(T)
    end

    # Build cost matrix
    cost = Matrix{Float64}(undef, n_x, n_y)
    for i in 1:n_x
        for j in 1:n_y
            cost[i, j] = Float64(distance_fn(x_support[i], y_support[j]))
        end
    end

    # Create optimization model
    model = Model(HiGHS.Optimizer)
    set_silent(model)  # Suppress solver output

    # Decision variables: transport plan γ[i,j] ≥ 0
    @variable(model, γ[1:n_x, 1:n_y] >= 0)

    # Objective: minimize total transport cost
    @objective(model, Min, sum(cost[i,j] * γ[i,j] for i in 1:n_x, j in 1:n_y))

    # Marginal constraints
    for i in 1:n_x
        @constraint(model, sum(γ[i,j] for j in 1:n_y) == x_probs[x_support[i]])
    end

    for j in 1:n_y
        @constraint(model, sum(γ[i,j] for i in 1:n_x) == y_probs[y_support[j]])
    end

    # Solve
    optimize!(model)

    if termination_status(model) != OPTIMAL
        @warn "LP solver did not find optimal solution" status=termination_status(model)
        # Fall back to greedy
        return _solve_greedy_ot(x_probs, y_probs, distance_fn, T)
    end

    return T(objective_value(model))
end
```

### Testing

```julia
@testset "LP vs Greedy OT" begin
    # Create non-uniform measures
    x_probs = Dict(1 => 0.5, 2 => 0.3, 3 => 0.2)
    y_probs = Dict(2 => 0.4, 3 => 0.35, 4 => 0.25)

    distance_fn = (i, j) -> abs(Float64(i - j))

    cost_lp = _solve_lp_ot(x_probs, y_probs, distance_fn, Float64)
    cost_greedy = _solve_greedy_ot(x_probs, y_probs, distance_fn, Float64)

    # LP should be ≤ greedy (optimal vs heuristic)
    @test cost_lp <= cost_greedy + 1e-6

    # For this simple case, verify by hand:
    # Optimal transport:
    #   1 → 2: 0.4 at cost 1 = 0.4
    #   1 → 3: 0.1 at cost 2 = 0.2
    #   2 → 3: 0.25 at cost 1 = 0.25
    #   2 → 4: 0.05 at cost 2 = 0.1
    #   3 → 4: 0.2 at cost 1 = 0.2
    # Total = 1.15
    @test cost_lp ≈ 1.15 atol=0.01
end
```

---

## 3. Comprehensive Solver Comparison

### Complexity Analysis

| Solver | Time Complexity | Space | Use Case |
|--------|----------------|-------|----------|
| **Brute Force** | O(k!) | O(k²) | k ≤ 6, verification |
| **Hungarian** | O(k³) | O(k²) | Uniform, equal degree, k ≤ 100 |
| **Greedy** | O(k² log k) | O(k²) | Fast approximation |
| **LP (HiGHS)** | O(k³) typical | O(k²) | Non-uniform, exact |
| **Sinkhorn** | O(k² × iterations) | O(k²) | Large k, approximate |

### Accuracy Comparison

Create a comprehensive comparison example:

**File**: `docs/examples/graphs/06-curvature-solver-comparison.jl`

```julia
#=
Example: Curvature Solver Comparison

Compares different solvers for computing Ollivier-Ricci curvature:
- FastMatchingSolver (Brute Force and Hungarian)
- GenericOTSolver (Greedy and LP)
- BruteForceSolver

Run with `julia --project=. docs/examples/graphs/06-curvature-solver-comparison.jl`
=#

using ManifoldANN
using LinearAlgebra
using Random
using Printf
using Statistics

println("=" ^ 70)
println("Curvature Solver Comparison")
println("=" ^ 70)
println()

# Generate test data
Random.seed!(42)
n_points = 100
data = randn(10, n_points)

# Build kNN graph
k = 8
index = build_index(BruteForceIndex, data)
graph = build_knn_graph(index, data; k=k)

distance_fn = (i, j) -> norm(data[:, i] - data[:, j])

# Sample edges for detailed comparison
sample_edges = [(i, graph[i][1]) for i in [1, 25, 50, 75, 100]]

println("Testing on $(length(sample_edges)) sample edges from kNN graph")
println("k = $k neighbors per node\n")

# ============================================================================
# Detailed Edge-by-Edge Comparison
# ============================================================================

solvers = [
    ("Brute Force", BruteForceSolver()),
    ("Fast Matching (Brute)", FastMatchingSolver(use_hungarian=false)),
    ("Fast Matching (Hungarian)", FastMatchingSolver(use_hungarian=true)),
    ("Generic OT (Greedy)", GenericOTSolver(method=:greedy)),
    ("Generic OT (LP)", GenericOTSolver(method=:lp))
]

println("=" ^ 70)
println("Edge-by-Edge Results")
println("=" ^ 70)
println()

for (x, y) in sample_edges
    x_nb = uniform_neighborhood(x, graph[x], Float64)
    y_nb = uniform_neighborhood(y, graph[y], Float64)
    edge_dist = distance_fn(x, y)
    edge_view = create_edge_view(x_nb, y_nb, edge_dist)

    println("Edge ($x, $y):")
    println("  Shared neighbors: $(length(edge_view.shared))")
    println("  Unique to x: $(length(edge_view.unique_x))")
    println("  Unique to y: $(length(edge_view.unique_y))")

    for (name, solver) in solvers
        if !can_handle(solver, edge_view) && solver isa FastMatchingSolver
            println("  %-30s: Cannot handle (degree mismatch)" % name)
            continue
        end

        time_ns = @elapsed result = compute_curvature(solver, edge_view, distance_fn)
        println(@sprintf("  %-30s: κ = %6.3f, W₁ = %6.3f, time = %6.3f ms",
                        name, result.curvature, result.wasserstein_distance, time_ns * 1000))
    end
    println()
end

# ============================================================================
# Performance Benchmark
# ============================================================================

println("=" ^ 70)
println("Performance Benchmark (100 edges)")
println("=" ^ 70)
println()

# Sample 100 random edges
benchmark_edges = [(rand(1:n_points), rand(1:n_points)) for _ in 1:100]

for (name, solver) in solvers
    times = Float64[]
    curvatures = Float64[]
    skipped = 0

    for (x, y) in benchmark_edges
        x in graph[x] || continue
        y_idx = findfirst(==(y), graph[x])
        y_idx === nothing && continue

        x_nb = uniform_neighborhood(x, graph[x], Float64)
        y_nb = uniform_neighborhood(y, graph[y], Float64)
        edge_dist = distance_fn(x, y)
        edge_view = create_edge_view(x_nb, y_nb, edge_dist)

        if !can_handle(solver, edge_view) && solver isa FastMatchingSolver
            skipped += 1
            continue
        end

        time_ns = @elapsed result = compute_curvature(solver, edge_view, distance_fn)
        push!(times, time_ns * 1000)  # Convert to ms
        push!(curvatures, result.curvature)
    end

    if !isempty(times)
        println(@sprintf("%-30s:", name))
        println(@sprintf("  Mean time:   %8.3f ms", mean(times)))
        println(@sprintf("  Median time: %8.3f ms", median(times)))
        println(@sprintf("  Min time:    %8.3f ms", minimum(times)))
        println(@sprintf("  Max time:    %8.3f ms", maximum(times)))
        println(@sprintf("  Mean κ:      %8.3f", mean(curvatures)))
        println(@sprintf("  Edges processed: %d / %d", length(times), length(benchmark_edges)))
        if skipped > 0
            println(@sprintf("  Skipped: %d (cannot handle)", skipped))
        end
    else
        println(@sprintf("%-30s: No edges processed", name))
    end
    println()
end

# ============================================================================
# Accuracy Comparison (vs ground truth)
# ============================================================================

println("=" ^ 70)
println("Accuracy Comparison")
println("=" ^ 70)
println()

println("Computing curvatures with all solvers...")
println()

# Use Brute Force as ground truth for small k
truth_solver = BruteForceSolver()

comparison_edges = sample_edges[1:3]  # Small subset for detailed comparison
errors = Dict{String, Vector{Float64}}()

for (name, solver) in solvers
    name == "Brute Force" && continue
    errors[name] = Float64[]
end

for (x, y) in comparison_edges
    x_nb = uniform_neighborhood(x, graph[x], Float64)
    y_nb = uniform_neighborhood(y, graph[y], Float64)
    edge_dist = distance_fn(x, y)
    edge_view = create_edge_view(x_nb, y_nb, edge_dist)

    truth = compute_curvature(truth_solver, edge_view, distance_fn)

    for (name, solver) in solvers
        name == "Brute Force" && continue

        if can_handle(solver, edge_view) || !(solver isa FastMatchingSolver)
            result = compute_curvature(solver, edge_view, distance_fn)
            error = abs(result.curvature - truth.curvature)
            push!(errors[name], error)
        end
    end
end

println("Mean Absolute Error vs Brute Force:")
for (name, errs) in sort(collect(errors), by=x->x[1])
    if !isempty(errs)
        println(@sprintf("  %-30s: MAE = %.6f", name, mean(errs)))
    end
end

println("\n" ^ 70)
println("=== Comparison Complete ===")
```

### Running the Comparison

```bash
# First, add dependencies (if implementing LP solver)
julia --project=. -e 'using Pkg; Pkg.add("JuMP"); Pkg.add("HiGHS")'

# For Hungarian algorithm
julia --project=. -e 'using Pkg; Pkg.add("Hungarian")'

# Run comparison
julia --project=. docs/examples/graphs/06-curvature-solver-comparison.jl
```

---

## 4. Integration Checklist

- [ ] Add Hungarian.jl dependency to Project.toml
- [ ] Implement `_solve_hungarian_matching` using Hungarian.jl
- [ ] Add tests comparing Hungarian vs Brute Force
- [ ] Add JuMP and HiGHS dependencies (optional, for LP)
- [ ] Implement `_solve_lp_ot` using JuMP
- [ ] Add tests comparing LP vs Greedy
- [ ] Create comprehensive comparison example
- [ ] Update documentation with performance guidelines
- [ ] Benchmark on real manifold data (Swiss roll, etc.)

---

## 5. Recommendations

### For Production Use

1. **k ≤ 10, uniform measures**: Use `FastMatchingSolver(use_hungarian=true)`
2. **k > 10, uniform measures**: Use `FastMatchingSolver(use_hungarian=true)` (still efficient)
3. **Non-uniform measures, small k**: Use `GenericOTSolver(method=:lp)`
4. **Non-uniform measures, large k or approximation OK**: Use `GenericOTSolver(method=:greedy)`

### Default Configuration

```julia
# Recommended default
solver = FastMatchingSolver(use_hungarian=true)
fallback = GenericOTSolver(method=:lp)

filtered = filter_graph(graph, data,
    solver=solver,
    fallback_solver=fallback
)
```

This provides:
- Fast exact solution for most edges (uniform neighborhoods)
- Exact fallback for non-uniform cases
- O(k³) complexity throughout
