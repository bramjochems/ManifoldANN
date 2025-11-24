#=
Example: Geodesic Curve Refinement

This example demonstrates different methods for refining discrete shortest paths
into smooth approximations of geodesic curves.

Run with: julia --project=. docs/examples/geodesic/03-curve-refinement.jl
=#

using ManifoldANN
using LinearAlgebra
using Random
using Printf

Random.seed!(42)

println("=" ^ 80)
println("Geodesic Curve Refinement Example")
println("=" ^ 80)
println()

# ============================================================================
# Step 1: Generate Swiss Roll Data
# ============================================================================
println("Step 1: Generating Swiss Roll manifold data")
println("-" ^ 80)

n = 200

# Load Swiss roll utilities for exact geodesic calculation
include("swiss_roll_utils.jl")

data, params = generate_swiss_roll(n; rng=Random.GLOBAL_RNG, t_min=1.5π, t_range=3π, h_scale=10.0)
t = params.t
height = params.h

println("Generated $n points on a Swiss Roll (2D manifold in 3D)")
println()

# ============================================================================
# Step 2: Build Geodesic Model
# ============================================================================
println("Step 2: Building geodesic distance model")
println("-" ^ 80)

index = build_index(BruteForceIndex, data)
method = PCAMethod(intrinsic_dim=2)
model = build_geodesic_model(method, index, data; k=15)

println("Built model with k=15 nearest neighbors")
L = mean_edge_weight(model.weighted_graph)
println("Mean edge weight: $(round(L, digits=3))")
println("  (This will be used as the length scale for curvature estimation)")
println()

# ============================================================================
# Step 3: Find a Discrete Shortest Path
# ============================================================================
println("Step 3: Computing discrete shortest path")
println("-" ^ 80)

# Find two points far apart on the manifold
start_idx = 1
end_idx = 100

result = shortest_path_with_path(model, data, start_idx, end_idx)
path = result.path
base_distance = result.distance

println("Path from node $start_idx to node $end_idx")
println("  Number of waypoints: $(length(path))")
println("  Graph shortest path distance: $(round(base_distance, digits=3))")
println()

# ============================================================================
# Step 4: Compare Refinement Methods
# ============================================================================
println("Step 4: Refining the path with different methods")
println("-" ^ 80)
println()

# Define refinement methods to compare
# Note: CurvatureCorrectedDistance can work standalone (on discrete path)
#       OR composed with another refinement (curvature-correct the smooth curve)
refinement_methods = [
    ("Discrete Path (no refinement)", NoRefinement()),
    ("Subdivision Smooth (5 subdivs)", SubdivisionSmoothing(
        subdivisions=5,
        max_iterations=20
    )),
    ("Subdivision Smooth (10 subdivs)", SubdivisionSmoothing(
        subdivisions=10,
        max_iterations=30
    )),
    ("Curvature Corrected (discrete)", CurvatureCorrectedDistance()),
    ("Curvature Corrected + Smooth", CurvatureCorrectedDistance(
        base_refinement=SubdivisionSmoothing(subdivisions=10, max_iterations=20)
    )),
]

# Store results

results = []

for (name, refinement) in refinement_methods
    refined = refine_path(refinement, model, data, path)

    # Compute smoothness metric (mean local curvature)
    mean_curv = if length(refined.points) >= 3
        curvatures = Float64[]
        for i in 2:length(refined.points)-1
            v1 = normalize(refined.points[i] - refined.points[i-1])
            v2 = normalize(refined.points[i+1] - refined.points[i])
            angle = acos(clamp(dot(v1, v2), -1, 1))
            push!(curvatures, angle)
        end
        isempty(curvatures) ? 0.0 : sum(curvatures) / length(curvatures)
    else
        0.0
    end

    push!(results, (name=name, refined=refined, mean_curv=mean_curv))
end

# ============================================================================
# Step 5: Display Results Table
# ============================================================================
println("Results Summary")
println("=" ^ 80)
println()

# Table header
@printf "%-35s %8s %10s %8s %8s\n" "Method" "Points" "Distance" "Change" "Smoothness"
println("-" ^ 80)

for (name, refined, mean_curv) in results
    change_pct = (refined.distance / base_distance - 1) * 100
    @printf "%-35s %8d %10.3f %+7.1f%% %8.4f\n" name length(refined.points) refined.distance change_pct mean_curv
end

println("-" ^ 80)
println()
println("Notes:")
println("  - Distance: Total arc length of the curve")
println("  - Change: Percent difference from graph shortest path ($(round(base_distance, digits=3)))")
println("  - Smoothness: Mean local curvature (rad) - lower is smoother")
println()
#
# ============================================================================
# Step 6: Curvature Correction Details
# ============================================================================
println("=" ^ 80)
println("Understanding Curvature Correction")
println("=" ^ 80)
println()

# Get the curvature-corrected results
curv_discrete = results[4].refined
curv_smooth = results[5].refined

println("Curvature correction formula:")
println("  d_geodesic ≈ d_tangent × (1 + (κ × d_tangent)² / 24)")
println()
println("Where:")
println("  • d_tangent: Distance measured in local tangent space")
println("  • κ: Local curvature (units: 1/length)")
println("  • κ = κ_indicator / length_scale")
println("    - κ_indicator: PCA eigenvalue spread (dimensionless, 0-1)")
println("    - length_scale: Converts to physical curvature (default: mean edge weight)")
println()
println("Why length_scale matters:")
println("  • Same eigenvalue spread means different things at different scales")
println("  • Example: Sphere radius 1 vs radius 100 → very different curvatures")
println("  • Auto-detected as mean edge weight: $(round(mean_edge_weight(model.weighted_graph), digits=3))")
println()

# Show the magnitude of correction
discrete_no_curv = results[1].refined
correction_amount = curv_discrete.distance - discrete_no_curv.distance
correction_pct = (curv_discrete.distance / discrete_no_curv.distance - 1) * 100

println("For this path:")
println("  • Discrete distance (no correction): $(round(discrete_no_curv.distance, digits=3))")
println("  • With curvature correction: $(round(curv_discrete.distance, digits=3))")
println("  • Correction amount: +$(round(correction_amount, digits=3)) (+$(round(correction_pct, digits=1))%)")
println()

if correction_pct > 5
    println("Note: Correction >5% suggests either high curvature or large edge lengths.")
    println("      You may want to tune the length_scale parameter explicitly.")
println()
end

# ============================================================================
# Step 7b: Accuracy Validation with Exact Geodesics
# ============================================================================
println("=" ^ 80)
println("Step 7b: Validating Against Exact Geodesics")
println("=" ^ 80)
println()

# Compute exact geodesic distance for the path
function compute_exact_path_distance(params, path)
    total = 0.0
    for i in 1:length(path)-1
        total += exact_swiss_roll_geodesic(params, path[i], path[i+1])
    end
    return total
end

exact_total = compute_exact_path_distance(params, path)

println("Distance estimates for the path:")
println()

@printf "%-35s %12s %14s\n" "Method" "Distance" "Error vs Exact"
println("-" ^ 80)

for (name, refined, _) in results
    error_pct = abs(refined.distance - exact_total) / exact_total * 100
    @printf "%-35s %12.3f %13.2f%%\n" name refined.distance error_pct
end

@printf "%-35s %12.3f %14s\n" "Exact (analytical)" exact_total "-"
println("-" ^ 80)
println()

println("Observations:")
println("  • Graph-based methods approximate true geodesic distance well (< 5% error)")
println("  • Curvature correction brings estimates closer to exact values")
println("  • Subdivision smoothing may shorten paths (averaging pulls points inward)")
println("  • Exact geodesic provides ground truth for validation")
println()

# ============================================================================
# Step 8: Performance Comparison
# ============================================================================
println("=" ^ 80)
println("Performance Comparison (10 runs each)")
println("=" ^ 80)
println()

@printf "%-35s %12s\n" "Method" "Mean Time"
println("-" ^ 80)

for (name, refinement) in refinement_methods
    times = Float64[]
    for _ in 1:10
        start_time = time()
        refine_path(refinement, model, data, path)
        elapsed = time() - start_time
        push!(times, elapsed)
    end

    mean_time = sum(times) / length(times)
    @printf "%-35s %10.2f ms\n" name (mean_time * 1000)
end

println("-" ^ 80)
println()

