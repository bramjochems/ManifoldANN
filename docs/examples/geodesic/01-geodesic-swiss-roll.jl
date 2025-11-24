#=
Example: Geodesic Distance on Swiss Roll

This example demonstrates how geodesic distance estimation works on a Swiss Roll
manifold, where Euclidean distance can be misleading.

The Swiss Roll is a classic manifold learning example where two points may be
close in 3D Euclidean space (through the "air") but far apart along the actual
manifold surface (walking along the roll).

Run with `julia --project=. docs/examples/geodesic/01-geodesic-swiss-roll.jl`
=#

using ManifoldANN
using LinearAlgebra
using Random
using Printf

# Load Swiss roll utilities (exact geodesic calculations)
include("swiss_roll_utils.jl")

# Set seed for reproducibility
rng = MersenneTwister(42)

# ============================================================================
# Step 1: Generate Swiss Roll Data
# ============================================================================
println("=" ^ 60)
println("Step 1: Generating Swiss Roll manifold data")
println("=" ^ 60)

n_points = 1000
data, params = generate_swiss_roll(n_points; rng=rng, t_min=1.5π, t_range=3π, h_scale=20.0)
t = params.t
height = params.h

println("Generated $(n_points) points on a Swiss Roll in 3D")
println("  - Intrinsic dimension: 2 (t and height)")
println("  - Ambient dimension: 3")
println("  - Parameters stored for exact geodesic computation")
println()

# ============================================================================
# Step 2: Build ANN Index
# ============================================================================
println("=" ^ 60)
println("Step 2: Building ANN index for fast neighbor queries")
println("=" ^ 60)

# BruteForceIndex for exact neighbors (HNSW works too for larger datasets)
index = build_index(BruteForceIndex, data)
println("Built BruteForceIndex for $(size(data, 2)) points")
println()

# ============================================================================
# Step 3: Configure Local Geometry Method
# ============================================================================
println("=" ^ 60)
println("Step 3: Configuring PCA-based local geometry estimation")
println("=" ^ 60)

# The Swiss Roll is intrinsically 2D, so we use intrinsic_dim=2
# This tells the PCA method to fit 2D tangent planes at each point
method = PCAMethod(intrinsic_dim=2)
println("Using PCAMethod with intrinsic_dim=2")
println("  - This fits a 2D tangent plane at each neighborhood")
println("  - Edge weights will measure tangent-space distances")
println()

# ============================================================================
# Alternative: Adaptive Neighborhood Selection
# ============================================================================
# The geometry method and neighborhood strategy are ORTHOGONAL.
# You can compose any strategy with any method:
#
#   strategy = AdaptiveNeighborhood(max_neighbors=30, min_neighbors=8, max_error=0.1)
#   estimator = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=2))
#
# AdaptiveNeighborhood iteratively removes neighbors with high reconstruction
# error, which helps when neighbors may come from different "layers" of a
# curved manifold like the Swiss Roll.
#
# For this example we use simple PCAMethod, but adaptive selection can improve
# geometry estimation in regions of high curvature.
println("Note: For curved manifolds, you can use adaptive neighborhood selection:")
println("  strategy = AdaptiveNeighborhood(max_neighbors=30, min_neighbors=8)")
println("  estimator = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=2))")
println()

# ============================================================================
# Step 4: Build Geodesic Model
# ============================================================================
println("=" ^ 60)
println("Step 4: Building geodesic distance model")
println("=" ^ 60)

k = 15  # Number of neighbors for the kNN graph
model = build_geodesic_model(method, index, data; k=k)

println("Built GeodesicDistanceModel:")
println("  - kNN graph with k=$(k)")
println("  - $(length(model.weighted_graph)) nodes with fitted geometries")
println("  - Edge weights computed using local PCA distances")
println()

# ============================================================================
# Step 5: Compare Euclidean vs Geodesic Distance
# ============================================================================
println("=" ^ 60)
println("Step 5: Comparing Euclidean vs Geodesic distances")
println("=" ^ 60)

# Find two points that are:
# - Close in Euclidean space (through the roll)
# - Far apart along the manifold surface
#
# Strategy: find points with similar x,z coordinates but very different t values
# (i.e., on different "layers" of the roll)

function find_shortcut_pair(data, t_values)
    best_pair = (0, 0)
    best_ratio = 0.0

    for i in 1:size(data, 2)
        for j in (i+1):size(data, 2)
            euclidean = norm(data[:, i] - data[:, j])
            t_diff = abs(t_values[i] - t_values[j])

            # We want small Euclidean but large t difference
            # (points on different "layers" of the roll)
            if euclidean < 5.0 && t_diff > 3.0
                ratio = t_diff / euclidean
                if ratio > best_ratio
                    best_ratio = ratio
                    best_pair = (i, j)
                end
            end
        end
    end

    return best_pair
end

point_a, point_b = find_shortcut_pair(data, t)

if point_a > 0 && point_b > 0
    println("Found 'shortcut' pair: points $point_a and $point_b")
else
    # Fall back to comparing points with very different t values
    sorted_t = sortperm(t)
    point_a = sorted_t[1]        # Smallest t (inner part of roll)
    point_b = sorted_t[end]      # Largest t (outer part of roll)
    println("Comparing inner vs outer roll: points $point_a and $point_b")
end

euclidean_dist = norm(data[:, point_a] - data[:, point_b])
geodesic_dist = geodesic_distance(model, data, point_a, point_b)
t_diff = abs(t[point_a] - t[point_b])

println()
println("  Euclidean distance: $(round(euclidean_dist, digits=3))")
println("  Geodesic distance:  $(round(geodesic_dist, digits=3))")
println("  True arc parameter difference: $(round(t_diff, digits=3))")
println()
println("  Ratio (geodesic/euclidean): $(round(geodesic_dist/euclidean_dist, digits=2))x")
println()
println("  -> Geodesic distance captures the true separation along the manifold!")
println()

# ============================================================================
# Step 5b: Validation with Exact Geodesic Distances
# ============================================================================
println("=" ^ 60)
println("Step 5b: Validating with exact geodesic distances")
println("=" ^ 60)

# The Swiss roll is isometric to a plane, so we can compute exact geodesics analytically
# Geodesic distance in the unrolled (s, h) coordinates is simply Euclidean distance

println("Computing exact geodesic distances for validation...")
println()

# Select diverse test pairs
test_pairs = [
    (point_a, point_b),                           # The shortcut pair
    (1, n_points ÷ 2),                           # Near vs middle
    (n_points ÷ 4, 3 * n_points ÷ 4),           # Middle regions
    (10, 100),                                    # Random pair
    (50, 500),                                    # Random pair
]

@printf "%-20s %12s %12s %12s %12s %12s\n" "Pair" "Ambient" "Graph" "Exact" "Graph Error" "Ambient/Exact"
println("-" ^ 90)

for (i, j) in test_pairs
    d_ambient = ambient_distance(data, i, j)
    d_graph = geodesic_distance(model, data, i, j)
    d_exact = exact_swiss_roll_geodesic(params, i, j)

    graph_error = abs(d_graph - d_exact) / d_exact * 100  # Percent error
    ambient_ratio = d_ambient / d_exact

    @printf "%-20s %12.3f %12.3f %12.3f %11.2f%% %12.2fx\n" "($i, $j)" d_ambient d_graph d_exact graph_error ambient_ratio
end

println("-" ^ 90)
println()

println("Observations:")
println("  • Ambient distance can be much smaller than geodesic (shortcuts through the roll)")
println("  • Graph approximation closely matches exact geodesic distances")
println("  • Typical error is < 5% for well-sampled manifolds")
println("  • Error decreases with larger k (more neighbors per node)")
println()

# ============================================================================
# Step 6: Query with New Points
# ============================================================================
println("=" ^ 60)
println("Step 6: Computing geodesic distance for new query points")
println("=" ^ 60)

# Generate a new point on the Swiss Roll
t_new = 2π
h_new = 5.0
new_point = [t_new * cos(t_new), h_new, t_new * sin(t_new)]

println("New query point at t=$(round(t_new, digits=2)), height=$(h_new)")
println("  Coordinates: $(round.(new_point, digits=3))")
println()

# Find geodesic distance to a few graph nodes
println("Geodesic distances from new point to selected graph nodes:")
for target_idx in [1, n_points ÷ 4, n_points ÷ 2, 3 * n_points ÷ 4]
    gdist = geodesic_distance(model, data, new_point, target_idx)
    edist = norm(new_point - data[:, target_idx])
    println("  -> Node $target_idx: geodesic=$(round(gdist, digits=3)), euclidean=$(round(edist, digits=3))")
end
println()

# ============================================================================
# Step 7: Path Reconstruction
# ============================================================================
println("=" ^ 60)
println("Step 7: Shortest path reconstruction (for visualization)")
println("=" ^ 60)

# Get shortest path between two points
if point_a > 0 && point_b > 0
    result = shortest_path_with_path(model, data, point_a, point_b)
    path = result.path

    println("Shortest path from $point_a to $point_b:")
    println("  Path length: $(length(path)) nodes")
    println("  Total distance: $(round(result.distance, digits=3))")
    println("  Path: $(join(path[1:min(10, length(path))], " -> "))" *
            (length(path) > 10 ? " -> ..." : ""))
    println()
    println("  This path follows the manifold surface, not a straight line through 3D!")
end

# ============================================================================
# Step 8: Adaptive Neighborhood Selection Demo
# ============================================================================
println("=" ^ 60)
println("Step 8: Comparing fixed vs adaptive neighborhood selection")
println("=" ^ 60)

# Build a model with adaptive neighborhood selection
# This can help when the manifold has high curvature
strategy = AdaptiveNeighborhood(
    max_neighbors=25,    # Start with more candidate neighbors
    min_neighbors=8,     # Keep at least this many
    max_error=0.15       # Target reconstruction error threshold
)
adaptive_method = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=2))

# Build weighted graph with adaptive selection
# candidate_k provides extra neighbors for the adaptive filtering process
adaptive_wg = build_weighted_graph(adaptive_method, index, data; k=15, candidate_k=25)

# Check how many neighbors were kept on average
neighbor_counts = [used_neighbor_count(g) for g in adaptive_wg.geometries]
avg_neighbors = sum(neighbor_counts) / length(neighbor_counts)
min_kept = minimum(neighbor_counts)
max_kept = maximum(neighbor_counts)

println("Adaptive neighborhood selection results:")
println("  - Average neighbors kept: $(round(avg_neighbors, digits=1)) / 25 candidates")
println("  - Range: $min_kept to $max_kept neighbors")
println()

# Compare reconstruction errors
errors = [max_reconstruction_error(g) for g in adaptive_wg.geometries]
println("Reconstruction error statistics:")
println("  - Mean: $(round(sum(errors)/length(errors), digits=4))")
println("  - Max: $(round(maximum(errors), digits=4))")
println()

println("The adaptive strategy filters out neighbors that don't fit the local")
println("tangent plane well, which can occur at high-curvature regions or when")
println("neighbors come from different 'layers' of the Swiss Roll.")

println()
println("=" ^ 60)
println("Done! The geodesic distance model successfully estimates manifold distances.")
println("=" ^ 60)
