#=
Example: Geodesic Robustness under Random Projections

We embed a Swiss roll (intrinsically 2D) into 100D with a random linear map
plus small noise, then apply additional random projections to 10D, 25D, and
50D. For each space we compare:
  - kNN neighbor overlap vs the 100D reference
  - Tangent plane rotation (principal angles)
  - Geodesic distance error vs the exact Swiss roll metric

Run with:
    julia --project=. docs/examples/geodesic/04-random-projection-robustness.jl
=#

using ManifoldANN
using LinearAlgebra
using Random
using Statistics
using Printf

# Load Swiss roll utilities (exact geodesic calculations)
include("swiss_roll_utils.jl")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
const RNG_SEED = 42
const N_POINTS = 800
const HI_DIM = 100
const PROJECTION_DIMS = [10, 25, 50]
const K_NEIGHBORS = 20
const CANDIDATE_K = 40
const NOISE_SCALE = 0.005
const INTRINSIC_DIM = 2
const T_MIN = Float64(π)        # Start closer to the center for higher curvature
const T_RANGE = Float64(12π)    # Even more turns -> stronger curvature/overlaps
const H_SCALE = 10.0    # Tighter height to emphasize the roll

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
function summarize(values::AbstractVector{<:Real})
    return (
        mean = mean(values),
        median = median(values),
        max = maximum(values),
    )
end

function print_stats(label::String, stats; scale::Real=1.0, suffix::String="")
    s = Float64(scale)
    @printf("  %-24s mean=%.3f  median=%.3f  max=%.3f%s\n",
            label, stats.mean * s, stats.median * s, stats.max * s, suffix)
end

function neighbor_overlap(base::KNNGraph, other::KNNGraph)
    length(base) == length(other) ||
        error("Graphs must have the same number of nodes for overlap comparison")

    overlaps = Vector{Float64}(undef, length(base))
    for i in 1:length(base)
        ref = Set(base[i])
        proj = Set(other[i])
        inter = length(intersect(ref, proj))
        uni = length(union(ref, proj))
        overlaps[i] = inter / uni
    end
    return overlaps
end

function tangent_angles(ref_geoms, proj_geoms)
    error("tangent_angles requires a projection matrix; use tangent_angles_proj")
end

function geodesic_errors(model::GeodesicDistanceModel, data, params, pairs)
    errors = Vector{Float64}(undef, length(pairs))
    for (idx, (i, j)) in enumerate(pairs)
        d_true = exact_swiss_roll_geodesic(params, i, j)
        d_est = geodesic_distance(model, data, i, j)
        errors[idx] = abs(d_est - d_true) / d_true
    end
    return errors
end

function ambient_errors(data, params, pairs)
    errors = Vector{Float64}(undef, length(pairs))
    for (idx, (i, j)) in enumerate(pairs)
        d_true = exact_swiss_roll_geodesic(params, i, j)
        d_est = norm(data[:, i] - data[:, j])
        errors[idx] = abs(d_est - d_true) / d_true
    end
    return errors
end

function sample_pairs(rng::AbstractRNG, n::Int; count::Int=16)
    pairs = [(1, n), (n ÷ 4, 3n ÷ 4)]
    while length(pairs) < count
        i = rand(rng, 1:n)
        j = rand(rng, 1:n)
        i == j && continue
        push!(pairs, (i, j))
    end
    return pairs
end

function build_model(data; k::Int=K_NEIGHBORS)
    index = build_index(BruteForceIndex, data)
    estimator = LocalGeometryEstimator(
        AdaptiveNeighborhood(
            max_neighbors=CANDIDATE_K,
            min_neighbors=8,
            criterion=SubspaceAngleCriterion(π/12),  # ~15 degrees
        ),
        PCAMethod(intrinsic_dim=INTRINSIC_DIM),
    )
    weighted_graph = build_weighted_graph(
        estimator,
        index,
        data;
        k=k,
        candidate_k=CANDIDATE_K,
        edge_weight=TangentProjectedSymmetricMean(),
    )
    return GeodesicDistanceModel(index, weighted_graph, estimator)
end

function project_geometry(geom_in::AbstractLocalGeometry, proj::AbstractMatrix)
    geom = unwrap_geometry(geom_in)
    # Project center and basis into the lower-dimensional space, then re-orthonormalize.
    projected_center = proj * geom.center
    B = proj * geom.basis
    Q = qr(B).Q
    d = size(B, 2)
    projected_basis = Matrix(Q[:, 1:d])
    eigenvalues = length(geom.eigenvalues) >= d ? geom.eigenvalues[1:d] : geom.eigenvalues
    return PCAGeometry(projected_center, projected_basis, eigenvalues)
end

function tangent_angles_proj(ref_geoms, proj_geoms, proj::AbstractMatrix)
    length(ref_geoms) == length(proj_geoms) ||
        error("Geometry vectors must align for tangent comparison")
    angles = Vector{Float64}(undef, length(ref_geoms))
    for i in 1:length(ref_geoms)
        ref_proj = project_geometry(ref_geoms[i], proj)
        geom_proj = unwrap_geometry(proj_geoms[i])
        angles[i] = subspace_angle(ref_proj, geom_proj)
    end
    return angles
end

# -----------------------------------------------------------------------------
# Step 1: Generate Swiss roll and embed to 100D
# -----------------------------------------------------------------------------
rng = MersenneTwister(RNG_SEED)

println("=" ^ 70)
println("Random Projection Robustness on a Swiss Roll")
println("=" ^ 70)
println("Config: k=$K_NEIGHBORS, candidate_k=$CANDIDATE_K, adaptive + SubspaceAngle(π/12)")
println()

base_data, params = generate_swiss_roll(N_POINTS; rng=rng, t_min=T_MIN, t_range=T_RANGE, h_scale=H_SCALE)

# Embed into 100D with a random linear map plus small Gaussian noise
lift_matrix = randn(rng, 3, HI_DIM)  # 3x100 matrix
noise = NOISE_SCALE .* randn(rng, HI_DIM, N_POINTS)
data_hi = lift_matrix' * base_data + noise

println("Generated Swiss roll and lifted to $HI_DIM dimensions:")
println("  - Points: $(N_POINTS)")
println("  - Intrinsic dimension: $INTRINSIC_DIM")
println("  - Ambient dimension (reference): $(size(data_hi, 1))")
println("  - Noise scale: $(NOISE_SCALE)")
println()

# -----------------------------------------------------------------------------
# Step 2: Reference model in 100D
# -----------------------------------------------------------------------------
println("Building reference geodesic model in 100D...")
model_hi = build_model(data_hi)
graph_hi = model_hi.weighted_graph.graph
geoms_hi = model_hi.weighted_graph.geometries

pairs = sample_pairs(rng, N_POINTS; count=18)
errors_hi = geodesic_errors(model_hi, data_hi, params, pairs)
ambient_hi = ambient_errors(data_hi, params, pairs)

println("Reference (100D) quality vs exact Swiss roll geodesics:")
print_stats("Geodesic rel. error", summarize(errors_hi); suffix="")
print_stats("Ambient rel. error", summarize(ambient_hi); suffix="")
println()

# -----------------------------------------------------------------------------
# Step 3: Random projections and comparisons
# -----------------------------------------------------------------------------
for dim in PROJECTION_DIMS
    println("-" ^ 70)
    println("Projecting to $dim dimensions (Gaussian JL matrix)...")
    proj = randn(rng, dim, HI_DIM) ./ sqrt(dim)
    data_rp = proj * data_hi

    model_rp = build_model(data_rp)
    graph_rp = model_rp.weighted_graph.graph
    geoms_rp = model_rp.weighted_graph.geometries

    overlaps = neighbor_overlap(graph_hi, graph_rp)
    angles = tangent_angles_proj(geoms_hi, geoms_rp, proj)
    errors_rp = geodesic_errors(model_rp, data_rp, params, pairs)
    ambient_rp = ambient_errors(data_rp, params, pairs)

    overlap_stats = summarize(overlaps)
    angle_stats = summarize(angles)
    error_stats = summarize(errors_rp)
    ambient_stats = summarize(ambient_rp)

    println("kNN agreement vs 100D (Jaccard overlap):")
    print_stats("Overlap fraction", overlap_stats; scale=100, suffix="%")

    println("Tangent plane rotation vs 100D (principal angles):")
    print_stats("Angle (degrees)", angle_stats; scale=180 / π, suffix="°")

    println("Distance errors vs exact Swiss roll:")
    print_stats("Geodesic rel. error", error_stats; suffix="")
    print_stats("Ambient rel. error", ambient_stats; suffix="")
    println()
end

println("-" ^ 70)
println("Lower overlap/angle = better agreement with the 100D reference.")
println("Compare geodesic vs ambient errors to see the benefit of geodesic modeling after projection.")
