#=
Example: Per-edge geodesic distance estimators

Demonstrates three `AbstractEdgeWeight` implementations on a small
Swiss-roll patch. The same kNN graph is scored three ways:

  - `EuclideanChord`              -- the classical Isomap-style edge length
  - `TangentProjectedSymmetricMean` -- the symmetrised tangent-projected length
  - `CurvatureFreeSymmetric`      -- the curvature-free symmetric estimator
                                      from Chapter 6 of the thesis (eq:geod-sym)

Run with `julia --project=. docs/examples/geodesic/05-edge-geodesic-estimators.jl`.
=#

using ManifoldANN
using LinearAlgebra
using Random
using Printf

include("swiss_roll_utils.jl")

rng = MersenneTwister(2024)

# A small patch is enough to make the curvature differences visible.
n_points = 400
data, params = generate_swiss_roll(n_points; rng=rng,
                                    t_min=1.5π, t_range=2.5π, h_scale=15.0)

index  = build_index(BruteForceIndex, data)
method = PCAMethod(intrinsic_dim=2)

# Pick a source/target pair that lies on the same sheet of the roll so the
# graph distance is a meaningful approximation of the manifold geodesic.
source, target = 1, n_points

# ---------------------------------------------------------------------------
# Build one model per estimator (same graph, same geometries -- only the
# per-edge weighting differs).
# ---------------------------------------------------------------------------
models = (
    euclidean = build_geodesic_model(method, index, data; k=10,
                                      edge_weight=EuclideanChord()),
    tangent   = build_geodesic_model(method, index, data; k=10,
                                      edge_weight=TangentProjectedSymmetricMean()),
    curvfree  = build_geodesic_model(method, index, data; k=10,
                                      edge_weight=CurvatureFreeSymmetric()),
)

println("=" ^ 64)
println("Edge geodesic estimators on a Swiss-roll patch ($(n_points) points)")
println("=" ^ 64)

# Reference: exact analytic geodesic on the roll for comparison context.
exact = exact_swiss_roll_geodesic(params, source, target)
@printf("Exact (analytic)           : %.6f\n", exact)
println("-" ^ 64)

for (name, model) in pairs(models)
    d  = geodesic_distance(model, data, source, target)
    dg = diagnostics(model)
    label = String(name)
    @printf("%-26s : %.6f   (negative-fallbacks=%d / %d edges)\n",
            label, d, dg.n_negative_fallbacks, dg.n_edges)
end

println("-" ^ 64)
println("eq:geod-sym (Chapter 6, §6.2.3) is implemented by the curvfree row.")
