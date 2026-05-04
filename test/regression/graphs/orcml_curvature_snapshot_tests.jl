using Test
using ManifoldANN
using Random

# Regression snapshot for ORCManL curvatures on a fixed seeded graph.
#
# Purpose: lock in the current numerical output of the ORCManL pipeline so
# that planned internal swaps — most immediately Floyd-Warshall →
# per-source Dijkstra in `compute_shortest_paths` (see TODO_cleanup.md) —
# can be validated as numerically equivalent rather than just "passes the
# existing non-emptiness/finiteness checks".
#
# The snapshot is built from:
#   - data: 5 × 30, `randn` with `Random.seed!(20260504)`
#   - graph: undirected k=6 kNN via BruteForceIndex
#   - variant: ORCManL() (geodesic_normalized cost, normalized denominator)
#   - solver: ClpSolver() (deterministic; rules out Sinkhorn/Hungarian noise)
#
# Tolerances: weighted shortest-path FP summation order can differ between
# FW and Dijkstra, so we check digest sums to ~1e-10 relative and per-edge
# spot values to ~1e-10 absolute. `:unit` (BFS-equivalent) and edge counts
# must match exactly. Bumping a tolerance to make a future swap pass is the
# wrong call — investigate the divergence first.

@testset "ORCManL curvature snapshot (FW baseline)" begin
    Random.seed!(20260504)
    data = randn(5, 30)
    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k = 6, directed = false)

    results = compute_all_curvatures(
        graph, data;
        variant = ORCManL(),
        solver = ClpSolver(),
        fallback_solver = ClpSolver(),
    )

    keys_sorted = sort(collect(keys(results)))
    curvs = [results[k].curvature for k in keys_sorted]
    wass = [results[k].wasserstein_distance for k in keys_sorted]
    edge_d = [results[k].edge_distance for k in keys_sorted]

    # ---- structural invariants (must be exact) ----
    @test length(keys_sorted) == 252
    @test all(isfinite, curvs)
    @test all(isfinite, wass)
    @test all(>(0.0), edge_d)

    # ---- digest snapshot (rtol = 1e-10) ----
    rtol = 1e-10
    @test isapprox(sum(curvs),  21.738724816031489;  rtol = rtol)
    @test isapprox(sum(wass),  426.773617463751123;  rtol = rtol)
    @test isapprox(sum(edge_d), 489.894743225469654; rtol = rtol)
    @test isapprox(minimum(curvs), -0.913061552453234; atol = 1e-12)
    @test isapprox(maximum(curvs),  0.848113343889226; atol = 1e-12)
    @test isapprox(minimum(wass),   0.451507507811391; atol = 1e-12)
    @test isapprox(maximum(wass),   3.481057922978104; atol = 1e-12)

    # ---- per-edge spot checks (atol = 1e-10) ----
    # Five edges spread across the sorted key range. If only one drifts,
    # that's a localised bug; if all five drift uniformly, suspect a
    # global change (e.g. summation order in shortest paths).
    spots = Dict(
        (1,  2)  => (curv = 0.222919741190823,  wass = 1.571770650049714, edged = 2.022661922281163),
        (8,  26) => (curv = -0.431806813889676, wass = 1.753291305625380, edged = 1.224530634033199),
        (15, 20) => (curv = 0.026633092149901,  wass = 2.021145519719138, edged = 2.076447743824879),
        (23, 9)  => (curv = -0.132676772846864, wass = 2.104214156147327, edged = 1.857735769453987),
        (30, 25) => (curv = -0.077738804443731, wass = 1.838681501182136, edged = 1.706054837777843),
    )

    atol = 1e-10
    for (edge, expected) in spots
        @test haskey(results, edge)
        r = results[edge]
        @test isapprox(r.curvature,            expected.curv;  atol = atol)
        @test isapprox(r.wasserstein_distance, expected.wass;  atol = atol)
        @test isapprox(r.edge_distance,        expected.edged; atol = atol)
    end
end
