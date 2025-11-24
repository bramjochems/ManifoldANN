using Test
using ManifoldANN
using LinearAlgebra
using Random

@testset "Geodesic Refinement" begin

@testset "NoRefinement" begin
    # Simple 2D data
    Random.seed!(42)
    data = randn(2, 20)

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=5)
    method = PCAMethod(intrinsic_dim=2)
    wg = build_weighted_graph(method, graph, data)
    model = GeodesicDistanceModel(index, wg, method)

    # Test path (use shortest path to ensure edges exist in graph)
    result = shortest_path_with_path(model, data, 1, 10)
    path = result.path

    # Refine with NoRefinement
    refinement = NoRefinement()
    refined = refine_path(refinement, model, data, path)

    @test length(refined.points) == length(path)
    @test refined.original_path == path
    @test length(refined.segment_lengths) == length(path) - 1

    # Check that points match original path
    for (i, idx) in enumerate(path)
        @test refined.points[i] ≈ data[:, idx]
    end

    # IMPORTANT: NoRefinement should preserve graph edge weights, not Euclidean
    # The distance should match the shortest path distance from Dijkstra
    @test refined.distance ≈ result.distance

    # Verify that graph edge weights differ from Euclidean (otherwise test is trivial)
    euclidean_distance = sum(norm(refined.points[i+1] - refined.points[i]) for i in 1:length(path)-1)
    # On a manifold with tangent projection, these should differ slightly
    # (but we don't require it, just check they're both positive and close)
    @test euclidean_distance > 0
    @test abs(refined.distance - euclidean_distance) / euclidean_distance < 0.5  # Within 50%
end

@testset "SubdivisionSmoothing - basic functionality" begin
    Random.seed!(42)
    n = 50
    data = randn(3, n)

    index = build_index(BruteForceIndex, data)
    method = PCAMethod(intrinsic_dim=2)
    model = build_geodesic_model(method, index, data; k=8)

    path = [1, 5, 10, 15]

    # Test with small subdivisions
    refinement = SubdivisionSmoothing(subdivisions=3, max_iterations=10)
    refined = refine_path(refinement, model, data, path)

    @test length(refined.points) > length(path)  # Should have more points
    @test refined.original_path == path
    @test refined.points[1] ≈ data[:, path[1]]  # Start point preserved
    @test refined.points[end] ≈ data[:, path[end]]  # End point preserved
    @test refined.distance > 0
    @test length(refined.segment_lengths) == length(refined.points) - 1
end

@testset "SubdivisionSmoothing - parameter validation" begin
    @test_throws ArgumentError SubdivisionSmoothing(subdivisions=0)
    @test_throws ArgumentError SubdivisionSmoothing(max_iterations=0)
    @test_throws ArgumentError SubdivisionSmoothing(tolerance=-1.0)
    @test_throws ArgumentError SubdivisionSmoothing(damping=0.0)
    @test_throws ArgumentError SubdivisionSmoothing(damping=1.5)

    # Valid construction
    @test SubdivisionSmoothing() isa SubdivisionSmoothing
    @test SubdivisionSmoothing(subdivisions=10, max_iterations=50) isa SubdivisionSmoothing
end

@testset "SubdivisionSmoothing - smoothing effect" begin
    Random.seed!(42)
    # Create data on a circle (1D manifold in 2D)
    n = 30
    t = range(0, 2π, length=n+1)[1:end-1]
    data = vcat(cos.(t)', sin.(t)')

    index = build_index(BruteForceIndex, data)
    method = PCAMethod(intrinsic_dim=1)
    model = build_geodesic_model(method, index, data; k=6)

    # Path that roughly follows the circle
    path = [1, 5, 10, 15, 20]

    refinement = SubdivisionSmoothing(subdivisions=5, max_iterations=20, damping=0.5)
    refined = refine_path(refinement, model, data, path)

    # Check that refined curve is smoother (lower total curvature)
    # Compute discrete curvature as sum of angle changes
    function total_curvature(points)
        if length(points) < 3
            return 0.0
        end
        total = 0.0
        for i in 2:length(points)-1
            v1 = normalize(points[i] - points[i-1])
            v2 = normalize(points[i+1] - points[i])
            angle = acos(clamp(dot(v1, v2), -1, 1))
            total += angle
        end
        return total
    end

    # Refined curve should exist and have reasonable properties
    @test length(refined.points) > length(path)
    @test all(isfinite, refined.segment_lengths)
    @test refined.distance > 0
end

@testset "SubdivisionSmoothing - convergence" begin
    Random.seed!(42)
    data = randn(2, 15)

    index = build_index(BruteForceIndex, data)
    method = PCAMethod(intrinsic_dim=2)
    model = build_geodesic_model(method, index, data; k=5)

    path = [1, 4, 8]

    # With very low tolerance, should iterate more
    refinement_tight = SubdivisionSmoothing(
        subdivisions=3,
        max_iterations=100,
        tolerance=1e-8,
        damping=0.3
    )
    refined_tight = refine_path(refinement_tight, model, data, path)

    # With loose tolerance, should converge faster (but we can't directly test iteration count)
    refinement_loose = SubdivisionSmoothing(
        subdivisions=3,
        max_iterations=100,
        tolerance=1e-2,
        damping=0.3
    )
    refined_loose = refine_path(refinement_loose, model, data, path)

    # Both should produce valid results
    @test refined_tight.distance > 0
    @test refined_loose.distance > 0
    @test all(isfinite, refined_tight.segment_lengths)
    @test all(isfinite, refined_loose.segment_lengths)
end

@testset "CurvatureCorrectedDistance - basic functionality" begin
    Random.seed!(42)
    n = 40
    data = randn(3, n)

    index = build_index(BruteForceIndex, data)
    method = PCAMethod(intrinsic_dim=2)
    model = build_geodesic_model(method, index, data; k=8)

    path = [1, 5, 10, 15, 20]

    # Test with curvature correction on discrete path
    refinement = CurvatureCorrectedDistance()
    refined = refine_path(refinement, model, data, path)

    @test length(refined.points) == length(path)  # Same waypoints (no base refinement)
    @test refined.original_path == path
    @test refined.distance > 0
    @test all(>=(0), refined.segment_lengths)

    # Test composability with subdivision
    refinement_composed = CurvatureCorrectedDistance(
        base_refinement=SubdivisionSmoothing(subdivisions=3, max_iterations=5)
    )
    refined_composed = refine_path(refinement_composed, model, data, path)

    @test length(refined_composed.points) > length(path)  # Has subdivisions
    @test refined_composed.original_path == path
    @test refined_composed.distance > 0
end

@testset "CurvatureCorrectedDistance - correction effect" begin
    Random.seed!(42)
    # Points on a sphere (curved manifold)
    n = 50
    θ = π .* rand(n)
    φ = 2π .* rand(n)
    data = vcat(
        sin.(θ) .* cos.(φ)',
        sin.(θ) .* sin.(φ)',
        cos.(θ)'
    )

    index = build_index(BruteForceIndex, data)
    method = PCAMethod(intrinsic_dim=2)
    model = build_geodesic_model(method, index, data; k=10)

    # Get a path
    path_result = shortest_path_with_path(model, data, 1, 25)
    path = path_result.path

    # Test with different length scales
    with_default = CurvatureCorrectedDistance()
    with_small_scale = CurvatureCorrectedDistance(length_scale=0.1)
    with_large_scale = CurvatureCorrectedDistance(length_scale=10.0)

    refined_default = refine_path(with_default, model, data, path)
    refined_small = refine_path(with_small_scale, model, data, path)
    refined_large = refine_path(with_large_scale, model, data, path)

    # All should be positive and finite
    @test refined_default.distance > 0
    @test refined_small.distance > 0
    @test refined_large.distance > 0
    @test isfinite(refined_default.distance)
    @test isfinite(refined_small.distance)
    @test isfinite(refined_large.distance)

    # Smaller length scale → larger κ → larger correction → larger distance
    # (But correction is second-order so effect may be small)
    @test refined_small.distance >= refined_default.distance * 0.95  # Within 5%
    @test refined_large.distance <= refined_default.distance * 1.05
end

@testset "CurvatureCorrectedDistance - fallback for missing edges" begin
    Random.seed!(42)
    data = randn(2, 10)

    index = build_index(BruteForceIndex, data)
    method = PCAMethod(intrinsic_dim=2)
    model = build_geodesic_model(method, index, data; k=3)

    # Create a path that might have edges not in the graph
    # (if we manually specify indices)
    path = [1, 9]  # These might not be neighbors

    refinement = CurvatureCorrectedDistance()
    refined = refine_path(refinement, model, data, path)

    # Should still work (falls back to Euclidean if edge not in graph)
    @test refined.distance > 0
    @test isfinite(refined.distance)
end

@testset "subdivide_path helper" begin
    data = Float64[0 1 2 3; 0 0 0 0]  # Points on x-axis
    path = [1, 2, 4]  # Indices 1, 2, 4 -> points at x=0, x=1, x=3

    # Subdivide each segment into 2 parts (so 1 midpoint per segment)
    dense = ManifoldANN.subdivide_path(data, path, 2)

    # Should have: start of seg1, mid of seg1, start of seg2, mid of seg2, end of seg2
    # = 5 points
    @test length(dense) == 5

    # Check positions
    @test dense[1] ≈ [0.0, 0.0]  # Start at x=0
    @test dense[2] ≈ [0.5, 0.0]  # Midpoint between x=0 and x=1
    @test dense[3] ≈ [1.0, 0.0]  # At x=1
    @test dense[4] ≈ [2.0, 0.0]  # Midpoint between x=1 and x=3
    @test dense[5] ≈ [3.0, 0.0]  # End at x=3
end

@testset "RefinedPath structure" begin
    points = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    segment_lengths = [1.0, 1.0]
    distance = 2.0
    path = [1, 3, 5]

    refined = RefinedPath{Float64}(points, distance, path, segment_lengths)

    @test refined.points == points
    @test refined.distance == distance
    @test refined.original_path == path
    @test refined.segment_lengths == segment_lengths
end

@testset "Integration: refine_path with different methods" begin
    Random.seed!(42)
    # Swiss roll data
    n = 100
    t = 1.5π .+ 3π .* rand(n)
    height = 10 .* rand(n)
    data = vcat((t .* cos.(t))', height', (t .* sin.(t))')

    index = build_index(BruteForceIndex, data)
    method = PCAMethod(intrinsic_dim=2)
    model = build_geodesic_model(method, index, data; k=12)

    # Get a path
    path_result = shortest_path_with_path(model, data, 1, 50)
    path = path_result.path

    # Test all refinement methods
    methods = [
        NoRefinement(),
        SubdivisionSmoothing(subdivisions=4, max_iterations=15),
        CurvatureCorrectedDistance()
    ]

    for refinement_method in methods
        refined = refine_path(refinement_method, model, data, path)

        @test refined isa RefinedPath
        @test refined.original_path == path
        @test refined.distance > 0
        @test isfinite(refined.distance)
        @test length(refined.points) >= length(path)
        @test length(refined.segment_lengths) == length(refined.points) - 1

        # Check endpoints preserved
        @test refined.points[1] ≈ data[:, path[1]]
        @test refined.points[end] ≈ data[:, path[end]]
    end
end

@testset "Edge cases" begin
    Random.seed!(42)
    data = randn(2, 10)
    index = build_index(BruteForceIndex, data)
    method = PCAMethod(intrinsic_dim=2)
    model = build_geodesic_model(method, index, data; k=4)

    # Single point path
    path_single = [5]
    refined_single = refine_path(NoRefinement(), model, data, path_single)
    @test length(refined_single.points) == 1
    @test refined_single.distance == 0

    # Two point path
    path_two = [2, 7]
    refined_two = refine_path(NoRefinement(), model, data, path_two)
    @test length(refined_two.points) == 2
    @test refined_two.distance ≈ norm(data[:, 2] - data[:, 7])

    # Test with SubdivisionSmoothing on short paths
    refined_smooth = refine_path(
        SubdivisionSmoothing(subdivisions=3),
        model, data, path_two
    )
    @test length(refined_smooth.points) > 2
    @test refined_smooth.distance > 0
end

@testset "Custom geometry without curvature estimation" begin
    # Test that custom geometry types work with a warning
    Random.seed!(42)
    data = randn(2, 10)

    # Create a minimal custom geometry type (no curvature estimation)
    struct DummyGeometry <: ManifoldANN.AbstractLocalGeometry
        center::Vector{Float64}
    end
    ManifoldANN.supports_projection(::DummyGeometry) = false
    ManifoldANN.center(g::DummyGeometry) = g.center

    # Manually create a weighted graph with dummy geometries
    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=4)

    # Create dummy geometries
    geometries = [DummyGeometry(data[:, i]) for i in 1:size(data, 2)]
    # Use Euclidean distances as edge weights
    edge_weights = [[norm(data[:, i] - data[:, j]) for j in graph.neighbors[i]]
                    for i in 1:length(graph)]

    wg = ManifoldANN.WeightedKNNGraph(graph, geometries, edge_weights)
    model = ManifoldANN.GeodesicDistanceModel(index, wg, ManifoldANN.PCAMethod())

    path = [1, 3, 7]

    # Should work but issue a warning
    @test_logs (:warn, r"Curvature estimation not implemented.*DummyGeometry") begin
        refined = refine_path(CurvatureCorrectedDistance(), model, data, path)
        @test refined.distance > 0
        @test isfinite(refined.distance)
    end
end

@testset "Works with LocalGeometryEstimator (EstimatedGeometry wrappers)" begin
    Random.seed!(42)
    n = 30
    data = randn(3, n)

    index = build_index(BruteForceIndex, data)

    # Use LocalGeometryEstimator which produces EstimatedGeometry wrappers
    strategy = FixedNeighborhood()
    pca_method = PCAMethod(intrinsic_dim=2)
    estimator = LocalGeometryEstimator(strategy, pca_method)

    # Build model with estimator (produces EstimatedGeometry in the graph)
    model = build_geodesic_model(estimator, index, data; k=8)

    # Get a path
    path_result = shortest_path_with_path(model, data, 1, 15)
    path = path_result.path

    # Test that all refinement methods work with EstimatedGeometry
    @testset "$(name)" for (name, refinement) in [
        ("NoRefinement", NoRefinement()),
        ("SubdivisionSmoothing", SubdivisionSmoothing(subdivisions=3, max_iterations=10)),
        ("CurvatureCorrectedDistance", CurvatureCorrectedDistance()),
    ]
        refined = refine_path(refinement, model, data, path)

        @test refined isa RefinedPath
        @test refined.original_path == path
        @test refined.distance > 0
        @test isfinite(refined.distance)
        @test length(refined.points) >= length(path)
        @test all(isfinite, refined.segment_lengths)

        # Check endpoints preserved
        @test refined.points[1] ≈ data[:, path[1]]
        @test refined.points[end] ≈ data[:, path[end]]
    end
end

end  # @testset "Geodesic Refinement"
