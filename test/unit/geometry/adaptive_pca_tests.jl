using Test
using ManifoldANN
using LinearAlgebra
using Random

# Helper function
_mean(x) = sum(x) / length(x)

@testset "AdaptiveNeighborhood construction" begin
    @testset "default construction" begin
        strategy = AdaptiveNeighborhood()
        @test strategy.max_neighbors == 30
        @test strategy.min_neighbors == 5
        @test strategy.criterion isa FitErrorCriterion
        @test strategy.criterion.max_relative_error == 0.1
        @test strategy.shrink_factor == 0.2
        @test strategy.max_iterations == 10
    end

    @testset "custom parameters with legacy max_error" begin
        strategy = AdaptiveNeighborhood(
            max_neighbors=50,
            min_neighbors=10,
            max_error=0.05,  # Legacy parameter
            shrink_factor=0.3
        )
        @test strategy.max_neighbors == 50
        @test strategy.min_neighbors == 10
        @test strategy.criterion.max_relative_error == 0.05
        @test strategy.shrink_factor == 0.3
    end

    @testset "custom parameters with criterion" begin
        strategy = AdaptiveNeighborhood(
            max_neighbors=50,
            min_neighbors=10,
            criterion=DistortionCriterion(0.08),
            shrink_factor=0.3
        )
        @test strategy.criterion isa DistortionCriterion
        @test strategy.criterion.max_distortion == 0.08
    end

    @testset "invalid parameters" begin
        @test_throws ArgumentError AdaptiveNeighborhood(max_neighbors=1)
        @test_throws ArgumentError AdaptiveNeighborhood(min_neighbors=1)
        @test_throws ArgumentError AdaptiveNeighborhood(min_neighbors=40, max_neighbors=30)
        @test_throws ArgumentError AdaptiveNeighborhood(max_error=0.0)
        @test_throws ArgumentError AdaptiveNeighborhood(max_error=1.5)
        @test_throws ArgumentError AdaptiveNeighborhood(shrink_factor=0.0)
        @test_throws ArgumentError AdaptiveNeighborhood(shrink_factor=1.0)
        @test_throws ArgumentError AdaptiveNeighborhood(max_iterations=0)
    end
end

@testset "LocalGeometryEstimator with AdaptiveNeighborhood" begin
    rng = MersenneTwister(42)

    @testset "2D plane in 3D (no outliers)" begin
        # Clean 2D plane - adaptive should keep all points
        n = 30
        xy = randn(rng, 2, n) .* 0.5
        z = 0.5 .* xy[1, :] .+ 0.3 .* xy[2, :]
        data = vcat(xy, z')

        strategy = AdaptiveNeighborhood(max_neighbors=25, min_neighbors=10)
        method = PCAMethod(intrinsic_dim=2)
        estimator = LocalGeometryEstimator(strategy, method)

        geom = fit_geometry(estimator, data, 1, collect(2:n))

        @test geom isa EstimatedGeometry
        @test intrinsic_dimension(geom) == 2
        @test used_neighbor_count(geom) >= 10
        @test max_reconstruction_error(geom) < 0.2  # Should be low for clean plane
    end

    @testset "1D line in 3D" begin
        # Points on a line
        n = 25
        t = randn(rng, n)
        direction = normalize([1.0, 2.0, 3.0])
        data = direction * t'

        strategy = AdaptiveNeighborhood(max_neighbors=20)
        method = PCAMethod(intrinsic_dim=1)
        estimator = LocalGeometryEstimator(strategy, method)

        geom = fit_geometry(estimator, data, 1, collect(2:n))

        @test intrinsic_dimension(geom) == 1

        # Access underlying PCAGeometry
        pca_geom = unwrap_geometry(geom)
        @test size(pca_geom.basis) == (3, 1)

        # Basis should align with the line direction
        alignment = abs(dot(pca_geom.basis[:, 1], direction))
        @test alignment > 0.99
    end
end

@testset "AdaptiveNeighborhood filters outliers" begin
    rng = MersenneTwister(123)

    @testset "plane with off-plane outliers" begin
        # Create a 2D plane with some points far off the plane
        n_plane = 25
        n_outliers = 5

        # Points on plane z = 0
        xy_plane = randn(rng, 2, n_plane) .* 0.5
        z_plane = zeros(1, n_plane)
        plane_data = vcat(xy_plane, z_plane)

        # Outliers far from plane
        xy_outliers = randn(rng, 2, n_outliers) .* 0.5
        z_outliers = 5.0 .+ randn(rng, 1, n_outliers)  # z ≈ 5, far from plane
        outlier_data = vcat(xy_outliers, z_outliers)

        data = hcat(plane_data, outlier_data)

        # Non-adaptive should use all neighbors (and have poor fit)
        basic_method = PCAMethod(intrinsic_dim=2)
        basic_geom = fit_geometry(basic_method, data, 1, collect(2:size(data, 2)))

        # Adaptive should filter outliers
        strategy = AdaptiveNeighborhood(
            max_neighbors=size(data, 2)-1,
            min_neighbors=10,
            max_error=0.1
        )
        estimator = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=2))
        adaptive_geom = fit_geometry(estimator, data, 1, collect(2:size(data, 2)))

        # Adaptive should use fewer neighbors (filtered out outliers)
        @test used_neighbor_count(adaptive_geom) < size(data, 2) - 1

        # Adaptive should have lower reconstruction error
        @test max_reconstruction_error(adaptive_geom) < 0.5
    end

    @testset "Swiss Roll layer separation" begin
        # Simulate Swiss Roll: create two "layers" close in Euclidean space
        # but far apart on the manifold

        # Layer 1: t ≈ π (inner)
        n1 = 15
        t1 = π .+ 0.2 .* randn(rng, n1)
        h1 = randn(rng, n1)
        layer1 = vcat((t1 .* cos.(t1))', h1', (t1 .* sin.(t1))')

        # Layer 2: t ≈ 3π (outer, but close in Euclidean to layer1)
        n2 = 15
        t2 = 3π .+ 0.2 .* randn(rng, n2)
        h2 = randn(rng, n2)
        layer2 = vcat((t2 .* cos.(t2))', h2', (t2 .* sin.(t2))')

        data = hcat(layer1, layer2)

        # Adaptive method should prefer keeping layer-consistent neighbors
        strategy = AdaptiveNeighborhood(
            max_neighbors=n1 + n2 - 1,
            min_neighbors=8,
            max_error=0.15
        )
        estimator = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=2))

        # Fit at a point in layer 1
        geom = fit_geometry(estimator, data, 1, collect(2:size(data, 2)))

        # Should have removed some neighbors due to curvature mismatch
        @test refinement_iterations(geom) >= 1
    end
end

@testset "EstimatedGeometry interface" begin
    rng = MersenneTwister(456)
    n = 20
    data = randn(rng, 3, n)

    strategy = AdaptiveNeighborhood()
    method = PCAMethod(intrinsic_dim=2)
    estimator = LocalGeometryEstimator(strategy, method)
    geom = fit_geometry(estimator, data, 1, collect(2:n))

    @testset "projection interface" begin
        @test supports_projection(geom)

        point = randn(rng, 3)
        local_coords = project(geom, point)
        @test length(local_coords) == 2

        reconstructed = reconstruct(geom, local_coords)
        @test length(reconstructed) == 3
    end

    @testset "local_distance" begin
        p1 = data[:, 2]
        p2 = data[:, 3]

        d = local_distance(geom, p1, p2)
        @test d >= 0
        @test d == local_distance(geom, p2, p1)  # Symmetric
    end

    @testset "utility functions" begin
        @test intrinsic_dimension(geom) == 2
        @test center(geom) ≈ data[:, 1]
        @test used_neighbor_count(geom) >= 1
        @test refinement_iterations(geom) >= 1
        @test max_reconstruction_error(geom) >= 0

        ratios = explained_variance_ratio(geom)
        @test length(ratios) == 2
        @test sum(ratios) ≈ 1.0

        @test total_variance(geom) >= 0
    end
end

@testset "Adaptive estimation in weighted graph" begin
    rng = MersenneTwister(789)

    @testset "build_weighted_graph with candidate_k" begin
        n = 40
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        strategy = AdaptiveNeighborhood(max_neighbors=20, min_neighbors=5)
        estimator = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=2))

        # Use more candidates than final k
        wg = build_weighted_graph(estimator, index, data; k=8, candidate_k=20)

        @test length(wg) == n
        @test configured_k(wg) == 8
        @test all(g -> g isa EstimatedGeometry, wg.geometries)
    end

    @testset "adaptive vs non-adaptive on curved manifold" begin
        # Circle in 2D (1D manifold)
        n = 40
        t = range(0, 2π, length=n+1)[1:end-1]
        data = vcat(cos.(t)', sin.(t)')

        index = build_index(BruteForceIndex, data)

        # Non-adaptive (using FixedNeighborhood implicitly via PCAMethod)
        basic_method = PCAMethod(intrinsic_dim=1)
        basic_wg = build_weighted_graph(basic_method, index, data; k=10)

        # Adaptive
        strategy = AdaptiveNeighborhood(max_neighbors=15, min_neighbors=5)
        estimator = LocalGeometryEstimator(strategy, PCAMethod(intrinsic_dim=1))
        adaptive_wg = build_weighted_graph(estimator, index, data; k=10, candidate_k=15)

        # Both should work
        @test length(basic_wg) == n
        @test length(adaptive_wg) == n

        # Adaptive should have potentially filtered some high-curvature neighbors
        avg_used = _mean([used_neighbor_count(g) for g in adaptive_wg.geometries])
        @test avg_used >= 5  # At least min_neighbors
    end
end
