using Test
using ManifoldANN
using LinearAlgebra
using Random

@testset "PCAMethod construction" begin
    @testset "default construction" begin
        method = PCAMethod()
        @test isnothing(method.intrinsic_dim)
        @test method.min_variance_ratio == 0.95
    end

    @testset "fixed dimension" begin
        method = PCAMethod(intrinsic_dim=2)
        @test method.intrinsic_dim == 2
        @test method.min_variance_ratio == 0.95
    end

    @testset "custom variance ratio" begin
        method = PCAMethod(min_variance_ratio=0.99)
        @test isnothing(method.intrinsic_dim)
        @test method.min_variance_ratio == 0.99
    end

    @testset "combined parameters" begin
        method = PCAMethod(intrinsic_dim=3, min_variance_ratio=0.8)
        @test method.intrinsic_dim == 3
        @test method.min_variance_ratio == 0.8
    end

    @testset "invalid parameters" begin
        @test_throws ArgumentError PCAMethod(intrinsic_dim=0)
        @test_throws ArgumentError PCAMethod(intrinsic_dim=-1)
        @test_throws ArgumentError PCAMethod(min_variance_ratio=0.0)
        @test_throws ArgumentError PCAMethod(min_variance_ratio=1.5)
        @test_throws ArgumentError PCAMethod(min_variance_ratio=-0.1)
    end
end

@testset "fit_geometry on simple data" begin
    rng = MersenneTwister(42)

    @testset "2D data in 2D space" begin
        # Simple 2D points
        n = 20
        data = randn(rng, 2, n)

        method = PCAMethod(intrinsic_dim=2)
        geom = fit_geometry(method, data, 1, collect(2:n))

        @test geom isa PCAGeometry{Float64}
        @test length(geom.center) == 2
        @test size(geom.basis) == (2, 2)
        @test length(geom.eigenvalues) == 2
        @test intrinsic_dimension(geom) == 2
    end

    @testset "2D plane in 3D space" begin
        # Create points on a 2D plane embedded in 3D: z = 0.5x + 0.3y
        n = 30
        xy = randn(rng, 2, n)
        z = 0.5 .* xy[1, :] .+ 0.3 .* xy[2, :]
        data = vcat(xy, z')

        method = PCAMethod(intrinsic_dim=2)
        geom = fit_geometry(method, data, 1, collect(2:n))

        @test length(geom.center) == 3
        @test size(geom.basis) == (3, 2)
        @test length(geom.eigenvalues) == 2
        @test intrinsic_dimension(geom) == 2

        # The third eigenvalue should be very small (near-zero for a perfect plane)
        # but we only kept 2 dimensions
    end

    @testset "1D line in 3D space" begin
        # Points along a line in 3D
        n = 20
        t = randn(rng, n)
        direction = [1.0, 2.0, 3.0] ./ norm([1.0, 2.0, 3.0])
        data = direction * t'

        method = PCAMethod(intrinsic_dim=1)
        geom = fit_geometry(method, data, 1, collect(2:n))

        @test length(geom.center) == 3
        @test size(geom.basis) == (3, 1)
        @test intrinsic_dimension(geom) == 1

        # The basis should be aligned with the direction
        alignment = abs(dot(geom.basis[:, 1], direction))
        @test alignment > 0.99
    end

    @testset "auto-dimension detection" begin
        # Create data with clear 2D structure plus noise
        n = 50
        # Main variation in first 2 dimensions
        main_data = randn(rng, 2, n) .* 10
        # Small noise in third dimension
        noise = randn(rng, 1, n) .* 0.01
        data = vcat(main_data, noise)

        method = PCAMethod(min_variance_ratio=0.99)
        geom = fit_geometry(method, data, 1, collect(2:n))

        # Should detect 2 dimensions (the noise dimension has tiny variance)
        @test intrinsic_dimension(geom) == 2
    end
end

@testset "fit_geometry for query points" begin
    rng = MersenneTwister(123)

    @testset "query point not in data" begin
        n = 20
        data = randn(rng, 3, n)
        query_point = randn(rng, 3)
        neighbor_indices = collect(1:10)

        method = PCAMethod(intrinsic_dim=2)
        geom = fit_geometry(method, data, query_point, neighbor_indices)

        @test geom isa PCAGeometry{Float64}
        @test geom.center ≈ query_point
        @test size(geom.basis) == (3, 2)
    end
end

@testset "project and reconstruct" begin
    rng = MersenneTwister(456)

    @testset "round-trip on tangent space" begin
        # 2D plane in 3D
        n = 30
        xy = randn(rng, 2, n)
        z = 0.5 .* xy[1, :] .+ 0.3 .* xy[2, :]
        data = vcat(xy, z')

        method = PCAMethod(intrinsic_dim=2)
        geom = fit_geometry(method, data, 1, collect(2:n))

        @test supports_projection(geom)

        # Project and reconstruct a point that's on the plane
        point = data[:, 5]
        local_coords = project(geom, point)
        reconstructed = reconstruct(geom, local_coords)

        @test length(local_coords) == 2
        @test length(reconstructed) == 3

        # For a point on the tangent plane, reconstruction should be close
        # (not exact because center may differ slightly)
        @test norm(reconstructed - point) < 1.0
    end

    @testset "projection dimensions" begin
        data = randn(rng, 5, 20)  # 5D ambient space
        method = PCAMethod(intrinsic_dim=3)  # 3D tangent space
        geom = fit_geometry(method, data, 1, collect(2:20))

        point = randn(rng, 5)
        local_coords = project(geom, point)

        @test length(local_coords) == 3
    end
end

@testset "local_distance" begin
    rng = MersenneTwister(789)

    @testset "basic properties" begin
        n = 20
        data = randn(rng, 3, n)

        method = PCAMethod(intrinsic_dim=2)
        geom = fit_geometry(method, data, 1, collect(2:n))

        p1 = data[:, 2]
        p2 = data[:, 3]

        d = local_distance(geom, p1, p2)
        @test d >= 0
        @test d isa Real

        # Distance to self should be zero
        d_self = local_distance(geom, p1, p1)
        @test d_self ≈ 0 atol=1e-10
    end

    @testset "local distance vs euclidean" begin
        # On a plane, local distance should be similar to Euclidean
        # for points near the center
        n = 30
        xy = randn(rng, 2, n) .* 0.1  # Small neighborhood
        z = 0.5 .* xy[1, :] .+ 0.3 .* xy[2, :]
        data = vcat(xy, z')

        method = PCAMethod(intrinsic_dim=2)
        geom = fit_geometry(method, data, 1, collect(2:n))

        p1 = data[:, 2]
        p2 = data[:, 3]

        local_d = local_distance(geom, p1, p2)
        euclidean_d = norm(p2 - p1)

        # For points on the manifold, local distance should be close to
        # the Euclidean distance in the tangent plane (≤ ambient Euclidean)
        @test local_d <= euclidean_d * 1.1  # Allow some tolerance
    end

    @testset "symmetry" begin
        n = 20
        data = randn(rng, 3, n)

        method = PCAMethod(intrinsic_dim=2)
        geom = fit_geometry(method, data, 1, collect(2:n))

        p1 = data[:, 2]
        p2 = data[:, 3]

        d_forward = local_distance(geom, p1, p2)
        d_backward = local_distance(geom, p2, p1)

        @test d_forward ≈ d_backward
    end
end

@testset "utility functions" begin
    rng = MersenneTwister(111)
    data = randn(rng, 4, 30)

    method = PCAMethod(intrinsic_dim=3)
    geom = fit_geometry(method, data, 1, collect(2:30))

    @testset "intrinsic_dimension" begin
        @test intrinsic_dimension(geom) == 3
    end

    @testset "center" begin
        c = center(geom)
        @test c ≈ data[:, 1]
    end

    @testset "explained_variance_ratio" begin
        ratios = explained_variance_ratio(geom)
        @test length(ratios) == 3
        @test all(r -> r >= 0, ratios)
        @test sum(ratios) ≈ 1.0
    end

    @testset "total_variance" begin
        tv = total_variance(geom)
        @test tv >= 0
        @test tv == sum(geom.eigenvalues)
    end
end

@testset "edge cases" begin
    rng = MersenneTwister(222)

    @testset "minimal neighbors" begin
        # Only 2 neighbors (minimum for non-trivial PCA)
        data = randn(rng, 3, 5)
        method = PCAMethod(intrinsic_dim=2)
        geom = fit_geometry(method, data, 1, [2, 3])

        @test geom isa PCAGeometry
        @test intrinsic_dimension(geom) <= 2
    end

    @testset "single neighbor fallback" begin
        # Only 1 neighbor - should use fallback
        data = randn(rng, 3, 3)
        method = PCAMethod(intrinsic_dim=2)
        geom = fit_geometry(method, data, 1, [2])

        @test geom isa PCAGeometry
        # Should still produce valid geometry
        @test size(geom.basis, 1) == 3  # Ambient dimension
    end

    @testset "intrinsic_dim > ambient_dim" begin
        # Request more dimensions than ambient space
        data = randn(rng, 2, 20)  # 2D ambient
        method = PCAMethod(intrinsic_dim=5)  # Request 5D
        geom = fit_geometry(method, data, 1, collect(2:20))

        # Should cap at ambient dimension
        @test intrinsic_dimension(geom) <= 2
    end

    @testset "high dimensional data" begin
        # Higher dimensional case
        data = randn(rng, 10, 50)
        method = PCAMethod(intrinsic_dim=5)
        geom = fit_geometry(method, data, 1, collect(2:50))

        @test length(geom.center) == 10
        @test size(geom.basis) == (10, 5)
        @test intrinsic_dimension(geom) == 5
    end
end
