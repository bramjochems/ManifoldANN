using Test
using ManifoldANN
using LinearAlgebra
using Random

@testset "GeodesicDistanceModel construction" begin
    rng = MersenneTwister(42)

    @testset "basic construction" begin
        n = 30
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=5)

        @test model isa GeodesicDistanceModel
        @test length(model) == n
        @test configured_k(model) == 5
    end

    @testset "different k values" begin
        n = 30
        data = randn(rng, 3, n)
        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)

        for k in [3, 5, 10]
            model = build_geodesic_model(method, index, data; k=k)
            @test configured_k(model) == k
        end
    end
end

@testset "geodesic_distance between graph nodes" begin
    rng = MersenneTwister(123)

    @testset "line of points (known geometry)" begin
        # Points on a 1D line: geodesic = Euclidean
        n = 20
        data = reshape(collect(1.0:Float64(n)), 1, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=1)
        model = build_geodesic_model(method, index, data; k=2)

        # Distance from point 1 to point n should be approximately n-1
        d = geodesic_distance(model, data, 1, n)
        @test d ≈ Float64(n - 1) atol=1.0

        # Adjacent points should have distance ~1
        d_adjacent = geodesic_distance(model, data, 1, 2)
        @test d_adjacent ≈ 1.0 atol=0.5
    end

    @testset "self distance is zero" begin
        n = 20
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=5)

        for i in 1:5
            d = geodesic_distance(model, data, i, i)
            @test d ≈ 0.0 atol=1e-10
        end
    end

    @testset "distance is non-negative" begin
        n = 30
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=8)

        for _ in 1:20
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)
            d = geodesic_distance(model, data, i, j)
            @test d >= 0
        end
    end

    @testset "connected graph has finite distances" begin
        # Dense enough graph should be connected
        n = 30
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=10)

        # Sample some pairs
        for _ in 1:10
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)
            d = geodesic_distance(model, data, i, j)
            @test isfinite(d)
        end
    end
end

@testset "shortest_path_with_path" begin
    rng = MersenneTwister(456)

    @testset "path includes endpoints" begin
        n = 20
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=8)

        result = shortest_path_with_path(model, data, 1, n)

        @test result.path[1] == 1
        @test result.path[end] == n
        @test result.distance >= 0
    end

    @testset "path distance matches geodesic_distance" begin
        n = 30
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=10)

        for _ in 1:5
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)

            d1 = geodesic_distance(model, data, i, j)
            result = shortest_path_with_path(model, data, i, j)

            @test d1 ≈ result.distance atol=1e-10
        end
    end

    @testset "self path is single node" begin
        n = 20
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=5)

        result = shortest_path_with_path(model, data, 5, 5)

        @test result.path == [5]
        @test result.distance ≈ 0.0 atol=1e-10
    end
end

@testset "geodesic_distance from new point to graph node" begin
    rng = MersenneTwister(789)

    @testset "basic new point query" begin
        n = 30
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=8)

        query_point = randn(rng, 3)
        d = geodesic_distance(model, data, query_point, 1)

        @test d >= 0
        @test isfinite(d)
    end

    @testset "query point at graph node location" begin
        # Query point exactly at a graph node should have small distance
        n = 30
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=8)

        # Query exactly at node 5
        query_point = data[:, 5]
        d = geodesic_distance(model, data, query_point, 5)

        # Should be very small (not exactly 0 due to path through entry nodes)
        @test d < 1.0
    end

    @testset "distance depends on entry_k" begin
        n = 30
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=10)

        query_point = randn(rng, 3)

        d1 = geodesic_distance(model, data, query_point, 1; entry_k=1)
        d5 = geodesic_distance(model, data, query_point, 1; entry_k=5)

        # More entry points should find equal or better path
        @test d5 <= d1 + 1e-10
    end
end

@testset "geodesic_distance between two new points" begin
    rng = MersenneTwister(111)

    @testset "basic two new points query" begin
        n = 30
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=8)

        point_a = randn(rng, 3)
        point_b = randn(rng, 3)

        d = geodesic_distance(model, data, point_a, point_b)

        @test d >= 0
        @test isfinite(d)
    end

    @testset "same point has small distance" begin
        n = 30
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=8)

        point = randn(rng, 3)
        d = geodesic_distance(model, data, point, point)

        # Distance to self should be small (not exactly 0 due to graph routing)
        @test d < 1.0
    end

    @testset "closer points have smaller distance" begin
        n = 50
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=10)

        # Points close to each other
        center = randn(rng, 3)
        close_point = center .+ 0.1 .* randn(rng, 3)
        far_point = center .+ 2.0 .* randn(rng, 3)

        d_close = geodesic_distance(model, data, center, close_point)
        d_far = geodesic_distance(model, data, center, far_point)

        # Generally, closer Euclidean distance means closer geodesic distance
        # (though not always for curved manifolds)
        @test d_close < d_far * 2  # Allow some tolerance
    end
end

@testset "geodesic distance on manifolds" begin
    rng = MersenneTwister(222)

    @testset "circle manifold" begin
        # Points on a unit circle
        n = 50
        t = range(0, 2π, length=n+1)[1:end-1]
        data = vcat(cos.(t)', sin.(t)')

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=1)
        model = build_geodesic_model(method, index, data; k=6)

        # Opposite points on circle: geodesic should be ~π (half circumference)
        # Point 1 is at angle 0, point n÷2+1 is at angle π
        opposite_idx = n ÷ 2 + 1
        d = geodesic_distance(model, data, 1, opposite_idx)

        # Should be close to π (arc length)
        @test d > 2.0   # At least greater than Euclidean (which is 2.0)
        @test d < 4.0   # But not too large
    end

    @testset "2D plane in 3D (geodesic ≈ Euclidean)" begin
        # Points on a flat 2D plane: geodesic should equal Euclidean
        n = 40
        xy = randn(rng, 2, n)
        z = zeros(1, n)
        data = vcat(xy, z)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=10)

        for _ in 1:5
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)

            d_geodesic = geodesic_distance(model, data, i, j)
            d_euclidean = norm(data[:, i] - data[:, j])

            # On a flat plane, geodesic should be close to Euclidean
            # (not exact due to graph discretization)
            @test d_geodesic ≈ d_euclidean rtol=0.5
        end
    end
end

@testset "all_pairs_geodesic_distances" begin
    rng = MersenneTwister(333)

    @testset "basic computation" begin
        n = 15  # Small for speed
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=5)

        D = all_pairs_geodesic_distances(model, data)

        @test size(D) == (n, n)
        @test all(D .>= 0)

        # Diagonal should be zero
        for i in 1:n
            @test D[i, i] ≈ 0.0 atol=1e-10
        end
    end

    @testset "matches individual queries" begin
        n = 10
        data = randn(rng, 3, n)

        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)
        model = build_geodesic_model(method, index, data; k=5)

        D = all_pairs_geodesic_distances(model, data)

        # Check a few pairs
        for _ in 1:5
            i = rand(rng, 1:n)
            j = rand(rng, 1:n)
            d = geodesic_distance(model, data, i, j)
            @test D[i, j] ≈ d atol=1e-10
        end
    end
end

@testset "GeodesicDistanceModel with different index types" begin
    rng = MersenneTwister(444)
    n = 30
    data = randn(rng, 3, n)
    method = PCAMethod(intrinsic_dim=2)

    @testset "BruteForceIndex" begin
        index = build_index(BruteForceIndex, data)
        model = build_geodesic_model(method, index, data; k=5)

        d = geodesic_distance(model, data, 1, n)
        @test isfinite(d)
    end

    @testset "HNSWIndex" begin
        index = build_index(HNSWIndex, data; M=8, ef_construction=50)
        model = build_geodesic_model(method, index, data; k=5)

        d = geodesic_distance(model, data, 1, n)
        @test isfinite(d)
    end

    @testset "KDTreeIndex" begin
        index = build_index(KDTreeIndex, data)
        model = build_geodesic_model(method, index, data; k=5)

        d = geodesic_distance(model, data, 1, n)
        @test isfinite(d)
    end
end
