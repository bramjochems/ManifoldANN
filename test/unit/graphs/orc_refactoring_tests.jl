using Test
using ManifoldANN
using LinearAlgebra
using Random

@testset "ORC Refactoring Tests" begin
    Random.seed!(42)

    # Create simple test data
    n_points = 50
    dim = 10
    data = randn(dim, n_points)

    # Build k-NN graphs
    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=10)  # Directed graph for backward compatibility tests
    graph_undirected = build_knn_graph(index, data; k=10, directed=false)  # Undirected for orcml/geodesic tests

    @testset "Backward Compatibility" begin
        # Default settings should match previous behavior
        # (This is already tested by the full test suite, but we verify explicitly)
        results_default = compute_all_curvatures(graph, data)
        @test !isempty(results_default)
        @test all(haskey(results_default, (i, j)) for i in 1:n_points for j in graph[i])
    end

    # Distance metric functions are tested indirectly through different metric configurations

    @testset "Endpoint Exclusion" begin
        # Test with endpoint exclusion
        results_excluded = compute_all_curvatures(
            graph, data;
            exclude_edge_endpoints=true
        )
        @test !isempty(results_excluded)

        # Results should be different from default (no exclusion)
        results_included = compute_all_curvatures(graph, data)

        # Some curvatures should differ
        n_different = 0
        for key in keys(results_excluded)
            if haskey(results_included, key)
                if abs(results_excluded[key].curvature - results_included[key].curvature) > 1e-10
                    n_different += 1
                end
            end
        end
        @test n_different > 0  # At least some curvatures should be different
    end

    @testset "orcml Configuration" begin
        # Test orcml approach: exclude endpoints + geodesic normalized
        # orcml requires undirected graphs for geodesic distances
        # Use NetworkSimplexSolver fallback for reliability (Sinkhorn can fail with geodesic distances)
        results_orcml = compute_all_curvatures(
            graph_undirected, data;
            exclude_edge_endpoints=true,
            cost_metric=:geodesic_normalized,
            denominator_metric=:normalized,
            solver=HungarianSolver(),
            fallback_solver=NetworkSimplexSolver()
        )
        @test !isempty(results_orcml)
        @test all(haskey(results_orcml, (i, j)) for i in 1:n_points for j in graph_undirected[i])

        # Curvatures should be finite
        @test all(isfinite(result.curvature) for result in values(results_orcml))
    end

    @testset "Geodesic Metrics" begin
        @testset "Geodesic Euclidean" begin
            # Geodesic metrics require undirected graphs
            results = compute_all_curvatures(
                graph_undirected, data;
                cost_metric=:geodesic_euclidean,
                denominator_metric=:euclidean
            )
            @test !isempty(results)
            @test all(isfinite(result.curvature) for result in values(results))
        end

        @testset "Geodesic Unit" begin
            # Geodesic metrics require undirected graphs
            results = compute_all_curvatures(
                graph_undirected, data;
                cost_metric=:geodesic_unit,
                denominator_metric=:euclidean
            )
            @test !isempty(results)
            @test all(isfinite(result.curvature) for result in values(results))
        end
    end

    @testset "Threading Control" begin
        # Test with threading disabled
        results_sequential = compute_all_curvatures(
            graph, data;
            use_threading=false
        )
        @test !isempty(results_sequential)

        # Test with threading enabled
        results_parallel = compute_all_curvatures(
            graph, data;
            use_threading=true
        )
        @test !isempty(results_parallel)

        # Results should be identical (within numerical precision)
        for key in keys(results_sequential)
            @test haskey(results_parallel, key)
            @test results_sequential[key].curvature ≈ results_parallel[key].curvature atol=1e-10
        end
    end

    # build_neighborhood and metric validation are tested indirectly through the public API

    @testset "Deprecated distance_fn Parameter" begin
        # The deprecated parameter should still work
        custom_dist = (i, j) -> 1.0  # Constant distance
        results = compute_all_curvatures(
            graph, data;
            distance_fn=custom_dist
        )
        @test !isempty(results)
        # All edge distances should be 1.0
        @test all(result.edge_distance ≈ 1.0 for result in values(results))
    end
end
