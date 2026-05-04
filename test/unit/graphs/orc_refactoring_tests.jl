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

    @testset "Variant: ORCManL differs from StandardORC" begin
        # Switching to the ORC-ManL variant changes endpoint handling and
        # the cost / denominator metrics; results should generally differ
        # from StandardORC (the default).
        # ORC-ManL requires undirected graphs for the geodesic metric.
        results_orcml = compute_all_curvatures(
            graph_undirected, data;
            variant=ORCManL(),
            solver=HungarianSolver(),
            fallback_solver=ClpSolver()
        )
        results_std = compute_all_curvatures(graph_undirected, data;
            variant=StandardORC(),
            solver=HungarianSolver(),
            fallback_solver=ClpSolver()
        )

        @test !isempty(results_orcml)
        @test all(haskey(results_orcml, (i, j))
                  for i in 1:n_points for j in graph_undirected[i])
        @test all(isfinite(result.curvature) for result in values(results_orcml))

        # At least some curvatures should differ between the two variants.
        n_different = 0
        for key in keys(results_orcml)
            if haskey(results_std, key) &&
               abs(results_orcml[key].curvature - results_std[key].curvature) > 1e-10
                n_different += 1
            end
        end
        @test n_different > 0
    end

    @testset "Variant: ORCManL with OrcmlExact profile" begin
        # The orcml-compatibility profile composes cleanly with the
        # ORC-ManL variant via ORCManL(profile=...).
        results = compute_all_curvatures(
            graph_undirected, data;
            variant=ORCManL(profile=OrcmlExact()),
            solver=HungarianSolver(),
            fallback_solver=ClpSolver()
        )
        @test !isempty(results)
        @test all(isfinite(result.curvature) for result in values(results))
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

    @testset "Variant unpacking pins old-kwarg semantics" begin
        # StandardORC must unpack to (false, :euclidean, :euclidean, *).
        # ORCManL (default profile) must unpack to
        # (true, :geodesic_normalized, :normalized, ManifoldANNDefault()).
        # ORCManL(profile=OrcmlExact()) must unpack to
        # (true, :geodesic_normalized, :normalized, OrcmlExact()).
        # This locks in the precise mapping from the trait to the
        # internal flags so the refactor cannot silently change the
        # algorithmic meaning of either preset.
        let (excl, cm, dm, prof) =
                ManifoldANN._unpack_variant(StandardORC())
            @test excl  == false
            @test cm    == :euclidean
            @test dm    == :euclidean
            @test prof isa AbstractOrcMLCompatibilityProfile
        end

        let (excl, cm, dm, prof) =
                ManifoldANN._unpack_variant(ORCManL())
            @test excl == true
            @test cm   == :geodesic_normalized
            @test dm   == :normalized
            @test prof === ManifoldANNDefault()
        end

        let (excl, cm, dm, prof) =
                ManifoldANN._unpack_variant(ORCManL(profile=OrcmlExact()))
            @test excl == true
            @test cm   == :geodesic_normalized
            @test dm   == :normalized
            @test prof === OrcmlExact()
        end

        # End-to-end: the default `compute_all_curvatures(graph, data)`
        # call must equal `variant=StandardORC()` exactly (same flags).
        c_default = compute_all_curvatures(graph, data)
        c_std = compute_all_curvatures(graph, data; variant=StandardORC())
        @test length(c_default) == length(c_std)
        for k in keys(c_default)
            @test haskey(c_std, k)
            @test c_default[k].curvature ≈ c_std[k].curvature atol=1e-12
        end
    end

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
