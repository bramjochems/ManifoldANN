using Test
using ManifoldANN
using LinearAlgebra
using Random

# Type-plumbing smoke tests for the ORC curvature pipeline.
#
# The pipeline used to hardcode `Float64` in `_process_edge!`,
# `_mirror_bidirectional_results!`, and the top-level `Dict{...,
# CurvatureResult{Float64}}` container, silently promoting Float32 input
# data to Float64 — a 2× memory cost on neighbourhood/distance containers
# and a type-stability surprise at the public API.
#
# These tests assert eltype-only: the user-facing `CurvatureResult` carries
# `eltype(data)` out of the pipeline. Numerical drift between Float32 and
# Float64 runs is expected (LP solvers internally promote to Float64 and
# cast back), so values are NOT pinned. The bug class to catch is "someone
# re-introduces a hardcoded `Float64` in the hot path".
#
# The Float64 numerical regression is covered by
# `orcml_curvature_snapshot_tests.jl` at 1e-10.

@testset "ORC eltype propagation" begin
    Random.seed!(0xC0FFEE)

    n_points = 25
    dim = 6
    k = 5

    # Same point cloud, two precisions.
    data_f64 = randn(dim, n_points)
    data_f32 = Float32.(data_f64)

    @testset "Float64 input → CurvatureResult{Float64}" begin
        index = build_index(BruteForceIndex, data_f64)
        graph = build_knn_graph(index, data_f64; k=k, directed=false)

        # StandardORC path
        results_std = compute_all_curvatures(
            graph, data_f64;
            variant=StandardORC(),
            solver=HungarianSolver(),
            fallback_solver=ClpSolver(),
            use_threading=false,
        )
        @test !isempty(results_std)
        @test eltype(values(results_std)) === CurvatureResult{Float64}
        @test all(isfinite(r.curvature) for r in values(results_std))
        @test all(r isa CurvatureResult{Float64} for r in values(results_std))

        # ORCManL path (exercises the geodesic/effective-epsilon distance
        # closures which produce Float64 internally — the cast back to T
        # at the edge_view boundary is what this test pins down.)
        results_manl = compute_all_curvatures(
            graph, data_f64;
            variant=ORCManL(),
            solver=HungarianSolver(),
            fallback_solver=ClpSolver(),
            use_threading=false,
        )
        @test !isempty(results_manl)
        @test eltype(values(results_manl)) === CurvatureResult{Float64}
        @test all(isfinite(r.curvature) for r in values(results_manl))
    end

    @testset "Float32 input → CurvatureResult{Float32}" begin
        index = build_index(BruteForceIndex, data_f32)
        graph = build_knn_graph(index, data_f32; k=k, directed=false)

        results_std = compute_all_curvatures(
            graph, data_f32;
            variant=StandardORC(),
            solver=HungarianSolver(),
            fallback_solver=ClpSolver(),
            use_threading=false,
        )
        @test !isempty(results_std)
        @test eltype(values(results_std)) === CurvatureResult{Float32}
        @test all(r isa CurvatureResult{Float32} for r in values(results_std))
        @test all(isfinite(r.curvature) for r in values(results_std))
        @test all(r.curvature isa Float32 for r in values(results_std))
        @test all(r.wasserstein_distance isa Float32 for r in values(results_std))
        @test all(r.edge_distance isa Float32 for r in values(results_std))

        results_manl = compute_all_curvatures(
            graph, data_f32;
            variant=ORCManL(),
            solver=HungarianSolver(),
            fallback_solver=ClpSolver(),
            use_threading=false,
        )
        @test !isempty(results_manl)
        @test eltype(values(results_manl)) === CurvatureResult{Float32}
        @test all(isfinite(r.curvature) for r in values(results_manl))
    end

    @testset "Float32 / Float64 produce the same edge set" begin
        # Same seed → same point cloud → same kNN structure at both
        # precisions for a small well-separated cloud, so the edge-key
        # set should match exactly. (Numerical curvature values may
        # drift between precisions; we don't pin those.)
        index_f64 = build_index(BruteForceIndex, data_f64)
        graph_f64 = build_knn_graph(index_f64, data_f64; k=k, directed=false)
        index_f32 = build_index(BruteForceIndex, data_f32)
        graph_f32 = build_knn_graph(index_f32, data_f32; k=k, directed=false)

        r64 = compute_all_curvatures(
            graph_f64, data_f64;
            variant=StandardORC(),
            solver=HungarianSolver(),
            fallback_solver=ClpSolver(),
            use_threading=false,
        )
        r32 = compute_all_curvatures(
            graph_f32, data_f32;
            variant=StandardORC(),
            solver=HungarianSolver(),
            fallback_solver=ClpSolver(),
            use_threading=false,
        )

        @test length(r64) == length(r32)
        @test keys(r64) == keys(r32)
    end

    @testset "filter_graph preserves eltype path" begin
        # filter_graph wraps compute_all_curvatures; verify it runs
        # without error on Float32 input (the returned KNNGraph itself
        # has no float eltype, but the internal curvature pipeline must
        # propagate Float32 cleanly).
        index = build_index(BruteForceIndex, data_f32)
        graph = build_knn_graph(index, data_f32; k=k, directed=false)
        filtered = filter_graph(
            graph, data_f32;
            curvature_threshold=-1.0,
            solver=HungarianSolver(),
            fallback_solver=ClpSolver(),
            use_threading=false,
        )
        @test filtered isa KNNGraph
        @test length(filtered) == n_points
    end
end
