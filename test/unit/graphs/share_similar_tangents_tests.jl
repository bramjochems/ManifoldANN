using Test
using ManifoldANN
using LinearAlgebra
using Random

# Smoke-test coverage for ShareSimilarTangents and its helpers. The bar here is
# "function executes without error on a small graph and returns the expected
# type / shape" — no pinned numerical comparisons. Closes the
# "ORC / refinement / tangent-sharing test coverage" smoke-test ask in
# TODO_cleanup.md (a broken constructor previously slipped through Pkg.test()
# because none of these paths were exercised).

const MA = ManifoldANN

@testset "ShareSimilarTangents constructor" begin
    crit = SubspaceAngleCriterion(π / 12)

    # Default max_graph_distance
    sharing = ShareSimilarTangents(crit)
    @test sharing isa ShareSimilarTangents
    @test sharing isa MA.AbstractTangentSharingMode
    @test sharing.criterion === crit
    @test sharing.max_graph_distance == 1

    # Explicit max_graph_distance
    sharing2 = ShareSimilarTangents(crit; max_graph_distance=3)
    @test sharing2.max_graph_distance == 3

    # Validation: max_graph_distance must be >= 1
    @test_throws ArgumentError ShareSimilarTangents(crit; max_graph_distance=0)
    @test_throws ArgumentError ShareSimilarTangents(crit; max_graph_distance=-2)

    # Works with other criterion types too (not just SubspaceAngleCriterion)
    @test ShareSimilarTangents(FitErrorCriterion(0.1)) isa ShareSimilarTangents
    @test ShareSimilarTangents(DistortionCriterion(0.1)) isa ShareSimilarTangents
end

@testset "_fit_geometries(::ShareSimilarTangents) smoke" begin
    rng = MersenneTwister(42)
    n, k = 25, 5
    data = randn(rng, 3, n)

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k)
    method = PCAMethod(intrinsic_dim=2)

    # Loose criterion → likely lots of sharing
    sharing_loose = ShareSimilarTangents(SubspaceAngleCriterion(π / 2))
    geoms_loose = MA._fit_geometries(sharing_loose, method, graph, data)
    @test geoms_loose isa Vector
    @test length(geoms_loose) == n
    @test eltype(geoms_loose) <: MA.AbstractLocalGeometry
    @test all(g -> g isa MA.AbstractLocalGeometry, geoms_loose)

    # Tight criterion → likely no sharing; same shape regardless
    sharing_tight = ShareSimilarTangents(SubspaceAngleCriterion(1e-8))
    geoms_tight = MA._fit_geometries(sharing_tight, method, graph, data)
    @test length(geoms_tight) == n
    @test eltype(geoms_tight) <: MA.AbstractLocalGeometry

    # Reference: NoSharing returns same length / type family
    geoms_ref = MA._fit_geometries(NoSharing(), method, graph, data)
    @test length(geoms_ref) == n

    # max_graph_distance > 1 path (BFS branch in _get_nearby_assigned_nodes)
    sharing_far = ShareSimilarTangents(SubspaceAngleCriterion(π / 12);
                                        max_graph_distance=2)
    geoms_far = MA._fit_geometries(sharing_far, method, graph, data)
    @test length(geoms_far) == n

    # Sharing should not produce more unique geometries than NoSharing.
    # (Smoke-only: just check counts are sane and within [1, n].)
    n_unique_loose = length(unique(objectid, geoms_loose))
    @test 1 <= n_unique_loose <= n
end

@testset "_find_shareable_geometry smoke" begin
    rng = MersenneTwister(7)
    n, k = 20, 5
    data = randn(rng, 3, n)

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k)
    method = PCAMethod(intrinsic_dim=2)

    # Build a partial geometry array with node 1 fitted, mirroring the state
    # machine inside _fit_geometries.
    g1 = MA.fit_geometry(method, data, 1, graph[1]; graph=graph)
    G = typeof(g1)
    geometries = fill!(Vector{Union{Nothing, G}}(undef, n), nothing)
    geometries[1] = g1
    assigned = Int[1]

    # Loose criterion: should find a shareable geometry (or nothing if node 2
    # isn't a graph-neighbor of node 1) — both outcomes are valid; we just
    # require the function to execute and return either nothing or a G.
    sharing_loose = ShareSimilarTangents(SubspaceAngleCriterion(π / 2))
    res_loose = MA._find_shareable_geometry(sharing_loose, geometries, assigned,
                                             graph, data, 2, method)
    @test res_loose === nothing || res_loose isa G

    # Tight criterion: no sharing. Should reliably return nothing for a
    # generic random point, but the contract we test is type-only.
    sharing_tight = ShareSimilarTangents(SubspaceAngleCriterion(1e-12))
    res_tight = MA._find_shareable_geometry(sharing_tight, geometries, assigned,
                                             graph, data, 2, method)
    @test res_tight === nothing || res_tight isa G

    # Empty assigned list: must short-circuit to nothing
    empty_assigned = Int[]
    res_empty = MA._find_shareable_geometry(sharing_loose, geometries,
                                             empty_assigned, graph, data, 2,
                                             method)
    @test res_empty === nothing
end

@testset "_fit_geometries_with_candidates(::ShareSimilarTangents) smoke" begin
    rng = MersenneTwister(123)
    n, k, candidate_k = 25, 5, 8
    data = randn(rng, 4, n)

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k)
    method = PCAMethod(intrinsic_dim=2)

    sharing = ShareSimilarTangents(SubspaceAngleCriterion(π / 6))

    geoms = MA._fit_geometries_with_candidates(sharing, method, graph, index,
                                                data, candidate_k)
    @test geoms isa Vector
    @test length(geoms) == n
    @test eltype(geoms) <: MA.AbstractLocalGeometry

    # NoSharing reference
    geoms_ref = MA._fit_geometries_with_candidates(NoSharing(), method, graph,
                                                    index, data, candidate_k)
    @test length(geoms_ref) == n

    # Vary candidate_k and max_graph_distance to catch parameter-handling
    # regressions.
    sharing_far = ShareSimilarTangents(SubspaceAngleCriterion(π / 6);
                                        max_graph_distance=2)
    geoms_far = MA._fit_geometries_with_candidates(sharing_far, method, graph,
                                                    index, data, k + 2)
    @test length(geoms_far) == n
end

@testset "_find_shareable_geometry_candidates smoke" begin
    rng = MersenneTwister(2024)
    n, k, candidate_k = 20, 5, 7
    data = randn(rng, 4, n)

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k)
    method = PCAMethod(intrinsic_dim=2)

    # Bootstrap with node 1 fitted using candidate-style indices.
    center_1 = @view data[:, 1]
    candidates_1 = MA.query(index, data, center_1, candidate_k + 1)
    candidate_indices_1 = [c.id for c in candidates_1 if c.id != 1]
    if length(candidate_indices_1) > candidate_k
        candidate_indices_1 = candidate_indices_1[1:candidate_k]
    end
    g1 = MA.fit_geometry(method, data, 1, candidate_indices_1; graph=graph)
    G = typeof(g1)
    geometries = fill!(Vector{Union{Nothing, G}}(undef, n), nothing)
    geometries[1] = g1
    assigned = Int[1]

    # Build candidate indices for node 2.
    center_2 = @view data[:, 2]
    candidates_2 = MA.query(index, data, center_2, candidate_k + 1)
    candidate_indices_2 = [c.id for c in candidates_2 if c.id != 2]
    if length(candidate_indices_2) > candidate_k
        candidate_indices_2 = candidate_indices_2[1:candidate_k]
    end

    sharing_loose = ShareSimilarTangents(SubspaceAngleCriterion(π / 2))
    res = MA._find_shareable_geometry_candidates(sharing_loose, geometries,
                                                  assigned, graph, data, 2,
                                                  candidate_indices_2, method)
    @test res === nothing || res isa G

    # Empty assigned list short-circuits to nothing.
    res_empty = MA._find_shareable_geometry_candidates(sharing_loose, geometries,
                                                       Int[], graph, data, 2,
                                                       candidate_indices_2,
                                                       method)
    @test res_empty === nothing

    # max_graph_distance=2 path
    sharing_far = ShareSimilarTangents(SubspaceAngleCriterion(π / 2);
                                        max_graph_distance=2)
    res_far = MA._find_shareable_geometry_candidates(sharing_far, geometries,
                                                      assigned, graph, data, 2,
                                                      candidate_indices_2,
                                                      method)
    @test res_far === nothing || res_far isa G
end

@testset "build_weighted_graph integration with ShareSimilarTangents" begin
    rng = MersenneTwister(99)
    n, k = 25, 5
    data = randn(rng, 3, n)

    index = build_index(BruteForceIndex, data)
    graph = build_knn_graph(index, data; k=k)
    method = PCAMethod(intrinsic_dim=2)

    sharing = ShareSimilarTangents(SubspaceAngleCriterion(π / 6))
    wg = build_weighted_graph(method, graph, data; tangent_sharing=sharing)
    @test wg isa WeightedKNNGraph
    @test length(wg) == n
    @test length(wg.geometries) == n
    @test unique_geometry_count(wg) <= n
    @test unique_geometry_count(wg) >= 1

    # Index-overload path with candidate_k > k (exercises
    # _fit_geometries_with_candidates)
    wg2 = build_weighted_graph(method, index, data; k=k, candidate_k=k + 3,
                                tangent_sharing=sharing)
    @test wg2 isa WeightedKNNGraph
    @test length(wg2) == n
end
