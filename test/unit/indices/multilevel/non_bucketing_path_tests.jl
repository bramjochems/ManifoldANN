using Test
using ManifoldANN
using Random
using LinearAlgebra

# Exercises the non-bucketing branch in _build_transformed where
# preserves_data=true. Resulting TransformedIndex has id_mappings=nothing AND
# child_data=nothing — the M3 dispatch path of _resolve_child_data /
# _child_distance_type. Other multilevel tests cover only the
# Vector{Vector{Int}} / IVF case.
@testset "Multilevel non-bucketing path (M3 dispatch)" begin
    rng = MersenneTwister(0xBEEF)
    X = randn(rng, Float32, 16, 200)
    q = randn(rng, Float32, 16)

    # IdentityTransform: preserves_data=true, has_bucketing=false.
    # ExhaustiveRouting works without an assignment because there's only
    # one child to probe.
    config = TransformedConfig(
        IdentityTransform(),
        ExhaustiveRouting(),
        TerminalConfig(HNSWIndex, (M=8, ef_construction=40)),
    )
    index = build_index(MultiLevelIndex, X, config)

    @test index isa MultiLevelIndex
    @test index.root isa TransformedIndex
    @test index.root.id_mappings === nothing
    @test index.root.child_data === nothing
    @test length(index.root.indices) == 1

    # The struct's M and C type parameters should both pin to Nothing here.
    M_param = typeof(index.root).parameters[4]
    C_param = typeof(index.root).parameters[3]
    @test M_param === Nothing
    @test C_param === Nothing

    # Query through the non-bucketing path; results should be valid global ids.
    neighbors = query(index, X, q, 5)
    ids = [n.id for n in neighbors]
    @test length(ids) == 5
    @test all(1 .<= ids .<= 200)
    @test length(unique(ids)) == 5

    # Sanity: recall vs brute-force on the same data should be high (the only
    # child is an HNSW over the full dataset).
    brute = build_index(BruteForceIndex, X)
    truth = Set(n.id for n in query(brute, X, q, 5))
    recall = length(intersect(Set(ids), truth)) / 5
    @test recall >= 0.6
end
