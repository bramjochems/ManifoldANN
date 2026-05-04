using Test
using Random
using LinearAlgebra
using ManifoldANN

# Smoke tests for ExpandingNeighborhood. The strategy lacked direct
# coverage; same risk profile as the prior ShareSimilarTangents
# broken-constructor incident.
@testset "ExpandingNeighborhood smoke" begin
    @testset "construct with defaults" begin
        strat = ExpandingNeighborhood()
        @test strat.initial_k == 8
        @test strat.max_neighbors == 50
        @test strat.max_shells == 3
        @test strat.criterion isa DistortionCriterion
    end

    @testset "construct with custom criterion" begin
        strat = ExpandingNeighborhood(
            initial_k = 4,
            max_neighbors = 20,
            criterion = SubspaceAngleCriterion(π/4),
            max_shells = 2,
        )
        @test strat.initial_k == 4
        @test strat.max_neighbors == 20
        @test strat.max_shells == 2
        @test strat.criterion isa SubspaceAngleCriterion
    end

    @testset "select_neighbors returns a non-empty result" begin
        # Small synthetic 2D plane embedded in 3D.
        rng = MersenneTwister(2025)
        n = 30
        U = randn(rng, 2, n)
        data = vcat(U, zeros(1, n))  # third coord ~ 0 → flat plane
        center_idx = 1
        candidates = collect(2:n)
        method = PCAMethod(intrinsic_dim = 2)

        # No-graph path: must terminate and return at least the initial_k
        # seed neighbours.
        strat = ExpandingNeighborhood(
            initial_k = 5,
            max_neighbors = 15,
            criterion = DistortionCriterion(0.5),
            max_shells = 2,
        )
        result = select_neighbors(strat, method, data, center_idx, candidates)
        @test result.indices isa Vector{Int}
        @test length(result.indices) >= 5
        @test length(result.indices) <= 15
        @test result.iterations >= 0
    end

    @testset "expanding loop terminates with tight criterion" begin
        # A very tight DistortionCriterion will reject expansions
        # immediately. The loop must stop, not infinite-loop.
        rng = MersenneTwister(7)
        data = randn(rng, 4, 25)
        method = PCAMethod(intrinsic_dim = 2)
        strat = ExpandingNeighborhood(
            initial_k = 3,
            max_neighbors = 20,
            criterion = DistortionCriterion(1e-9),
            max_shells = 5,
        )
        result = select_neighbors(strat, method, data, 1, collect(2:25))
        @test length(result.indices) >= 3      # initial_k seed
        @test result.iterations <= strat.max_shells
    end
end
