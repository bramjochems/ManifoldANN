using Test
using LinearAlgebra
using ManifoldANN

@testset "SubspaceAngleCriterion smoke" begin
    @testset "construct" begin
        c_default = SubspaceAngleCriterion()
        @test c_default.max_angle ≈ π/6

        c_custom = SubspaceAngleCriterion(π/4)
        @test c_custom.max_angle ≈ π/4
    end

    @testset "subspace_angle on identical bases is ≈ 0" begin
        # Two PCAGeometry with the same orthonormal basis.
        basis = Matrix{Float64}(I, 4, 2)  # span of e1, e2 in R^4
        center = zeros(4)
        eigvals = [2.0, 1.0]
        g1 = PCAGeometry{Float64}(center, basis, eigvals)
        g2 = PCAGeometry{Float64}(center, copy(basis), copy(eigvals))
        @test subspace_angle(g1, g2) < 1e-9
    end

    @testset "subspace_angle on orthogonal bases is ≈ π/2" begin
        # span(e1, e2) vs span(e3, e4) in R^4 — fully orthogonal.
        b1 = Matrix{Float64}(I, 4, 4)[:, 1:2]
        b2 = Matrix{Float64}(I, 4, 4)[:, 3:4]
        center = zeros(4)
        g1 = PCAGeometry{Float64}(center, b1, [1.0, 1.0])
        g2 = PCAGeometry{Float64}(center, b2, [1.0, 1.0])
        @test subspace_angle(g1, g2) ≈ π/2 atol = 1e-9
    end

    @testset "dimension-mismatch lock-in (silent truncation)" begin
        # 2D vs 3D bases with the 2D one nested inside the 3D one.
        # Current behaviour silently truncates to min(d1,d2)=2 and so
        # returns angle ≈ 0. This test locks that in — a future change
        # to error on mismatch, or to use a different reduction, will
        # produce a noisy diff here.
        b2 = Matrix{Float64}(I, 5, 5)[:, 1:2]
        b3 = Matrix{Float64}(I, 5, 5)[:, 1:3]
        center = zeros(5)
        g_short = PCAGeometry{Float64}(center, b2, [1.0, 1.0])
        g_long  = PCAGeometry{Float64}(center, b3, [1.0, 1.0, 1.0])

        # Should not crash.
        angle = subspace_angle(g_short, g_long)
        @test angle isa Real
        # The first two columns coincide → truncated principal angle ≈ 0.
        @test angle < 1e-9
    end
end
