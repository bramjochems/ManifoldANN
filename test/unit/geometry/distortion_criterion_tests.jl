using Test
using LinearAlgebra
using ManifoldANN

@testset "DistortionCriterion smoke" begin
    @testset "construct" begin
        c_default = DistortionCriterion()
        @test c_default.max_distortion ≈ 0.05

        c_custom = DistortionCriterion(0.2)
        @test c_custom.max_distortion ≈ 0.2

        @test_throws ArgumentError DistortionCriterion(0.0)
        @test_throws ArgumentError DistortionCriterion(1.0)
    end

    @testset "zero distortion when points lie in the tangent plane" begin
        # 4 points already in span(e1, e2) of R^3. Tangent-plane projection
        # is the identity on these points → distortion exactly 0.
        data = Float64[
            0.0  1.0  0.0  1.0;
            0.0  0.0  1.0  1.0;
            0.0  0.0  0.0  0.0;
        ]
        center = zeros(3)
        basis = Matrix{Float64}(I, 3, 3)[:, 1:2]
        geom = PCAGeometry{Float64}(center, basis, [1.0, 1.0])
        crit = DistortionCriterion(0.05)

        d = evaluate_neighborhood(crit, geom, data, center, [1, 2, 3, 4])
        @test d ≈ 0.0 atol = 1e-12
    end

    @testset "non-zero distortion when points have off-plane component" begin
        # Same xy-coords as above but with non-trivial z. Projecting onto
        # span(e1, e2) drops z, so pairwise distances shrink and the
        # criterion returns a positive distortion. Hand-checkable case:
        # (0,0,0) and (0,0,1) are at distance 1 in ambient space, but
        # both project to (0,0) — relative distortion = 1.
        data = Float64[
            0.0  0.0;
            0.0  0.0;
            0.0  1.0;
        ]
        center = zeros(3)
        basis = Matrix{Float64}(I, 3, 3)[:, 1:2]
        geom = PCAGeometry{Float64}(center, basis, [1.0, 1.0])
        crit = DistortionCriterion(0.05)

        d = evaluate_neighborhood(crit, geom, data, center, [1, 2])
        @test d ≈ 1.0 atol = 1e-9
    end

    @testset "single-point neighbourhood returns 0" begin
        # n_pairs == 0 path.
        data = randn(3, 5)
        center = zeros(3)
        basis = Matrix{Float64}(I, 3, 3)[:, 1:2]
        geom = PCAGeometry{Float64}(center, basis, [1.0, 1.0])
        crit = DistortionCriterion(0.05)
        @test evaluate_neighborhood(crit, geom, data, center, [1]) == 0.0
    end
end
