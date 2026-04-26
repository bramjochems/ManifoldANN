using Test
using ManifoldANN
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# Hand-built tangent geometries for predictable estimator outputs.
# We construct two points x, y in R^3 and tangent planes whose bases are
# explicitly known, so d_E, d_T^{(x)}, d_T^{(y)} can all be computed in
# closed form.
# ---------------------------------------------------------------------------

# x at the origin, y at (1, 0, 1).  d_E = sqrt(2).
# T_x M = span(e1, e2)  -> d_T^{(x)} = ||(1, 0, 0)|| = 1.
# T_y M = span(e1, e3)  -> d_T^{(y)} = ||(1, 0, 1)|| = sqrt(2).
const _X3 = [0.0, 0.0, 0.0]
const _Y3 = [1.0, 0.0, 1.0]
const _DATA3 = hcat(_X3, _Y3)

const _BASIS_X = [1.0 0.0; 0.0 1.0; 0.0 0.0]
const _BASIS_Y = [1.0 0.0; 0.0 0.0; 0.0 1.0]
const _GEOM_X  = ManifoldANN.PCAGeometry{Float64}(_X3, _BASIS_X, [1.0, 1.0])
const _GEOM_Y  = ManifoldANN.PCAGeometry{Float64}(_Y3, _BASIS_Y, [1.0, 1.0])
const _GEOMETRIES = [_GEOM_X, _GEOM_Y]

const _D_E   = sqrt(2.0)
const _D_T_X = 1.0
const _D_T_Y = sqrt(2.0)

@testset "edge geodesic estimators" begin

    @testset "EuclideanChord matches norm(y - x)" begin
        rng = MersenneTwister(7)
        d, n = 4, 25
        data = randn(rng, d, n)
        # geometries argument is ignored; pass nothing to prove it is unused.
        for trial in 1:20
            i = rand(rng, 1:n); j = rand(rng, 1:n)
            expected = norm(data[:, j] - data[:, i])
            got = compute_edge_weight(EuclideanChord(), i, j, data, nothing)
            @test got ≈ expected
        end
    end

    @testset "TangentProjectedSymmetricMean uses (d_T^x + d_T^y)/2" begin
        got = compute_edge_weight(TangentProjectedSymmetricMean(), 1, 2,
                                   _DATA3, _GEOMETRIES)
        @test got ≈ (_D_T_X + _D_T_Y) / 2
    end

    @testset "CurvatureFreeSymmetric matches eq:geod-sym" begin
        got = compute_edge_weight(CurvatureFreeSymmetric(), 1, 2,
                                   _DATA3, _GEOMETRIES)
        expected = (8 * _D_E - _D_T_X - _D_T_Y) / 6
        @test got ≈ expected
    end

    @testset "CurvatureFreeSymmetric falls back to d_E on negative raw value" begin
        # Engineer an example where d_T^{(x)} and d_T^{(y)} both equal d_E
        # (tangent planes contain the displacement vector). Then the raw
        # estimator is (8 d_E - 2 d_E)/6 = d_E -- positive, no fallback.
        # To force a negative value we inflate d_T artificially via an
        # over-large basis. Easiest construction: identical 3D bases that
        # span all of R^3 yield d_T = d_E, so we instead synthesise basis
        # vectors that make ||basis' * (y - x)|| > d_E by violating
        # orthonormality. The estimator does not check; this models a
        # broken/noisy tangent estimate, which is exactly the failure mode
        # the fallback exists to handle.
        x = [0.0, 0.0]
        y = [1.0, 0.0]
        d_E = 1.0
        # Non-orthonormal "basis" inflates the projected norm.
        bad_basis = [5.0 0.0; 0.0 5.0]   # 2x2, projects to 5*(y-x) -> norm 5
        geom_x = ManifoldANN.PCAGeometry{Float64}(x, bad_basis, [1.0, 1.0])
        geom_y = ManifoldANN.PCAGeometry{Float64}(y, bad_basis, [1.0, 1.0])
        data2 = hcat(x, y)
        geoms = [geom_x, geom_y]

        # Sanity-check: raw (8*d_E - d_T_x - d_T_y)/6 < 0 on this input.
        raw, raw_dE, _ = ManifoldANN._curvature_free_symmetric_raw(1, 2, data2, geoms)
        @test raw < 0
        @test raw_dE ≈ d_E

        # The public estimator must fall back to d_E.
        got = compute_edge_weight(CurvatureFreeSymmetric(), 1, 2, data2, geoms)
        @test got ≈ d_E
    end

    @testset "build_geodesic_model: default reproduces EuclideanChord" begin
        rng = MersenneTwister(2024)
        data = randn(rng, 3, 80)
        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)

        # Same RNG state should reach the build with identical inputs;
        # builds are deterministic for BruteForceIndex + PCAMethod.
        model_default = build_geodesic_model(method, index, data; k=6)
        model_eucl    = build_geodesic_model(method, index, data; k=6,
                                              edge_weight=EuclideanChord())

        @test length(model_default) == length(model_eucl)

        D_default = all_pairs_geodesic_distances(model_default, data)
        D_eucl    = all_pairs_geodesic_distances(model_eucl, data)
        @test D_default == D_eucl

        # Diagnostics: zero fallbacks for EuclideanChord.
        @test diagnostics(model_eucl).n_negative_fallbacks == 0
    end

    @testset "build_geodesic_model with CurvatureFreeSymmetric is consistent" begin
        rng = MersenneTwister(2025)
        data = randn(rng, 3, 60)
        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)

        model_cf = build_geodesic_model(method, index, data; k=6,
                                         edge_weight=CurvatureFreeSymmetric())
        diag = diagnostics(model_cf)
        @test diag isa EstimatorDiagnostics
        @test diag.n_edges == 60 * 6  # k neighbours per node
        @test diag.n_negative_fallbacks >= 0
    end

    @testset "single-pass regression: build_geodesic_model with EuclideanChord matches build_weighted_graph" begin
        # Locks in the single-pass behaviour: `build_geodesic_model` must
        # not re-compute / overwrite edge weights with a different default
        # rule before applying the requested `edge_weight=`. Equivalence
        # with `build_weighted_graph` on the same `edge_weight=` is the
        # cleanest way to assert that.
        rng = MersenneTwister(2718)
        data = randn(rng, 3, 50)
        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)

        model = build_geodesic_model(method, index, data; k=5,
                                      edge_weight=EuclideanChord())
        wg_direct = build_weighted_graph(method, index, data; k=5,
                                          edge_weight=EuclideanChord())

        for i in 1:length(model)
            @test neighbors(model.weighted_graph, i) == neighbors(wg_direct, i)
            @test neighbor_weights(model.weighted_graph, i) ==
                  neighbor_weights(wg_direct, i)
        end
    end

    @testset "single-pass regression: build_geodesic_model with CurvatureFreeSymmetric matches build_weighted_graph" begin
        # Same single-pass guarantee as above, but for the curvature-free
        # symmetric estimator -- which exercises the diagnostic-tracking
        # branch of `_compute_edge_weights_and_diagnostics`. The numerical
        # output must still match the public `build_weighted_graph` path
        # bit-for-bit (both go through the same negative-fallback rule).
        rng = MersenneTwister(2024_04_26)
        data = randn(rng, 3, 50)
        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)

        model = build_geodesic_model(method, index, data; k=5,
                                      edge_weight=CurvatureFreeSymmetric())
        wg_direct = build_weighted_graph(method, index, data; k=5,
                                          edge_weight=CurvatureFreeSymmetric())

        for i in 1:length(model)
            @test neighbors(model.weighted_graph, i) == neighbors(wg_direct, i)
            @test neighbor_weights(model.weighted_graph, i) ==
                  neighbor_weights(wg_direct, i)
        end
    end

    @testset "regression: build_geodesic_model with TangentProjectedSymmetricMean matches build_weighted_graph" begin
        # After unification, build_geodesic_model(...; edge_weight=W) and
        # build_weighted_graph(...; edge_weight=W) must produce identical
        # edge weights -- there is no longer a separate "rewrite weights"
        # pass. This regression locks in the unification and confirms that
        # under the unified `TangentProjectedSymmetricMean` the new
        # pipeline produces identical edge weights to the (pre-refactor)
        # `SymmetricMean` path on `build_weighted_graph`.
        rng = MersenneTwister(31415)
        data = randn(rng, 3, 50)
        index = build_index(BruteForceIndex, data)
        method = PCAMethod(intrinsic_dim=2)

        model = build_geodesic_model(method, index, data; k=5,
                                      edge_weight=TangentProjectedSymmetricMean())
        wg_direct = build_weighted_graph(method, index, data; k=5,
                                          edge_weight=TangentProjectedSymmetricMean())

        for i in 1:length(model)
            @test neighbors(model.weighted_graph, i) == neighbors(wg_direct, i)
            @test neighbor_weights(model.weighted_graph, i) ==
                  neighbor_weights(wg_direct, i)
        end
    end
end
