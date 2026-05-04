using Test
using ManifoldANN
using LinearAlgebra
using Random
using Statistics

@testset "ORC-ManL Compatibility Profiles" begin

    @testset "Preset construction" begin
        d  = ManifoldANNDefault()
        o  = OrcmlExact()

        @test d isa AbstractOrcMLCompatibilityProfile
        @test o isa AbstractOrcMLCompatibilityProfile

        # ManifoldANNDefault axes (current Julia behaviour)
        @test d.endpoint_inclusion_in_eff_eps == false
        @test d.slice_drops_smallest          == false
        @test d.asymmetric_target_exclusion   == false

        # OrcmlExact axes (orcml-matching)
        @test o.endpoint_inclusion_in_eff_eps == true
        @test o.slice_drops_smallest          == true
        @test o.asymmetric_target_exclusion   == true
    end

    # Build a small deterministic swiss-roll-like dataset and check that
    # (a) ManifoldANNDefault reproduces the historical (no-profile) result,
    # (b) OrcmlExact differs from it but is highly correlated.
    @testset "Curvature regression: default vs OrcmlExact" begin
        Random.seed!(0xC0FFEE)
        n = 120
        # 2D swiss-roll-ish manifold embedded in 3D
        t = range(0, stop=3π, length=n)
        u = range(-1.0, stop=1.0, length=n)
        data = Matrix{Float64}(undef, 3, n)
        for i in 1:n
            ti = t[i] + 0.05 * randn()
            ui = u[i] + 0.05 * randn()
            data[1, i] = ti * cos(ti)
            data[2, i] = ui
            data[3, i] = ti * sin(ti)
        end

        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=10, directed=false)

        # 1. Calling `ORCManL()` (which defaults to the
        #    `ManifoldANNDefault()` profile) must equal explicitly
        #    passing `ORCManL(profile=ManifoldANNDefault())`.
        c_implicit = compute_all_curvatures(
            graph, data;
            variant=ORCManL(),
            solver=HungarianSolver(),
            fallback_solver=NetworkSimplexSolver(),
            use_threading=false,
        )
        c_default = compute_all_curvatures(
            graph, data;
            variant=ORCManL(profile=ManifoldANNDefault()),
            solver=HungarianSolver(),
            fallback_solver=NetworkSimplexSolver(),
            use_threading=false,
        )
        @test length(c_implicit) == length(c_default)
        for k in keys(c_default)
            @test haskey(c_implicit, k)
            @test c_implicit[k].curvature ≈ c_default[k].curvature atol=1e-12
        end

        # 2. OrcmlExact is a different config: it must produce a result
        #    that is highly correlated with — but not identical to — the
        #    default. This locks in the relative behaviour of the two
        #    presets so future refactors can't silently break either.
        c_orcml = compute_all_curvatures(
            graph, data;
            variant=ORCManL(profile=OrcmlExact()),
            solver=HungarianSolver(),
            fallback_solver=NetworkSimplexSolver(),
            use_threading=false,
        )

        common = collect(intersect(keys(c_default), keys(c_orcml)))
        @test length(common) > 50

        v_default = [c_default[k].curvature for k in common]
        v_orcml   = [c_orcml[k].curvature   for k in common]

        # Highly correlated (we expect ~0.99+ on small datasets).
        r = cor(v_default, v_orcml)
        @test r > 0.95

        # Not bitwise identical: the two presets are genuinely different.
        @test maximum(abs.(v_default .- v_orcml)) > 1e-6
    end

    @testset "ManifoldANNDefault is the default everywhere" begin
        # Check the `effective_epsilon` thin wrapper too.
        Random.seed!(7)
        n = 30
        data = randn(3, n)
        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=5, directed=false)

        e1 = ManifoldANN.effective_epsilon(1, 2, graph, data)
        e2 = ManifoldANN.effective_epsilon(1, 2, graph, data;
                                           profile=ManifoldANNDefault())
        @test e1 ≈ e2
    end

    # Regression: `_eff_eps_average` previously had a `Vector{Float64}`
    # signature, so calling `compute_effective_epsilon` with a Float32
    # data matrix MethodError-ed (norm-of-Float32-difference returns
    # Float32). Both compatibility profiles must work on Float32.
    @testset "Float32 effective_epsilon regression" begin
        Random.seed!(11)
        n = 30
        data = randn(Float32, 3, n)
        index = build_index(BruteForceIndex, data)
        graph = build_knn_graph(index, data; k=5, directed=false)

        e_default = ManifoldANN.effective_epsilon(1, 2, graph, data)
        e_orcml   = ManifoldANN.effective_epsilon(1, 2, graph, data;
                                                  profile=OrcmlExact())
        @test isfinite(e_default) && e_default > 0
        @test isfinite(e_orcml)   && e_orcml   > 0
        @test eltype([e_default]) === Float32
    end
end
