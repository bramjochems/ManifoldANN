using ManifoldANN
using Random
using Test

@testset "NN-Descent bounded_candidates query knob" begin
    Random.seed!(42)
    data = randn(Float32, 16, 400)
    queries = randn(Float32, 16, 50)
    k = 10

    index = build_index(
        NNDescentIndex, data;
        k = 24, max_iterations = 15,
        rng = Random.MersenneTwister(0xBEEF),
    )

    @testset "default behaviour unchanged (regression)" begin
        # Bit-equal: not passing the kwarg vs passing nothing.
        res_default = query(index, data, queries, k;
            ef_search = 40, rng = Random.MersenneTwister(7))
        res_explicit_nothing = query(index, data, queries, k;
            ef_search = 40, bounded_candidates = nothing,
            rng = Random.MersenneTwister(7))
        @test length(res_default) == length(res_explicit_nothing)
        for i in eachindex(res_default)
            ids_a = [n.id for n in res_default[i]]
            ids_b = [n.id for n in res_explicit_nothing[i]]
            @test ids_a == ids_b
        end
    end

    @testset "single-query API accepts the knob" begin
        q = view(queries, :, 1)
        res = query(index, data, q, k;
            ef_search = 40, bounded_candidates = 20,
            rng = Random.MersenneTwister(11))
        @test length(res) == k
        @test all(n -> 1 <= n.id <= size(data, 2), res)
    end

    @testset "bounded mode runs and returns valid results" begin
        res = query(index, data, queries, k;
            ef_search = 40, bounded_candidates = 20,
            rng = Random.MersenneTwister(11))
        @test length(res) == size(queries, 2)
        for r in res
            @test length(r) == k
            ids = [n.id for n in r]
            @test length(unique(ids)) == k
            @test all(1 .<= ids .<= size(data, 2))
        end
    end

    @testset "bounded mode trades recall for speed (sanity)" begin
        # At a fixed ef_search, a tight bound should yield ≤ recall of
        # the unbounded path. The brute-force baseline supplies ground truth.
        brute = build_index(BruteForceIndex, data)
        gt = query(brute, data, queries, k)
        gt_ids_per_q = [Set(nb.id for nb in r) for r in gt]

        res_unbounded = query(index, data, queries, k;
            ef_search = 60, rng = Random.MersenneTwister(11))
        res_bounded = query(index, data, queries, k;
            ef_search = 60, bounded_candidates = k,
            rng = Random.MersenneTwister(11))

        function recall(res)
            hits = 0
            for (i, r) in enumerate(res)
                for nb in r
                    nb.id in gt_ids_per_q[i] && (hits += 1)
                end
            end
            return hits / (length(res) * k)
        end
        r_unb = recall(res_unbounded)
        r_bnd = recall(res_bounded)
        @test r_bnd <= r_unb
    end
end
