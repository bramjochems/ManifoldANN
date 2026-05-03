using Test
using Random
using ManifoldANN

# Generic batch-query equivalence: for every concrete index registered, the
# matrix-input batch query must return the same ids as the per-query loop.
# This locks in the AbstractANNIndex thread-safety / re-entrancy contract:
# if a future index breaks re-entrancy, this test catches it.
@testset "AbstractANNIndex: batch query matches single-query elementwise" begin
    rng = MersenneTwister(0xBA7CE5)
    data = randn(rng, Float32, 16, 1000)
    n_queries = 80  # > BATCH_THREAD_THRESHOLD (64) to exercise threading
    queries = randn(rng, Float32, 16, n_queries)
    k = 5

    # Each entry is (label, build_fn, single_kwargs, batch_kwargs).
    # For most indices single and batch kwargs match; NN-Descent batch spawns
    # child RNGs internally so its serial reference uses the same machinery.
    cases = [
        ("BruteForce",
            () -> build_index(BruteForceIndex, data),
            NamedTuple(),
            NamedTuple()),
        ("KDTree",
            () -> build_index(KDTreeIndex, data; axis_selector = :variance),
            NamedTuple(),
            NamedTuple()),
        ("LSH",
            () -> build_index(LSHIndex, data; n_tables = 4, hash_length = 4, T = Float32),
            NamedTuple(),
            NamedTuple()),
        ("IVFFlat",
            () -> build_index(IVFFlatIndex, data; nlist = 8, nprobe = 4),
            NamedTuple(),
            NamedTuple()),
        ("HNSW",
            () -> build_index(HNSWIndex, data; M = 4, ef_construction = 20, threaded = false),
            (ef_search = 20,),
            (ef_search = 20,)),
    ]

    for (label, build_fn, single_kwargs, batch_kwargs) in cases
        @testset "$label" begin
            idx = build_fn()
            ref = [query(idx, data, view(queries, :, i), k; single_kwargs...) for i in 1:n_queries]
            got = query(idx, data, queries, k; batch_kwargs...)
            @test length(got) == n_queries
            for i in 1:n_queries
                @test [n.id for n in got[i]] == [n.id for n in ref[i]]
            end

            # Also exercise vector-of-vectors path (routes through abstract
            # convenience overload).
            qvec = [Vector(view(queries, :, i)) for i in 1:n_queries]
            got_vv = query(idx, data, qvec, k; batch_kwargs...)
            @test length(got_vv) == n_queries
            for i in 1:n_queries
                @test [n.id for n in got_vv[i]] == [n.id for n in ref[i]]
            end
        end
    end

    @testset "NNDescent (deterministic via shared rng seed)" begin
        # NN-Descent keeps its own matrix batch method to spawn child RNGs.
        # Equivalence test uses an explicit rng so single and batch paths agree.
        idx = build_index(NNDescentIndex, data; k = 8, max_iterations = 3, threaded = false)
        # Use the same RNG-spawning the batch method does, so the serial
        # reference matches.
        seed_rng = MersenneTwister(42)
        parent_seed = ManifoldANN.derive_child_seed(seed_rng)
        ref = [query(idx, data, view(queries, :, i), k; ef_search = 20,
                     rng = ManifoldANN.query_child_rng(parent_seed, i))
               for i in 1:n_queries]

        got = query(idx, data, queries, k; ef_search = 20, rng = MersenneTwister(42))
        @test length(got) == n_queries
        for i in 1:n_queries
            @test [n.id for n in got[i]] == [n.id for n in ref[i]]
        end
    end

    @testset "Empty batch returns empty vector (BruteForce)" begin
        idx = build_index(BruteForceIndex, data)
        empty_q = Matrix{Float32}(undef, 16, 0)
        got = query(idx, data, empty_q, k)
        @test got isa Vector
        @test isempty(got)
    end

    @testset "Empty batch still validates dimensions" begin
        # Regression for the brutal-critic finding: empty batch with WRONG
        # dimensions used to silently return [] for the generic dispatch
        # while HNSW (specialised) threw DimensionMismatch. Now both paths
        # validate up front.
        bf = build_index(BruteForceIndex, data)
        bad_empty = Matrix{Float32}(undef, 999, 0)  # wrong dim, empty
        @test_throws DimensionMismatch query(bf, data, bad_empty, k)

        hnsw = build_index(HNSWIndex, data; M = 4, ef_construction = 20, threaded = false)
        @test_throws DimensionMismatch query(hnsw, data, bad_empty, k; ef_search = 20)
    end

    @testset "Small batch (< BATCH_THREAD_THRESHOLD) takes serial path" begin
        # Coverage for the serial branch even under -t > 1. We can't directly
        # observe which branch ran, but we can verify equivalence at small
        # batch sizes (the threshold is 64) so that branch is exercised.
        idx = build_index(BruteForceIndex, data)
        small_n = 10  # well below BATCH_THREAD_THRESHOLD
        small_qs = randn(MersenneTwister(0x57A11), Float32, 16, small_n)
        ref = [query(idx, data, view(small_qs, :, i), k) for i in 1:small_n]
        got = query(idx, data, small_qs, k)
        @test length(got) == small_n
        for i in 1:small_n
            @test [n.id for n in got[i]] == [n.id for n in ref[i]]
        end
    end

    @testset "HNSW routes to its specialised batch method (dispatch lock-in)" begin
        # If HNSW silently lost its specialised query(::HNSWIndex, ::Matrix, ...)
        # method (the one with BatchQueryScratch pooling), the equivalence test
        # above would still pass because the generic path produces identical
        # results — just slower and without pool amortisation. Lock dispatch in
        # by inspecting which method handles the call.
        idx = build_index(HNSWIndex, data; M = 4, ef_construction = 20, threaded = false)
        m = which(query, (typeof(idx), typeof(data), typeof(queries), Int))
        @test m.module === ManifoldANN
        # The specialised method's first argument is HNSWIndex{T,...}, NOT
        # AbstractANNIndex. If dispatch silently downgrades to the generic,
        # this assertion catches it.
        @test occursin("HNSWIndex", string(m.sig))
    end

    @testset "MultiLevelIndex routes to generic batch path" begin
        # Regression for the brutal-critic ambiguity finding (commit 906d72a):
        # MultiLevelIndex used to ship its own matrix and Vector{Vector}
        # overloads which were ambiguous with the new generic. Both deleted;
        # MultiLevelIndex now goes through the generic threaded fallback.
        ml = build_ivf_hnsw_index(
            data; nlist = 4, routing_k = 2,
            kmeans_max_iters = 2, hnsw_M = 4, hnsw_ef_construction = 20,
        )
        # Matrix path
        ref = [query(ml, data, view(queries, :, i), k) for i in 1:n_queries]
        got = query(ml, data, queries, k)
        @test length(got) == n_queries
        for i in 1:n_queries
            @test [n.id for n in got[i]] == [n.id for n in ref[i]]
        end
        # Vector-of-vectors path (the one that used to be ambiguous)
        qvec = [Vector(view(queries, :, i)) for i in 1:n_queries]
        got_vv = query(ml, data, qvec, k)
        @test length(got_vv) == n_queries
        for i in 1:n_queries
            @test [n.id for n in got_vv[i]] == [n.id for n in ref[i]]
        end
    end
end
