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
        child_rngs = ManifoldANN.spawn_child_rngs(seed_rng, n_queries)
        ref = [query(idx, data, view(queries, :, i), k; ef_search = 20, rng = child_rngs[i])
               for i in 1:n_queries]

        got = query(idx, data, queries, k; ef_search = 20, rng = MersenneTwister(42))
        @test length(got) == n_queries
        for i in 1:n_queries
            @test [n.id for n in got[i]] == [n.id for n in ref[i]]
        end
    end

    @testset "Empty batch returns empty vector" begin
        idx = build_index(BruteForceIndex, data)
        empty_q = Matrix{Float32}(undef, 16, 0)
        got = query(idx, data, empty_q, k)
        @test got isa Vector
        @test isempty(got)
    end
end
