# Independent stress harness for the threaded NN-Descent local-join.
# Runs many builds at meaningful n/k under high thread count and asserts
# structural invariants (no self-loop, allunique, ids in 1:n) on every node
# of every build. If the heap-race fix is correct, this should be solid.
#
# Usage:
#   julia --project=. -t 8 scripts/nndescent_stress.jl
#   TRIALS=200 N=5000 K=20 julia --project=. -t 16 scripts/nndescent_stress.jl

using Random
using ManifoldANN

const TRIALS = parse(Int, get(ENV, "TRIALS", "100"))
const N      = parse(Int, get(ENV, "N",      "5000"))
const K      = parse(Int, get(ENV, "K",      "20"))
const D      = parse(Int, get(ENV, "D",      "32"))
const MAX_IT = parse(Int, get(ENV, "MAX_IT", "10"))

println("nthreads=$(Threads.nthreads()) trials=$TRIALS n=$N k=$K d=$D max_it=$MAX_IT")

data = randn(MersenneTwister(0xDEADBEEF), Float32, D, N)

failures = 0
t0 = time()
for trial in 1:TRIALS
    rng = MersenneTwister(UInt64(0x1000 + trial))
    idx = build_index(NNDescentIndex, data;
                      k=K, threaded=true,
                      max_iterations=MAX_IT,
                      rng=rng)
    bad = 0
    for i in 1:N
        adj = idx.neighbors[i]
        if i in adj
            @warn "self-loop" trial i
            bad += 1
        end
        if !allunique(adj)
            @warn "duplicates" trial i adj
            bad += 1
        end
        if !all(1 <= id <= N for id in adj)
            @warn "out-of-range id" trial i adj
            bad += 1
        end
    end
    if bad > 0
        global failures += 1
    end
    if trial % 10 == 0
        println("  trial $trial: ok ($(round(time()-t0; digits=1))s elapsed)")
    end
end

println("DONE: $failures/$TRIALS trials had invariant violations")
exit(failures == 0 ? 0 : 1)
