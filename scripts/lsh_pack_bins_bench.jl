#!/usr/bin/env julia
# Microbench for the inner LSH hash path: pack_bins.
#
# Goal: confirm the FNV-1a fold over Int bins allocates zero bytes per call
# vs. the original `UInt64(hash(Tuple(bins)))` which allocated a Tuple per
# call.
#
# Run:
#   julia --project=. scripts/lsh_pack_bins_bench.jl
#
# Uses only stdlib (`@allocated`, `@elapsed`) so no extra dependencies.

using Random
using ManifoldANN

const pack_bins_new = ManifoldANN.pack_bins

# Reference: previous implementation, kept here purely for comparison.
@inline pack_bins_old(bins::AbstractVector{Int}) = UInt64(hash(Tuple(bins)))

function measure(f, bins, n_calls)
    f(bins) # warmup / compile
    sink = Ref(UInt64(0))
    bytes = @allocated for _ in 1:n_calls
        sink[] ⊻= f(bins)
    end
    sink[] = UInt64(0)
    elapsed = @elapsed for _ in 1:n_calls
        sink[] ⊻= f(bins)
    end
    return bytes, elapsed, sink[]
end

function bench_one(hash_length::Int, n_calls::Int = 1_000_000)
    rng = MersenneTwister(0xBEEF)
    bins = rand(rng, -1000:1000, hash_length)

    println("\n--- hash_length = $hash_length, n_calls = $n_calls ---")

    bo, to, _ = measure(pack_bins_old, bins, n_calls)
    bn, tn, _ = measure(pack_bins_new, bins, n_calls)

    println("OLD (hash(Tuple(...))): bytes=$(bo)  bytes/call=$(bo / n_calls)  ns/call=$(round(to * 1e9 / n_calls; digits=2))")
    println("NEW (FNV-1a fold)    : bytes=$(bn)  bytes/call=$(bn / n_calls)  ns/call=$(round(tn * 1e9 / n_calls; digits=2))")
end

function main()
    println("pack_bins microbench")
    for L in (4, 12, 32)
        bench_one(L)
    end
end

main()
