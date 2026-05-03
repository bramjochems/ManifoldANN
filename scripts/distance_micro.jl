#!/usr/bin/env julia
# Micro-benchmark: per-call cost of `default_distance(@view(data[:,c]), q)` vs
# a column-indexed kernel that bypasses SubArray construction. Both compile
# the same SIMD reduction; the question is whether the SubArray wrapping
# costs measurable time per call.
#
# Run: julia --project=. -t 1 scripts/distance_micro.jl

using Random, Printf, LinearAlgebra
using ManifoldANN: default_distance, default_squared_distance

BLAS.set_num_threads(1)
Random.seed!(0xC0FFEE)

# Drop-in column-indexed kernel: same FLOP count, no view construction.
@inline function squared_distance_col(data::Matrix{T}, col::Int, q::AbstractVector{T}) where {T}
    s = zero(T)
    @inbounds @simd for i in 1:size(data, 1)
        diff = data[i, col] - q[i]
        s += diff * diff
    end
    return s
end

@inline function default_distance_col(data::Matrix{T}, col::Int, q::AbstractVector{T}) where {T}
    return sqrt(squared_distance_col(data, col, q))
end

function bench(d, n, calls; warm=true)
    data = randn(Float32, d, n)
    q    = randn(Float32, d)
    cols = rand(1:n, calls)

    # warmup
    if warm
        s1 = 0.0f0
        for c in cols[1:min(1000, calls)]
            s1 += default_distance(@view(data[:, c]), q)
        end
        s2 = 0.0f0
        for c in cols[1:min(1000, calls)]
            s2 += default_distance_col(data, c, q)
        end
    end

    GC.gc()
    a0 = Base.gc_bytes()
    t_view = @elapsed begin
        s = 0.0f0
        for c in cols
            s += default_distance(@view(data[:, c]), q)
        end
        # consume to prevent dead-code elimination
        global _sink_view = s
    end
    a1 = Base.gc_bytes()

    GC.gc()
    a2 = Base.gc_bytes()
    t_col = @elapsed begin
        s = 0.0f0
        for c in cols
            s += default_distance_col(data, c, q)
        end
        global _sink_col = s
    end
    a3 = Base.gc_bytes()

    @printf "  d=%-3d  view: %6.2f ns/call  alloc %5.0f B/call  |  col: %6.2f ns/call  alloc %5.0f B/call  |  speedup %.2fx\n" d (t_view*1e9/calls) ((a1-a0)/calls) (t_col*1e9/calls) ((a3-a2)/calls) (t_view/t_col)
end

println("Single-call distance micro (warm)")
println("=" ^ 60)
for d in (8, 16, 32, 64, 128, 256, 512)
    bench(d, 10_000, 5_000_000)
end

# Also: squared_distance variants — relevant because internal HNSW search
# typically uses *squared* Euclidean for ranking (no sqrt).
println()
println("Squared-distance variant (no sqrt):")
println("=" ^ 60)
function bench_sq(d, n, calls)
    data = randn(Float32, d, n)
    q    = randn(Float32, d)
    cols = rand(1:n, calls)
    # warm
    for c in cols[1:1000]; default_squared_distance(@view(data[:, c]), q); end
    for c in cols[1:1000]; squared_distance_col(data, c, q); end

    GC.gc()
    t_view = @elapsed begin
        s = 0.0f0
        for c in cols; s += default_squared_distance(@view(data[:, c]), q); end
        global _sink_sqv = s
    end
    GC.gc()
    t_col = @elapsed begin
        s = 0.0f0
        for c in cols; s += squared_distance_col(data, c, q); end
        global _sink_sqc = s
    end
    @printf "  d=%-3d  view: %6.2f ns/call  |  col: %6.2f ns/call  |  speedup %.2fx\n" d (t_view*1e9/calls) (t_col*1e9/calls) (t_view/t_col)
end
for d in (8, 16, 32, 64, 128, 256, 512)
    bench_sq(d, 10_000, 5_000_000)
end

println("\nDone.")
