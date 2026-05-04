#!/usr/bin/env julia
# Microbench: ManifoldANN's default_distance vs Distances.Euclidean per-call.
# Run: julia --project=benchmarking/julia scripts/distance_kernel_micro.jl

using Random, LinearAlgebra, Printf
using ManifoldANN
using Distances
const MA = ManifoldANN

BLAS.set_num_threads(1)

const D_SIZES = (8, 32, 128)

# Median of REPS runs of N_CALLS-loop, return ns/call
function bench_kernel(f, args...; reps=15, n_calls=2_000_000)
    times = Float64[]
    for _ in 1:reps
        # warm
        s = 0.0f0
        for _ in 1:1000
            s += f(args...)
        end
        # measure
        t = @elapsed begin
            s2 = 0.0f0
            for _ in 1:n_calls
                s2 += f(args...)
            end
            s2
        end
        push!(times, t * 1e9 / n_calls)
    end
    return minimum(times), times
end

for d in D_SIZES
    println("\n=== d = $d ===")
    Random.seed!(0xC0FFEE)
    x = randn(Float32, d)
    y = randn(Float32, d)

    r_mann = MA.default_distance(x, y)
    r_nnd  = Distances.evaluate(Distances.Euclidean(), x, y)
    @assert isapprox(r_mann, r_nnd; rtol=1e-5)

    f_mann() = MA.default_distance(x, y)
    f_nnd()  = Distances.evaluate(Distances.Euclidean(), x, y)

    m_mann, _ = bench_kernel(f_mann)
    m_nnd, _  = bench_kernel(f_nnd)

    @printf "  MANN default_distance:        %.2f ns/call\n" m_mann
    @printf "  Distances.Euclidean evaluate: %.2f ns/call\n" m_nnd
    @printf "  ratio: %.2fx (MANN/NND, lower is better for MANN)\n" (m_mann / m_nnd)
end

println("\n=== view-into-Matrix vs Vector-of-Vectors access (d=32) ===")
const N = 20_000
data_mat = randn(Float32, 32, N)
data_cols = [collect(data_mat[:, i]) for i in 1:N]
q = randn(Float32, 32)

function bench_loop(f, args...; reps=15, n_loops=10_000)
    times = Float64[]
    for _ in 1:reps
        s = 0.0f0; for _ in 1:1000; s += f(args...); end  # warm
        t = @elapsed begin
            s2 = 0.0f0
            for _ in 1:n_loops
                s2 += f(args...)
            end
            s2
        end
        push!(times, t * 1e6 / n_loops)
    end
    return minimum(times)
end

# Each f-call evaluates 1000 distances against q
function loop_mann()
    s = 0.0f0
    @inbounds for i in 1:1000
        s += MA.default_distance(view(data_mat, :, i), q)
    end
    s
end
function loop_nnd()
    s = 0.0f0
    @inbounds for i in 1:1000
        s += Distances.evaluate(Distances.Euclidean(), data_cols[i], q)
    end
    s
end

m_mann = bench_loop(loop_mann)
m_nnd  = bench_loop(loop_nnd)

@printf "  MANN view(data,:,i): %.2f us / 1000 calls = %.2f ns/call\n" m_mann (m_mann*1000)
@printf "  NND Vector-of-Vec:   %.2f us / 1000 calls = %.2f ns/call\n" m_nnd  (m_nnd*1000)
@printf "  ratio: %.2fx\n" (m_mann / m_nnd)

println("\nDone.")
