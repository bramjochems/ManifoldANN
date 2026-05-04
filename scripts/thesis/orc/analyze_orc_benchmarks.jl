"""
Analyze and compare ORC benchmark results from Julia and Python implementations.

Reads:
- benchmark_results/orc_benchmark_julia.csv
- benchmark_results/orc_benchmark_python.csv (if available)

Generates:
- Comparative performance tables
- Speedup analysis
- Recommendations for best solver/config combinations
"""

using DelimitedFiles
using Printf
using Statistics

println("="^80)
println("ORC Benchmark Analysis")
println("="^80)

# Load Julia results
julia_file = "benchmark_results/orc_benchmark_julia.csv"
if !isfile(julia_file)
    println("Error: Julia benchmark results not found at $julia_file")
    println("Run: julia --project=. scripts/benchmark_orc_comprehensive.jl")
    exit(1)
end

julia_data = readdlm(julia_file, ',', skipstart=1)
println("\nLoaded Julia results: $(size(julia_data, 1)) entries")

# Load Python results (if available)
python_file = "benchmark_results/orc_benchmark_python.csv"
has_python = isfile(python_file)

python_data = nothing
if has_python
    python_data = readdlm(python_file, ',', skipstart=1)
    println("Loaded Python results: $(size(python_data, 1)) entries")
else
    println("Python results not available (run scripts/benchmark_orc_python.py)")
end

# Parse Julia results
struct JuliaResult
    n::Int
    config::String
    solver::String
    time_sec::Float64
    n_edges::Int
    n_nan::Int
    ms_per_edge::Float64
    mean_curv::Float64
    std_curv::Float64
end

julia_results = JuliaResult[]
for row in eachrow(julia_data)
    push!(julia_results, JuliaResult(
        Int(row[1]),      # n
        String(row[2]),   # config
        String(row[3]),   # solver
        Float64(row[4]),  # time_sec
        Int(row[5]),      # n_edges
        Int(row[6]),      # n_nan
        Float64(row[7]),  # ms_per_edge
        Float64(row[8]),  # mean_curv
        Float64(row[9])   # std_curv
    ))
end

# Parse Python results (if available)
struct PythonResult
    n::Int
    implementation::String
    time_sec::Float64
    n_edges::Int
    ms_per_edge::Float64
    mean_curv::Float64
    std_curv::Float64
end

python_results = PythonResult[]
if has_python
    for row in eachrow(python_data)
        push!(python_results, PythonResult(
            Int(row[1]),           # n
            String(row[2]),        # implementation
            Float64(row[3]),       # time_sec
            Int(row[4]),           # n_edges
            Float64(row[5]),       # ms_per_edge
            Float64(row[6]),       # mean_curv
            Float64(row[7])        # std_curv
        ))
    end
end

# Get unique sizes
sizes = sort(unique([r.n for r in julia_results]))

println("\n" * "="^80)
println("Performance Comparison: Julia Solvers")
println("="^80)

for n in sizes
    println("\n" * "-"^80)
    println("Graph Size: n=$n")
    println("-"^80)

    for config in ["original", "orcml"]
        results_for_config = filter(r -> r.n == n && r.config == config, julia_results)

        if isempty(results_for_config)
            continue
        end

        println("\nConfiguration: $config")
        println("  Solver              | Time (s) | Edges | ms/edge | NaN | Mean κ  | Std κ")
        println("  " * "-"^74)

        # Sort by time
        sort!(results_for_config, by = r -> r.time_sec)

        for r in results_for_config
            if r.n_nan > 0
                @printf("  %-18s | %8.3f | %5d | %7.3f | %3d | %7s | %6s\n",
                        r.solver, r.time_sec, r.n_edges, r.ms_per_edge, r.n_nan, "N/A", "N/A")
            else
                @printf("  %-18s | %8.3f | %5d | %7.3f | %3d | %7.4f | %6.4f\n",
                        r.solver, r.time_sec, r.n_edges, r.ms_per_edge, r.n_nan, r.mean_curv, r.std_curv)
            end
        end

        # Find fastest
        valid_results = filter(r -> r.n_nan == 0, results_for_config)
        if !isempty(valid_results)
            fastest = first(valid_results)
            println("\n  → Fastest: $(fastest.solver) ($(fastest.ms_per_edge) ms/edge)")
        end
    end
end

# Cross-language comparison
if has_python
    println("\n" * "="^80)
    println("Cross-Language Performance Comparison")
    println("="^80)

    for n in sizes
        println("\n" * "-"^80)
        println("Graph Size: n=$n")
        println("-"^80)

        # Get Julia orcml results (most comparable to Python implementations)
        julia_orcml = filter(r -> r.n == n && r.config == "orcml" && r.n_nan == 0, julia_results)
        python_n = filter(r -> r.n == n, python_results)

        if !isempty(julia_orcml) && !isempty(python_n)
            println("\n  Implementation      | Language | Time (s) | ms/edge | Mean κ")
            println("  " * "-"^64)

            # Julia results
            for r in julia_orcml
                @printf("  %-18s | Julia    | %8.3f | %7.3f | %7.4f\n",
                        "ManifoldANN-$(r.solver)", r.time_sec, r.ms_per_edge, r.mean_curv)
            end

            # Python results
            for r in python_n
                @printf("  %-18s | Python   | %8.3f | %7.3f | %7.4f\n",
                        r.implementation, r.time_sec, r.ms_per_edge, r.mean_curv)
            end

            # Compute speedups
            if !isempty(python_n)
                println("\n  Speedup vs Python:")
                for py_result in python_n
                    println("\n    vs $(py_result.implementation):")
                    for ju_result in julia_orcml
                        speedup = py_result.time_sec / ju_result.time_sec
                        @printf("      ManifoldANN-%-12s: %.2fx %s\n",
                                ju_result.solver, speedup,
                                speedup >= 1.0 ? "faster" : "slower")
                    end
                end
            end
        end
    end
end

# Recommendations
println("\n" * "="^80)
println("Recommendations")
println("="^80)

println("\n1. Best Julia Solver by Use Case:")

for config in ["original", "orcml"]
    println("\n  Configuration: $config")

    # Find fastest for each size
    for n in sizes
        results_for_n = filter(r -> r.n == n && r.config == config && r.n_nan == 0, julia_results)

        if !isempty(results_for_n)
            fastest = argmin([r.time_sec for r in results_for_n])
            best_result = results_for_n[fastest]

            @printf("    n=%4d: %-15s (%.3f ms/edge)\n",
                    n, best_result.solver, best_result.ms_per_edge)
        end
    end
end

println("\n2. Solver Reliability:")
# Check which solvers produced NaN values
nan_solvers = Set{String}()
for r in julia_results
    if r.n_nan > 0
        push!(nan_solvers, r.solver)
    end
end

if isempty(nan_solvers)
    println("  ✓ All solvers reliable (no NaN values)")
else
    println("  ⚠ Solvers with NaN issues: ", join(nan_solvers, ", "))
    println("  → Use NetworkSimplexSolver or LPReferenceSolver for production")
end

println("\n3. Configuration Trade-offs:")
println("  • 'original': Faster (directed graph, simple metrics)")
println("  • 'orcml': Slower but matches published research (undirected, geodesic)")

if has_python
    println("\n4. Julia vs Python:")

    # Average speedup across all sizes
    speedups = Float64[]
    for n in sizes
        julia_orcml = filter(r -> r.n == n && r.config == "orcml" && r.n_nan == 0, julia_results)
        python_n = filter(r -> r.n == n, python_results)

        if !isempty(julia_orcml) && !isempty(python_n)
            # Compare fastest Julia solver vs each Python implementation
            fastest_julia = minimum([r.time_sec for r in julia_orcml])

            for py_result in python_n
                push!(speedups, py_result.time_sec / fastest_julia)
            end
        end
    end

    if !isempty(speedups)
        avg_speedup = mean(speedups)
        @printf("  • Average speedup: %.2fx\n", avg_speedup)

        if avg_speedup >= 2.0
            println("  → Julia significantly faster")
        elseif avg_speedup >= 1.2
            println("  → Julia moderately faster")
        else
            println("  → Julia and Python comparable")
        end
    end
end

println("\n" * "="^80)
println("Analysis Complete!")
println("="^80)
