"""
Generate benchmark data for cross-language comparison.
Saves data in a format both Julia and Python can read.
"""

using Random
using DelimitedFiles

const SIZES = [100, 500, 1000]
const DIM = 50
const SEED = 42

println("="^80)
println("Generating Benchmark Data")
println("="^80)

for n in SIZES
    Random.seed!(SEED)
    data = randn(DIM, n)  # d x n

    # Save as CSV (transpose for Python which expects n x d)
    data_transposed = data'  # n x d

    filename = "benchmark_results/benchmark_data_n$n.csv"
    writedlm(filename, data_transposed, ',')

    println("✓ Generated n=$n: $(filename) ($(n) x $(DIM))")
end

println("\n" * "="^80)
println("Data generation complete!")
println("Files can be used by both Julia and Python benchmarks.")
println("="^80)
