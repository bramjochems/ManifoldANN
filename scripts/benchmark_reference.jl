using Pkg
Pkg.add("NearestNeighborDescent")
using NearestNeighborDescent
using Random

Random.seed!(42)
data = randn(Float32, 784, 10000)

println("Building NearestNeighborDescent.jl index (k=32)...")
@time graph = nndescent(data, 32, NearestNeighborDescent.SqEuclidean(); max_iters=10, sample_rate=0.5)
println("Completed!")
