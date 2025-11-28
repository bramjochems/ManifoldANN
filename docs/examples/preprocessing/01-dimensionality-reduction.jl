#=
Example: Dimensionality Reduction with PCA and Random Projections

This example demonstrates how to use the preprocessing transforms to reduce
dimensionality before building ANN indices.
=#

using ManifoldANN
using LinearAlgebra
using Random

println("=== Dimensionality Reduction Example ===\n")

# Generate high-dimensional data with intrinsic low-dimensional structure
Random.seed!(42)
n_samples = 1000
ambient_dim = 500
intrinsic_dim = 10

# Create data with structure in first 10 dimensions + noise
println("Generating $(n_samples) samples in $(ambient_dim) dimensions")
println("True intrinsic dimension: $(intrinsic_dim)\n")

X_intrinsic = randn(intrinsic_dim, n_samples) .* (10.0 .- collect(1:intrinsic_dim))
X_noise = randn(ambient_dim - intrinsic_dim, n_samples) .* 0.1
X = vcat(X_intrinsic, X_noise)

# ============================================================================
# Example 1: PCA with automatic dimension selection
# ============================================================================

println("--- PCA with Automatic Dimension Selection ---")
pca_auto = PCATransform(variance_threshold=0.95)
fit!(pca_auto, X)

println("Dimensions retained: $(target_dimension(pca_auto))")
println("Variance ratios: ", round.(explained_variance_ratio(pca_auto)[1:5], digits=3))
println()

# Transform data
X_pca = hcat([transform(pca_auto, X[:, i]).data for i in 1:n_samples]...)
println("Original shape: $(size(X))")
println("PCA-reduced shape: $(size(X_pca))\n")

# ============================================================================
# Example 2: PCA with fixed dimension
# ============================================================================

println("--- PCA with Fixed Dimension (20) ---")
pca_fixed = PCATransform(target_dim=20)
fit!(pca_fixed, X)

X_pca_20 = hcat([transform(pca_fixed, X[:, i]).data for i in 1:n_samples]...)
println("Reduced shape: $(size(X_pca_20))")
println("Top 5 explained variance ratios: ", round.(explained_variance_ratio(pca_fixed)[1:5], digits=3))
println()

# ============================================================================
# Example 3: Random Projection (Johnson-Lindenstrauss)
# ============================================================================

println("--- Random Projection ---")

# Suggested dimension for 10% distortion
target_dim = suggested_dimension(n_samples, epsilon=0.1)
println("Suggested dimension for ε=0.1: $(target_dim)")

# Create random projection transform
rp = RandomProjectionTransform(target_dim=target_dim, projection_type=:gaussian)
fit!(rp, X)

X_rp = hcat([transform(rp, X[:, i]).data for i in 1:n_samples]...)
println("Random projection shape: $(size(X_rp))")

# Verify distance preservation (sample a few pairs)
println("\nDistance preservation check:")
for _ in 1:5
    i, j = rand(1:n_samples, 2)
    d_orig = norm(X[:, i] - X[:, j])
    d_proj = norm(X_rp[:, i] - X_rp[:, j])
    distortion = abs(d_proj - d_orig) / d_orig
    println("  Pair ($i, $j): distortion = $(round(distortion, digits=3))")
end
println()

# ============================================================================
# Example 4: Sparse Random Projection
# ============================================================================

println("--- Sparse Random Projection ---")

rp_sparse = RandomProjectionTransform(
    target_dim=100,
    projection_type=:sparse,
    density=1/3
)
fit!(rp_sparse, X)

# Check sparsity
n_nonzero = count(!iszero, rp_sparse.projection)
actual_density = n_nonzero / length(rp_sparse.projection)
println("Target density: 0.333")
println("Actual density: $(round(actual_density, digits=3))")

X_rp_sparse = hcat([transform(rp_sparse, X[:, i]).data for i in 1:n_samples]...)
println("Sparse projection shape: $(size(X_rp_sparse))\n")

# ============================================================================
# Example 5: Using reduced data with ANN index
# ============================================================================

println("--- Building ANN Index on Reduced Data ---")

# Build index on PCA-reduced data
index_pca = build_index(BruteForceIndex, X_pca_20)
println("Built BruteForceIndex on PCA-reduced data")

# Query
query_point_orig = X[:, 1]
query_point_pca = transform(pca_fixed, query_point_orig).data

neighbors_pca = query(index_pca, X_pca_20, query_point_pca, 10)
println("Found $(length(neighbors_pca)) neighbors")
println("Neighbor IDs: ", neighbor_ids(neighbors_pca)[1:5], "...")
println()

# ============================================================================
# Example 6: Reconstruction from PCA
# ============================================================================

println("--- PCA Reconstruction ---")

# Take a point, reduce it, then reconstruct
original = X[:, 1]
reduced = transform(pca_fixed, original).data
reconstructed = inverse_transform(pca_fixed, reduced)

reconstruction_error = norm(original - reconstructed)
relative_error = reconstruction_error / norm(original)

println("Reconstruction error: $(round(reconstruction_error, digits=3))")
println("Relative error: $(round(relative_error, digits=3))")
println()

println("=== Example Complete ===")
