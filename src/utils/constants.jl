"""
Default parameter constants for ANN indices.

This module defines default values for various algorithm parameters along with
their rationale. These constants improve code clarity and make it easier to
tune parameters globally.
"""

# HNSW Index Defaults
# ===================

"""
Default maximum number of bidirectional links per HNSW node.

From the original HNSW paper (Malkov & Yashunin, 2018), M=16 provides a good
balance between recall and memory usage. Lower values (M=8) reduce memory but
may decrease recall; higher values (M=32) improve recall but increase memory.

See: https://arxiv.org/abs/1603.09320
"""
const HNSW_DEFAULT_M = 16

"""
Default size of the dynamic candidate list during HNSW construction.

Higher values improve graph quality but slow down construction. The original
paper recommends ef_construction ≥ 100 for good recall. 200 provides high
quality graphs for most datasets.
"""
const HNSW_DEFAULT_EF_CONSTRUCTION = 200

"""
Default size of the dynamic candidate list during HNSW search.

Controls the recall-speed tradeoff at query time. Values around M to 4M work
well in practice. ef_search = 64 gives ~95% recall for typical datasets.
"""
const HNSW_DEFAULT_EF_SEARCH = 64

"""
Normalization factor for HNSW layer probability calculation.

The layer selection probability is computed as ml = ML_FACTOR / log(max(M, 2)).
The value 1.0 comes from the original HNSW paper's exponential distribution
for layer assignment.
"""
const HNSW_ML_NORMALIZATION_FACTOR = 1.0

# NN-Descent Index Defaults
# =========================

"""
Default number of neighbors per node in NN-Descent graph construction.

k=32 provides a good balance between graph quality and construction time for
most datasets. For high-dimensional data, larger values (64-100) may improve
recall.
"""
const NNDESCENT_DEFAULT_K = 32

"""
Default maximum NN-Descent iterations.

Most datasets converge within 5-10 iterations. Setting this to 10 provides
good results while avoiding excessive computation.
"""
const NNDESCENT_DEFAULT_MAX_ITERATIONS = 10

"""
Default convergence threshold for NN-Descent.

Algorithm stops when relative improvement < 0.001 (0.1%). This balances
convergence quality with computational cost.
"""
const NNDESCENT_DEFAULT_CONVERGENCE_THRESHOLD = 1e-3

"""
Default pruning-degree multiplier for NN-Descent's local-join sampling.

Per-iteration candidate set per node is sampled from `B[v] ∪ R[v]` (forward
∪ reverse neighbors) and capped at `ceil(pruning_degree_multiplier × k)`.
PyNNDescent uses 1.5 as its default; this gives a good balance between
recall (the larger sample lets reverse-neighbor sampling discover better
edges) and build cost (the local-join inner loop is O(|sample|²) per node).

Higher values (2.0-3.0) improve recall further at the cost of substantially
slower build (the cost is roughly quadratic in the multiplier).
"""
const NNDESCENT_DEFAULT_PRUNING_DEGREE_MULTIPLIER = 1.5

# LSH Index Defaults
# ==================

"""
Default hash length for LSH hash functions.

16 bits provides a good balance between bucket granularity and collision rate.
Longer hashes (32+) reduce collisions but may create too many empty buckets.
"""
const LSH_DEFAULT_HASH_LENGTH = 16

"""
Default number of LSH tables for multi-table hashing.

8 tables provides good recall for most datasets. More tables improve recall
but increase memory and query time.
"""
const LSH_DEFAULT_N_TABLES = 8

"""
Expected number of candidates per LSH table (for sizing hints).

Used for pre-allocating candidate sets. 32 is a reasonable estimate based on
typical bucket sizes.
"""
const LSH_EXPECTED_CANDIDATES_PER_TABLE = 32

# KMeans Transform Defaults
# =========================

"""
Default maximum iterations for Lloyd's K-means algorithm.

100 iterations is usually sufficient for convergence on most datasets.
Early stopping via tolerance typically kicks in before reaching this limit.
"""
const KMEANS_DEFAULT_MAX_ITERATIONS = 100

"""
Default convergence tolerance for K-means.

Algorithm stops when centroid movement < 1e-6. This provides good convergence
while avoiding excessive iterations on nearly-converged solutions.
"""
const KMEANS_DEFAULT_TOLERANCE = 1e-6

# Multi-Level Index Defaults
# ==========================

"""
Default number of clusters for IVF (Inverted File) indices.

For datasets of ~100K-1M points, 100-1000 clusters work well. Rule of thumb:
sqrt(n) to n/100, where n is the dataset size.
"""
const IVF_DEFAULT_NLIST = 100

"""
Default number of clusters to probe in IVF search.

Probing 5 clusters provides 85-90% recall for typical distributions. Increase
for higher recall at the cost of query time.
"""
const IVF_DEFAULT_NPROBE = 5

# Hash Function Limits
# ====================

"""
Maximum hash bits supported by LSH functions.

Limited to 64 bits to fit in UInt64 for efficient hashing and storage.
"""
const MAX_HASH_BITS = 64
