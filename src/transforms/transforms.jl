"""
    Transforms

Data transformation module for multi-level indexing.

This module provides transform abstractions for hierarchical ANN indices:
- Transform interface (AbstractTransform, TransformResult)
- Identity transform (pass-through)
- KMeans transform (IVF-style bucketing)
- Utility functions for partitioning and batch transformations

# Exports
- `AbstractTransform`, `TransformResult`
- `IdentityTransform`
- `KMeansTransform`, `KMeansAssignment`
- `fit!`, `transform`
- `has_bucketing`, `get_bucket_assignment`
- `preserves_data`
- `partition_by_transform`, `apply_transform_batch`
"""

using Distances

# Core interface
include("Transform.jl")
export AbstractTransform, TransformResult
export fit!, transform

# Concrete transforms
include("IdentityTransform.jl")
export IdentityTransform

include("KMeansTransform.jl")
export KMeansTransform, KMeansAssignment

# Utilities
include("utils.jl")
export has_bucketing, get_bucket_assignment, preserves_data
export partition_by_transform, apply_transform_batch
