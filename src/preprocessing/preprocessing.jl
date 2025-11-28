"""
Preprocessing Module

Provides dimensionality reduction transforms for preprocessing data before
building ANN indices or kNN graphs.

# Transforms

- `PCATransform`: Principal Component Analysis with three modes
- `RandomProjectionTransform`: Johnson-Lindenstrauss random projections

See individual transform files for details.
"""

include("PCATransform.jl")
include("RandomProjectionTransform.jl")
