# Union{Nothing, T} Usage Rationale

This document explains the rationale for `Union{Nothing, T}` usage in ManifoldANN and why it's appropriate in most cases.

## Overview

The codebase contains ~20 instances of `Union{Nothing, T}` types. After analysis, these are intentional design choices that improve API ergonomics without significant performance impact.

## Appropriate Use Cases

### 1. Optional Function Parameters

**Pattern:** Default parameter values that may be omitted
```julia
function query(index, data, q, k; ef_search::Union{Nothing,Int} = nothing)
    # Use index's default if nothing provided
    ef = ef_search === nothing ? index.traversal_policy.ef_search : ef_search
end
```

**Rationale:**
- Allows users to omit parameter entirely
- Alternative (sentinel value like -1) requires validation and is less clear
- Not in hot path (only checked once per query, not per operation)

**Examples:**
- `ef_search::Union{Nothing,Int}` - HNSW search parameter
- `candidate_cap::Union{Nothing,Int}` - LSH candidate limit
- `k::Union{Nothing,Integer}` - Graph construction parameter
- `metadata::Union{Nothing,AbstractVector}` - Optional node metadata

### 2. Unfitted vs Fitted State

**Pattern:** Transform state before/after fitting
```julia
mutable struct KMeansTransform
    centroids::Union{Nothing, Matrix{TC}}  # nothing before fit!, matrix after
end
```

**Rationale:**
- Clear distinction between unfitted and fitted state
- Attempting to use unfitted transform throws clear error
- Alternative (requiring separate types) adds complexity
- Not performance-critical (checked once during transform, not in tight loops)

**Examples:**
- `KMeansTransform.centroids`

### 3. Optional Data Structures

**Pattern:** Data that only exists for certain configurations
```julia
struct TransformedIndex
    id_mappings::Union{Nothing, Vector{Vector{Int}}}  # Only for bucketing transforms
    bucket_lookup::Union{Nothing, Vector{Int}}        # Only for bucketing transforms
end
```

**Rationale:**
- Bucketing transforms need ID mapping, non-bucketing don't
- Alternative (always allocating) wastes memory
- Alternative (separate types) duplicates code
- Access patterns check `isnothing()` once, not in tight loops

**Examples:**
- `TransformedIndex.id_mappings`
- `TransformedIndex.bucket_lookup`

### 4. Capability Introspection Returns

**Pattern:** Functions that query index capabilities
```julia
configured_k(index::AbstractANNIndex) -> Union{Int, Nothing}
index_distance(index::AbstractANNIndex) -> Union{Nothing, Function}
```

**Rationale:**
- Not all indices have a configured `k` value (LSH doesn't, HNSW does)
- Not all indices store distance function (LSH does, BruteForce doesn't always)
- Alternative (always returning a value) requires arbitrary defaults
- Callers can branch based on capability

**Examples:**
- `configured_k(index)` - Returns k for graph indices, nothing otherwise
- `index_distance(index)` - Returns distance if stored, nothing otherwise

### 5. Policy Resolution (Internal)

**Pattern:** Accept multiple types, resolve to concrete type internally
```julia
function build_index(
    ::Type{HNSWIndex},
    data;
    neighbor_policy::Union{AbstractNeighborPolicy,Symbol,Nothing} = nothing
)
    # Resolve to concrete policy
    policy = _resolve_neighbor_policy(neighbor_policy, M)
    # ... rest of function uses concrete type
end
```

**Rationale:**
- User-friendly API (accept symbol, type, or nothing for default)
- Resolved to concrete type immediately, so no runtime overhead
- Index stores concrete type, maintaining type stability

## Performance Considerations

### Where Union{Nothing, T} is Safe

1. **Function parameters** - Checked once at function entry, not in loops
2. **One-time initialization** - State checked during setup, not repeatedly
3. **Immediately resolved** - Union type converted to concrete type before hot path

### Where Union{Nothing, T} Could Be Problematic

1. **Return types in tight loops** - Can cause type instability if not handled carefully
2. **Struct fields accessed repeatedly** - If accessed in performance-critical loops

### Our Usage Analysis

**All instances are safe because:**
- Optional parameters are resolved at function entry (before hot paths)
- Struct fields with Union types are either:
  - Checked once during operation (not in loops)
  - Only present in setup/query dispatch logic (not tight loops)
- Return types from capability functions are used for dispatch/branching, not arithmetic

## Alternative Approaches Considered

### Sentinel Values (e.g., k = -1)

**Rejected because:**
- Less clear than `nothing`
- Requires validation ("is -1 valid?")
- Type system can't enforce checking

### Separate Types for States

**Rejected because:**
- Duplicates code (FittedKMeans vs UnfittedKMeans)
- More complex API
- Negligible performance benefit

### Multiple Dispatch Instead of Optional Fields

**Rejected because:**
- Would require many method variants
- Harder to maintain
- Current approach is more Julian

## Conclusion

The `Union{Nothing, T}` usage in ManifoldANN is **appropriate and intentional**. Each instance provides:
- Clearer API semantics
- Better error messages
- No measurable performance impact

## When to Avoid Union{Nothing, T}

Future contributors should avoid `Union{Nothing, T}` in these cases:
1. Return types for functions called in tight inner loops
2. Struct fields accessed repeatedly in performance-critical paths
3. When a sentinel value would be equally clear (rare)

## Type Stability Verification

Key hot paths have been verified for type stability:
- Distance computations (type-stable)
- Neighbor heap operations (type-stable)
- Query dispatch (resolves to concrete types before loops)

Use `@code_warntype` to verify new code doesn't introduce instabilities.
