# ADR-0011: Optimal Transport Solver Refactoring

## Context

The original Ollivier-Ricci curvature (ORC) implementation used custom solvers
(`FastMatchingSolver`, `BruteForceSolver`) with wrapper overhead around external
packages. Performance profiling revealed that the Hungarian algorithm wrapper
was 9x slower than calling `Hungarian.jl` directly. Additionally, the solver
abstraction mixed domain concepts ("curvature solver") with the underlying
algorithm (optimal transport).

Upcoming work to replicate the orcml approach (ICLR 2025) requires supporting:
1. Multiple OT solvers (Hungarian, Sinkhorn, Network Simplex, LP)
2. Approximate vs exact OT methods
3. Solver selection based on distribution properties

## Decision

1. **Rename Abstraction**: `AbstractCurvatureSolver` → `AbstractOTSolver`
   - Reflects that these solve optimal transport problems, not curvature directly
   - Aligns with academic literature and external packages
   - Curvature is computed using OT as a subroutine

2. **Direct Package Integration**:
   - `HungarianSolver`: Call `Hungarian.jl` directly (9x speedup)
   - `SinkhornSolver`: Use `OptimalTransport.jl` sinkhorn2()
   - `NetworkSimplexSolver`: Use `OptimalTransport.jl` emd2() with Tulip backend
   - `LPReferenceSolver`: Use `HiGHS` for debugging/reference
   - `GreedySolver`: Keep custom implementation for maximum speed

3. **Solver Capabilities**:
   - `can_handle(solver, edge_view)`: Determine if solver supports edge structure
   - Fallback solver system: Primary solver → fallback if requirements not met
   - Each solver documents its requirements (uniform distributions, equal sizes, etc.)

4. **Deprecate Old Names**:
   - Remove `FastMatchingSolver` (use `HungarianSolver`)
   - Remove `BruteForceSolver` (O(k!) is impractical for k > 10)
   - Keep `GenericOTSolver` as convenience wrapper

## Consequences

**Benefits**:
- **9x speedup** for Hungarian algorithm (measured)
- Access to community-maintained OT implementations
- Clear separation: curvature computation vs OT solving
- Easier to add new solvers (just implement `AbstractOTSolver` interface)
- Better alignment with optimal transport literature

**Migration**:
- Tests updated to use new solver names
- Benchmarks compare multiple solvers
- Documentation clarifies OT vs curvature relationship

**Future Work**:
- Consider adding GPU-accelerated OT solvers (e.g., CuPy integration)
- Explore entropic regularization variants for large k
- Profile memory usage for different solver choices

## References

- Hungarian.jl: https://github.com/Gnimuc/Hungarian.jl
- OptimalTransport.jl: https://github.com/JuliaOptimalTransport/OptimalTransport.jl
- orcml: https://github.com/TristanSaidi/orcml
- Tulip (network simplex): https://github.com/ds4dm/Tulip.jl
