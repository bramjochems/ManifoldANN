# Agent Guidance

This project favors clear module boundaries, high code quality, and comprehensive testing. When contributing automated or manual assistance, adhere to the following principles:

1. **Keep files short and coherent**: follow the layout in `docs/design/ADR-0001-module-organization.md`, adding new code to the appropriate submodule (`indices/`, `graphs/`, `preprocessing/`, etc.) instead of growing monolithic files.
2. **Respect documented decisions**: review the ADRs in `docs/design/` before introducing structural changes; update or add ADRs when decisions evolve.
3. **Specialize through types**: use parametric structs with concrete type parameters so Julia can specialize aggressively, especially for BLAS-heavy routines and node metadata payloads.
4. **High test coverage**: aim for near-100% coverage in new or modified modules. Place tests under `test/`, exercising success paths, edge cases, and error handling.
5. **Property-based testing**: when algorithms have invariants (e.g. symmetry of distance matrices, monotonicity of shortest-path distances, stability of kNN memberships), add property-based tests using `Random` or dedicated packages to validate behavior under varied inputs.
6. **Test execution**: always run `make test` before handing off changes. If the environment prevents executing the test suite (e.g. Julia cannot create lockfiles), state this explicitly in your report and ask the user to run `make test` so results are still validated.
7. **Deterministic builders**: expose RNG seeds or deterministic construction modes so tests remain reproducible.
8. **Examples**: keep runnable examples in `docs/examples/`, organized by topic (e.g. `docs/examples/indices/ex01-...jl`). Scripts should work with `julia --project=..` and demonstrate the canonical API usage.
9. **Documentation first**: update relevant ADRs or module docstrings when making structural changes. Keep high-level design notes under `docs/design/`.
10. **No silent data capture**: indices must not store raw data matrices; always require the caller to supply data at query time, matching the interface defined in `src/ann_index.jl`.

When in doubt, ask for clarification before implementing large-scale changes.
