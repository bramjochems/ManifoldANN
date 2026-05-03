"""
    validate_index_dimensions(index, data, q)

Ensure the provided `data` matrix and query vector `q` match the index
dimension and hold at least as many points as the index tracks. Returns
`nothing` on success; otherwise throws `DimensionMismatch` or `ArgumentError`.
"""
function validate_index_dimensions(index, data, q)
    size(data, 1) == index.dimension ||
        throw(DimensionMismatch("Expected data with $(index.dimension) rows"))
    length(q) == index.dimension ||
        throw(DimensionMismatch("Expected query length $(index.dimension)"))
    size(data, 2) >= index.n_points || throw(
        ArgumentError(
            "Data contains $(size(data, 2)) points but index tracks $(index.n_points)",
        ),
    )
    return nothing
end

"""
    validate_index_query_matrix(index, queries)

Matrix-input variant of [`validate_index_dimensions`](@ref): verifies that the
column dimension of `queries` matches `index.dimension`. Used by batch query
paths where each query is a column of the supplied matrix.
"""
function validate_index_query_matrix(index, queries)
    size(queries, 1) == index.dimension || throw(
        DimensionMismatch(
            "Expected queries with $(index.dimension) rows, got $(size(queries, 1))",
        ),
    )
    return nothing
end
