public enum LinAlgError: Error, Equatable, Sendable {
    case emptyInput
    case raggedRows
    case invalidShape(rows: Int, columns: Int)
    case dimensionMismatch(String)
    case indexOutOfBounds(row: Int, column: Int)
    case singularMatrix
    case notSquare
    case notPositiveDefinite
    case unsupported(String)
}
