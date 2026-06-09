//
//  Errors.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

/// Errors reported by MxNumerics linear algebra operations.
///
/// The cases describe structural failures, such as incompatible dimensions, and
/// numerical failures, such as singular or non-positive-definite matrices. Use
/// these errors for recoverable validation paths; bounds checks that indicate a
/// programmer error are currently enforced with preconditions.
public enum LinAlgError: Error, Equatable, Sendable {
    /// The operation expected at least one row or element.
    case emptyInput

    /// A nested array initializer received rows with different lengths.
    case raggedRows

    /// A shape contains a negative dimension or does not match its storage.
    /// - Parameters:
    ///   - rows: The requested row count.
    ///   - columns: The requested column count.
    case invalidShape(rows: Int, columns: Int)

    /// The operands have incompatible dimensions.
    /// - Parameter message: A human-readable description of the mismatch.
    case dimensionMismatch(String)

    /// The requested element index is outside the matrix bounds.
    /// - Parameters:
    ///   - row: The requested row index.
    ///   - column: The requested column index.
    case indexOutOfBounds(row: Int, column: Int)

    /// The matrix is singular, or numerically singular for the requested operation.
    case singularMatrix

    /// The operation requires a square matrix.
    case notSquare

    /// The operation requires a positive-definite matrix.
    case notPositiveDefinite

    /// The selected backend or scalar type does not support the operation yet.
    /// - Parameter message: A human-readable explanation of the unsupported path.
    case unsupported(String)
}
