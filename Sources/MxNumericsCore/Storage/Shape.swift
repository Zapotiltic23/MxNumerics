//
//  Shape.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

/// The dimensions of a dense matrix.
///
/// A shape is valid when both dimensions are nonnegative. MxNumerics permits
/// empty dimensions, which makes 0-by-n, n-by-0, and 0-by-0 matrices representable
/// in the type system for algorithms that need degenerate cases.
public struct Shape: Equatable, Sendable {
    /// The number of matrix rows.
    public let rows: Int

    /// The number of matrix columns.
    public let columns: Int

    /// Creates a shape with the specified dimensions.
    ///
    /// - Parameters:
    ///   - rows: The row count. Must be nonnegative.
    ///   - columns: The column count. Must be nonnegative.
    /// - Throws: ``LinAlgError/invalidShape(rows:columns:)`` if either
    ///   dimension is negative.
    public init(rows: Int, columns: Int) throws {
        guard rows >= 0, columns >= 0 else {
            throw LinAlgError.invalidShape(rows: rows, columns: columns)
        }
        self.rows = rows
        self.columns = columns
    }

    /// The total number of scalar entries in a dense matrix with this shape.
    public var count: Int { rows * columns }

    /// A Boolean value indicating whether the matrix has the same number of rows
    /// and columns.
    public var isSquare: Bool { rows == columns }
}

/// Row and column strides for indexing into a matrix buffer.
///
/// Strides are measured in scalar elements rather than bytes. Contiguous
/// row-major storage has `row == columns` and `column == 1`; a transpose view
/// swaps these values.
public struct Strides: Equatable, Sendable {
    /// The element-distance between adjacent rows.
    public let row: Int

    /// The element-distance between adjacent columns.
    public let column: Int

    /// Creates a stride pair.
    ///
    /// - Parameters:
    ///   - row: The element-distance between adjacent rows.
    ///   - column: The element-distance between adjacent columns.
    public init(row: Int, column: Int) {
        self.row = row
        self.column = column
    }
}
