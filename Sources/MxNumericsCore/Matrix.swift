//
//  Matrix.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

/// A dense, two-dimensional matrix with value semantics.
///
/// `Matrix` stores scalar values in canonical row-major order and uses
/// copy-on-write storage internally. Element indexing is zero-based. The type
/// provides both tuple-style access, such as `a[i, j]`, and MATLAB-like chained
/// access, such as `a[i][j]`.
///
/// Transposes are represented as views by swapping strides. Operations that
/// require contiguous row-major storage can call ``rowMajorElements()`` to
/// materialize a dense array in logical row-major order.
public struct Matrix<Scalar: MatrixScalar>: Sendable {
    /// The matrix dimensions.
    public private(set) var shape: Shape
    var strides: Strides
    var offset: Int
    var buffer: MatrixBuffer<Scalar>

    /// The number of matrix rows.
    public var rows: Int { shape.rows }

    /// The number of matrix columns.
    public var columns: Int { shape.columns }

    /// The number of stored scalar entries, equal to `rows * columns`.
    public var count: Int { shape.count }

    /// A Boolean value indicating whether the matrix has equal row and column counts.
    public var isSquare: Bool { shape.isSquare }

    /// Creates a matrix by repeating a scalar value.
    ///
    /// - Parameters:
    ///   - rows: The row count. Must be nonnegative.
    ///   - columns: The column count. Must be nonnegative.
    ///   - value: The value used for every entry. The default is zero.
    /// - Precondition: `rows >= 0` and `columns >= 0`.
    public init(rows: Int, columns: Int, repeating value: Scalar = .zero) {
        precondition(rows >= 0 && columns >= 0, "Matrix shape cannot be negative.")
        self.shape = try! Shape(rows: rows, columns: columns)
        self.strides = Strides(row: columns, column: 1)
        self.offset = 0
        self.buffer = MatrixBuffer(Array(repeating: value, count: rows * columns))
    }

    /// Creates a matrix from row-major storage.
    ///
    /// The first `columns` values form row 0, the next `columns` values form row
    /// 1, and so on.
    ///
    /// - Parameters:
    ///   - elements: Scalar entries in row-major order.
    ///   - rows: The row count.
    ///   - columns: The column count.
    /// - Throws: ``LinAlgError/invalidShape(rows:columns:)`` if a dimension is
    ///   negative, or ``LinAlgError/dimensionMismatch(_:)`` if `elements.count`
    ///   does not equal `rows * columns`.
    public init(rowMajor elements: [Scalar], rows: Int, columns: Int) throws {
        let shape = try Shape(rows: rows, columns: columns)
        guard elements.count == shape.count else {
            throw LinAlgError.dimensionMismatch("Expected \(shape.count) elements, received \(elements.count).")
        }
        self.shape = shape
        self.strides = Strides(row: columns, column: 1)
        self.offset = 0
        self.buffer = MatrixBuffer(elements)
    }

    /// Creates a matrix from nested row arrays.
    ///
    /// - Parameter rows: An array whose elements represent matrix rows.
    /// - Throws: ``LinAlgError/emptyInput`` if `rows` is empty, or
    ///   ``LinAlgError/raggedRows`` if the rows do not all have the same length.
    public init(_ rows: [[Scalar]]) throws {
        guard let first = rows.first else { throw LinAlgError.emptyInput }
        let columnCount = first.count
        guard rows.allSatisfy({ $0.count == columnCount }) else { throw LinAlgError.raggedRows }
        self.shape = try Shape(rows: rows.count, columns: columnCount)
        self.strides = Strides(row: columnCount, column: 1)
        self.offset = 0
        self.buffer = MatrixBuffer(rows.flatMap { $0 })
    }

    /// Creates a matrix filled with zeros.
    ///
    /// - Parameters:
    ///   - rows: The row count.
    ///   - columns: The column count.
    /// - Returns: A `rows` by `columns` matrix whose entries are all zero.
    public static func zeros(_ rows: Int, _ columns: Int) -> Matrix {
        Matrix(rows: rows, columns: columns, repeating: .zero)
    }

    /// Creates a matrix filled with ones.
    ///
    /// - Parameters:
    ///   - rows: The row count.
    ///   - columns: The column count.
    /// - Returns: A `rows` by `columns` matrix whose entries are all one.
    public static func ones(_ rows: Int, _ columns: Int) -> Matrix {
        Matrix(rows: rows, columns: columns, repeating: .one)
    }

    /// Creates an identity matrix.
    ///
    /// The returned matrix has ones on the main diagonal and zeros elsewhere.
    ///
    /// - Parameter n: The matrix order.
    /// - Returns: The `n` by `n` identity matrix.
    public static func eye(_ n: Int) -> Matrix {
        var matrix = Matrix(rows: n, columns: n, repeating: .zero)
        for index in 0..<n {
            matrix[index, index] = .one
        }
        return matrix
    }

    /// Creates a diagonal matrix from a vector.
    ///
    /// - Parameter diagonal: The values placed on the main diagonal.
    /// - Returns: A square matrix with `diagonal[i]` at entry `(i, i)` and zero
    ///   elsewhere.
    public static func diag(_ diagonal: Vector<Scalar>) -> Matrix {
        var matrix = Matrix(rows: diagonal.count, columns: diagonal.count, repeating: .zero)
        for index in 0..<diagonal.count {
            matrix[index, index] = diagonal[index]
        }
        return matrix
    }

    /// Accesses an element by row and column.
    ///
    /// Matrix indices are zero-based. Mutation uses copy-on-write semantics, so
    /// assigning through this subscript never mutates another independent matrix
    /// value that shares the same storage.
    ///
    /// - Parameters:
    ///   - row: The row index.
    ///   - column: The column index.
    /// - Precondition: `row` and `column` are inside the matrix bounds.
    public subscript(_ row: Int, _ column: Int) -> Scalar {
        _read {
            yield buffer.elements[linearIndex(row, column)]
        }
        _modify {
            ensureUnique()
            let index = linearIndex(row, column)
            yield &buffer.elements[index]
        }
    }

    /// Accesses a row view for chained element indexing.
    ///
    /// This subscript enables `matrix[i][j]` syntax. Assigning through the
    /// returned row view writes back into the matrix and preserves value
    /// semantics through copy-on-write storage.
    ///
    /// - Parameter row: The row index.
    /// - Precondition: `row` is inside the matrix bounds.
    public subscript(_ row: Int) -> RowView<Scalar> {
        _read {
            precondition(row >= 0 && row < rows, "Row index \(row) is out of bounds.")
            yield RowView(buffer: buffer, offset: offset + row * strides.row, stride: strides.column, count: columns)
        }
        _modify {
            precondition(row >= 0 && row < rows, "Row index \(row) is out of bounds.")
            ensureUnique()
            var view = RowView(buffer: buffer, offset: offset + row * strides.row, stride: strides.column, count: columns)
            yield &view
        }
    }

    /// Returns a copied submatrix selected by closed row and column ranges.
    public subscript(_ rowRange: ClosedRange<Int>, _ columnRange: ClosedRange<Int>) -> Matrix {
        copiedRows(Array(rowRange), columns: Array(columnRange))
    }

    /// Returns a copied submatrix selected by half-open row and column ranges.
    public subscript(_ rowRange: Range<Int>, _ columnRange: Range<Int>) -> Matrix {
        copiedRows(Array(rowRange), columns: Array(columnRange))
    }

    /// Returns a copied row matrix with shape `1 x columns`.
    public subscript(_ row: Int, _ columns: MatrixSlice) -> Matrix {
        switch columns {
        case .all:
            copiedRows([row], columns: Array(0..<self.columns))
        }
    }

    /// Returns a copied column matrix with shape `rows x 1`.
    public subscript(_ rows: MatrixSlice, _ column: Int) -> Matrix {
        switch rows {
        case .all:
            copiedRows(Array(0..<self.rows), columns: [column])
        }
    }

    /// Returns a copied submatrix for a closed row range and all columns.
    public subscript(_ rowRange: ClosedRange<Int>, _ columns: MatrixSlice) -> Matrix {
        switch columns {
        case .all:
            copiedRows(Array(rowRange), columns: Array(0..<self.columns))
        }
    }

    /// Returns a copied submatrix for all rows and a closed column range.
    public subscript(_ rows: MatrixSlice, _ columnRange: ClosedRange<Int>) -> Matrix {
        switch rows {
        case .all:
            copiedRows(Array(0..<self.rows), columns: Array(columnRange))
        }
    }

    /// The transpose view of the matrix.
    ///
    /// The transpose has shape `columns x rows` and entry `(i, j)` equal to the
    /// original entry `(j, i)`. This property swaps strides and shares storage;
    /// it does not immediately copy scalar values.
    public var t: Matrix {
        Matrix(shape: try! Shape(rows: columns, columns: rows), strides: Strides(row: strides.column, column: strides.row), offset: offset, buffer: buffer)
    }

    /// The transpose view of the matrix.
    public var transposed: Matrix { t }

    /// Returns a logical row-major copy of the matrix entries.
    ///
    /// For a transpose view, the returned array follows the transposed logical
    /// order rather than the physical order of the shared buffer.
    public func rowMajorElements() -> [Scalar] {
        var result: [Scalar] = []
        result.reserveCapacity(count)
        for row in 0..<rows {
            for column in 0..<columns {
                result.append(self[row, column])
            }
        }
        return result
    }

    mutating func ensureUnique() {
        if !isKnownUniquelyReferenced(&buffer) {
            buffer = buffer.copy()
        }
    }

    func linearIndex(_ row: Int, _ column: Int) -> Int {
        precondition(row >= 0 && row < rows, "Row index \(row) is out of bounds.")
        precondition(column >= 0 && column < columns, "Column index \(column) is out of bounds.")
        return offset + row * strides.row + column * strides.column
    }

    func copiedRows(_ selectedRows: [Int], columns selectedColumns: [Int]) -> Matrix {
        var elements: [Scalar] = []
        elements.reserveCapacity(selectedRows.count * selectedColumns.count)
        for row in selectedRows {
            for column in selectedColumns {
                elements.append(self[row, column])
            }
        }
        return try! Matrix(rowMajor: elements, rows: selectedRows.count, columns: selectedColumns.count)
    }

    init(shape: Shape, strides: Strides, offset: Int, buffer: MatrixBuffer<Scalar>) {
        self.shape = shape
        self.strides = strides
        self.offset = offset
        self.buffer = buffer
    }
}

extension Matrix: Equatable where Scalar: Equatable {
    public static func == (lhs: Matrix, rhs: Matrix) -> Bool {
        lhs.shape == rhs.shape && lhs.rowMajorElements() == rhs.rowMajorElements()
    }
}

/// A mutable view over one matrix row.
///
/// Row views support the chained syntax `matrix[i][j]`. They are short-lived
/// accessors over matrix storage rather than standalone arrays.
public struct RowView<Scalar: MatrixScalar>: Sendable {
    var buffer: MatrixBuffer<Scalar>
    let offset: Int
    let stride: Int
    /// The number of elements in the row.
    public let count: Int

    /// Accesses an element in the row.
    ///
    /// - Parameter column: The zero-based column index.
    /// - Precondition: `column` is inside the row bounds.
    public subscript(_ column: Int) -> Scalar {
        _read {
            precondition(column >= 0 && column < count, "Column index \(column) is out of bounds.")
            yield buffer.elements[offset + column * stride]
        }
        _modify {
            precondition(column >= 0 && column < count, "Column index \(column) is out of bounds.")
            yield &buffer.elements[offset + column * stride]
        }
    }
}

extension Matrix where Scalar: FloatingScalar {
    /// The Frobenius norm of the matrix.
    ///
    /// The Frobenius norm is `sqrt(sum(abs(a_ij)^2))`. This implementation uses
    /// a direct sum of squared magnitudes, which is mathematically correct for
    /// ordinary inputs but is not yet the scaled BLAS `nrm2` algorithm planned
    /// for backend kernels. Extremely large or small entries can therefore
    /// overflow or underflow in the intermediate sum.
    public var frobeniusNorm: Scalar.Magnitude {
        var sum = Scalar.Magnitude.zero
        for value in rowMajorElements() {
            sum += value.magnitude * value.magnitude
        }
        return Scalar.Magnitude.sqrt(sum)
    }

    /// The trace of a square matrix.
    ///
    /// The trace is the sum of the main diagonal entries, `sum(a_ii)`.
    ///
    /// - Precondition: The matrix is square.
    public var trace: Scalar {
        precondition(isSquare, "Trace requires a square matrix.")
        var result = Scalar.zero
        for index in 0..<rows {
            result += self[index, index]
        }
        return result
    }
}

extension Matrix where Scalar: BinaryFloatingPoint, Scalar.RawSignificand: FixedWidthInteger {
    /// Creates a matrix with uniformly distributed pseudorandom entries.
    ///
    /// Each entry is sampled independently from `range` using Swift's
    /// `SystemRandomNumberGenerator`. This initializer is intended for examples,
    /// tests, and exploratory work; it is not a reproducible statistical random
    /// stream.
    ///
    /// - Parameters:
    ///   - rows: The row count.
    ///   - columns: The column count.
    ///   - range: The half-open sampling range. The default is `0..<1`.
    /// - Returns: A random dense matrix.
    public static func rand(_ rows: Int, _ columns: Int, in range: Range<Scalar> = 0..<1) -> Matrix {
        var generator = SystemRandomNumberGenerator()
        let elements = (0..<(rows * columns)).map { _ in
            let unit = Scalar.random(in: 0..<1, using: &generator)
            return range.lowerBound + (range.upperBound - range.lowerBound) * unit
        }
        return try! Matrix(rowMajor: elements, rows: rows, columns: columns)
    }
}
