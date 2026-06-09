public struct Matrix<Scalar: MatrixScalar>: Sendable {
    public private(set) var shape: Shape
    var strides: Strides
    var offset: Int
    var buffer: MatrixBuffer<Scalar>

    public var rows: Int { shape.rows }
    public var columns: Int { shape.columns }
    public var count: Int { shape.count }
    public var isSquare: Bool { shape.isSquare }

    public init(rows: Int, columns: Int, repeating value: Scalar = .zero) {
        precondition(rows >= 0 && columns >= 0, "Matrix shape cannot be negative.")
        self.shape = try! Shape(rows: rows, columns: columns)
        self.strides = Strides(row: columns, column: 1)
        self.offset = 0
        self.buffer = MatrixBuffer(Array(repeating: value, count: rows * columns))
    }

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

    public init(_ rows: [[Scalar]]) throws {
        guard let first = rows.first else { throw LinAlgError.emptyInput }
        let columnCount = first.count
        guard rows.allSatisfy({ $0.count == columnCount }) else { throw LinAlgError.raggedRows }
        self.shape = try Shape(rows: rows.count, columns: columnCount)
        self.strides = Strides(row: columnCount, column: 1)
        self.offset = 0
        self.buffer = MatrixBuffer(rows.flatMap { $0 })
    }

    public static func zeros(_ rows: Int, _ columns: Int) -> Matrix {
        Matrix(rows: rows, columns: columns, repeating: .zero)
    }

    public static func ones(_ rows: Int, _ columns: Int) -> Matrix {
        Matrix(rows: rows, columns: columns, repeating: .one)
    }

    public static func eye(_ n: Int) -> Matrix {
        var matrix = Matrix(rows: n, columns: n, repeating: .zero)
        for index in 0..<n {
            matrix[index, index] = .one
        }
        return matrix
    }

    public static func diag(_ diagonal: Vector<Scalar>) -> Matrix {
        var matrix = Matrix(rows: diagonal.count, columns: diagonal.count, repeating: .zero)
        for index in 0..<diagonal.count {
            matrix[index, index] = diagonal[index]
        }
        return matrix
    }

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

    public subscript(_ rowRange: ClosedRange<Int>, _ columnRange: ClosedRange<Int>) -> Matrix {
        copiedRows(Array(rowRange), columns: Array(columnRange))
    }

    public subscript(_ rowRange: Range<Int>, _ columnRange: Range<Int>) -> Matrix {
        copiedRows(Array(rowRange), columns: Array(columnRange))
    }

    public subscript(_ row: Int, _ columns: MatrixSlice) -> Matrix {
        switch columns {
        case .all:
            copiedRows([row], columns: Array(0..<self.columns))
        }
    }

    public subscript(_ rows: MatrixSlice, _ column: Int) -> Matrix {
        switch rows {
        case .all:
            copiedRows(Array(0..<self.rows), columns: [column])
        }
    }

    public subscript(_ rowRange: ClosedRange<Int>, _ columns: MatrixSlice) -> Matrix {
        switch columns {
        case .all:
            copiedRows(Array(rowRange), columns: Array(0..<self.columns))
        }
    }

    public subscript(_ rows: MatrixSlice, _ columnRange: ClosedRange<Int>) -> Matrix {
        switch rows {
        case .all:
            copiedRows(Array(0..<self.rows), columns: Array(columnRange))
        }
    }

    public var t: Matrix {
        Matrix(shape: try! Shape(rows: columns, columns: rows), strides: Strides(row: strides.column, column: strides.row), offset: offset, buffer: buffer)
    }

    public var transposed: Matrix { t }

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

public struct RowView<Scalar: MatrixScalar>: Sendable {
    var buffer: MatrixBuffer<Scalar>
    let offset: Int
    let stride: Int
    public let count: Int

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
    public var frobeniusNorm: Scalar.Magnitude {
        var sum = Scalar.Magnitude.zero
        for value in rowMajorElements() {
            sum += value.magnitude * value.magnitude
        }
        return Scalar.Magnitude.sqrt(sum)
    }

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
    public static func rand(_ rows: Int, _ columns: Int, in range: Range<Scalar> = 0..<1) -> Matrix {
        var generator = SystemRandomNumberGenerator()
        let elements = (0..<(rows * columns)).map { _ in
            let unit = Scalar.random(in: 0..<1, using: &generator)
            return range.lowerBound + (range.upperBound - range.lowerBound) * unit
        }
        return try! Matrix(rowMajor: elements, rows: rows, columns: columns)
    }
}
