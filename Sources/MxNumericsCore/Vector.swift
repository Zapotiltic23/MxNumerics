public struct Vector<Scalar: MatrixScalar>: Sendable, Equatable where Scalar: Equatable {
    public private(set) var matrix: Matrix<Scalar>

    public var count: Int { matrix.rows }

    public init(_ elements: [Scalar]) {
        self.matrix = try! Matrix(rowMajor: elements, rows: elements.count, columns: 1)
    }

    public init(column matrix: Matrix<Scalar>) {
        precondition(matrix.columns == 1, "Vector column initializer requires an n x 1 matrix.")
        self.matrix = matrix
    }

    public subscript(_ index: Int) -> Scalar {
        _read {
            yield matrix[index, 0]
        }
        _modify {
            yield &matrix[index, 0]
        }
    }

    public func asColumnMatrix() -> Matrix<Scalar> {
        matrix
    }
}

extension Vector where Scalar: FloatingScalar {
    public func dot(_ other: Vector<Scalar>) -> Scalar {
        precondition(count == other.count, "Dot product requires equal vector lengths.")
        var result = Scalar.zero
        for index in 0..<count {
            result += self[index] * other[index]
        }
        return result
    }

    public var norm: Scalar.Magnitude {
        var sum = Scalar.Magnitude.zero
        for index in 0..<count {
            let magnitude = self[index].magnitude
            sum += magnitude * magnitude
        }
        return Scalar.Magnitude.sqrt(sum)
    }
}
