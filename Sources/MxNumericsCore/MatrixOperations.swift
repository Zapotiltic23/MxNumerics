public func + <S: MatrixScalar>(lhs: Matrix<S>, rhs: Matrix<S>) -> Matrix<S> {
    precondition(lhs.shape == rhs.shape, "Matrix addition requires equal shapes.")
    let elements = zip(lhs.rowMajorElements(), rhs.rowMajorElements()).map(+)
    return try! Matrix(rowMajor: elements, rows: lhs.rows, columns: lhs.columns)
}

public func - <S: MatrixScalar>(lhs: Matrix<S>, rhs: Matrix<S>) -> Matrix<S> {
    precondition(lhs.shape == rhs.shape, "Matrix subtraction requires equal shapes.")
    let elements = zip(lhs.rowMajorElements(), rhs.rowMajorElements()).map(-)
    return try! Matrix(rowMajor: elements, rows: lhs.rows, columns: lhs.columns)
}

public func * <S: MatrixScalar>(lhs: Matrix<S>, rhs: Matrix<S>) -> Matrix<S> {
    precondition(lhs.columns == rhs.rows, "Matrix multiplication dimension mismatch.")
    var result = Matrix<S>(rows: lhs.rows, columns: rhs.columns, repeating: .zero)
    for i in 0..<lhs.rows {
        for k in 0..<lhs.columns {
            let aik = lhs[i, k]
            guard aik != .zero else { continue }
            for j in 0..<rhs.columns {
                result[i, j] += aik * rhs[k, j]
            }
        }
    }
    return result
}

public func * <S: MatrixScalar>(lhs: S, rhs: Matrix<S>) -> Matrix<S> {
    rhs.map { lhs * $0 }
}

public func * <S: MatrixScalar>(lhs: Matrix<S>, rhs: S) -> Matrix<S> {
    lhs.map { $0 * rhs }
}

extension Matrix {
    public func map<T: MatrixScalar>(_ transform: (Scalar) throws -> T) rethrows -> Matrix<T> {
        let elements = try rowMajorElements().map(transform)
        return try! Matrix<T>(rowMajor: elements, rows: rows, columns: columns)
    }

    public func power(_ exponent: Int) -> Matrix {
        precondition(isSquare, "Matrix powers require a square matrix.")
        precondition(exponent >= 0, "Negative powers require inverse support.")
        var result = Matrix.eye(rows)
        var base = self
        var remaining = exponent
        while remaining > 0 {
            if remaining & 1 == 1 {
                result = result * base
            }
            remaining >>= 1
            if remaining > 0 {
                base = base * base
            }
        }
        return result
    }
}

infix operator ^^: MultiplicationPrecedence

public func ^^ <S: MatrixScalar>(lhs: Matrix<S>, rhs: Int) -> Matrix<S> {
    lhs.power(rhs)
}
