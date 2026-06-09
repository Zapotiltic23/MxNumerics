//
//  MatrixOperations.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

/// Adds two matrices element by element.
///
/// - Parameters:
///   - lhs: The left matrix.
///   - rhs: The right matrix.
/// - Returns: A matrix whose `(i, j)` entry is `lhs[i, j] + rhs[i, j]`.
/// - Precondition: The matrices have identical shapes.
public func + <S: MatrixScalar>(lhs: Matrix<S>, rhs: Matrix<S>) -> Matrix<S> {
    precondition(lhs.shape == rhs.shape, "Matrix addition requires equal shapes.")
    let elements = zip(lhs.rowMajorElements(), rhs.rowMajorElements()).map(+)
    return try! Matrix(rowMajor: elements, rows: lhs.rows, columns: lhs.columns)
}

/// Subtracts two matrices element by element.
///
/// - Parameters:
///   - lhs: The left matrix.
///   - rhs: The right matrix.
/// - Returns: A matrix whose `(i, j)` entry is `lhs[i, j] - rhs[i, j]`.
/// - Precondition: The matrices have identical shapes.
public func - <S: MatrixScalar>(lhs: Matrix<S>, rhs: Matrix<S>) -> Matrix<S> {
    precondition(lhs.shape == rhs.shape, "Matrix subtraction requires equal shapes.")
    let elements = zip(lhs.rowMajorElements(), rhs.rowMajorElements()).map(-)
    return try! Matrix(rowMajor: elements, rows: lhs.rows, columns: lhs.columns)
}

/// Multiplies two matrices with the standard matrix product.
///
/// If `lhs` has shape `m x k` and `rhs` has shape `k x n`, the result has shape
/// `m x n` and entry `(i, j) = sum(lhs[i, p] * rhs[p, j])` over `p = 0..<k`.
/// This is the mathematical operation represented by BLAS GEMM with `alpha = 1`
/// and `beta = 0`, although this overload currently uses the in-core reference
/// loop rather than dispatching to a tuned backend.
///
/// - Parameters:
///   - lhs: The left matrix.
///   - rhs: The right matrix.
/// - Returns: The matrix product `lhs * rhs`.
/// - Precondition: `lhs.columns == rhs.rows`.
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

/// Multiplies every matrix entry by a scalar on the left.
public func * <S: MatrixScalar>(lhs: S, rhs: Matrix<S>) -> Matrix<S> {
    rhs.map { lhs * $0 }
}

/// Multiplies every matrix entry by a scalar on the right.
public func * <S: MatrixScalar>(lhs: Matrix<S>, rhs: S) -> Matrix<S> {
    lhs.map { $0 * rhs }
}

extension Matrix {
    /// Returns a matrix by transforming every entry.
    ///
    /// The result preserves the shape of the receiver. The transform is applied
    /// in logical row-major order.
    ///
    /// - Parameter transform: A closure that maps each scalar to a result scalar.
    /// - Returns: A matrix containing the transformed entries.
    public func map<T: MatrixScalar>(_ transform: (Scalar) throws -> T) rethrows -> Matrix<T> {
        let elements = try rowMajorElements().map(transform)
        return try! Matrix<T>(rowMajor: elements, rows: rows, columns: columns)
    }

    /// Raises a square matrix to a nonnegative integer power.
    ///
    /// The implementation uses exponentiation by squaring. `power(0)` returns
    /// the identity matrix with the same order as the receiver.
    ///
    /// - Parameter exponent: A nonnegative integer exponent.
    /// - Returns: The matrix power `self` raised to `exponent`.
    /// - Precondition: The matrix is square and `exponent >= 0`.
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

/// Matrix integer-power operator.
///
/// This operator is equivalent to calling ``Matrix/power(_:)``.
infix operator ^^: MultiplicationPrecedence

/// Raises a square matrix to a nonnegative integer power.
public func ^^ <S: MatrixScalar>(lhs: Matrix<S>, rhs: Int) -> Matrix<S> {
    lhs.power(rhs)
}

public func ^ <S: MatrixScalar>(lhs: Matrix<S>, rhs: Int) -> Matrix<S> {
    lhs.power(rhs)
}
