//
//  LAPACKKernels.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

import Accelerate
import MxNumericsCore

private typealias LAPACKInteger = Int32

/// The result of a LAPACK LU factorization.
///
/// LAPACK `getrf` computes a partial-pivoting factorization `P A = L U`.
/// The permutation is stored as zero-based row indices matching the public
/// routines in `MxNumerics`.
public struct LAPACKLUResult<Scalar: RealFloatingScalar>: Sendable {
    /// The unit lower-triangular factor.
    public let l: Matrix<Scalar>

    /// The upper-triangular factor.
    public let u: Matrix<Scalar>

    /// The zero-based permutation applied to the right-hand side.
    public let permutation: [Int]

    /// The sign of the permutation, either `1` or `-1`.
    public let parity: Int
}

/// The result of a LAPACK QR factorization.
///
/// The factorization is returned in economy form: `q` has orthonormal columns
/// and `r` has `min(m, n)` rows, so `q * r` reconstructs the original matrix.
public struct LAPACKQRResult<Scalar: RealFloatingScalar>: Sendable {
    /// The orthonormal factor.
    public let q: Matrix<Scalar>

    /// The upper-trapezoidal factor.
    public let r: Matrix<Scalar>
}

/// The result of a LAPACK singular value decomposition.
///
/// For the thin form, `u` is `m x min(m, n)` and `v` is `n x min(m, n)`.
/// For the full form, `u` is `m x m` and `v` is `n x n`.
public struct LAPACKSVDResult<Scalar: RealFloatingScalar>: Sendable {
    /// The left singular vectors.
    public let u: Matrix<Scalar>

    /// The singular values in descending order.
    public let singularValues: Vector<Scalar>

    /// The right singular vectors.
    public let v: Matrix<Scalar>
}

/// The result of a symmetric eigenvalue decomposition.
public struct LAPACKSymmetricEigenResult<Scalar: RealFloatingScalar>: Sendable {
    /// The eigenvalues in ascending order.
    public let values: Vector<Scalar>

    /// The orthonormal eigenvectors, stored as columns.
    public let vectors: Matrix<Scalar>
}

/// The result of a real, general eigenvalue decomposition.
///
/// LAPACK `geev` returns real and imaginary parts separately. For complex
/// conjugate pairs, `rightEigenvectorsPacked` follows LAPACK's packed real
/// convention: adjacent columns hold the real and imaginary parts.
public struct LAPACKGeneralEigenResult<Scalar: RealFloatingScalar>: Sendable {
    /// The real parts of the eigenvalues.
    public let realParts: Vector<Scalar>

    /// The imaginary parts of the eigenvalues.
    public let imaginaryParts: Vector<Scalar>

    /// The packed right eigenvectors returned by LAPACK.
    public let rightEigenvectorsPacked: Matrix<Scalar>
}

/// Synchronous wrappers around Accelerate's LAPACK routines.
///
/// These routines accept and return `Matrix` and `Vector` values in MxNumerics'
/// row-major value model. LAPACK's Fortran column-major storage is handled
/// internally with ``ColumnMajorBridge``.
public enum LAPACKKernels {
    /// Computes `P A = L U` with partial pivoting.
    public static func lu(_ matrix: Matrix<Double>) throws -> LAPACKLUResult<Double> {
        try luDouble(matrix)
    }

    /// Computes `P A = L U` with partial pivoting.
    public static func lu(_ matrix: Matrix<Float>) throws -> LAPACKLUResult<Float> {
        try luFloat(matrix)
    }

    /// Solves `A x = b` with LU factorization and partial pivoting.
    public static func solve(_ matrix: Matrix<Double>, _ b: Vector<Double>) throws -> Vector<Double> {
        let solution = try solve(matrix, b.asColumnMatrix())
        return Vector(solution.rowMajorElements())
    }

    /// Solves `A x = b` with LU factorization and partial pivoting.
    public static func solve(_ matrix: Matrix<Float>, _ b: Vector<Float>) throws -> Vector<Float> {
        let solution = try solve(matrix, b.asColumnMatrix())
        return Vector(solution.rowMajorElements())
    }

    /// Solves `A X = B` with one LU factorization and multiple right-hand sides.
    public static func solve(_ matrix: Matrix<Double>, _ rhs: Matrix<Double>) throws -> Matrix<Double> {
        try solveDouble(matrix, rhs)
    }

    /// Solves `A X = B` with one LU factorization and multiple right-hand sides.
    public static func solve(_ matrix: Matrix<Float>, _ rhs: Matrix<Float>) throws -> Matrix<Float> {
        try solveFloat(matrix, rhs)
    }

    /// Computes `A^-1` by applying `getrf` followed by `getri`.
    public static func inverse(_ matrix: Matrix<Double>) throws -> Matrix<Double> {
        try inverseDouble(matrix)
    }

    /// Computes `A^-1` by applying `getrf` followed by `getri`.
    public static func inverse(_ matrix: Matrix<Float>) throws -> Matrix<Float> {
        try inverseFloat(matrix)
    }

    /// Computes the determinant from the diagonal of a `getrf` factorization.
    public static func determinant(_ matrix: Matrix<Double>) throws -> Double {
        try determinantDouble(matrix)
    }

    /// Computes the determinant from the diagonal of a `getrf` factorization.
    public static func determinant(_ matrix: Matrix<Float>) throws -> Float {
        try determinantFloat(matrix)
    }

    /// Computes an economy QR factorization.
    public static func qr(_ matrix: Matrix<Double>) throws -> LAPACKQRResult<Double> {
        try qrDouble(matrix)
    }

    /// Computes an economy QR factorization.
    public static func qr(_ matrix: Matrix<Float>) throws -> LAPACKQRResult<Float> {
        try qrFloat(matrix)
    }

    /// Computes the lower Cholesky factor of a symmetric positive-definite matrix.
    public static func cholesky(_ matrix: Matrix<Double>) throws -> Matrix<Double> {
        try choleskyDouble(matrix)
    }

    /// Computes the lower Cholesky factor of a symmetric positive-definite matrix.
    public static func cholesky(_ matrix: Matrix<Float>) throws -> Matrix<Float> {
        try choleskyFloat(matrix)
    }

    /// Solves an SPD system from a Cholesky factorization.
    public static func solveSPD(_ matrix: Matrix<Double>, _ rhs: Matrix<Double>) throws -> Matrix<Double> {
        try solveSPDDouble(matrix, rhs)
    }

    /// Solves an SPD system from a Cholesky factorization.
    public static func solveSPD(_ matrix: Matrix<Float>, _ rhs: Matrix<Float>) throws -> Matrix<Float> {
        try solveSPDFloat(matrix, rhs)
    }

    /// Computes a singular value decomposition with `gesdd`.
    public static func svd(_ matrix: Matrix<Double>, fullVectors: Bool = false) throws -> LAPACKSVDResult<Double> {
        try svdDouble(matrix, fullVectors: fullVectors)
    }

    /// Computes a singular value decomposition with `gesdd`.
    public static func svd(_ matrix: Matrix<Float>, fullVectors: Bool = false) throws -> LAPACKSVDResult<Float> {
        try svdFloat(matrix, fullVectors: fullVectors)
    }

    /// Computes eigenvalues and eigenvectors of a real symmetric matrix.
    public static func symmetricEigen(_ matrix: Matrix<Double>) throws -> LAPACKSymmetricEigenResult<Double> {
        try symmetricEigenDouble(matrix)
    }

    /// Computes eigenvalues and eigenvectors of a real symmetric matrix.
    public static func symmetricEigen(_ matrix: Matrix<Float>) throws -> LAPACKSymmetricEigenResult<Float> {
        try symmetricEigenFloat(matrix)
    }

    /// Computes eigenvalues and right eigenvectors of a real general matrix.
    public static func eigen(_ matrix: Matrix<Double>) throws -> LAPACKGeneralEigenResult<Double> {
        try eigenDouble(matrix)
    }

    /// Computes eigenvalues and right eigenvectors of a real general matrix.
    public static func eigen(_ matrix: Matrix<Float>) throws -> LAPACKGeneralEigenResult<Float> {
        try eigenFloat(matrix)
    }

    /// Solves the ordinary least-squares problem `min_x ||A x - b||_2`.
    public static func leastSquares(_ matrix: Matrix<Double>, _ b: Vector<Double>) throws -> Vector<Double> {
        try leastSquaresDouble(matrix, b)
    }

    /// Solves the ordinary least-squares problem `min_x ||A x - b||_2`.
    public static func leastSquares(_ matrix: Matrix<Float>, _ b: Vector<Float>) throws -> Vector<Float> {
        try leastSquaresFloat(matrix, b)
    }

    /// Computes a matrix norm with LAPACK's scaled norm kernels.
    public static func norm(_ matrix: Matrix<Double>, kind: LAPACKMatrixNorm) -> Double {
        normDouble(matrix, kind: kind)
    }

    /// Computes a matrix norm with LAPACK's scaled norm kernels.
    public static func norm(_ matrix: Matrix<Float>, kind: LAPACKMatrixNorm) -> Float {
        normFloat(matrix, kind: kind)
    }

    /// Balances a square matrix with LAPACK `gebal`.
    public static func balance(_ matrix: Matrix<Double>) throws -> (balanced: Matrix<Double>, scales: Vector<Double>) {
        try balanceDouble(matrix)
    }

    /// Balances a square matrix with LAPACK `gebal`.
    public static func balance(_ matrix: Matrix<Float>) throws -> (balanced: Matrix<Float>, scales: Vector<Float>) {
        try balanceFloat(matrix)
    }
}

/// Matrix norm selectors supported by LAPACK `lange`.
public enum LAPACKMatrixNorm: Sendable {
    /// Maximum absolute column sum.
    case one

    /// Maximum absolute row sum.
    case infinity

    /// Frobenius norm using LAPACK's scaled sum of squares.
    case frobenius

    /// Maximum absolute entry.
    case maximum
}

private func luDouble(_ matrix: Matrix<Double>) throws -> LAPACKLUResult<Double> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var rows = n
    var columns = n
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var lda = max(n, 1)
    var pivots = [LAPACKInteger](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    dgetrf_(&rows, &columns, &a, &lda, &pivots, &info)
    try requireFactorizationSuccess(info)
    return try luResult(fromColumnMajorFactor: a, pivots: pivots, order: matrix.rows)
}

private func luFloat(_ matrix: Matrix<Float>) throws -> LAPACKLUResult<Float> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var rows = n
    var columns = n
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var lda = max(n, 1)
    var pivots = [LAPACKInteger](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    sgetrf_(&rows, &columns, &a, &lda, &pivots, &info)
    try requireFactorizationSuccess(info)
    return try luResult(fromColumnMajorFactor: a, pivots: pivots, order: matrix.rows)
}

private func solveDouble(_ matrix: Matrix<Double>, _ rhs: Matrix<Double>) throws -> Matrix<Double> {
    try requireSquare(matrix)
    guard matrix.rows == rhs.rows else {
        throw LinAlgError.dimensionMismatch("Solve requires A.rows == B.rows.")
    }
    let n = checkedLAPACKInt(matrix.rows)
    let nrhs = checkedLAPACKInt(rhs.columns)
    var rows = n
    var columns = n
    var rightHandSides = nrhs
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var b = ColumnMajorBridge.columnMajor(from: rhs)
    var lda = max(n, 1)
    var ldb = max(n, 1)
    var pivots = [LAPACKInteger](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    dgetrf_(&rows, &columns, &a, &lda, &pivots, &info)
    try requireFactorizationSuccess(info)
    var trans = lapackCharacter("N")
    var solveOrder = n
    dgetrs_(&trans, &solveOrder, &rightHandSides, &a, &lda, &pivots, &b, &ldb, &info)
    try requireSuccess(info)
    return try ColumnMajorBridge.rowMajor(fromColumnMajor: b, rows: rhs.rows, columns: rhs.columns)
}

private func solveFloat(_ matrix: Matrix<Float>, _ rhs: Matrix<Float>) throws -> Matrix<Float> {
    try requireSquare(matrix)
    guard matrix.rows == rhs.rows else {
        throw LinAlgError.dimensionMismatch("Solve requires A.rows == B.rows.")
    }
    let n = checkedLAPACKInt(matrix.rows)
    let nrhs = checkedLAPACKInt(rhs.columns)
    var rows = n
    var columns = n
    var rightHandSides = nrhs
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var b = ColumnMajorBridge.columnMajor(from: rhs)
    var lda = max(n, 1)
    var ldb = max(n, 1)
    var pivots = [LAPACKInteger](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    sgetrf_(&rows, &columns, &a, &lda, &pivots, &info)
    try requireFactorizationSuccess(info)
    var trans = lapackCharacter("N")
    var solveOrder = n
    sgetrs_(&trans, &solveOrder, &rightHandSides, &a, &lda, &pivots, &b, &ldb, &info)
    try requireSuccess(info)
    return try ColumnMajorBridge.rowMajor(fromColumnMajor: b, rows: rhs.rows, columns: rhs.columns)
}

private func inverseDouble(_ matrix: Matrix<Double>) throws -> Matrix<Double> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var rows = n
    var columns = n
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var lda = max(n, 1)
    var pivots = [LAPACKInteger](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    dgetrf_(&rows, &columns, &a, &lda, &pivots, &info)
    try requireFactorizationSuccess(info)
    var lwork: LAPACKInteger = -1
    var workQuery = [Double](repeating: 0, count: 1)
    var inverseOrder = n
    dgetri_(&inverseOrder, &a, &lda, &pivots, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Double](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    inverseOrder = n
    dgetri_(&inverseOrder, &a, &lda, &pivots, &work, &lwork, &info)
    try requireSuccess(info)
    return try ColumnMajorBridge.rowMajor(fromColumnMajor: a, rows: matrix.rows, columns: matrix.columns)
}

private func inverseFloat(_ matrix: Matrix<Float>) throws -> Matrix<Float> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var rows = n
    var columns = n
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var lda = max(n, 1)
    var pivots = [LAPACKInteger](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    sgetrf_(&rows, &columns, &a, &lda, &pivots, &info)
    try requireFactorizationSuccess(info)
    var lwork: LAPACKInteger = -1
    var workQuery = [Float](repeating: 0, count: 1)
    var inverseOrder = n
    sgetri_(&inverseOrder, &a, &lda, &pivots, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Float](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    inverseOrder = n
    sgetri_(&inverseOrder, &a, &lda, &pivots, &work, &lwork, &info)
    try requireSuccess(info)
    return try ColumnMajorBridge.rowMajor(fromColumnMajor: a, rows: matrix.rows, columns: matrix.columns)
}

private func determinantDouble(_ matrix: Matrix<Double>) throws -> Double {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var rows = n
    var columns = n
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var lda = max(n, 1)
    var pivots = [LAPACKInteger](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    dgetrf_(&rows, &columns, &a, &lda, &pivots, &info)
    try requireFactorizationSuccess(info)
    let (_, parity) = permutationAndParity(from: pivots, count: matrix.rows)
    var determinant = Double(parity)
    for index in 0..<matrix.rows {
        determinant *= a[index + index * matrix.rows]
    }
    return determinant
}

private func determinantFloat(_ matrix: Matrix<Float>) throws -> Float {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var rows = n
    var columns = n
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var lda = max(n, 1)
    var pivots = [LAPACKInteger](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    sgetrf_(&rows, &columns, &a, &lda, &pivots, &info)
    try requireFactorizationSuccess(info)
    let (_, parity) = permutationAndParity(from: pivots, count: matrix.rows)
    var determinant = Float(parity)
    for index in 0..<matrix.rows {
        determinant *= a[index + index * matrix.rows]
    }
    return determinant
}

private func qrDouble(_ matrix: Matrix<Double>) throws -> LAPACKQRResult<Double> {
    let m = checkedLAPACKInt(matrix.rows)
    let n = checkedLAPACKInt(matrix.columns)
    let k = min(matrix.rows, matrix.columns)
    guard k > 0 else {
        return LAPACKQRResult(q: Matrix(rows: matrix.rows, columns: 0), r: Matrix(rows: 0, columns: matrix.columns))
    }

    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var rows = m
    var columns = n
    var lda = max(m, 1)
    var tau = [Double](repeating: 0, count: k)
    var info: LAPACKInteger = 0
    var lwork: LAPACKInteger = -1
    var workQuery = [Double](repeating: 0, count: 1)
    dgeqrf_(&rows, &columns, &a, &lda, &tau, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Double](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    dgeqrf_(&rows, &columns, &a, &lda, &tau, &work, &lwork, &info)
    try requireSuccess(info)

    var r = Matrix<Double>.zeros(k, matrix.columns)
    for row in 0..<k {
        for column in row..<matrix.columns {
            r[row, column] = a[column * matrix.rows + row]
        }
    }

    var qStorage = [Double](repeating: 0, count: matrix.rows * k)
    for column in 0..<k {
        for row in 0..<matrix.rows {
            qStorage[column * matrix.rows + row] = a[column * matrix.rows + row]
        }
    }
    var qRows = m
    var qColumns = checkedLAPACKInt(k)
    var reflectorCount = checkedLAPACKInt(k)
    lwork = -1
    workQuery = [Double](repeating: 0, count: 1)
    dorgqr_(&qRows, &qColumns, &reflectorCount, &qStorage, &lda, &tau, &workQuery, &lwork, &info)
    try requireSuccess(info)
    work = [Double](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    dorgqr_(&qRows, &qColumns, &reflectorCount, &qStorage, &lda, &tau, &work, &lwork, &info)
    try requireSuccess(info)
    let q = try ColumnMajorBridge.rowMajor(fromColumnMajor: qStorage, rows: matrix.rows, columns: k)
    return LAPACKQRResult(q: q, r: r)
}

private func qrFloat(_ matrix: Matrix<Float>) throws -> LAPACKQRResult<Float> {
    let m = checkedLAPACKInt(matrix.rows)
    let n = checkedLAPACKInt(matrix.columns)
    let k = min(matrix.rows, matrix.columns)
    guard k > 0 else {
        return LAPACKQRResult(q: Matrix(rows: matrix.rows, columns: 0), r: Matrix(rows: 0, columns: matrix.columns))
    }

    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var rows = m
    var columns = n
    var lda = max(m, 1)
    var tau = [Float](repeating: 0, count: k)
    var info: LAPACKInteger = 0
    var lwork: LAPACKInteger = -1
    var workQuery = [Float](repeating: 0, count: 1)
    sgeqrf_(&rows, &columns, &a, &lda, &tau, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Float](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    sgeqrf_(&rows, &columns, &a, &lda, &tau, &work, &lwork, &info)
    try requireSuccess(info)

    var r = Matrix<Float>.zeros(k, matrix.columns)
    for row in 0..<k {
        for column in row..<matrix.columns {
            r[row, column] = a[column * matrix.rows + row]
        }
    }

    var qStorage = [Float](repeating: 0, count: matrix.rows * k)
    for column in 0..<k {
        for row in 0..<matrix.rows {
            qStorage[column * matrix.rows + row] = a[column * matrix.rows + row]
        }
    }
    var qRows = m
    var qColumns = checkedLAPACKInt(k)
    var reflectorCount = checkedLAPACKInt(k)
    lwork = -1
    workQuery = [Float](repeating: 0, count: 1)
    sorgqr_(&qRows, &qColumns, &reflectorCount, &qStorage, &lda, &tau, &workQuery, &lwork, &info)
    try requireSuccess(info)
    work = [Float](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    sorgqr_(&qRows, &qColumns, &reflectorCount, &qStorage, &lda, &tau, &work, &lwork, &info)
    try requireSuccess(info)
    let q = try ColumnMajorBridge.rowMajor(fromColumnMajor: qStorage, rows: matrix.rows, columns: k)
    return LAPACKQRResult(q: q, r: r)
}

private func choleskyDouble(_ matrix: Matrix<Double>) throws -> Matrix<Double> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var order = n
    var lda = max(n, 1)
    var info: LAPACKInteger = 0
    var uplo = lapackCharacter("L")
    dpotrf_(&uplo, &order, &a, &lda, &info)
    try requirePositiveDefiniteSuccess(info)
    return try lowerTriangle(fromColumnMajor: a, order: matrix.rows)
}

private func choleskyFloat(_ matrix: Matrix<Float>) throws -> Matrix<Float> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var order = n
    var lda = max(n, 1)
    var info: LAPACKInteger = 0
    var uplo = lapackCharacter("L")
    spotrf_(&uplo, &order, &a, &lda, &info)
    try requirePositiveDefiniteSuccess(info)
    return try lowerTriangle(fromColumnMajor: a, order: matrix.rows)
}

private func solveSPDDouble(_ matrix: Matrix<Double>, _ rhs: Matrix<Double>) throws -> Matrix<Double> {
    try requireSquare(matrix)
    guard matrix.rows == rhs.rows else {
        throw LinAlgError.dimensionMismatch("SPD solve requires A.rows == B.rows.")
    }
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var b = ColumnMajorBridge.columnMajor(from: rhs)
    var order = n
    var nrhs = checkedLAPACKInt(rhs.columns)
    var lda = max(n, 1)
    var ldb = max(n, 1)
    var info: LAPACKInteger = 0
    var uplo = lapackCharacter("L")
    dpotrf_(&uplo, &order, &a, &lda, &info)
    try requirePositiveDefiniteSuccess(info)
    dpotrs_(&uplo, &order, &nrhs, &a, &lda, &b, &ldb, &info)
    try requireSuccess(info)
    return try ColumnMajorBridge.rowMajor(fromColumnMajor: b, rows: rhs.rows, columns: rhs.columns)
}

private func solveSPDFloat(_ matrix: Matrix<Float>, _ rhs: Matrix<Float>) throws -> Matrix<Float> {
    try requireSquare(matrix)
    guard matrix.rows == rhs.rows else {
        throw LinAlgError.dimensionMismatch("SPD solve requires A.rows == B.rows.")
    }
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var b = ColumnMajorBridge.columnMajor(from: rhs)
    var order = n
    var nrhs = checkedLAPACKInt(rhs.columns)
    var lda = max(n, 1)
    var ldb = max(n, 1)
    var info: LAPACKInteger = 0
    var uplo = lapackCharacter("L")
    spotrf_(&uplo, &order, &a, &lda, &info)
    try requirePositiveDefiniteSuccess(info)
    spotrs_(&uplo, &order, &nrhs, &a, &lda, &b, &ldb, &info)
    try requireSuccess(info)
    return try ColumnMajorBridge.rowMajor(fromColumnMajor: b, rows: rhs.rows, columns: rhs.columns)
}

private func svdDouble(_ matrix: Matrix<Double>, fullVectors: Bool) throws -> LAPACKSVDResult<Double> {
    let m = checkedLAPACKInt(matrix.rows)
    let n = checkedLAPACKInt(matrix.columns)
    let k = min(matrix.rows, matrix.columns)
    guard k > 0 else {
        return LAPACKSVDResult(u: Matrix(rows: matrix.rows, columns: 0), singularValues: Vector([]), v: Matrix(rows: matrix.columns, columns: 0))
    }

    var jobz = lapackCharacter(fullVectors ? "A" : "S")
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var rows = m
    var columns = n
    var lda = max(m, 1)
    let uColumns = fullVectors ? matrix.rows : k
    let vtRows = fullVectors ? matrix.columns : k
    var s = [Double](repeating: 0, count: k)
    var u = [Double](repeating: 0, count: matrix.rows * uColumns)
    var vt = [Double](repeating: 0, count: vtRows * matrix.columns)
    var ldu = max(m, 1)
    var ldvt = max(checkedLAPACKInt(vtRows), 1)
    var lwork: LAPACKInteger = -1
    var workQuery = [Double](repeating: 0, count: 1)
    var iwork = [LAPACKInteger](repeating: 0, count: max(1, 8 * k))
    var info: LAPACKInteger = 0
    dgesdd_(&jobz, &rows, &columns, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &workQuery, &lwork, &iwork, &info)
    try requireSuccess(info)
    var work = [Double](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    dgesdd_(&jobz, &rows, &columns, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &iwork, &info)
    try requireSuccess(info)
    let uMatrix = try ColumnMajorBridge.rowMajor(fromColumnMajor: u, rows: matrix.rows, columns: uColumns)
    let vtMatrix = try ColumnMajorBridge.rowMajor(fromColumnMajor: vt, rows: vtRows, columns: matrix.columns)
    return LAPACKSVDResult(u: uMatrix, singularValues: Vector(s), v: vtMatrix.t)
}

private func svdFloat(_ matrix: Matrix<Float>, fullVectors: Bool) throws -> LAPACKSVDResult<Float> {
    let m = checkedLAPACKInt(matrix.rows)
    let n = checkedLAPACKInt(matrix.columns)
    let k = min(matrix.rows, matrix.columns)
    guard k > 0 else {
        return LAPACKSVDResult(u: Matrix(rows: matrix.rows, columns: 0), singularValues: Vector([]), v: Matrix(rows: matrix.columns, columns: 0))
    }

    var jobz = lapackCharacter(fullVectors ? "A" : "S")
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var rows = m
    var columns = n
    var lda = max(m, 1)
    let uColumns = fullVectors ? matrix.rows : k
    let vtRows = fullVectors ? matrix.columns : k
    var s = [Float](repeating: 0, count: k)
    var u = [Float](repeating: 0, count: matrix.rows * uColumns)
    var vt = [Float](repeating: 0, count: vtRows * matrix.columns)
    var ldu = max(m, 1)
    var ldvt = max(checkedLAPACKInt(vtRows), 1)
    var lwork: LAPACKInteger = -1
    var workQuery = [Float](repeating: 0, count: 1)
    var iwork = [LAPACKInteger](repeating: 0, count: max(1, 8 * k))
    var info: LAPACKInteger = 0
    sgesdd_(&jobz, &rows, &columns, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &workQuery, &lwork, &iwork, &info)
    try requireSuccess(info)
    var work = [Float](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    sgesdd_(&jobz, &rows, &columns, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &iwork, &info)
    try requireSuccess(info)
    let uMatrix = try ColumnMajorBridge.rowMajor(fromColumnMajor: u, rows: matrix.rows, columns: uColumns)
    let vtMatrix = try ColumnMajorBridge.rowMajor(fromColumnMajor: vt, rows: vtRows, columns: matrix.columns)
    return LAPACKSVDResult(u: uMatrix, singularValues: Vector(s), v: vtMatrix.t)
}

private func symmetricEigenDouble(_ matrix: Matrix<Double>) throws -> LAPACKSymmetricEigenResult<Double> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var order = n
    var lda = max(n, 1)
    var values = [Double](repeating: 0, count: matrix.rows)
    var jobz = lapackCharacter("V")
    var uplo = lapackCharacter("L")
    var lwork: LAPACKInteger = -1
    var workQuery = [Double](repeating: 0, count: 1)
    var info: LAPACKInteger = 0
    dsyev_(&jobz, &uplo, &order, &a, &lda, &values, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Double](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    dsyev_(&jobz, &uplo, &order, &a, &lda, &values, &work, &lwork, &info)
    try requireSuccess(info)
    let vectors = try ColumnMajorBridge.rowMajor(fromColumnMajor: a, rows: matrix.rows, columns: matrix.columns)
    return LAPACKSymmetricEigenResult(values: Vector(values), vectors: vectors)
}

private func symmetricEigenFloat(_ matrix: Matrix<Float>) throws -> LAPACKSymmetricEigenResult<Float> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var order = n
    var lda = max(n, 1)
    var values = [Float](repeating: 0, count: matrix.rows)
    var jobz = lapackCharacter("V")
    var uplo = lapackCharacter("L")
    var lwork: LAPACKInteger = -1
    var workQuery = [Float](repeating: 0, count: 1)
    var info: LAPACKInteger = 0
    ssyev_(&jobz, &uplo, &order, &a, &lda, &values, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Float](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    ssyev_(&jobz, &uplo, &order, &a, &lda, &values, &work, &lwork, &info)
    try requireSuccess(info)
    let vectors = try ColumnMajorBridge.rowMajor(fromColumnMajor: a, rows: matrix.rows, columns: matrix.columns)
    return LAPACKSymmetricEigenResult(values: Vector(values), vectors: vectors)
}

private func eigenDouble(_ matrix: Matrix<Double>) throws -> LAPACKGeneralEigenResult<Double> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var order = n
    var lda = max(n, 1)
    var real = [Double](repeating: 0, count: matrix.rows)
    var imaginary = [Double](repeating: 0, count: matrix.rows)
    var vl = [Double](repeating: 0, count: 1)
    var vr = [Double](repeating: 0, count: matrix.count)
    var ldvl: LAPACKInteger = 1
    var ldvr = max(n, 1)
    var jobvl = lapackCharacter("N")
    var jobvr = lapackCharacter("V")
    var lwork: LAPACKInteger = -1
    var workQuery = [Double](repeating: 0, count: 1)
    var info: LAPACKInteger = 0
    dgeev_(&jobvl, &jobvr, &order, &a, &lda, &real, &imaginary, &vl, &ldvl, &vr, &ldvr, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Double](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    dgeev_(&jobvl, &jobvr, &order, &a, &lda, &real, &imaginary, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)
    try requireSuccess(info)
    let vectors = try ColumnMajorBridge.rowMajor(fromColumnMajor: vr, rows: matrix.rows, columns: matrix.columns)
    return LAPACKGeneralEigenResult(realParts: Vector(real), imaginaryParts: Vector(imaginary), rightEigenvectorsPacked: vectors)
}

private func eigenFloat(_ matrix: Matrix<Float>) throws -> LAPACKGeneralEigenResult<Float> {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var order = n
    var lda = max(n, 1)
    var real = [Float](repeating: 0, count: matrix.rows)
    var imaginary = [Float](repeating: 0, count: matrix.rows)
    var vl = [Float](repeating: 0, count: 1)
    var vr = [Float](repeating: 0, count: matrix.count)
    var ldvl: LAPACKInteger = 1
    var ldvr = max(n, 1)
    var jobvl = lapackCharacter("N")
    var jobvr = lapackCharacter("V")
    var lwork: LAPACKInteger = -1
    var workQuery = [Float](repeating: 0, count: 1)
    var info: LAPACKInteger = 0
    sgeev_(&jobvl, &jobvr, &order, &a, &lda, &real, &imaginary, &vl, &ldvl, &vr, &ldvr, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Float](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    sgeev_(&jobvl, &jobvr, &order, &a, &lda, &real, &imaginary, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)
    try requireSuccess(info)
    let vectors = try ColumnMajorBridge.rowMajor(fromColumnMajor: vr, rows: matrix.rows, columns: matrix.columns)
    return LAPACKGeneralEigenResult(realParts: Vector(real), imaginaryParts: Vector(imaginary), rightEigenvectorsPacked: vectors)
}

private func leastSquaresDouble(_ matrix: Matrix<Double>, _ b: Vector<Double>) throws -> Vector<Double> {
    guard matrix.rows == b.count else {
        throw LinAlgError.dimensionMismatch("Least squares requires A.rows == b.count.")
    }
    let m = checkedLAPACKInt(matrix.rows)
    let n = checkedLAPACKInt(matrix.columns)
    let ldbInt = max(matrix.rows, matrix.columns)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var rhs = [Double](repeating: 0, count: max(1, ldbInt))
    for index in 0..<b.count { rhs[index] = b[index] }
    var rows = m
    var columns = n
    var nrhs: LAPACKInteger = 1
    var lda = max(m, 1)
    var ldb = checkedLAPACKInt(max(1, ldbInt))
    var trans = lapackCharacter("N")
    var lwork: LAPACKInteger = -1
    var workQuery = [Double](repeating: 0, count: 1)
    var info: LAPACKInteger = 0
    dgels_(&trans, &rows, &columns, &nrhs, &a, &lda, &rhs, &ldb, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Double](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    dgels_(&trans, &rows, &columns, &nrhs, &a, &lda, &rhs, &ldb, &work, &lwork, &info)
    try requireSuccess(info)
    return Vector(Array(rhs.prefix(matrix.columns)))
}

private func leastSquaresFloat(_ matrix: Matrix<Float>, _ b: Vector<Float>) throws -> Vector<Float> {
    guard matrix.rows == b.count else {
        throw LinAlgError.dimensionMismatch("Least squares requires A.rows == b.count.")
    }
    let m = checkedLAPACKInt(matrix.rows)
    let n = checkedLAPACKInt(matrix.columns)
    let ldbInt = max(matrix.rows, matrix.columns)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var rhs = [Float](repeating: 0, count: max(1, ldbInt))
    for index in 0..<b.count { rhs[index] = b[index] }
    var rows = m
    var columns = n
    var nrhs: LAPACKInteger = 1
    var lda = max(m, 1)
    var ldb = checkedLAPACKInt(max(1, ldbInt))
    var trans = lapackCharacter("N")
    var lwork: LAPACKInteger = -1
    var workQuery = [Float](repeating: 0, count: 1)
    var info: LAPACKInteger = 0
    sgels_(&trans, &rows, &columns, &nrhs, &a, &lda, &rhs, &ldb, &workQuery, &lwork, &info)
    try requireSuccess(info)
    var work = [Float](repeating: 0, count: max(1, Int(workQuery[0])))
    lwork = checkedLAPACKInt(work.count)
    sgels_(&trans, &rows, &columns, &nrhs, &a, &lda, &rhs, &ldb, &work, &lwork, &info)
    try requireSuccess(info)
    return Vector(Array(rhs.prefix(matrix.columns)))
}

private func normDouble(_ matrix: Matrix<Double>, kind: LAPACKMatrixNorm) -> Double {
    guard matrix.count > 0 else { return 0 }
    var norm = lapackCharacter(kind.lapackCode)
    var m = checkedLAPACKInt(matrix.rows)
    var n = checkedLAPACKInt(matrix.columns)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var lda = max(m, 1)
    var work = [Double](repeating: 0, count: max(1, matrix.rows))
    return dlange_(&norm, &m, &n, &a, &lda, &work)
}

private func normFloat(_ matrix: Matrix<Float>, kind: LAPACKMatrixNorm) -> Float {
    guard matrix.count > 0 else { return 0 }
    var norm = lapackCharacter(kind.lapackCode)
    var m = checkedLAPACKInt(matrix.rows)
    var n = checkedLAPACKInt(matrix.columns)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var lda = max(m, 1)
    var work = [Float](repeating: 0, count: max(1, matrix.rows))
    return Float(slange_(&norm, &m, &n, &a, &lda, &work))
}

private func balanceDouble(_ matrix: Matrix<Double>) throws -> (balanced: Matrix<Double>, scales: Vector<Double>) {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var order = n
    var lda = max(n, 1)
    var ilo: LAPACKInteger = 0
    var ihi: LAPACKInteger = 0
    var scales = [Double](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    var job = lapackCharacter("B")
    dgebal_(&job, &order, &a, &lda, &ilo, &ihi, &scales, &info)
    try requireSuccess(info)
    let balanced = try ColumnMajorBridge.rowMajor(fromColumnMajor: a, rows: matrix.rows, columns: matrix.columns)
    return (balanced, Vector(scales))
}

private func balanceFloat(_ matrix: Matrix<Float>) throws -> (balanced: Matrix<Float>, scales: Vector<Float>) {
    try requireSquare(matrix)
    let n = checkedLAPACKInt(matrix.rows)
    var a = ColumnMajorBridge.columnMajor(from: matrix)
    var order = n
    var lda = max(n, 1)
    var ilo: LAPACKInteger = 0
    var ihi: LAPACKInteger = 0
    var scales = [Float](repeating: 0, count: matrix.rows)
    var info: LAPACKInteger = 0
    var job = lapackCharacter("B")
    sgebal_(&job, &order, &a, &lda, &ilo, &ihi, &scales, &info)
    try requireSuccess(info)
    let balanced = try ColumnMajorBridge.rowMajor(fromColumnMajor: a, rows: matrix.rows, columns: matrix.columns)
    return (balanced, Vector(scales))
}

private func luResult<Scalar: RealFloatingScalar>(
    fromColumnMajorFactor factor: [Scalar],
    pivots: [LAPACKInteger],
    order: Int
) throws -> LAPACKLUResult<Scalar> {
    let combined = try ColumnMajorBridge.rowMajor(fromColumnMajor: factor, rows: order, columns: order)
    var l = Matrix<Scalar>.eye(order)
    var u = Matrix<Scalar>.zeros(order, order)
    for row in 0..<order {
        for column in 0..<order {
            if row > column {
                l[row, column] = combined[row, column]
            } else {
                u[row, column] = combined[row, column]
            }
        }
    }
    let (permutation, parity) = permutationAndParity(from: pivots, count: order)
    return LAPACKLUResult(l: l, u: u, permutation: permutation, parity: parity)
}

private func lowerTriangle<Scalar: RealFloatingScalar>(
    fromColumnMajor factor: [Scalar],
    order: Int
) throws -> Matrix<Scalar> {
    var columnMajor = Array(repeating: Scalar.zero, count: order * order)
    for column in 0..<order {
        for row in column..<order {
            columnMajor[column * order + row] = factor[column * order + row]
        }
    }
    return try ColumnMajorBridge.rowMajor(fromColumnMajor: columnMajor, rows: order, columns: order)
}

private func permutationAndParity(from pivots: [LAPACKInteger], count: Int) -> ([Int], Int) {
    var permutation = Array(0..<count)
    var parity = 1
    for (index, pivotValue) in pivots.enumerated() {
        let pivot = Int(pivotValue) - 1
        if pivot != index {
            permutation.swapAt(index, pivot)
            parity = -parity
        }
    }
    return (permutation, parity)
}

private func requireSquare<S: MatrixScalar>(_ matrix: Matrix<S>) throws {
    guard matrix.isSquare else { throw LinAlgError.notSquare }
}

private func requireFactorizationSuccess(_ info: LAPACKInteger) throws {
    if info < 0 {
        throw LinAlgError.unsupported("LAPACK reported an illegal argument at position \(-info).")
    }
    if info > 0 {
        throw LinAlgError.singularMatrix
    }
}

private func requirePositiveDefiniteSuccess(_ info: LAPACKInteger) throws {
    if info < 0 {
        throw LinAlgError.unsupported("LAPACK reported an illegal argument at position \(-info).")
    }
    if info > 0 {
        throw LinAlgError.notPositiveDefinite
    }
}

private func requireSuccess(_ info: LAPACKInteger) throws {
    if info < 0 {
        throw LinAlgError.unsupported("LAPACK reported an illegal argument at position \(-info).")
    }
    if info > 0 {
        throw LinAlgError.unsupported("LAPACK routine did not converge; info=\(info).")
    }
}

private func checkedLAPACKInt(_ value: Int) -> LAPACKInteger {
    precondition(value <= Int(Int32.max), "Accelerate LAPACK wrapper only supports Int32-sized dimensions.")
    return LAPACKInteger(value)
}

private func lapackCharacter(_ character: Character) -> [Int8] {
    Array(String(character).utf8CString)
}

private extension LAPACKMatrixNorm {
    var lapackCode: Character {
        switch self {
        case .one: "1"
        case .infinity: "I"
        case .frobenius: "F"
        case .maximum: "M"
        }
    }
}
