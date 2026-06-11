//
//  NumericalRoutines.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

public import ComplexModule
internal import Foundation

public enum MatrixNorm: Sendable {
    case one
    case infinity
    case frobenius
    case two
}

public enum VectorNorm: Sendable {
    case one
    case two
    case infinity
    case p(Double)
}

public enum GramSchmidtMode: Sendable {
    case modified
    case classical
}

public enum EigenSortKey: Sendable {
    case realPart
    case magnitude
}

public struct QRDecomposition<S: RealFloatingScalar>: Sendable {
    public let q: Matrix<S>
    public let r: Matrix<S>

    public init(q: Matrix<S>, r: Matrix<S>) {
        self.q = q
        self.r = r
    }
}

public struct LUDecomposition<S: RealFloatingScalar>: Sendable {
    public let l: Matrix<S>
    public let u: Matrix<S>
    public let permutation: [Int]
    public let parity: Int

    public init(l: Matrix<S>, u: Matrix<S>, permutation: [Int], parity: Int) {
        self.l = l
        self.u = u
        self.permutation = permutation
        self.parity = parity
    }
}

public struct CholeskyDecomposition<S: RealFloatingScalar>: Sendable {
    public let lower: Matrix<S>

    public init(lower: Matrix<S>) {
        self.lower = lower
    }
}

public struct SVDDecomposition<S: RealFloatingScalar>: Sendable {
    public let u: Matrix<S>
    public let singularValues: Vector<S>
    public let v: Matrix<S>

    public init(u: Matrix<S>, singularValues: Vector<S>, v: Matrix<S>) {
        self.u = u
        self.singularValues = singularValues
        self.v = v
    }

    public var sigma: Matrix<S> {
        Matrix.diag(singularValues)
    }
}

public struct EigenDecomposition<S: RealFloatingScalar>: Sendable {
    public let values: [Complex<S>]
    public let vectors: Matrix<Complex<S>>?

    public init(values: [Complex<S>], vectors: Matrix<Complex<S>>? = nil) {
        self.values = values
        self.vectors = vectors
    }
}

public struct RealSchurDecomposition<S: RealFloatingScalar>: Sendable {
    public let q: Matrix<S>
    public let t: Matrix<S>

    public init(q: Matrix<S>, t: Matrix<S>) {
        self.q = q
        self.t = t
    }
}

public struct FundamentalSubspaces<S: RealFloatingScalar>: Sendable {
    public let columnSpace: Matrix<S>
    public let leftNullSpace: Matrix<S>
    public let rowSpace: Matrix<S>
    public let nullSpace: Matrix<S>

    public init(columnSpace: Matrix<S>, leftNullSpace: Matrix<S>, rowSpace: Matrix<S>, nullSpace: Matrix<S>) {
        self.columnSpace = columnSpace
        self.leftNullSpace = leftNullSpace
        self.rowSpace = rowSpace
        self.nullSpace = nullSpace
    }
}

public struct IterativeSolverResult<S: RealFloatingScalar>: Sendable {
    public let solution: Vector<S>
    public let iterations: Int
    public let residualNorm: S
    public let converged: Bool

    public init(solution: Vector<S>, iterations: Int, residualNorm: S, converged: Bool) {
        self.solution = solution
        self.iterations = iterations
        self.residualNorm = residualNorm
        self.converged = converged
    }
}

public struct PolynomialDivisionResult<S: RealFloatingScalar>: Sendable {
    public let quotient: [S]
    public let remainder: [S]

    public init(quotient: [S], remainder: [S]) {
        self.quotient = quotient
        self.remainder = remainder
    }
}

public struct RationalApproximation: Equatable, Sendable {
    public let numerator: Int
    public let denominator: Int

    public init(numerator: Int, denominator: Int) {
        precondition(denominator != 0, "The denominator cannot be zero.")
        let divisor = RationalApproximation.gcd(abs(numerator), abs(denominator))
        let sign = denominator < 0 ? -1 : 1
        self.numerator = sign * numerator / divisor
        self.denominator = sign * denominator / divisor
    }

    private static func gcd(_ a: Int, _ b: Int) -> Int {
        var x = a
        var y = b
        while y != 0 {
            let r = x % y
            x = y
            y = r
        }
        return max(x, 1)
    }
}

public func / <S: FloatingScalar>(lhs: Matrix<S>, rhs: S) -> Matrix<S> {
    lhs.map { $0 / rhs }
}

public func / <S: RealFloatingScalar>(lhs: Matrix<S>, rhs: Matrix<S>) -> Matrix<S> {
    try! solveMatrixInverseLU(rhs.t, lhs.t).t
}

public func / <S: FloatingScalar>(lhs: Vector<S>, rhs: S) -> Vector<S> {
    Vector((lhs.asColumnMatrix() / rhs).rowMajorElements())
}

public func + <S: MatrixScalar>(lhs: Matrix<S>, rhs: S) -> Matrix<S> {
    lhs.map { $0 + rhs }
}

public func - <S: MatrixScalar>(lhs: Matrix<S>, rhs: S) -> Matrix<S> {
    lhs.map { $0 - rhs }
}

public func hadamard<S: MatrixScalar>(_ lhs: Matrix<S>, _ rhs: Matrix<S>) -> Matrix<S> {
    if let accelerated: Matrix<S> = try? acceleratedHadamard(lhs, rhs) {
        return accelerated
    }
    precondition(lhs.shape == rhs.shape, "Hadamard product requires equal shapes.")
    return try! Matrix(rowMajor: zip(lhs.rowMajorElements(), rhs.rowMajorElements()).map(*), rows: lhs.rows, columns: lhs.columns)
}

public func elementwiseDivide<S: FloatingScalar>(_ lhs: Matrix<S>, _ rhs: Matrix<S>) -> Matrix<S> {
    if let accelerated: Matrix<S> = try? acceleratedElementwiseDivide(lhs, rhs) {
        return accelerated
    }
    precondition(lhs.shape == rhs.shape, "Elementwise division requires equal shapes.")
    return try! Matrix(rowMajor: zip(lhs.rowMajorElements(), rhs.rowMajorElements()).map(/), rows: lhs.rows, columns: lhs.columns)
}

public func dividePwMatrix<S: FloatingScalar>(_ lhs: Matrix<S>, _ rhs: Matrix<S>) -> Matrix<S> {
    elementwiseDivide(lhs, rhs)
}

public func swapElements<S>(_ values: inout [S], _ i: Int, _ j: Int) {
    values.swapAt(i, j)
}

public func absMatrix<S: FloatingScalar>(_ matrix: Matrix<S>) -> Matrix<S.Magnitude> {
    try! Matrix<S.Magnitude>(rowMajor: matrix.rowMajorElements().map(\.magnitude), rows: matrix.rows, columns: matrix.columns)
}

public func sum<S: MatrixScalar>(_ matrix: Matrix<S>) -> S {
    if let accelerated: S = acceleratedSum(matrix) {
        return accelerated
    }
    return matrix.rowMajorElements().reduce(.zero, +)
}

public func trace<S: FloatingScalar>(_ matrix: Matrix<S>) -> S {
    matrix.trace
}

public func transpose<S: MatrixScalar>(_ matrix: Matrix<S>) -> Matrix<S> {
    matrix.t
}

public func constantMatrixMultiplication<S: MatrixScalar>(_ scalar: S, _ matrix: Matrix<S>) -> Matrix<S> {
    scalar * matrix
}

public func multiplyConstantMatrix<S: MatrixScalar>(_ matrix: Matrix<S>, _ scalar: S) -> Matrix<S> {
    if let accelerated: Matrix<S> = try? acceleratedScalarMultiply(matrix, scalar) {
        return accelerated
    }
    return matrix * scalar
}

public func divideConstantMatrix<S: FloatingScalar>(_ matrix: Matrix<S>, _ scalar: S) -> Matrix<S> {
    matrix / scalar
}

public func multiply<S: MatrixScalar>(_ lhs: Matrix<S>, _ rhs: Matrix<S>) -> Matrix<S> {
    if let accelerated: Matrix<S> = try? acceleratedGemm(lhs, rhs) {
        return accelerated
    }
    return lhs * rhs
}

public func multiplyAddInPlace<S: MatrixScalar>(_ a: Matrix<S>, _ b: Matrix<S>, into c: inout Matrix<S>) {
    c = c + a * b
}

public func powerMatrix<S: MatrixScalar>(_ matrix: Matrix<S>, _ exponent: Int) -> Matrix<S> {
    matrix.power(exponent)
}

public func getMatrixRow<S: MatrixScalar>(_ matrix: Matrix<S>, _ row: Int) -> Vector<S> {
    Vector(matrix[row, .all].rowMajorElements())
}

public func getMatrixColumn<S: MatrixScalar>(_ matrix: Matrix<S>, _ column: Int) -> Vector<S> {
    Vector(matrix[.all, column].rowMajorElements())
}

public func getDiagonal<S: MatrixScalar>(_ matrix: Matrix<S>) -> Vector<S> {
    let n = min(matrix.rows, matrix.columns)
    return Vector((0..<n).map { matrix[$0, $0] })
}

public func diagonal<S: MatrixScalar>(_ vector: Vector<S>) -> Matrix<S> {
    Matrix.diag(vector)
}

public func repmat<S: MatrixScalar>(_ matrix: Matrix<S>, rows rowRepeats: Int, columns columnRepeats: Int) -> Matrix<S> {
    precondition(rowRepeats >= 0 && columnRepeats >= 0, "Repeat counts must be nonnegative.")
    var result = Matrix<S>(rows: matrix.rows * rowRepeats, columns: matrix.columns * columnRepeats)
    for rr in 0..<rowRepeats {
        for cc in 0..<columnRepeats {
            for i in 0..<matrix.rows {
                for j in 0..<matrix.columns {
                    result[rr * matrix.rows + i, cc * matrix.columns + j] = matrix[i, j]
                }
            }
        }
    }
    return result
}

public func reshape<S: MatrixScalar>(_ matrix: Matrix<S>, rows: Int, columns: Int) throws -> Matrix<S> {
    try Matrix(rowMajor: matrix.rowMajorElements(), rows: rows, columns: columns)
}

public func reshapeV2<S: MatrixScalar>(_ matrix: Matrix<S>, rowRange: ClosedRange<Int>, columnRange: ClosedRange<Int>) -> Matrix<S> {
    matrix[rowRange, columnRange]
}

public func embedMatrix<S: MatrixScalar>(_ block: Matrix<S>, into matrix: Matrix<S>, row: Int, column: Int) -> Matrix<S> {
    var result = matrix
    precondition(row >= 0 && column >= 0 && row + block.rows <= matrix.rows && column + block.columns <= matrix.columns, "Block does not fit.")
    for i in 0..<block.rows {
        for j in 0..<block.columns {
            result[row + i, column + j] = block[i, j]
        }
    }
    return result
}

public func dropRow<S: MatrixScalar>(_ matrix: Matrix<S>, _ rowToDrop: Int) -> Matrix<S> {
    precondition(rowToDrop >= 0 && rowToDrop < matrix.rows, "Row index is out of bounds.")
    let rows = (0..<matrix.rows).filter { $0 != rowToDrop }
    return matrix.copiedPublicRows(rows, columns: Array(0..<matrix.columns))
}

public func dropColumn<S: MatrixScalar>(_ matrix: Matrix<S>, _ columnToDrop: Int) -> Matrix<S> {
    precondition(columnToDrop >= 0 && columnToDrop < matrix.columns, "Column index is out of bounds.")
    let columns = (0..<matrix.columns).filter { $0 != columnToDrop }
    return matrix.copiedPublicRows(Array(0..<matrix.rows), columns: columns)
}

public func obtainSubMatrix<S: MatrixScalar>(_ matrix: Matrix<S>, droppingRow row: Int, column: Int) -> Matrix<S> {
    dropColumn(dropRow(matrix, row), column)
}

public func principalSubmatrices<S: MatrixScalar>(_ matrix: Matrix<S>) -> [Matrix<S>] {
    precondition(matrix.isSquare, "Principal submatrices require a square matrix.")
    return (1...matrix.rows).map { matrix[0...($0 - 1), 0...($0 - 1)] }
}

public func principalMinors<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> [S] {
    try principalSubmatrices(matrix).map { try determinant($0) }
}

public func norm<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ kind: MatrixNorm = .frobenius) -> S {
    switch kind {
    case .one:
        if let accelerated: S = acceleratedMatrixNorm(matrix, kind) {
            return accelerated
        }
        var best = S.zero
        for j in 0..<matrix.columns {
            var total = S.zero
            for i in 0..<matrix.rows { total += S.abs(matrix[i, j]) }
            best = max(best, total)
        }
        return best
    case .infinity:
        if let accelerated: S = acceleratedMatrixNorm(matrix, kind) {
            return accelerated
        }
        var best = S.zero
        for i in 0..<matrix.rows {
            var total = S.zero
            for j in 0..<matrix.columns { total += S.abs(matrix[i, j]) }
            best = max(best, total)
        }
        return best
    case .frobenius:
        if let accelerated: S = acceleratedMatrixNorm(matrix, kind) ?? acceleratedFrobeniusNorm(matrix) {
            return accelerated
        }
        return matrix.frobeniusNorm
    case .two:
        return singularValueDecomposition(matrix).singularValues[0]
    }
}

public func norm<S: RealFloatingScalar>(_ vector: Vector<S>, _ kind: VectorNorm = .two) -> S {
    switch kind {
    case .one:
        if let accelerated: S = acceleratedVectorASum(vector) {
            return accelerated
        }
        return (0..<vector.count).reduce(S.zero) { $0 + S.abs(vector[$1]) }
    case .two:
        if let accelerated: S = acceleratedVectorNorm(vector) {
            return accelerated
        }
        return vector.norm
    case .infinity:
        return (0..<vector.count).reduce(S.zero) { max($0, S.abs(vector[$1])) }
    case .p(let p):
        precondition(p > 0, "p-norm requires p > 0.")
        let total = (0..<vector.count).reduce(0.0) { $0 + Foundation.pow(Double(S.abs(vector[$1])), p) }
        return S(Foundation.pow(total, 1.0 / p))
    }
}

public func householderVector<S: RealFloatingScalar>(_ x: Vector<S>) -> (v: Vector<S>, beta: S) {
    var values = (0..<x.count).map { x[$0] }
    let sigma = values.dropFirst().reduce(S.zero) { $0 + $1 * $1 }
    guard sigma != .zero || values[0] != .zero else {
        return (Vector(Array(repeating: .zero, count: x.count)), .zero)
    }

    let length = S.sqrt(values[0] * values[0] + sigma)
    values[0] += sign(length, values[0])
    let denominator = values.reduce(S.zero) { $0 + $1 * $1 }
    let beta = denominator == .zero ? S.zero : S(2) / denominator
    return (Vector(values), beta)
}

private func qrRoutine<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> QRDecomposition<S> {
    var r = matrix
    var q = Matrix<S>.eye(matrix.rows)
    let limit = min(matrix.rows, matrix.columns)

    for k in 0..<limit {
        let x = Vector((k..<matrix.rows).map { r[$0, k] })
        let reflector = householderVector(x)
        let v = reflector.v
        let beta = reflector.beta
        guard beta != .zero else { continue }

        for j in k..<matrix.columns {
            var projection = S.zero
            for i in 0..<v.count {
                projection += v[i] * r[k + i, j]
            }
            projection *= beta
            for i in 0..<v.count {
                r[k + i, j] -= projection * v[i]
            }
        }

        for row in 0..<q.rows {
            var projection = S.zero
            for i in 0..<v.count {
                projection += q[row, k + i] * v[i]
            }
            projection *= beta
            for i in 0..<v.count {
                q[row, k + i] -= projection * v[i]
            }
        }
    }

    return QRDecomposition(q: q, r: r)
}

public func qr<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> QRDecomposition<S> {
    if let accelerated: QRDecomposition<S> = try? acceleratedQR(matrix) {
        return accelerated
    }
    return qrRoutine(matrix)
}

public func gramSchmidtFactorization<S: RealFloatingScalar>(
    _ matrix: Matrix<S>,
    mode: GramSchmidtMode = .modified
) -> QRDecomposition<S> {
    switch mode {
    case .modified:
        return modifiedGramSchmidt(matrix)
    case .classical:
        return classicalGramSchmidt(matrix)
    }
}

func gramSmchmidtFactorization<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> QRDecomposition<S> {
    gramSchmidtFactorization(matrix)
}

public func isOrthogonal<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    guard matrix.rows >= matrix.columns else { return false }
    let gram = matrix.t * matrix
    let identity = Matrix<S>.eye(gram.rows)
    return norm(gram - identity, .frobenius) <= tolerance * max(S(1), S(matrix.rows))
}

public func isGramSchmidt<S: RealFloatingScalar>(_ decomposition: QRDecomposition<S>, of matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    norm(decomposition.q * decomposition.r - matrix, .frobenius) <= tolerance * max(S(1), norm(matrix, .frobenius))
        && isOrthogonal(decomposition.q, tolerance: tolerance * max(S(1), S(matrix.rows)))
}

public func luDecompositionDoolittle<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> LUDecomposition<S> {
    guard matrix.isSquare else { throw LinAlgError.notSquare }
    let n = matrix.rows
    var l = Matrix<S>.eye(n)
    var u = Matrix<S>.zeros(n, n)
    for i in 0..<n {
        for k in i..<n {
            var sum = S.zero
            for j in 0..<i { sum += l[i, j] * u[j, k] }
            u[i, k] = matrix[i, k] - sum
        }
        guard S.abs(u[i, i]) > S.tolerance else { throw LinAlgError.singularMatrix }
        if i + 1 < n {
            for k in (i + 1)..<n {
                var sum = S.zero
                for j in 0..<i { sum += l[k, j] * u[j, i] }
                l[k, i] = (matrix[k, i] - sum) / u[i, i]
            }
        }
    }
    return LUDecomposition(l: l, u: u, permutation: Array(0..<n), parity: 1)
}

public func luWithScaledRowPivoting<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> LUDecomposition<S> {
    guard matrix.isSquare else { throw LinAlgError.notSquare }
    if let accelerated: LUDecomposition<S> = try acceleratedLU(matrix) {
        return accelerated
    }
    let n = matrix.rows
    var a = matrix
    var permutation = Array(0..<n)
    var parity = 1
    var scales = Array(repeating: S.zero, count: n)

    for i in 0..<n {
        var rowMax = S.zero
        for j in 0..<n { rowMax = max(rowMax, S.abs(a[i, j])) }
        guard rowMax > S.tolerance else { throw LinAlgError.singularMatrix }
        scales[i] = rowMax
    }

    for k in 0..<n {
        var pivot = k
        var best = S.zero
        for i in k..<n {
            let score = S.abs(a[i, k]) / scales[i]
            if score > best {
                best = score
                pivot = i
            }
        }
        guard S.abs(a[pivot, k]) > S.tolerance else { throw LinAlgError.singularMatrix }
        if pivot != k {
            a.swapRows(k, pivot)
            permutation.swapAt(k, pivot)
            scales.swapAt(k, pivot)
            parity = -parity
        }

        if k + 1 < n {
            for i in (k + 1)..<n {
                a[i, k] /= a[k, k]
                for j in (k + 1)..<n {
                    a[i, j] -= a[i, k] * a[k, j]
                }
            }
        }
    }

    var l = Matrix<S>.eye(n)
    var u = Matrix<S>.zeros(n, n)
    for i in 0..<n {
        for j in 0..<n {
            if i > j { l[i, j] = a[i, j] }
            else { u[i, j] = a[i, j] }
        }
    }
    return LUDecomposition(l: l, u: u, permutation: permutation, parity: parity)
}

public func croutsLUwithPartialImplicitPivoting<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> LUDecomposition<S> {
    try luWithScaledRowPivoting(matrix)
}

public func solveSystemPLU<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>) throws -> Vector<S> {
    if let accelerated: Vector<S> = try acceleratedSolve(matrix, b) {
        return accelerated
    }
    let lu = try luWithScaledRowPivoting(matrix)
    return try solve(lu: lu, b)
}

public func solveVectorInverseLU<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>) throws -> Vector<S> {
    try solveSystemPLU(matrix, b)
}

public func solveMatrixInverseLU<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Matrix<S>) throws -> Matrix<S> {
    precondition(matrix.rows == b.rows, "Solve requires A.rows == B.rows.")
    if let accelerated: Matrix<S> = try acceleratedSolve(matrix, b) {
        return accelerated
    }
    var columns: [[S]] = []
    for j in 0..<b.columns {
        let solution = try solveSystemPLU(matrix, getMatrixColumn(b, j))
        columns.append((0..<solution.count).map { solution[$0] })
    }
    return columnsToMatrix(columns, rows: b.rows)
}

private func determinantRoutine<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> S {
    if let accelerated: S = try acceleratedDeterminant(matrix) {
        return accelerated
    }
    let lu = try luWithScaledRowPivoting(matrix)
    var det = S(lu.parity)
    for i in 0..<matrix.rows {
        det *= lu.u[i, i]
    }
    return det
}

public func determinant<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> S {
    try determinantRoutine(matrix)
}

private func inverseRoutine<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> Matrix<S> {
    guard matrix.isSquare else { throw LinAlgError.notSquare }
    if let accelerated: Matrix<S> = try acceleratedInverse(matrix) {
        return accelerated
    }
    let n = matrix.rows
    var columns: [[S]] = []
    for i in 0..<n {
        var e = Array(repeating: S.zero, count: n)
        e[i] = .one
        let solution = try solveSystemPLU(matrix, Vector(e))
        columns.append((0..<n).map { solution[$0] })
    }
    return columnsToMatrix(columns, rows: n)
}

public func inverse<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> Matrix<S> {
    try inverseRoutine(matrix)
}

public func inverseLU<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> Matrix<S> {
    try inverse(matrix)
}

public func choleskyDecomposition<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> CholeskyDecomposition<S> {
    guard matrix.isSquare else { throw LinAlgError.notSquare }
    if let accelerated: CholeskyDecomposition<S> = try acceleratedCholesky(matrix) {
        return accelerated
    }
    let n = matrix.rows
    var lower = Matrix<S>.zeros(n, n)

    for i in 0..<n {
        for j in 0...i {
            var sum = S.zero
            for k in 0..<j { sum += lower[i, k] * lower[j, k] }
            if i == j {
                let value = matrix[i, i] - sum
                guard value > S.tolerance else { throw LinAlgError.notPositiveDefinite }
                lower[i, j] = S.sqrt(value)
            } else {
                lower[i, j] = (matrix[i, j] - sum) / lower[j, j]
            }
        }
    }

    return CholeskyDecomposition(lower: lower)
}

public func isPositiveDefinite<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> Bool {
    (try? choleskyDecomposition(matrix)) != nil
}

public func isCholesky<S: RealFloatingScalar>(_ factor: Matrix<S>, of matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    factor.rows == matrix.rows
        && factor.columns == matrix.columns
        && isLowerTriangular(factor, tolerance: tolerance)
        && norm(factor * factor.t - matrix, .frobenius) <= tolerance * max(S(1), norm(matrix, .frobenius))
}

public func singularValueDecomposition<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> SVDDecomposition<S> {
    if let accelerated: SVDDecomposition<S> = try? acceleratedSVD(matrix) {
        return accelerated
    }
    let ata = matrix.t * matrix
    let eigen = jacobiEigen(ata)
    let pairs = eigen.values.enumerated().map { index, value in
        (index: index, sigma: S.sqrt(max(value.real, .zero)))
    }.sorted { lhs, rhs in lhs.sigma > rhs.sigma }

    var singulars = Array(repeating: S.zero, count: matrix.columns)
    var v = Matrix<S>.zeros(matrix.columns, matrix.columns)
    for (newIndex, pair) in pairs.enumerated() {
        singulars[newIndex] = pair.sigma
        for row in 0..<matrix.columns {
            v[row, newIndex] = eigen.vectors![row, pair.index].real
        }
    }

    var u = Matrix<S>.zeros(matrix.rows, matrix.columns)
    for column in 0..<matrix.columns {
        let sigma = singulars[column]
        guard sigma > S.tolerance else { continue }
        let vColumn = getMatrixColumn(v, column)
        let av = matrix * vColumn.asColumnMatrix()
        for row in 0..<matrix.rows {
            u[row, column] = av[row, 0] / sigma
        }
    }

    return reorderSVD(SVDDecomposition(u: u, singularValues: Vector(singulars), v: v))
}

public func reorderSVD<S: RealFloatingScalar>(_ svd: SVDDecomposition<S>) -> SVDDecomposition<S> {
    let order = (0..<svd.singularValues.count).sorted { svd.singularValues[$0] > svd.singularValues[$1] }
    var s = Array(repeating: S.zero, count: svd.singularValues.count)
    var u = Matrix<S>.zeros(svd.u.rows, svd.u.columns)
    var v = Matrix<S>.zeros(svd.v.rows, svd.v.columns)
    for (newIndex, oldIndex) in order.enumerated() {
        s[newIndex] = svd.singularValues[oldIndex]
        for row in 0..<svd.u.rows { u[row, newIndex] = svd.u[row, oldIndex] }
        for row in 0..<svd.v.rows { v[row, newIndex] = svd.v[row, oldIndex] }
    }
    return SVDDecomposition(u: u, singularValues: Vector(s), v: v)
}

private func rankRoutine<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S? = nil) -> Int {
    let svd = singularValueDecomposition(matrix)
    let maxSigma = svd.singularValues.count == 0 ? S.zero : svd.singularValues[0]
    let tol = tolerance ?? S(max(matrix.rows, matrix.columns)) * S.tolerance * maxSigma
    return (0..<svd.singularValues.count).filter { svd.singularValues[$0] > tol }.count
}

public func rank<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S? = nil) -> Int {
    rankRoutine(matrix, tolerance: tolerance)
}

public func nullity<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S? = nil) -> Int {
    matrix.columns - rank(matrix, tolerance: tolerance)
}

public func conditionNumber<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> S {
    let svd = singularValueDecomposition(matrix)
    let values = (0..<svd.singularValues.count).map { svd.singularValues[$0] }.filter { $0 > S.tolerance }
    guard let maxValue = values.first, let minValue = values.last, minValue > .zero else { return .infinity }
    return maxValue / minValue
}

public func pseudoInverseMoorePenrose<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S? = nil) -> Matrix<S> {
    let svd = singularValueDecomposition(matrix)
    let maxSigma = svd.singularValues.count == 0 ? S.zero : svd.singularValues[0]
    let tol = tolerance ?? S(max(matrix.rows, matrix.columns)) * S.tolerance * maxSigma
    var sigmaPlus = Matrix<S>.zeros(svd.v.columns, svd.u.columns)
    for i in 0..<svd.singularValues.count where svd.singularValues[i] > tol {
        sigmaPlus[i, i] = S.one / svd.singularValues[i]
    }
    return svd.v * sigmaPlus * svd.u.t
}

public func solveSystemPseudoInverse<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>) -> Vector<S> {
    Vector((pseudoInverseMoorePenrose(matrix) * b.asColumnMatrix()).rowMajorElements())
}

public func solveSystemOrdinaryLeastSquares<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>) throws -> Vector<S> {
    if let accelerated: Vector<S> = try acceleratedLeastSquares(matrix, b) {
        return accelerated
    }
    let normal = matrix.t * matrix
    let rhs = matrix.t * b.asColumnMatrix()
    return try solveSystemPLU(normal, Vector(rhs.rowMajorElements()))
}

public func fundamentalSubspaces<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S? = nil) -> FundamentalSubspaces<S> {
    if let accelerated: FundamentalSubspaces<S> = try? acceleratedFundamentalSubspaces(matrix, tolerance: tolerance) {
        return accelerated
    }
    let svd = singularValueDecomposition(matrix)
    let maxSigma = svd.singularValues.count == 0 ? S.zero : svd.singularValues[0]
    let tol = tolerance ?? S(max(matrix.rows, matrix.columns)) * S.tolerance * maxSigma
    let positive = (0..<svd.singularValues.count).filter { svd.singularValues[$0] > tol }
    let zero = (0..<svd.singularValues.count).filter { svd.singularValues[$0] <= tol }

    let columnSpace = columnsFrom(svd.u, indices: positive)
    let rowSpace = columnsFrom(svd.v, indices: positive)
    let nullSpace = columnsFrom(svd.v, indices: zero)
    let leftNullSpace = Matrix<S>.zeros(matrix.rows, max(matrix.rows - positive.count, 0))
    return FundamentalSubspaces(columnSpace: columnSpace, leftNullSpace: leftNullSpace, rowSpace: rowSpace, nullSpace: nullSpace)
}

func fundemantalSubspaces<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S? = nil) -> FundamentalSubspaces<S> {
    fundamentalSubspaces(matrix, tolerance: tolerance)
}

public func jacobiEigen<S: RealFloatingScalar>(_ matrix: Matrix<S>, maxIterations: Int = 100, tolerance: S = S.tolerance * S(100)) -> EigenDecomposition<S> {
    precondition(matrix.isSquare, "Jacobi eigen decomposition requires a square matrix.")
    let n = matrix.rows
    var a = matrix
    var v = Matrix<S>.eye(n)
    guard n > 0 else { return EigenDecomposition(values: [], vectors: v.map { Complex<S>($0) }) }

    for _ in 0..<maxIterations {
        var p = 0
        var q = min(1, n - 1)
        var largest = S.zero
        if n >= 2 {
            for i in 0..<(n - 1) {
                for j in (i + 1)..<n {
                    let value = S.abs(a[i, j])
                    if value > largest {
                        largest = value
                        p = i
                        q = j
                    }
                }
            }
        }
        if largest <= tolerance { break }

        let app = a[p, p]
        let aqq = a[q, q]
        let apq = a[p, q]
        let tau = (aqq - app) / (S(2) * apq)
        let t = tau >= .zero
            ? S.one / (tau + S.sqrt(S.one + tau * tau))
            : -S.one / (-tau + S.sqrt(S.one + tau * tau))
        let c = S.one / S.sqrt(S.one + t * t)
        let s = t * c

        for k in 0..<n where k != p && k != q {
            let aik = a[k, p]
            let akq = a[k, q]
            a[k, p] = c * aik - s * akq
            a[p, k] = a[k, p]
            a[k, q] = s * aik + c * akq
            a[q, k] = a[k, q]
        }
        a[p, p] = c * c * app - S(2) * s * c * apq + s * s * aqq
        a[q, q] = s * s * app + S(2) * s * c * apq + c * c * aqq
        a[p, q] = .zero
        a[q, p] = .zero

        for k in 0..<n {
            let vip = v[k, p]
            let viq = v[k, q]
            v[k, p] = c * vip - s * viq
            v[k, q] = s * vip + c * viq
        }
    }

    let values = (0..<n).map { Complex<S>(a[$0, $0]) }
    return sortEigenpairs(EigenDecomposition(values: values, vectors: v.map { Complex<S>($0) }), key: .realPart)
}

public func balanceMatrix<S: RealFloatingScalar>(_ matrix: Matrix<S>, radix: S = S(2)) -> (balanced: Matrix<S>, scales: Vector<S>) {
    precondition(matrix.isSquare, "Balancing requires a square matrix.")
    if radix == S(2), let accelerated: (balanced: Matrix<S>, scales: Vector<S>) = try? acceleratedBalance(matrix) {
        return accelerated
    }
    let n = matrix.rows
    var balanced = matrix
    var scales = Array(repeating: S.one, count: n)
    var changed = true

    while changed {
        changed = false
        for i in 0..<n {
            var rowNorm = S.zero
            var columnNorm = S.zero
            for j in 0..<n where j != i {
                rowNorm += S.abs(balanced[i, j])
                columnNorm += S.abs(balanced[j, i])
            }
            guard rowNorm > .zero && columnNorm > .zero else { continue }
            var f = S.one
            var c = columnNorm
            var r = rowNorm
            while c < r / radix {
                c *= radix
                r /= radix
                f *= radix
            }
            while c >= r * radix {
                c /= radix
                r *= radix
                f /= radix
            }
            if f != .one && (rowNorm + columnNorm) > S(0.95) * (r + c) {
                changed = true
                scales[i] *= f
                for j in 0..<n { balanced[i, j] /= f }
                for j in 0..<n { balanced[j, i] *= f }
            }
        }
    }

    return (balanced, Vector(scales))
}

public func upperHessenberg<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> Matrix<S> {
    precondition(matrix.isSquare, "Hessenberg reduction requires a square matrix.")
    let n = matrix.rows
    var h = matrix
    guard n > 2 else { return h }

    for k in 0..<(n - 2) {
        let x = Vector(((k + 1)..<n).map { h[$0, k] })
        let reflector = householderVector(x)
        let v = reflector.v
        let beta = reflector.beta
        guard beta != .zero else { continue }

        for j in k..<n {
            var projection = S.zero
            for i in 0..<v.count { projection += v[i] * h[k + 1 + i, j] }
            projection *= beta
            for i in 0..<v.count { h[k + 1 + i, j] -= projection * v[i] }
        }
        for i in 0..<n {
            var projection = S.zero
            for j in 0..<v.count { projection += h[i, k + 1 + j] * v[j] }
            projection *= beta
            for j in 0..<v.count { h[i, k + 1 + j] -= projection * v[j] }
        }
    }
    return h
}

public func qrAlgorithmBasic<S: RealFloatingScalar>(_ matrix: Matrix<S>, iterations: Int = 100) -> Matrix<S> {
    precondition(matrix.isSquare, "QR algorithm requires a square matrix.")
    var a = matrix
    for _ in 0..<iterations {
        let decomposition = qr(a)
        a = decomposition.r * decomposition.q
    }
    return a
}

public func qrAlgorithm<S: RealFloatingScalar>(_ matrix: Matrix<S>, iterations: Int = 100) -> Matrix<S> {
    precondition(matrix.isSquare, "QR algorithm requires a square matrix.")
    var a = upperHessenberg(matrix)
    let n = a.rows
    for _ in 0..<iterations {
        let shift = a[n - 1, n - 1]
        let shifted = a - shiftMatrix(n, shift)
        let decomposition = qr(shifted)
        a = decomposition.r * decomposition.q + shiftMatrix(n, shift)
    }
    return a
}

public func francisQRStep<S: RealFloatingScalar>(_ hessenberg: Matrix<S>) -> Matrix<S> {
    qrAlgorithm(hessenberg, iterations: 1)
}

public func hessRealSchurForm<S: RealFloatingScalar>(_ matrix: Matrix<S>, iterations: Int = 100) -> Matrix<S> {
    qrAlgorithm(upperHessenberg(matrix), iterations: iterations)
}

public func realSchurFormDecomposition<S: RealFloatingScalar>(_ matrix: Matrix<S>, iterations: Int = 100) -> RealSchurDecomposition<S> {
    precondition(matrix.isSquare, "Schur decomposition requires a square matrix.")
    var a = upperHessenberg(matrix)
    var qTotal = Matrix<S>.eye(matrix.rows)
    let n = matrix.rows
    for _ in 0..<iterations {
        let shift = a[n - 1, n - 1]
        let decomposition = qr(a - shiftMatrix(n, shift))
        a = decomposition.r * decomposition.q + shiftMatrix(n, shift)
        qTotal = qTotal * decomposition.q
    }
    return RealSchurDecomposition(q: qTotal, t: a)
}

public func similarityTransformsRealSchurForm<S: RealFloatingScalar>(_ matrix: Matrix<S>, iterations: Int = 100) -> RealSchurDecomposition<S> {
    realSchurFormDecomposition(matrix, iterations: iterations)
}

public func eigenPairsRealSchurWithExceptionalShift<S: RealFloatingScalar>(_ matrix: Matrix<S>, iterations: Int = 100) -> EigenDecomposition<S> {
    eigenValuesVectors(matrix, iterations: iterations)
}

public func eigenValuesVectors<S: RealFloatingScalar>(_ matrix: Matrix<S>, iterations: Int = 200) -> EigenDecomposition<S> {
    precondition(matrix.isSquare, "Eigen decomposition requires a square matrix.")
    if let accelerated: EigenDecomposition<S> = try? acceleratedEigen(matrix) {
        return accelerated
    }
    if isSymmetric(matrix, tolerance: S.tolerance * S(100)) {
        return jacobiEigen(matrix, maxIterations: iterations)
    }
    let schur = realSchurFormDecomposition(matrix, iterations: iterations)
    let values = eigenvaluesFromQuasiTriangular(schur.t)
    return EigenDecomposition(values: values, vectors: nil)
}

public func powerMethod<S: RealFloatingScalar>(_ matrix: Matrix<S>, maxIterations: Int = 1000, tolerance: S = S.tolerance * S(100)) -> (value: S, vector: Vector<S>) {
    precondition(matrix.isSquare, "Power method requires a square matrix.")
    var vector = Vector(Array(repeating: S.one / S.sqrt(S(matrix.rows)), count: matrix.rows))
    var value = S.zero
    for _ in 0..<maxIterations {
        let nextMatrix = matrix * vector.asColumnMatrix()
        var next = Vector(nextMatrix.rowMajorElements())
        let length = norm(next)
        guard length > .zero else { break }
        next = next / length
        let nextValue = next.dot(Vector((matrix * next.asColumnMatrix()).rowMajorElements()))
        if S.abs(nextValue - value) <= tolerance {
            return (nextValue, next)
        }
        value = nextValue
        vector = next
    }
    return (value, vector)
}

public func shiftedInversePM<S: RealFloatingScalar>(_ matrix: Matrix<S>, shift: S, maxIterations: Int = 100, tolerance: S = S.tolerance * S(100)) throws -> (value: S, vector: Vector<S>) {
    precondition(matrix.isSquare, "Inverse iteration requires a square matrix.")
    var vector = Vector(Array(repeating: S.one / S.sqrt(S(matrix.rows)), count: matrix.rows))
    var value = shift
    let shifted = matrix - shiftMatrix(matrix.rows, shift)
    for _ in 0..<maxIterations {
        var next = try solveSystemPLU(shifted, vector)
        next = next / norm(next)
        let av = Vector((matrix * next.asColumnMatrix()).rowMajorElements())
        let nextValue = next.dot(av) / next.dot(next)
        if S.abs(nextValue - value) <= tolerance {
            return (nextValue, next)
        }
        value = nextValue
        vector = next
    }
    return (value, vector)
}

public func spectralRadius<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> S {
    eigenValuesVectors(matrix).values.reduce(S.zero) { max($0, complexMagnitude($1)) }
}

public func sortEigenpairs<S: RealFloatingScalar>(_ decomposition: EigenDecomposition<S>, key: EigenSortKey = .realPart) -> EigenDecomposition<S> {
    let order = decomposition.values.indices.sorted { lhs, rhs in
        switch key {
        case .realPart:
            if decomposition.values[lhs].real == decomposition.values[rhs].real {
                return decomposition.values[lhs].imaginary < decomposition.values[rhs].imaginary
            }
            return decomposition.values[lhs].real < decomposition.values[rhs].real
        case .magnitude:
            return complexMagnitude(decomposition.values[lhs]) > complexMagnitude(decomposition.values[rhs])
        }
    }
    let values = order.map { decomposition.values[$0] }
    guard let vectors = decomposition.vectors else { return EigenDecomposition(values: values) }
    return EigenDecomposition(values: values, vectors: columnsFrom(vectors, indices: order))
}

public func normalizeEigenVectors<S: RealFloatingScalar>(_ vectors: Matrix<S>) -> Matrix<S> {
    var result = vectors
    for j in 0..<vectors.columns {
        let column = getMatrixColumn(vectors, j)
        let length = norm(column)
        guard length > .zero else { continue }
        for i in 0..<vectors.rows {
            result[i, j] /= length
        }
    }
    return result
}

public func scaleVectors<S: MatrixScalar>(_ vectors: Matrix<S>, scales: Vector<S>) -> Matrix<S> {
    precondition(vectors.columns == scales.count, "Scale vector length must match number of columns.")
    var result = vectors
    for j in 0..<vectors.columns {
        for i in 0..<vectors.rows {
            result[i, j] *= scales[j]
        }
    }
    return result
}

public func charPolyCoefficientsFaddeevLeverrier<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> [S] {
    precondition(matrix.isSquare, "Characteristic polynomial requires a square matrix.")
    let n = matrix.rows
    var coefficients = Array(repeating: S.zero, count: n + 1)
    coefficients[n] = .one
    var b = Matrix<S>.eye(n)
    for k in 1...n {
        let ab = matrix * b
        let ck = -trace(ab) / S(k)
        coefficients[n - k] = ck
        b = ab + ck * Matrix<S>.eye(n)
    }
    return coefficients
}

public func characteristicPolynomialRoots<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> [Complex<S>] {
    let coefficients = charPolyCoefficientsFaddeevLeverrier(matrix)
    let n = coefficients.count - 1
    guard n > 0 else { return [] }
    var companion = Matrix<S>.zeros(n, n)
    if n > 1 {
        for i in 1..<n { companion[i, i - 1] = .one }
    }
    let leading = coefficients[n]
    for i in 0..<n {
        companion[i, n - 1] = -coefficients[i] / leading
    }
    return eigenValuesVectors(companion).values
}

public func adjoint<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> Matrix<S> {
    try determinant(matrix) * inverse(matrix)
}

public func sortPolyCoefficients<S: RealFloatingScalar>(_ roots: [Complex<S>], key: EigenSortKey = .realPart) -> [Complex<S>] {
    sortEigenpairs(EigenDecomposition(values: roots), key: key).values
}

public func solveSystemIterativelyReweightedLeastSquares<S: RealFloatingScalar>(
    _ matrix: Matrix<S>,
    _ b: Vector<S>,
    p: S = S(1),
    delta: S = S.tolerance.squareRoot(),
    maxIterations: Int = 50,
    tolerance: S = S.tolerance * S(100)
) throws -> IterativeSolverResult<S> {
    var x = try solveSystemOrdinaryLeastSquares(matrix, b)
    for iteration in 0..<maxIterations {
        let residual = Vector((matrix * x.asColumnMatrix()).rowMajorElements()) - b
        var weightedA = matrix
        var weightedB = b
        for i in 0..<matrix.rows {
            let weight = S(Foundation.pow(Double(max(S.abs(residual[i]), delta)), Double(p - S(2))))
            let scale = S.sqrt(weight)
            for j in 0..<matrix.columns { weightedA[i, j] *= scale }
            weightedB[i] *= scale
        }
        let next = try solveSystemOrdinaryLeastSquares(weightedA, weightedB)
        let step = next - x
        x = next
        if norm(step) <= tolerance {
            return IterativeSolverResult(solution: x, iterations: iteration + 1, residualNorm: norm(residual), converged: true)
        }
    }
    let residual = Vector((matrix * x.asColumnMatrix()).rowMajorElements()) - b
    return IterativeSolverResult(solution: x, iterations: maxIterations, residualNorm: norm(residual), converged: false)
}

public func solveSystemJacobi<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>, maxIterations: Int = 500, tolerance: S = S.tolerance * S(100)) -> IterativeSolverResult<S> {
    precondition(matrix.isSquare && matrix.rows == b.count, "Jacobi solve dimension mismatch.")
    var x = Vector(Array(repeating: S.zero, count: b.count))
    for iteration in 0..<maxIterations {
        var next = x
        for i in 0..<matrix.rows {
            var sigma = S.zero
            for j in 0..<matrix.columns where j != i { sigma += matrix[i, j] * x[j] }
            next[i] = (b[i] - sigma) / matrix[i, i]
        }
        let residual = residualNorm(matrix, next, b)
        x = next
        if residual <= tolerance { return IterativeSolverResult(solution: x, iterations: iteration + 1, residualNorm: residual, converged: true) }
    }
    return IterativeSolverResult(solution: x, iterations: maxIterations, residualNorm: residualNorm(matrix, x, b), converged: false)
}

public func solveSystemGaussSeidel<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>, maxIterations: Int = 500, tolerance: S = S.tolerance * S(100)) -> IterativeSolverResult<S> {
    precondition(matrix.isSquare && matrix.rows == b.count, "Gauss-Seidel solve dimension mismatch.")
    var x = Vector(Array(repeating: S.zero, count: b.count))
    for iteration in 0..<maxIterations {
        for i in 0..<matrix.rows {
            var sigma = S.zero
            for j in 0..<matrix.columns where j != i { sigma += matrix[i, j] * x[j] }
            x[i] = (b[i] - sigma) / matrix[i, i]
        }
        let residual = residualNorm(matrix, x, b)
        if residual <= tolerance { return IterativeSolverResult(solution: x, iterations: iteration + 1, residualNorm: residual, converged: true) }
    }
    return IterativeSolverResult(solution: x, iterations: maxIterations, residualNorm: residualNorm(matrix, x, b), converged: false)
}

public func solveSystemSuccessiveOverRelaxation<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>, omega: S, maxIterations: Int = 500, tolerance: S = S.tolerance * S(100)) -> IterativeSolverResult<S> {
    precondition(omega > .zero && omega < S(2), "SOR relaxation must satisfy 0 < omega < 2.")
    var x = Vector(Array(repeating: S.zero, count: b.count))
    for iteration in 0..<maxIterations {
        for i in 0..<matrix.rows {
            var sigma = S.zero
            for j in 0..<matrix.columns where j != i { sigma += matrix[i, j] * x[j] }
            let gs = (b[i] - sigma) / matrix[i, i]
            x[i] = (S.one - omega) * x[i] + omega * gs
        }
        let residual = residualNorm(matrix, x, b)
        if residual <= tolerance { return IterativeSolverResult(solution: x, iterations: iteration + 1, residualNorm: residual, converged: true) }
    }
    return IterativeSolverResult(solution: x, iterations: maxIterations, residualNorm: residualNorm(matrix, x, b), converged: false)
}

public func solveSystemConjugateGradient<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>, maxIterations: Int? = nil, tolerance: S = S.tolerance * S(100)) -> IterativeSolverResult<S> {
    precondition(matrix.isSquare && matrix.rows == b.count, "CG solve dimension mismatch.")
    var x = Vector(Array(repeating: S.zero, count: b.count))
    var r = b
    var p = r
    var rsOld = r.dot(r)
    let limit = maxIterations ?? matrix.rows
    for iteration in 0..<limit {
        let ap = Vector((matrix * p.asColumnMatrix()).rowMajorElements())
        let denom = p.dot(ap)
        guard S.abs(denom) > S.tolerance else { break }
        let alpha = rsOld / denom
        x = x + alpha * p
        r = r - alpha * ap
        let rsNew = r.dot(r)
        let residual = S.sqrt(rsNew)
        if residual <= tolerance { return IterativeSolverResult(solution: x, iterations: iteration + 1, residualNorm: residual, converged: true) }
        p = r + (rsNew / rsOld) * p
        rsOld = rsNew
    }
    return IterativeSolverResult(solution: x, iterations: limit, residualNorm: residualNorm(matrix, x, b), converged: false)
}

public func solveSystemKaczmarz<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>, relaxation: S = .one, maxIterations: Int = 500, tolerance: S = S.tolerance * S(100)) -> IterativeSolverResult<S> {
    var x = Vector(Array(repeating: S.zero, count: matrix.columns))
    for iteration in 0..<maxIterations {
        let row = iteration % matrix.rows
        let a = getMatrixRow(matrix, row)
        let denom = a.dot(a)
        if denom > .zero {
            let correction = relaxation * (b[row] - a.dot(x)) / denom
            x = x + correction * a
        }
        let residual = residualNorm(matrix, x, b)
        if residual <= tolerance { return IterativeSolverResult(solution: x, iterations: iteration + 1, residualNorm: residual, converged: true) }
    }
    return IterativeSolverResult(solution: x, iterations: maxIterations, residualNorm: residualNorm(matrix, x, b), converged: false)
}

public func polynomialEvaluation<S: RealFloatingScalar>(_ coefficients: [S], at x: S) -> S {
    coefficients.reversed().reduce(S.zero) { $0 * x + $1 }
}

public func polynomialEvaluation<R: RealFloatingScalar>(_ coefficients: [Complex<R>], at x: Complex<R>) -> Complex<R> {
    coefficients.reversed().reduce(Complex<R>(0)) { $0 * x + $1 }
}

public func polynomialDivision<S: RealFloatingScalar>(_ numerator: [S], by denominator: [S]) -> PolynomialDivisionResult<S> {
    precondition(!denominator.isEmpty && denominator.last! != .zero, "Denominator polynomial cannot be zero.")
    guard numerator.count >= denominator.count else { return PolynomialDivisionResult(quotient: [], remainder: numerator) }
    var remainder = numerator
    var quotient = Array(repeating: S.zero, count: numerator.count - denominator.count + 1)
    let divisorLeading = denominator.last!
    for k in stride(from: quotient.count - 1, through: 0, by: -1) {
        let coefficient = remainder[denominator.count - 1 + k] / divisorLeading
        quotient[k] = coefficient
        for j in 0..<denominator.count {
            remainder[j + k] -= coefficient * denominator[j]
        }
    }
    while let last = remainder.last, S.abs(last) <= S.tolerance { remainder.removeLast() }
    return PolynomialDivisionResult(quotient: quotient, remainder: remainder)
}

public func mullerRoots<S: RealFloatingScalar>(_ coefficients: [S], maxIterations: Int = 80, tolerance: S = S.tolerance * S(100)) -> [Complex<S>] {
    var complexCoefficients = coefficients.map { Complex<S>($0) }
    var roots: [Complex<S>] = []
    while complexCoefficients.count > 2 {
        var x0 = Complex<S>(0)
        var x1 = Complex<S>(1)
        var x2 = Complex<S>(2)
        for _ in 0..<maxIterations {
            let y0 = polynomialEvaluation(complexCoefficients, at: x0)
            let y1 = polynomialEvaluation(complexCoefficients, at: x1)
            let y2 = polynomialEvaluation(complexCoefficients, at: x2)
            let h0 = x1 - x0
            let h1 = x2 - x1
            let d0 = (y1 - y0) / h0
            let d1 = (y2 - y1) / h1
            let a = (d1 - d0) / (h1 + h0)
            let b = a * h1 + d1
            let c = y2
            let discriminant = sqrtComplex(b * b - Complex<S>(4) * a * c)
            let den1 = b + discriminant
            let den2 = b - discriminant
            let denominator = complexMagnitude(den1) > complexMagnitude(den2) ? den1 : den2
            guard complexMagnitude(denominator) > tolerance else { break }
            let dx = -Complex<S>(2) * c / denominator
            let next = x2 + dx
            if complexMagnitude(dx) <= tolerance {
                x2 = next
                break
            }
            x0 = x1
            x1 = x2
            x2 = next
        }
        roots.append(x2)
        complexCoefficients = syntheticDivide(complexCoefficients, root: x2)
    }
    if complexCoefficients.count == 2 {
        roots.append(-complexCoefficients[0] / complexCoefficients[1])
    }
    return roots
}

public func cheb<S: RealFloatingScalar>(_ n: Int) -> (differentiation: Matrix<S>, nodes: Vector<S>) {
    precondition(n >= 0, "N must be nonnegative.")
    if n == 0 { return (Matrix<S>.zeros(1, 1), Vector([.one])) }
    let nodes = (0...n).map { j in S(Foundation.cos(Double.pi * Double(j) / Double(n))) }
    let c = (0...n).map { j -> S in
        let endpoint = (j == 0 || j == n) ? S(2) : S.one
        return j % 2 == 0 ? endpoint : -endpoint
    }
    var d = Matrix<S>.zeros(n + 1, n + 1)
    for i in 0...n {
        for j in 0...n where i != j {
            d[i, j] = (c[i] / c[j]) / (nodes[i] - nodes[j])
        }
    }
    for i in 0...n {
        var rowSum = S.zero
        for j in 0...n where i != j { rowSum += d[i, j] }
        d[i, i] = -rowSum
    }
    return (d, Vector(nodes))
}

public func rationalApproximation(_ x: Double, tolerance: Double = 1e-12, maxDenominator: Int = 1_000_000) -> RationalApproximation {
    var lowerN = 0
    var lowerD = 1
    var upperN = 1
    var upperD = 0
    let sign = x < 0 ? -1 : 1
    let target = abs(x)
    while true {
        let mediantN = lowerN + upperN
        let mediantD = lowerD + upperD
        if mediantD > maxDenominator {
            let lower = Double(lowerN) / Double(lowerD)
            let upper = Double(upperN) / Double(upperD)
            return abs(target - lower) < abs(target - upper)
                ? RationalApproximation(numerator: sign * lowerN, denominator: lowerD)
                : RationalApproximation(numerator: sign * upperN, denominator: upperD)
        }
        let mediant = Double(mediantN) / Double(mediantD)
        if abs(mediant - target) <= tolerance {
            return RationalApproximation(numerator: sign * mediantN, denominator: mediantD)
        } else if mediant < target {
            lowerN = mediantN
            lowerD = mediantD
        } else {
            upperN = mediantN
            upperD = mediantD
        }
    }
}

public func elementStringtoArray(_ text: String) throws -> Vector<Double> {
    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.first == "[", trimmed.last == "]" else {
        throw LinAlgError.unsupported("Expected bracketed vector text, for example [1, 2, 3].")
    }
    let body = trimmed.dropFirst().dropLast()
    if body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return Vector([]) }
    let values = try body.split(separator: ",").map { part -> Double in
        guard let value = Double(part.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            throw LinAlgError.unsupported("Invalid numeric element: \(part).")
        }
        return value
    }
    return Vector(values)
}

public func + <S: MatrixScalar>(lhs: Vector<S>, rhs: Vector<S>) -> Vector<S> {
    precondition(lhs.count == rhs.count, "Vector addition requires equal lengths.")
    return Vector((0..<lhs.count).map { lhs[$0] + rhs[$0] })
}

public func - <S: MatrixScalar>(lhs: Vector<S>, rhs: Vector<S>) -> Vector<S> {
    precondition(lhs.count == rhs.count, "Vector subtraction requires equal lengths.")
    return Vector((0..<lhs.count).map { lhs[$0] - rhs[$0] })
}

public func * <S: MatrixScalar>(lhs: S, rhs: Vector<S>) -> Vector<S> {
    Vector((0..<rhs.count).map { lhs * rhs[$0] })
}

public func * <S: MatrixScalar>(lhs: Vector<S>, rhs: S) -> Vector<S> {
    rhs * lhs
}

public func innerProduct<S: RealFloatingScalar>(_ lhs: Vector<S>, _ rhs: Vector<S>) -> S {
    if let accelerated: S = try? acceleratedDot(lhs, rhs) {
        return accelerated
    }
    return lhs.dot(rhs)
}

public func innerProduct<R: RealFloatingScalar>(_ lhs: Vector<Complex<R>>, _ rhs: Vector<Complex<R>>) -> Complex<R> {
    precondition(lhs.count == rhs.count, "Inner product requires equal lengths.")
    var result = Complex<R>(0)
    for i in 0..<lhs.count {
        result += lhs[i].conjugate * rhs[i]
    }
    return result
}

public func crossProductVector<S: RealFloatingScalar>(_ lhs: Vector<S>, _ rhs: Vector<S>) -> Vector<S> {
    precondition(lhs.count == 3 && rhs.count == 3, "Cross product is defined for 3D vectors.")
    return Vector([
        lhs[1] * rhs[2] - lhs[2] * rhs[1],
        lhs[2] * rhs[0] - lhs[0] * rhs[2],
        lhs[0] * rhs[1] - lhs[1] * rhs[0],
    ])
}

public func angleBetweenVectors<S: RealFloatingScalar>(_ lhs: Vector<S>, _ rhs: Vector<S>) -> S {
    let denominator = norm(lhs) * norm(rhs)
    precondition(denominator > .zero, "Angle is undefined for the zero vector.")
    let cosine = min(S.one, max(-S.one, lhs.dot(rhs) / denominator))
    return S(Foundation.acos(Double(cosine)))
}

public func orthogonalProjectionUontoV<S: RealFloatingScalar>(_ u: Vector<S>, _ v: Vector<S>) -> Vector<S> {
    let denominator = v.dot(v)
    precondition(denominator > .zero, "Projection vector cannot be zero.")
    return (u.dot(v) / denominator) * v
}

public func meanOfVector<S: RealFloatingScalar>(_ vector: Vector<S>) -> S {
    precondition(vector.count > 0, "Mean requires a nonempty vector.")
    if let accelerated: S = acceleratedMean(vector) {
        return accelerated
    }
    return (0..<vector.count).reduce(S.zero) { $0 + vector[$1] } / S(vector.count)
}

public func vectorCorrelation<S: RealFloatingScalar>(_ lhs: Vector<S>, _ rhs: Vector<S>) -> S {
    precondition(lhs.count == rhs.count && lhs.count > 0, "Correlation requires equal nonempty lengths.")
    let lhsMean = meanOfVector(lhs)
    let rhsMean = meanOfVector(rhs)
    let centeredL = Vector((0..<lhs.count).map { lhs[$0] - lhsMean })
    let centeredR = Vector((0..<rhs.count).map { rhs[$0] - rhsMean })
    let denominator = norm(centeredL) * norm(centeredR)
    precondition(denominator > .zero, "Correlation is undefined for zero variance.")
    return centeredL.dot(centeredR) / denominator
}

public func complexAbsoluteValue<R: RealFloatingScalar>(_ value: Complex<R>) -> R {
    R.hypot(value.real, value.imaginary)
}

public func absComplex<R: RealFloatingScalar>(_ value: Complex<R>) -> R {
    complexAbsoluteValue(value)
}

public func sqrtComplex<R: RealFloatingScalar>(_ value: Complex<R>) -> Complex<R> {
    if value.imaginary == .zero {
        if value.real >= .zero { return Complex(R.sqrt(value.real)) }
        return Complex(.zero, R.sqrt(-value.real))
    }
    let magnitude = complexAbsoluteValue(value)
    let realPart = R.sqrt((magnitude + R.abs(value.real)) / R(2))
    let imaginaryMagnitude = R.sqrt((magnitude - R.abs(value.real)) / R(2))
    if value.real >= .zero {
        let imaginary = value.imaginary >= .zero ? imaginaryMagnitude : -imaginaryMagnitude
        return Complex(realPart, imaginary)
    } else {
        let real = value.imaginary >= .zero ? imaginaryMagnitude : -imaginaryMagnitude
        return Complex(real, realPart)
    }
}

public func conjugateMatrix<R: RealFloatingScalar>(_ matrix: Matrix<Complex<R>>) -> Matrix<Complex<R>> {
    matrix.map(\.conjugate)
}

public func complexTranspose<R: RealFloatingScalar>(_ matrix: Matrix<Complex<R>>) -> Matrix<Complex<R>> {
    matrix.t
}

public func conjugateTranspose<R: RealFloatingScalar>(_ matrix: Matrix<Complex<R>>) -> Matrix<Complex<R>> {
    conjugateMatrix(matrix.t)
}

public func conjugateTransposeVector<R: RealFloatingScalar>(_ vector: Vector<Complex<R>>) -> Matrix<Complex<R>> {
    let elements = (0..<vector.count).map { vector[$0].conjugate }
    return try! Matrix(rowMajor: elements, rows: 1, columns: vector.count)
}

public func multiplyComplexMatrix<R: RealFloatingScalar>(_ lhs: Matrix<Complex<R>>, _ rhs: Matrix<Complex<R>>) -> Matrix<Complex<R>> {
    lhs * rhs
}

public func subtractComplexMatrix<R: RealFloatingScalar>(_ lhs: Matrix<Complex<R>>, _ rhs: Matrix<Complex<R>>) -> Matrix<Complex<R>> {
    lhs - rhs
}

public func diagonalComplexMatrix<R: RealFloatingScalar>(_ diagonal: Vector<Complex<R>>) -> Matrix<Complex<R>> {
    Matrix.diag(diagonal)
}

public func getComplexMatrixColumn<R: RealFloatingScalar>(_ matrix: Matrix<Complex<R>>, _ column: Int) -> Vector<Complex<R>> {
    getMatrixColumn(matrix, column)
}

public func complexVectorNorm<R: RealFloatingScalar>(_ vector: Vector<Complex<R>>) -> R {
    var sum = R.zero
    for i in 0..<vector.count {
        let magnitude = complexAbsoluteValue(vector[i])
        sum += magnitude * magnitude
    }
    return R.sqrt(sum)
}

public func realMatrix<R: RealFloatingScalar>(_ matrix: Matrix<Complex<R>>) -> Matrix<R> {
    matrix.map(\.real)
}

public func imaginaryMatrix<R: RealFloatingScalar>(_ matrix: Matrix<Complex<R>>) -> Matrix<R> {
    matrix.map(\.imaginary)
}

public func realToComplex<R: RealFloatingScalar>(_ matrix: Matrix<R>) -> Matrix<Complex<R>> {
    matrix.map { Complex<R>($0) }
}

public func isUpperTriangular<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    for i in 0..<matrix.rows {
        for j in 0..<min(i, matrix.columns) where S.abs(matrix[i, j]) > tolerance {
            return false
        }
    }
    return true
}

public func isLowerTriangular<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    for i in 0..<matrix.rows {
        if i + 1 < matrix.columns {
            for j in (i + 1)..<matrix.columns where S.abs(matrix[i, j]) > tolerance {
                return false
            }
        }
    }
    return true
}

public func isDiagonal<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    isUpperTriangular(matrix, tolerance: tolerance) && isLowerTriangular(matrix, tolerance: tolerance)
}

public func isSymmetric<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    guard matrix.isSquare else { return false }
    for i in 0..<matrix.rows {
        for j in 0..<i where S.abs(matrix[i, j] - matrix[j, i]) > tolerance {
            return false
        }
    }
    return true
}

public func isUpperHessenberg<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    for i in 0..<matrix.rows {
        for j in 0..<matrix.columns where i > j + 1 && S.abs(matrix[i, j]) > tolerance {
            return false
        }
    }
    return true
}

public func isZeroMatrix<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    matrix.rowMajorElements().allSatisfy { S.abs($0) <= tolerance }
}

public func isToeplitz<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> Bool {
    for i in 1..<matrix.rows {
        for j in 1..<matrix.columns where S.abs(matrix[i, j] - matrix[i - 1, j - 1]) > tolerance {
            return false
        }
    }
    return true
}

public func isConvergentMatrix<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> Bool {
    norm(matrix, .infinity) < S.one
}

public func isRowDiagonallyDominant<S: RealFloatingScalar>(_ matrix: Matrix<S>, strict: Bool = false) -> Bool {
    for i in 0..<matrix.rows {
        var off = S.zero
        for j in 0..<matrix.columns where j != i { off += S.abs(matrix[i, j]) }
        if strict {
            if !(S.abs(matrix[i, i]) > off) { return false }
        } else if !(S.abs(matrix[i, i]) >= off) { return false }
    }
    return true
}

public func isColumnDiagonallyDominant<S: RealFloatingScalar>(_ matrix: Matrix<S>, strict: Bool = false) -> Bool {
    for j in 0..<matrix.columns {
        var off = S.zero
        for i in 0..<matrix.rows where i != j { off += S.abs(matrix[i, j]) }
        if strict {
            if !(S.abs(matrix[j, j]) > off) { return false }
        } else if !(S.abs(matrix[j, j]) >= off) { return false }
    }
    return true
}

public func isAbsoluteDiagonallyDominant<S: RealFloatingScalar>(_ matrix: Matrix<S>, strict: Bool = false) -> Bool {
    isRowDiagonallyDominant(matrix, strict: strict) && isColumnDiagonallyDominant(matrix, strict: strict)
}

public func isSingular<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S? = nil) -> Bool {
    guard matrix.isSquare else { return true }
    return rank(matrix, tolerance: tolerance) < matrix.rows
}

extension Matrix where Scalar: RealFloatingScalar {
    public func qr() -> QRDecomposition<Scalar> { qrRoutine(self) }

    public func svd() -> SVDDecomposition<Scalar> { singularValueDecomposition(self) }
    public func eig() -> EigenDecomposition<Scalar> { eigenValuesVectors(self) }
    public func cond() -> Scalar { conditionNumber(self) }
    public func solve(_ b: Vector<Scalar>) throws -> Vector<Scalar> { try solveSystemPLU(self, b) }
    public var inverse: Matrix<Scalar> { try! inverseRoutine(self) }

    public func cholesky() throws -> CholeskyDecomposition<Scalar> { try choleskyDecomposition(self) }
    public func determinant() throws -> Scalar { try determinantRoutine(self) }

    public func rank(tolerance: Scalar? = nil) -> Int { rankRoutine(self, tolerance: tolerance) }
}

extension Matrix where Scalar: ComplexFloatingScalar {
    public var h: Matrix<Scalar> {
        var result = self.t
        for i in 0..<result.rows {
            for j in 0..<result.columns {
                result[i, j] = result[i, j].conjugate
            }
        }
        return result
    }
}

private func modifiedGramSchmidt<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> QRDecomposition<S> {
    let m = matrix.rows
    let n = matrix.columns
    var qColumns = (0..<n).map { getMatrixColumn(matrix, $0) }
    var r = Matrix<S>.zeros(n, n)
    for i in 0..<n {
        r[i, i] = norm(qColumns[i])
        if r[i, i] > .zero { qColumns[i] = qColumns[i] / r[i, i] }
        if i + 1 < n {
            for j in (i + 1)..<n {
                r[i, j] = qColumns[i].dot(qColumns[j])
                qColumns[j] = qColumns[j] - r[i, j] * qColumns[i]
            }
        }
    }
    let columns = qColumns.map { column in (0..<column.count).map { index in column[index] } }
    return QRDecomposition(q: columnsToMatrix(columns, rows: m), r: r)
}

private func classicalGramSchmidt<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> QRDecomposition<S> {
    let m = matrix.rows
    let n = matrix.columns
    var qColumns: [Vector<S>] = []
    var r = Matrix<S>.zeros(n, n)
    for j in 0..<n {
        var v = getMatrixColumn(matrix, j)
        for i in 0..<j {
            r[i, j] = qColumns[i].dot(getMatrixColumn(matrix, j))
            v = v - r[i, j] * qColumns[i]
        }
        r[j, j] = norm(v)
        if r[j, j] > .zero { v = v / r[j, j] }
        qColumns.append(v)
    }
    let columns = qColumns.map { column in (0..<column.count).map { index in column[index] } }
    return QRDecomposition(q: columnsToMatrix(columns, rows: m), r: r)
}

private func solve<S: RealFloatingScalar>(lu: LUDecomposition<S>, _ b: Vector<S>) throws -> Vector<S> {
    let n = lu.l.rows
    precondition(b.count == n, "Solve dimension mismatch.")
    var y = Array(repeating: S.zero, count: n)
    var x = Array(repeating: S.zero, count: n)
    for i in 0..<n {
        var sum = b[lu.permutation[i]]
        for j in 0..<i { sum -= lu.l[i, j] * y[j] }
        y[i] = sum
    }
    for i in stride(from: n - 1, through: 0, by: -1) {
        var sum = y[i]
        if i + 1 < n {
            for j in (i + 1)..<n { sum -= lu.u[i, j] * x[j] }
        }
        guard S.abs(lu.u[i, i]) > S.tolerance else { throw LinAlgError.singularMatrix }
        x[i] = sum / lu.u[i, i]
    }
    return Vector(x)
}

private func residualNorm<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ x: Vector<S>, _ b: Vector<S>) -> S {
    norm(Vector((matrix * x.asColumnMatrix()).rowMajorElements()) - b)
}

private enum AcceleratedElementwiseOperation {
    case multiply
    case divide
}

private func acceleratedHadamard<S: MatrixScalar>(_ lhs: Matrix<S>, _ rhs: Matrix<S>) throws -> Matrix<S>? {
    try acceleratedElementwise(lhs, rhs, operation: .multiply)
}

private func acceleratedElementwiseDivide<S: FloatingScalar>(_ lhs: Matrix<S>, _ rhs: Matrix<S>) throws -> Matrix<S>? {
    try acceleratedElementwise(lhs, rhs, operation: .divide)
}

private func acceleratedElementwise<S: MatrixScalar>(
    _ lhs: Matrix<S>,
    _ rhs: Matrix<S>,
    operation: AcceleratedElementwiseOperation
) throws -> Matrix<S>? {
    if S.self == Double.self {
        let left = try doubleMatrix(lhs)
        let right = try doubleMatrix(rhs)
        let result = switch operation {
        case .multiply: try VDSPKernels.multiply(left, right)
        case .divide: try VDSPKernels.divide(left, right)
        }
        return try castDoubleMatrix(result, as: S.self)
    }

    if S.self == Float.self {
        let left = try floatMatrix(lhs)
        let right = try floatMatrix(rhs)
        let result = switch operation {
        case .multiply: try VDSPKernels.multiply(left, right)
        case .divide: try VDSPKernels.divide(left, right)
        }
        return try castFloatMatrix(result, as: S.self)
    }

    if S.self == Float16.self {
        let left = try promotedFloatMatrix(lhs)
        let right = try promotedFloatMatrix(rhs)
        let result = switch operation {
        case .multiply: try VDSPKernels.multiply(left, right)
        case .divide: try VDSPKernels.divide(left, right)
        }
        return try demotedFloat16Matrix(result, as: S.self)
    }

    return nil
}

private func acceleratedSum<S: MatrixScalar>(_ matrix: Matrix<S>) -> S? {
    if S.self == Double.self {
        return VDSPKernels.sum(try! doubleMatrix(matrix)) as? S
    }

    if S.self == Float.self {
        return VDSPKernels.sum(try! floatMatrix(matrix)) as? S
    }

    if S.self == Float16.self {
        let result = VDSPKernels.sum(try! promotedFloatMatrix(matrix))
        return Float16(result) as? S
    }

    return nil
}

private func acceleratedScalarMultiply<S: MatrixScalar>(_ matrix: Matrix<S>, _ scalar: S) throws -> Matrix<S>? {
    if S.self == Double.self {
        let result = try VDSPKernels.multiply(try doubleMatrix(matrix), scalar: scalar as! Double)
        return try castDoubleMatrix(result, as: S.self)
    }

    if S.self == Float.self {
        let result = try VDSPKernels.multiply(try floatMatrix(matrix), scalar: scalar as! Float)
        return try castFloatMatrix(result, as: S.self)
    }

    if S.self == Float16.self {
        let result = try VDSPKernels.multiply(try promotedFloatMatrix(matrix), scalar: Float(scalar as! Float16))
        return try demotedFloat16Matrix(result, as: S.self)
    }

    return nil
}

private func acceleratedGemm<S: MatrixScalar>(_ lhs: Matrix<S>, _ rhs: Matrix<S>) throws -> Matrix<S>? {
    if S.self == Double.self {
        let result = try BLASKernels.gemm(try doubleMatrix(lhs), try doubleMatrix(rhs))
        return try castDoubleMatrix(result, as: S.self)
    }

    if S.self == Float.self {
        let result = try BLASKernels.gemm(try floatMatrix(lhs), try floatMatrix(rhs))
        return try castFloatMatrix(result, as: S.self)
    }

    if S.self == Float16.self {
        let result = try BLASKernels.gemm(try promotedFloatMatrix(lhs), try promotedFloatMatrix(rhs))
        return try demotedFloat16Matrix(result, as: S.self)
    }

    return nil
}

private func acceleratedFrobeniusNorm<S: RealFloatingScalar>(_ matrix: Matrix<S>) -> S? {
    if S.self == Double.self {
        return BLASKernels.nrm2(try! doubleMatrix(matrix)) as? S
    }

    if S.self == Float.self {
        return BLASKernels.nrm2(try! floatMatrix(matrix)) as? S
    }

    if S.self == Float16.self {
        return Float16(BLASKernels.nrm2(try! promotedFloatMatrix(matrix))) as? S
    }

    return nil
}

private func acceleratedVectorASum<S: RealFloatingScalar>(_ vector: Vector<S>) -> S? {
    if S.self == Double.self {
        return BLASKernels.asum(doubleVector(vector)) as? S
    }

    if S.self == Float.self {
        return BLASKernels.asum(floatVector(vector)) as? S
    }

    if S.self == Float16.self {
        return Float16(BLASKernels.asum(promotedFloatVector(vector))) as? S
    }

    return nil
}

private func acceleratedVectorNorm<S: RealFloatingScalar>(_ vector: Vector<S>) -> S? {
    if S.self == Double.self {
        return BLASKernels.nrm2(doubleVector(vector)) as? S
    }

    if S.self == Float.self {
        return BLASKernels.nrm2(floatVector(vector)) as? S
    }

    if S.self == Float16.self {
        return Float16(BLASKernels.nrm2(promotedFloatVector(vector))) as? S
    }

    return nil
}

private func acceleratedDot<S: RealFloatingScalar>(_ lhs: Vector<S>, _ rhs: Vector<S>) throws -> S? {
    if S.self == Double.self {
        return try BLASKernels.dot(doubleVector(lhs), doubleVector(rhs)) as? S
    }

    if S.self == Float.self {
        return try BLASKernels.dot(floatVector(lhs), floatVector(rhs)) as? S
    }

    if S.self == Float16.self {
        let result = try BLASKernels.dot(promotedFloatVector(lhs), promotedFloatVector(rhs))
        return Float16(result) as? S
    }

    return nil
}

private func acceleratedMean<S: RealFloatingScalar>(_ vector: Vector<S>) -> S? {
    if S.self == Double.self {
        return VDSPKernels.mean(doubleVector(vector)) as? S
    }

    if S.self == Float.self {
        return VDSPKernels.mean(floatVector(vector)) as? S
    }

    if S.self == Float16.self {
        return Float16(VDSPKernels.mean(promotedFloatVector(vector))) as? S
    }

    return nil
}

private func acceleratedMatrixNorm<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ kind: MatrixNorm) -> S? {
    let lapackKind: LAPACKMatrixNorm
    switch kind {
    case .one:
        lapackKind = .one
    case .infinity:
        lapackKind = .infinity
    case .frobenius:
        lapackKind = .frobenius
    case .two:
        return nil
    }

    if S.self == Double.self {
        return LAPACKKernels.norm(try! doubleMatrix(matrix), kind: lapackKind) as? S
    }

    if S.self == Float.self {
        return LAPACKKernels.norm(try! floatMatrix(matrix), kind: lapackKind) as? S
    }

    if S.self == Float16.self {
        let result = LAPACKKernels.norm(try! promotedFloatMatrix(matrix), kind: lapackKind)
        return Float16(result) as? S
    }

    return nil
}

private func acceleratedLU<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> LUDecomposition<S>? {
    if S.self == Double.self {
        let result = try LAPACKKernels.lu(try doubleMatrix(matrix))
        return try LUDecomposition(
            l: castDoubleMatrix(result.l, as: S.self),
            u: castDoubleMatrix(result.u, as: S.self),
            permutation: result.permutation,
            parity: result.parity
        )
    }

    if S.self == Float.self {
        let result = try LAPACKKernels.lu(try floatMatrix(matrix))
        return try LUDecomposition(
            l: castFloatMatrix(result.l, as: S.self),
            u: castFloatMatrix(result.u, as: S.self),
            permutation: result.permutation,
            parity: result.parity
        )
    }

    if S.self == Float16.self {
        let result = try LAPACKKernels.lu(try promotedFloatMatrix(matrix))
        return try LUDecomposition(
            l: demotedFloat16Matrix(result.l, as: S.self),
            u: demotedFloat16Matrix(result.u, as: S.self),
            permutation: result.permutation,
            parity: result.parity
        )
    }

    return nil
}

private func acceleratedSolve<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>) throws -> Vector<S>? {
    if S.self == Double.self {
        return try castDoubleVector(LAPACKKernels.solve(try doubleMatrix(matrix), doubleVector(b)), as: S.self)
    }

    if S.self == Float.self {
        return try castFloatVector(LAPACKKernels.solve(try floatMatrix(matrix), floatVector(b)), as: S.self)
    }

    if S.self == Float16.self {
        return demotedFloat16Vector(try LAPACKKernels.solve(try promotedFloatMatrix(matrix), promotedFloatVector(b)), as: S.self)
    }

    return nil
}

private func acceleratedSolve<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ rhs: Matrix<S>) throws -> Matrix<S>? {
    if S.self == Double.self {
        return try castDoubleMatrix(LAPACKKernels.solve(try doubleMatrix(matrix), try doubleMatrix(rhs)), as: S.self)
    }

    if S.self == Float.self {
        return try castFloatMatrix(LAPACKKernels.solve(try floatMatrix(matrix), try floatMatrix(rhs)), as: S.self)
    }

    if S.self == Float16.self {
        return try demotedFloat16Matrix(
            LAPACKKernels.solve(try promotedFloatMatrix(matrix), try promotedFloatMatrix(rhs)),
            as: S.self
        )
    }

    return nil
}

private func acceleratedDeterminant<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> S? {
    if S.self == Double.self {
        return try LAPACKKernels.determinant(try doubleMatrix(matrix)) as? S
    }

    if S.self == Float.self {
        return try LAPACKKernels.determinant(try floatMatrix(matrix)) as? S
    }

    if S.self == Float16.self {
        return try Float16(LAPACKKernels.determinant(try promotedFloatMatrix(matrix))) as? S
    }

    return nil
}

private func acceleratedInverse<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> Matrix<S>? {
    if S.self == Double.self {
        return try castDoubleMatrix(LAPACKKernels.inverse(try doubleMatrix(matrix)), as: S.self)
    }

    if S.self == Float.self {
        return try castFloatMatrix(LAPACKKernels.inverse(try floatMatrix(matrix)), as: S.self)
    }

    if S.self == Float16.self {
        return try demotedFloat16Matrix(LAPACKKernels.inverse(try promotedFloatMatrix(matrix)), as: S.self)
    }

    return nil
}

private func acceleratedQR<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> QRDecomposition<S>? {
    if S.self == Double.self {
        let result = try LAPACKKernels.qr(try doubleMatrix(matrix))
        return try QRDecomposition(q: castDoubleMatrix(result.q, as: S.self), r: castDoubleMatrix(result.r, as: S.self))
    }

    if S.self == Float.self {
        let result = try LAPACKKernels.qr(try floatMatrix(matrix))
        return try QRDecomposition(q: castFloatMatrix(result.q, as: S.self), r: castFloatMatrix(result.r, as: S.self))
    }

    if S.self == Float16.self {
        let result = try LAPACKKernels.qr(try promotedFloatMatrix(matrix))
        return try QRDecomposition(q: demotedFloat16Matrix(result.q, as: S.self), r: demotedFloat16Matrix(result.r, as: S.self))
    }

    return nil
}

private func acceleratedCholesky<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> CholeskyDecomposition<S>? {
    if S.self == Double.self {
        return try CholeskyDecomposition(lower: castDoubleMatrix(LAPACKKernels.cholesky(try doubleMatrix(matrix)), as: S.self))
    }

    if S.self == Float.self {
        return try CholeskyDecomposition(lower: castFloatMatrix(LAPACKKernels.cholesky(try floatMatrix(matrix)), as: S.self))
    }

    if S.self == Float16.self {
        return try CholeskyDecomposition(lower: demotedFloat16Matrix(LAPACKKernels.cholesky(try promotedFloatMatrix(matrix)), as: S.self))
    }

    return nil
}

private func acceleratedSVD<S: RealFloatingScalar>(
    _ matrix: Matrix<S>,
    fullVectors: Bool = false
) throws -> SVDDecomposition<S>? {
    if S.self == Double.self {
        let result = try LAPACKKernels.svd(try doubleMatrix(matrix), fullVectors: fullVectors)
        return try SVDDecomposition(
            u: castDoubleMatrix(result.u, as: S.self),
            singularValues: castDoubleVector(result.singularValues, as: S.self),
            v: castDoubleMatrix(result.v, as: S.self)
        )
    }

    if S.self == Float.self {
        let result = try LAPACKKernels.svd(try floatMatrix(matrix), fullVectors: fullVectors)
        return try SVDDecomposition(
            u: castFloatMatrix(result.u, as: S.self),
            singularValues: castFloatVector(result.singularValues, as: S.self),
            v: castFloatMatrix(result.v, as: S.self)
        )
    }

    if S.self == Float16.self {
        let result = try LAPACKKernels.svd(try promotedFloatMatrix(matrix), fullVectors: fullVectors)
        return try SVDDecomposition(
            u: demotedFloat16Matrix(result.u, as: S.self),
            singularValues: demotedFloat16Vector(result.singularValues, as: S.self),
            v: demotedFloat16Matrix(result.v, as: S.self)
        )
    }

    return nil
}

private func acceleratedFundamentalSubspaces<S: RealFloatingScalar>(
    _ matrix: Matrix<S>,
    tolerance: S?
) throws -> FundamentalSubspaces<S>? {
    guard let svd: SVDDecomposition<S> = try acceleratedSVD(matrix, fullVectors: true) else {
        return nil
    }

    let maxSigma = svd.singularValues.count == 0 ? S.zero : svd.singularValues[0]
    let tol = tolerance ?? S(max(matrix.rows, matrix.columns)) * S.tolerance * maxSigma
    let rank = (0..<svd.singularValues.count).filter { svd.singularValues[$0] > tol }.count

    return FundamentalSubspaces(
        columnSpace: columnsFrom(svd.u, indices: Array(0..<rank)),
        leftNullSpace: columnsFrom(svd.u, indices: Array(rank..<svd.u.columns)),
        rowSpace: columnsFrom(svd.v, indices: Array(0..<rank)),
        nullSpace: columnsFrom(svd.v, indices: Array(rank..<svd.v.columns))
    )
}

private func acceleratedLeastSquares<S: RealFloatingScalar>(_ matrix: Matrix<S>, _ b: Vector<S>) throws -> Vector<S>? {
    if S.self == Double.self {
        return try castDoubleVector(LAPACKKernels.leastSquares(try doubleMatrix(matrix), doubleVector(b)), as: S.self)
    }

    if S.self == Float.self {
        return try castFloatVector(LAPACKKernels.leastSquares(try floatMatrix(matrix), floatVector(b)), as: S.self)
    }

    if S.self == Float16.self {
        return demotedFloat16Vector(try LAPACKKernels.leastSquares(try promotedFloatMatrix(matrix), promotedFloatVector(b)), as: S.self)
    }

    return nil
}

private func acceleratedBalance<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> (balanced: Matrix<S>, scales: Vector<S>)? {
    if S.self == Double.self {
        let result = try LAPACKKernels.balance(try doubleMatrix(matrix))
        return try (castDoubleMatrix(result.balanced, as: S.self), castDoubleVector(result.scales, as: S.self))
    }

    if S.self == Float.self {
        let result = try LAPACKKernels.balance(try floatMatrix(matrix))
        return try (castFloatMatrix(result.balanced, as: S.self), castFloatVector(result.scales, as: S.self))
    }

    if S.self == Float16.self {
        let result = try LAPACKKernels.balance(try promotedFloatMatrix(matrix))
        return try (demotedFloat16Matrix(result.balanced, as: S.self), demotedFloat16Vector(result.scales, as: S.self))
    }

    return nil
}

private func acceleratedEigen<S: RealFloatingScalar>(_ matrix: Matrix<S>) throws -> EigenDecomposition<S>? {
    if S.self == Double.self {
        let realMatrix = try doubleMatrix(matrix)
        if isSymmetric(matrix, tolerance: S.tolerance * S(100)) {
            let result = try LAPACKKernels.symmetricEigen(realMatrix)
            return try symmetricEigenDecomposition(
                values: castDoubleVector(result.values, as: S.self),
                vectors: castDoubleMatrix(result.vectors, as: S.self)
            )
        }

        let result = try LAPACKKernels.eigen(realMatrix)
        return try generalEigenDecomposition(
            realParts: castDoubleVector(result.realParts, as: S.self),
            imaginaryParts: castDoubleVector(result.imaginaryParts, as: S.self),
            packedVectors: castDoubleMatrix(result.rightEigenvectorsPacked, as: S.self)
        )
    }

    if S.self == Float.self {
        let realMatrix = try floatMatrix(matrix)
        if isSymmetric(matrix, tolerance: S.tolerance * S(100)) {
            let result = try LAPACKKernels.symmetricEigen(realMatrix)
            return try symmetricEigenDecomposition(
                values: castFloatVector(result.values, as: S.self),
                vectors: castFloatMatrix(result.vectors, as: S.self)
            )
        }

        let result = try LAPACKKernels.eigen(realMatrix)
        return try generalEigenDecomposition(
            realParts: castFloatVector(result.realParts, as: S.self),
            imaginaryParts: castFloatVector(result.imaginaryParts, as: S.self),
            packedVectors: castFloatMatrix(result.rightEigenvectorsPacked, as: S.self)
        )
    }

    if S.self == Float16.self {
        let realMatrix = try promotedFloatMatrix(matrix)
        if isSymmetric(matrix, tolerance: S.tolerance * S(100)) {
            let result = try LAPACKKernels.symmetricEigen(realMatrix)
            return try symmetricEigenDecomposition(
                values: demotedFloat16Vector(result.values, as: S.self),
                vectors: demotedFloat16Matrix(result.vectors, as: S.self)
            )
        }

        let result = try LAPACKKernels.eigen(realMatrix)
        return try generalEigenDecomposition(
            realParts: demotedFloat16Vector(result.realParts, as: S.self),
            imaginaryParts: demotedFloat16Vector(result.imaginaryParts, as: S.self),
            packedVectors: demotedFloat16Matrix(result.rightEigenvectorsPacked, as: S.self)
        )
    }

    return nil
}

private func symmetricEigenDecomposition<S: RealFloatingScalar>(
    values: Vector<S>,
    vectors: Matrix<S>
) throws -> EigenDecomposition<S> {
    EigenDecomposition(
        values: (0..<values.count).map { Complex<S>(values[$0]) },
        vectors: vectors.map { Complex<S>($0) }
    )
}

private func generalEigenDecomposition<S: RealFloatingScalar>(
    realParts: Vector<S>,
    imaginaryParts: Vector<S>,
    packedVectors: Matrix<S>
) throws -> EigenDecomposition<S> {
    let n = realParts.count
    var vectors = Matrix<Complex<S>>.zeros(packedVectors.rows, packedVectors.columns)
    var column = 0
    while column < n {
        let imaginary = imaginaryParts[column]
        if imaginary == .zero {
            for row in 0..<packedVectors.rows {
                vectors[row, column] = Complex<S>(packedVectors[row, column])
            }
            column += 1
        } else if imaginary > .zero && column + 1 < n {
            for row in 0..<packedVectors.rows {
                let real = packedVectors[row, column]
                let imag = packedVectors[row, column + 1]
                vectors[row, column] = Complex<S>(real, imag)
                vectors[row, column + 1] = Complex<S>(real, -imag)
            }
            column += 2
        } else {
            column += 1
        }
    }

    let values = (0..<n).map { Complex<S>(realParts[$0], imaginaryParts[$0]) }
    return EigenDecomposition(values: values, vectors: vectors)
}

private func doubleMatrix<S: MatrixScalar>(_ matrix: Matrix<S>) throws -> Matrix<Double> {
    try Matrix<Double>(
        rowMajor: matrix.rowMajorElements().map { $0 as! Double },
        rows: matrix.rows,
        columns: matrix.columns
    )
}

private func floatMatrix<S: MatrixScalar>(_ matrix: Matrix<S>) throws -> Matrix<Float> {
    try Matrix<Float>(
        rowMajor: matrix.rowMajorElements().map { $0 as! Float },
        rows: matrix.rows,
        columns: matrix.columns
    )
}

private func promotedFloatMatrix<S: MatrixScalar>(_ matrix: Matrix<S>) throws -> Matrix<Float> {
    try Matrix<Float>(
        rowMajor: matrix.rowMajorElements().map { Float($0 as! Float16) },
        rows: matrix.rows,
        columns: matrix.columns
    )
}

private func castDoubleMatrix<S: MatrixScalar>(_ matrix: Matrix<Double>, as scalar: S.Type) throws -> Matrix<S> {
    try Matrix<S>(
        rowMajor: matrix.rowMajorElements().map { $0 as! S },
        rows: matrix.rows,
        columns: matrix.columns
    )
}

private func castFloatMatrix<S: MatrixScalar>(_ matrix: Matrix<Float>, as scalar: S.Type) throws -> Matrix<S> {
    try Matrix<S>(
        rowMajor: matrix.rowMajorElements().map { $0 as! S },
        rows: matrix.rows,
        columns: matrix.columns
    )
}

private func demotedFloat16Matrix<S: MatrixScalar>(_ matrix: Matrix<Float>, as scalar: S.Type) throws -> Matrix<S> {
    try Matrix<S>(
        rowMajor: matrix.rowMajorElements().map { Float16($0) as! S },
        rows: matrix.rows,
        columns: matrix.columns
    )
}

private func doubleVector<S: RealFloatingScalar>(_ vector: Vector<S>) -> Vector<Double> {
    Vector((0..<vector.count).map { vector[$0] as! Double })
}

private func floatVector<S: RealFloatingScalar>(_ vector: Vector<S>) -> Vector<Float> {
    Vector((0..<vector.count).map { vector[$0] as! Float })
}

private func promotedFloatVector<S: RealFloatingScalar>(_ vector: Vector<S>) -> Vector<Float> {
    Vector((0..<vector.count).map { Float(vector[$0] as! Float16) })
}

private func castDoubleVector<S: RealFloatingScalar>(_ vector: Vector<Double>, as scalar: S.Type) -> Vector<S> {
    Vector((0..<vector.count).map { vector[$0] as! S })
}

private func castFloatVector<S: RealFloatingScalar>(_ vector: Vector<Float>, as scalar: S.Type) -> Vector<S> {
    Vector((0..<vector.count).map { vector[$0] as! S })
}

private func demotedFloat16Vector<S: RealFloatingScalar>(_ vector: Vector<Float>, as scalar: S.Type) -> Vector<S> {
    Vector((0..<vector.count).map { Float16(vector[$0]) as! S })
}

private func shiftMatrix<S: RealFloatingScalar>(_ n: Int, _ shift: S) -> Matrix<S> {
    shift * Matrix<S>.eye(n)
}

private func complexMagnitude<S: RealFloatingScalar>(_ value: Complex<S>) -> S {
    S.hypot(value.real, value.imaginary)
}

private func eigenvaluesFromQuasiTriangular<S: RealFloatingScalar>(_ matrix: Matrix<S>, tolerance: S = S.tolerance * S(100)) -> [Complex<S>] {
    var values: [Complex<S>] = []
    var i = 0
    while i < matrix.rows {
        if i + 1 < matrix.rows && S.abs(matrix[i + 1, i]) > tolerance {
            let a = matrix[i, i]
            let b = matrix[i, i + 1]
            let c = matrix[i + 1, i]
            let d = matrix[i + 1, i + 1]
            let tr = a + d
            let det = a * d - b * c
            let discriminant = tr * tr - S(4) * det
            if discriminant >= .zero {
                let root = S.sqrt(discriminant)
                values.append(Complex((tr + root) / S(2)))
                values.append(Complex((tr - root) / S(2)))
            } else {
                let imaginary = S.sqrt(-discriminant) / S(2)
                values.append(Complex(tr / S(2), imaginary))
                values.append(Complex(tr / S(2), -imaginary))
            }
            i += 2
        } else {
            values.append(Complex(matrix[i, i]))
            i += 1
        }
    }
    return values
}

private func syntheticDivide<R: RealFloatingScalar>(_ coefficients: [Complex<R>], root: Complex<R>) -> [Complex<R>] {
    var result = Array(repeating: Complex<R>(0), count: coefficients.count - 1)
    result[result.count - 1] = coefficients.last!
    if result.count >= 2 {
        for i in stride(from: result.count - 2, through: 0, by: -1) {
            result[i] = coefficients[i + 1] + root * result[i + 1]
        }
    }
    return result
}

private func columnsToMatrix<S: MatrixScalar>(_ columns: [[S]], rows: Int) -> Matrix<S> {
    let columnCount = columns.count
    var result = Matrix<S>(rows: rows, columns: columnCount)
    for j in 0..<columnCount {
        for i in 0..<rows {
            result[i, j] = columns[j][i]
        }
    }
    return result
}

private func columnsToMatrix<S: MatrixScalar>(_ columns: [Vector<S>]) -> Matrix<S> {
    let rows = columns.first?.count ?? 0
    return columnsToMatrix(columns.map { vector in (0..<vector.count).map { vector[$0] } }, rows: rows)
}

private func columnsFrom<S: MatrixScalar>(_ matrix: Matrix<S>, indices: [Int]) -> Matrix<S> {
    guard !indices.isEmpty else { return Matrix<S>.zeros(matrix.rows, 0) }
    return columnsToMatrix(indices.map { index in (0..<matrix.rows).map { matrix[$0, index] } }, rows: matrix.rows)
}

private extension Matrix {
    func copiedPublicRows(_ rows: [Int], columns: [Int]) -> Matrix<Scalar> {
        var elements: [Scalar] = []
        elements.reserveCapacity(rows.count * columns.count)
        for row in rows {
            for column in columns {
                elements.append(self[row, column])
            }
        }
        return try! Matrix(rowMajor: elements, rows: rows.count, columns: columns.count)
    }

    mutating func swapRows(_ lhs: Int, _ rhs: Int) {
        guard lhs != rhs else { return }
        for column in 0..<columns {
            let temp = self[lhs, column]
            self[lhs, column] = self[rhs, column]
            self[rhs, column] = temp
        }
    }
}
