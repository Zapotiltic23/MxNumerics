//
//  NumericalRoutinesTests.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

import ComplexModule
import MxNumerics
import Testing

@Suite
struct NumericalRoutinesTests {
    @Test
    func qrReconstructsInputAndProducesOrthogonalQ() throws {
        let a = try Matrix<Double>([
            [12, -51, 4],
            [6, 167, -68],
            [-4, 24, -41],
        ])

        let decomposition = qr(a)

        #expect(approximatelyEqual(decomposition.q * decomposition.r, a, tolerance: 1e-8))
        #expect(isOrthogonal(decomposition.q, tolerance: 1e-8))
    }

    @Test
    func pivotedLUSolvesDeterminantAndInverse() throws {
        let a = try Matrix<Double>([
            [4, 3],
            [6, 3],
        ])
        let b = Vector<Double>([10, 12])

        let x = try solveSystemPLU(a, b)
        let inverseA = try inverse(a)

        #expect(approximatelyEqual(Vector((a * x.asColumnMatrix()).rowMajorElements()), b, tolerance: 1e-9))
        #expect(abs(try determinant(a) + 6) < 1e-9)
        #expect(approximatelyEqual(a * inverseA, Matrix<Double>.eye(2), tolerance: 1e-9))
    }

    @Test
    func choleskyFactorsSPDMatrix() throws {
        let a = try Matrix<Double>([
            [25, 15, -5],
            [15, 18, 0],
            [-5, 0, 11],
        ])

        let factor = try choleskyDecomposition(a).lower

        #expect(isPositiveDefinite(a))
        #expect(isCholesky(factor, of: a, tolerance: 1e-9))
    }

    @Test
    func svdReconstructsDiagonalMatrixAndComputesRank() throws {
        let a = try Matrix<Double>([
            [3, 0],
            [0, 2],
            [0, 0],
        ])

        let decomposition = singularValueDecomposition(a)
        let reconstructed = decomposition.u * decomposition.sigma * decomposition.v.t

        #expect(approximatelyEqual(reconstructed, a, tolerance: 1e-9))
        #expect(decomposition.singularValues[0] == 3)
        #expect(decomposition.singularValues[1] == 2)
        #expect(rank(a) == 2)
        #expect(nullity(a) == 0)
    }

    @Test
    func symmetricEigenDecompositionAndSpectralRadius() throws {
        let a = try Matrix<Double>([
            [2, 0],
            [0, 5],
        ])

        let eigen = eigenValuesVectors(a)
        let values = eigen.values.map(\.real).sorted()

        #expect(approximatelyEqual(values[0], 2, tolerance: 1e-9))
        #expect(approximatelyEqual(values[1], 5, tolerance: 1e-9))
        #expect(approximatelyEqual(spectralRadius(a), 5, tolerance: 1e-9))
    }

    @Test
    func iterativeSolversConvergeOnSPDSystem() throws {
        let a = try Matrix<Double>([
            [4, 1],
            [1, 3],
        ])
        let b = Vector<Double>([1, 2])

        let cg = solveSystemConjugateGradient(a, b, tolerance: 1e-10)
        let jacobi = solveSystemJacobi(a, b, tolerance: 1e-8)

        #expect(cg.converged)
        #expect(jacobi.converged)
        #expect(approximatelyEqual(cg.solution, Vector([1.0 / 11.0, 7.0 / 11.0]), tolerance: 1e-8))
    }

    @Test
    func polynomialChebyshevAndParsingRoutinesWork() throws {
        #expect(polynomialEvaluation([1.0, 2.0, 3.0], at: 2.0) == 17)

        let division = polynomialDivision([-1.0, 0.0, 1.0], by: [-1.0, 1.0])
        #expect(approximatelyEqual(division.quotient, [1, 1], tolerance: 1e-9))

        let chebResult = cheb(4) as (differentiation: Matrix<Double>, nodes: Vector<Double>)
        #expect(chebResult.differentiation.rows == 5)
        #expect(chebResult.nodes.count == 5)

        let parsed = try elementStringtoArray("[1, 2.5, -3]")
        #expect(approximatelyEqual(parsed, Vector([1, 2.5, -3]), tolerance: 1e-12))
    }

    @Test
    func complexHelpersUseHermitianConventionsWhereDocumented() throws {
        let z = Complex<Double>(3, 4)
        let vector = Vector([Complex<Double>(1, 2), Complex<Double>(3, -1)])
        let matrix = try Matrix<Complex<Double>>([
            [Complex(1, 1), Complex(2, -1)],
        ])

        #expect(complexAbsoluteValue(z) == 5)
        #expect(innerProduct(vector, vector).imaginary == 0)
        #expect(approximatelyEqual(innerProduct(vector, vector).real, 15, tolerance: 1e-12))
        #expect(conjugateTranspose(matrix).rows == 2)
        #expect(conjugateTranspose(matrix)[0, 0] == Complex(1, -1))
    }

    @Test
    func sectionSixCatalogSmokeTest() throws {
        let a: Matrix<Double> = [
            [2, 1],
            [1, 2],
        ]
        let b = Vector<Double>([1, 0])

        _ = try Matrix(rowMajor: [1.0, 2.0, 3.0, 4.0], shape: Shape(rows: 2, columns: 2))
        #expect(((a ^ 2).rowMajorElements()) == [5, 4, 4, 5])
        #expect(approximatelyEqual((a / a), Matrix<Double>.eye(2), tolerance: 1e-9))
        #expect(repmat(a, rows: 2, columns: 2).rows == 4)
        #expect(try reshape(a, rows: 1, columns: 4).columns == 4)
        #expect(embedMatrix(Matrix<Double>.eye(1), into: a, row: 0, column: 0)[0, 0] == 1)
        #expect(dropRow(a, 0).rows == 1)
        #expect(dropColumn(a, 0).columns == 1)
        #expect(obtainSubMatrix(a, droppingRow: 0, column: 0).rowMajorElements() == [2])
        #expect(getDiagonal(a).count == 2)
        #expect(try principalMinors(a).count == 2)

        _ = try luDecompositionDoolittle(a)
        _ = try croutsLUwithPartialImplicitPivoting(a)
        _ = gramSchmidtFactorization(a)
        _ = balanceMatrix(a)
        #expect(isUpperHessenberg(upperHessenberg(a)))
        _ = francisQRStep(a)
        _ = qrAlgorithmBasic(a, iterations: 2)
        _ = qrAlgorithm(a, iterations: 2)
        _ = hessRealSchurForm(a, iterations: 2)
        _ = realSchurFormDecomposition(a, iterations: 2)
        _ = similarityTransformsRealSchurForm(a, iterations: 2)
        _ = eigenPairsRealSchurWithExceptionalShift(a, iterations: 2)
        _ = powerMethod(a, maxIterations: 10)
        _ = try shiftedInversePM(a, shift: 1.5, maxIterations: 10)
        _ = normalizeEigenVectors(a)
        _ = scaleVectors(a, scales: Vector([2, 3]))

        let coefficients = charPolyCoefficientsFaddeevLeverrier(a)
        #expect(coefficients.count == 3)
        _ = characteristicPolynomialRoots(a)
        _ = try adjoint(a)
        _ = sortPolyCoefficients([Complex<Double>(1), Complex<Double>(0)])

        #expect(norm(a, .one) == 3)
        #expect(norm(a, .infinity) == 3)
        #expect(conditionNumber(a) > 1)
        _ = fundamentalSubspaces(a)
        _ = fundamentalSubspaces(a)
        _ = solveSystemPseudoInverse(a, b)
        _ = try solveSystemOrdinaryLeastSquares(a, b)
        _ = try solveSystemIterativelyReweightedLeastSquares(a, b, maxIterations: 2)
        _ = solveSystemGaussSeidel(a, b, maxIterations: 20)
        _ = solveSystemSuccessiveOverRelaxation(a, b, omega: 1.1, maxIterations: 20)
        _ = solveSystemKaczmarz(a, b, maxIterations: 20)

        #expect(isLowerTriangular(try Matrix<Double>([[1, 0], [2, 3]])))
        #expect(isUpperTriangular(try Matrix<Double>([[1, 2], [0, 3]])))
        #expect(isDiagonal(Matrix<Double>.eye(2)))
        #expect(isSymmetric(a))
        #expect(isToeplitz(try Matrix<Double>([[1, 2, 3], [4, 1, 2]])))
        #expect(isZeroMatrix(Matrix<Double>.zeros(2, 2)))
        #expect(isRowDiagonallyDominant(a))
        #expect(isColumnDiagonallyDominant(a))
        #expect(isAbsoluteDiagonallyDominant(a))
        #expect(!isSingular(a))
        #expect(isConvergentMatrix(0.1 * Matrix<Double>.eye(2)))

        #expect(innerProduct(Vector([1.0, 2.0]), Vector([3.0, 4.0])) == 11)
        #expect(crossProductVector(Vector([1.0, 0, 0]), Vector([0.0, 1, 0])).asColumnMatrix().rowMajorElements() == [0, 0, 1])
        #expect(approximatelyEqual(angleBetweenVectors(Vector([1.0, 0]), Vector([0.0, 1])), Double.pi / 2, tolerance: 1e-12))
        #expect(orthogonalProjectionUontoV(Vector([2.0, 2]), Vector([1.0, 0])).asColumnMatrix().rowMajorElements() == [2, 0])
        #expect(meanOfVector(Vector([1.0, 2, 3])) == 2)
        #expect(approximatelyEqual(vectorCorrelation(Vector([1.0, 2, 3]), Vector([1.0, 2, 3])), 1, tolerance: 1e-12))

        let rational = rationalApproximation(0.333333333333, tolerance: 1e-9)
        #expect(rational == RationalApproximation(numerator: 1, denominator: 3))
    }
}

private func approximatelyEqual(_ lhs: Double, _ rhs: Double, tolerance: Double) -> Bool {
    abs(lhs - rhs) <= tolerance
}

private func approximatelyEqual(_ lhs: [Double], _ rhs: [Double], tolerance: Double) -> Bool {
    lhs.count == rhs.count && zip(lhs, rhs).allSatisfy { abs($0 - $1) <= tolerance }
}

private func approximatelyEqual(_ lhs: Vector<Double>, _ rhs: Vector<Double>, tolerance: Double) -> Bool {
    lhs.count == rhs.count && (0..<lhs.count).allSatisfy { abs(lhs[$0] - rhs[$0]) <= tolerance }
}

private func approximatelyEqual(_ lhs: Matrix<Double>, _ rhs: Matrix<Double>, tolerance: Double) -> Bool {
    lhs.rows == rhs.rows
        && lhs.columns == rhs.columns
        && zip(lhs.rowMajorElements(), rhs.rowMajorElements()).allSatisfy { abs($0 - $1) <= tolerance }
}
