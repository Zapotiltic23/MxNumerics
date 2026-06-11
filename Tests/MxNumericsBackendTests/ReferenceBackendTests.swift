//
//  ReferenceBackendTests.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
@testable import MxNumerics
import Testing

@Suite
struct ReferenceBackendTests {
    @Test
    func asyncGemmMatchesMatrixMultiplication() async throws {
        let a = try Matrix<Double>([
            [1, 2, 3],
            [4, 5, 6],
        ])
        let b = try Matrix<Double>([
            [7, 8],
            [9, 10],
            [11, 12],
        ])

        let aRef = BufferRef(a)
        let bRef = BufferRef(b)
        let productRef = try await ReferenceBackend().gemm(aRef, bRef)
        let result = productRef.matrix()

        #expect(a.rows == 2)
        #expect(a.columns == 3)
        #expect(aRef.rows == 2)
        #expect(aRef.columns == 3)
        #expect(productRef.rows == 2)
        #expect(productRef.columns == 2)
        #expect(result.rowMajorElements() == [58, 64, 139, 154])
    }

    @Test
    func routerPrefersAccelerateForDoubleAndFactorizations() {
        let router = BackendRouter(gpuAvailable: true)

        #expect(router.choose(op: .gemm, scalar: Double.self, rows: 8, columns: 8).backend == .accelerate)
        #expect(router.choose(op: .svd, scalar: Float.self, rows: 256, columns: 256).backend == .accelerate)
        #expect(router.choose(op: .gemm, scalar: Float.self, rows: 512, columns: 512).backend == .mlx)
    }

    @Test
    func accelerateBackendGemmUsesNativePathForDouble() async throws {
        let a = try Matrix<Double>([
            [1, 2, 3],
            [4, 5, 6],
        ])
        let b = try Matrix<Double>([
            [7, 8],
            [9, 10],
            [11, 12],
        ])

        let result = try await AccelerateBackend().gemm(BufferRef(a), BufferRef(b)).matrix()

        #expect(result.rowMajorElements() == [58, 64, 139, 154])
    }

    @Test
    func accelerateBackendPromotesFloat16Gemm() async throws {
        let a = try Matrix<Float16>([
            [1, 2],
            [3, 4],
        ])
        let b = Matrix<Float16>.eye(2)

        let result = try await AccelerateBackend().gemm(BufferRef(a), BufferRef(b)).matrix()

        #expect(AccelerateBackend.supports(.svd, scalar: Float16.self))
        #expect(result == a)
    }

    @Test
    func columnMajorBridgeRoundTripsStridedViews() throws {
        let matrix = try Matrix<Double>([
            [1, 2, 3],
            [4, 5, 6],
        ]).t

        let columnMajor = ColumnMajorBridge.columnMajor(from: matrix)
        let roundTrip = try ColumnMajorBridge.rowMajor(
            fromColumnMajor: columnMajor,
            rows: matrix.rows,
            columns: matrix.columns
        )

        #expect(roundTrip == matrix)
    }

    @Test
    func lapackKernelsCoverCoreFactorizations() throws {
        let a = try Matrix<Double>([
            [4, 3],
            [6, 3],
        ])
        let b = Vector<Double>([10, 12])

        let lu = try LAPACKKernels.lu(a)
        let x = try LAPACKKernels.solve(a, b)
        let inverse = try LAPACKKernels.inverse(a)
        let qr = try LAPACKKernels.qr(a)
        let cholesky = try LAPACKKernels.cholesky(try Matrix<Double>([
            [25, 15, -5],
            [15, 18, 0],
            [-5, 0, 11],
        ]))
        let svd = try LAPACKKernels.svd(a)
        let eigen = try LAPACKKernels.symmetricEigen(try Matrix<Double>([
            [2, 0],
            [0, 5],
        ]))
        let leastSquares = try LAPACKKernels.leastSquares(a, b)

        #expect(lu.permutation == [1, 0])
        #expect(approximatelyEqual(a * x.asColumnMatrix(), b.asColumnMatrix(), tolerance: 1e-9))
        #expect(approximatelyEqual(a * inverse, Matrix<Double>.eye(2), tolerance: 1e-9))
        #expect(approximatelyEqual(qr.q * qr.r, a, tolerance: 1e-9))
        #expect(approximatelyEqual(cholesky * cholesky.t, try Matrix<Double>([
            [25, 15, -5],
            [15, 18, 0],
            [-5, 0, 11],
        ]), tolerance: 1e-9))
        #expect(svd.singularValues.count == 2)
        #expect(eigen.values[0] == 2)
        #expect(eigen.values[1] == 5)
        #expect(approximatelyEqual(a * leastSquares.asColumnMatrix(), b.asColumnMatrix(), tolerance: 1e-9))
    }
}

private func approximatelyEqual(_ lhs: Matrix<Double>, _ rhs: Matrix<Double>, tolerance: Double) -> Bool {
    lhs.rows == rhs.rows
        && lhs.columns == rhs.columns
        && zip(lhs.rowMajorElements(), rhs.rowMajorElements()).allSatisfy { abs($0 - $1) <= tolerance }
}
