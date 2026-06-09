//
//  ReferenceBackendTests.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
import MxNumericsBackend
import MxNumericsCore
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

        let result = try await ReferenceBackend().gemm(BufferRef(a), BufferRef(b)).matrix()

        #expect(result.rowMajorElements() == [58, 64, 139, 154])
    }

    @Test
    func routerPrefersAccelerateForDoubleAndFactorizations() {
        let router = BackendRouter(gpuAvailable: true)

        #expect(router.choose(op: .gemm, scalar: Double.self, rows: 8, columns: 8).backend == .accelerate)
        #expect(router.choose(op: .svd, scalar: Float.self, rows: 256, columns: 256).backend == .accelerate)
        #expect(router.choose(op: .gemm, scalar: Float.self, rows: 512, columns: 512).backend == .mlx)
    }
}
