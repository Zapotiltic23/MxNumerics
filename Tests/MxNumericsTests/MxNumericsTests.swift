//
//  MxNumericsTests.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
import MxNumerics
import Testing

@Suite(.serialized)
struct MxNumericsTests {
    @Test
    func runtimeStateIsConcurrencySafeAndAffectsDispatch() throws {
        let previousDeterministicMode = MxNumericsRuntime.deterministicMode
        let previousPolicy = MxNumericsRuntime.backendPolicy
        defer {
            MxNumericsRuntime.deterministicMode = previousDeterministicMode
            MxNumericsRuntime.backendPolicy = previousPolicy
        }

        MxNumericsRuntime.deterministicMode = true
        MxNumericsRuntime.backendPolicy = .forceMLX

        let matrix = try Matrix<Float>([
            [1, 2],
            [3, 4],
        ])

        #expect(matrix.backendDecision(for: .gemm).backend == .accelerate)
    }

    @Test
    func publicAsyncMultiplyWorksThroughUmbrellaModule() async throws {
        let a = try Matrix<Int>([
            [1, 2],
            [3, 4],
        ])
        let result = try await a.multipliedConcurrently(by: Matrix<Int>.eye(2))

        #expect(result == a)
    }
}
