import MxNumerics
import Testing

@Suite(.serialized)
struct MxNumericsTests {
    @Test
    func runtimeStateIsConcurrencySafeAndAffectsDispatch() throws {
        let previousDeterministicMode = MxNumerics.deterministicMode
        let previousPolicy = MxNumerics.backendPolicy
        defer {
            MxNumerics.deterministicMode = previousDeterministicMode
            MxNumerics.backendPolicy = previousPolicy
        }

        MxNumerics.deterministicMode = true
        MxNumerics.backendPolicy = .forceMLX

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
