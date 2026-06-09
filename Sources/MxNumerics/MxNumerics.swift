@_exported import MxNumericsBackend
@_exported import MxNumericsCore

import MxNumericsMLX
import os

public enum MxNumerics {
    private struct RuntimeState: Sendable {
        var deterministicMode = false
        var backendPolicy = BackendPolicy.auto
    }

    private static let state = OSAllocatedUnfairLock(initialState: RuntimeState())

    public static var deterministicMode: Bool {
        get { state.withLock { $0.deterministicMode } }
        set { state.withLock { $0.deterministicMode = newValue } }
    }

    public static var backendPolicy: BackendPolicy {
        get { state.withLock { $0.backendPolicy } }
        set { state.withLock { $0.backendPolicy = newValue } }
    }

    public static func router() -> BackendRouter {
        state.withLock {
            BackendRouter(
                policy: $0.backendPolicy,
                deterministicMode: $0.deterministicMode,
                gpuAvailable: MLXBackend.gpuAvailable
            )
        }
    }
}

extension Matrix {
    public func backendDecision(for operation: BackendOperation) -> DispatchDecision {
        MxNumerics.router().choose(op: operation, scalar: Scalar.self, rows: rows, columns: columns)
    }

    public func multipliedConcurrently(by other: Matrix<Scalar>) async throws -> Matrix<Scalar> {
        let result = try await ReferenceBackend().gemm(BufferRef(self), BufferRef(other))
        return result.matrix()
    }
}
