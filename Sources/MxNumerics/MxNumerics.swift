//
//  MxNumerics.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
internal import os

/// A concurrency-safe snapshot of the current runtime configuration.
public struct RuntimeConfiguration: Sendable, Equatable {
    /// A Boolean value that forces deterministic CPU backend routing.
    public let deterministicMode: Bool

    /// The global backend selection policy.
    public let backendPolicy: BackendPolicy
}

/// Global runtime controls for MxNumerics.
///
/// Use this namespace to configure backend policy for operations that route
/// through the umbrella module. The stored state is protected by an unfair lock
/// so synchronous callers can read and update it safely under Swift concurrency.
public enum MxNumericsRuntime {
    private struct RuntimeState: Sendable {
        var deterministicMode = false
        var backendPolicy = BackendPolicy.auto
    }

    private static let state = OSAllocatedUnfairLock(initialState: RuntimeState())

    /// A Boolean value that forces deterministic CPU backend routing.
    ///
    /// Deterministic mode is useful for regression tests where backend changes
    /// and GPU execution could introduce small, valid floating-point differences.
    public static var deterministicMode: Bool {
        get { state.withLock { $0.deterministicMode } }
        set { state.withLock { $0.deterministicMode = newValue } }
    }

    /// The global backend selection policy.
    public static var backendPolicy: BackendPolicy {
        get { state.withLock { $0.backendPolicy } }
        set { state.withLock { $0.backendPolicy = newValue } }
    }

    /// Returns an immutable snapshot of the current runtime configuration.
    public static var configuration: RuntimeConfiguration {
        state.withLock {
            RuntimeConfiguration(
                deterministicMode: $0.deterministicMode,
                backendPolicy: $0.backendPolicy
            )
        }
    }

    /// Creates an internal router snapshot from the current runtime state.
    static func router() -> BackendRouter {
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
    /// Returns the backend decision for an operation on this matrix.
    ///
    /// This method is diagnostic: it reports routing policy without executing the
    /// operation.
    ///
    /// - Parameter operation: The primitive operation to route.
    /// - Returns: The selected backend and a reason string.
    public func backendDecision(for operation: BackendOperation) -> DispatchDecision {
        MxNumericsRuntime.router().choose(op: operation, scalar: Scalar.self, rows: rows, columns: columns)
    }

    /// Multiplies this matrix by another matrix using the async reference backend.
    ///
    /// The operation computes the standard matrix product through the library's
    /// structured-concurrency execution path.
    ///
    /// - Parameter other: The right-hand matrix.
    /// - Returns: The product `self * other`.
    /// - Throws: ``LinAlgError/dimensionMismatch(_:)`` when the inner dimensions
    ///   do not agree.
    public func multipliedConcurrently(by other: Matrix<Scalar>) async throws -> Matrix<Scalar> {
        let result = try await ReferenceBackend().gemm(BufferRef(self), BufferRef(other))
        return result.matrix()
    }
}
