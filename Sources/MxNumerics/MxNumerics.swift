//
//  MxNumerics.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
@_exported import MxNumericsBackend
@_exported import MxNumericsCore

import MxNumericsMLX
import os

/// Global runtime controls for MxNumerics.
///
/// Use this namespace to configure backend policy for operations that route
/// through the umbrella module. The stored state is protected by an unfair lock
/// so synchronous callers can read and update it safely under Swift concurrency.
public enum MxNumerics {
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

    /// Creates a router snapshot from the current global runtime state.
    ///
    /// - Returns: A ``BackendRouter`` initialized with the current policy,
    ///   deterministic-mode flag, and MLX availability.
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
    /// Returns the backend decision for an operation on this matrix.
    ///
    /// This method is diagnostic: it reports routing policy without executing the
    /// operation.
    ///
    /// - Parameter operation: The primitive operation to route.
    /// - Returns: The selected backend and a reason string.
    public func backendDecision(for operation: BackendOperation) -> DispatchDecision {
        MxNumerics.router().choose(op: operation, scalar: Scalar.self, rows: rows, columns: columns)
    }

    /// Multiplies this matrix by another matrix using the async reference backend.
    ///
    /// The operation computes the standard matrix product. It currently calls
    /// ``ReferenceBackend`` directly, so it is useful for validating async
    /// backend behavior but is not yet the optimized public multiply path.
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
