//
//  Dispatch.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

/// Controls backend selection.
public enum BackendPolicy: Sendable, Equatable {
    /// Lets the router choose a backend from scalar type, operation, size, and availability.
    case auto

    /// Routes eligible operations to Accelerate.
    case forceAccelerate

    /// Routes eligible operations to MLX.
    case forceMLX

    /// Routes operations to the portable Swift reference implementation.
    case forceReference
}

/// The result of a backend routing decision.
public struct DispatchDecision: Sendable, Equatable {
    /// The selected backend.
    public let backend: BackendID

    /// A short explanation of why the backend was selected.
    public let reason: String
}

/// Chooses a backend for a primitive operation.
///
/// The router encodes the package's numerical policy. Double precision and
/// factorization routines prefer Accelerate, because LAPACK-style CPU kernels
/// are the reference path for those operations. Large single-precision GEMM can
/// be routed to MLX when GPU support is available. Deterministic mode overrides
/// the automatic policy and forces the CPU reference path chosen for regression
/// stability.
struct BackendRouter: Sendable {
    /// The explicit backend policy.
    var policy: BackendPolicy

    /// A Boolean value that forces deterministic CPU routing when true.
    var deterministicMode: Bool

    /// A Boolean value indicating whether the MLX backend reports GPU availability.
    var gpuAvailable: Bool

    /// The minimum result element count for routing Float GEMM to MLX in auto mode.
    var gpuGemmElementThreshold: Int

    /// Creates a backend router.
    ///
    /// - Parameters:
    ///   - policy: The backend policy.
    ///   - deterministicMode: Whether to force deterministic CPU routing.
    ///   - gpuAvailable: Whether GPU-backed MLX routing is available.
    ///   - gpuGemmElementThreshold: The minimum result element count for
    ///     auto-routing Float GEMM to MLX.
    init(
        policy: BackendPolicy = .auto,
        deterministicMode: Bool = false,
        gpuAvailable: Bool = false,
        gpuGemmElementThreshold: Int = 128 * 128
    ) {
        self.policy = policy
        self.deterministicMode = deterministicMode
        self.gpuAvailable = gpuAvailable
        self.gpuGemmElementThreshold = gpuGemmElementThreshold
    }

    /// Chooses a backend for an operation.
    ///
    /// - Parameters:
    ///   - op: The primitive operation.
    ///   - scalar: The scalar type stored in the operands.
    ///   - rows: The operation's representative row count.
    ///   - columns: The operation's representative column count.
    /// - Returns: The backend decision and the reason for it.
    func choose<S: MatrixScalar>(
        op: BackendOperation,
        scalar: S.Type,
        rows: Int,
        columns: Int
    ) -> DispatchDecision {
        if deterministicMode {
            return DispatchDecision(backend: .accelerate, reason: "deterministic mode forces CPU-backed Accelerate")
        }

        switch policy {
        case .forceAccelerate:
            return DispatchDecision(backend: .accelerate, reason: "policy override")
        case .forceMLX:
            return DispatchDecision(backend: .mlx, reason: "policy override")
        case .forceReference:
            return DispatchDecision(backend: .reference, reason: "policy override")
        case .auto:
            break
        }

        if S.self == Double.self {
            return DispatchDecision(backend: .accelerate, reason: "Double precision is routed to Accelerate")
        }

        if op == .gemm, S.self == Float.self, gpuAvailable, rows * columns >= gpuGemmElementThreshold {
            return DispatchDecision(backend: .mlx, reason: "large Float GEMM is eligible for GPU residency")
        }

        if [.lu, .qr, .svd, .eig, .cholesky].contains(op) {
            return DispatchDecision(backend: .accelerate, reason: "factorizations use LAPACK as the reference path")
        }

        return DispatchDecision(backend: .reference, reason: "portable fallback")
    }
}
