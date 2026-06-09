import MxNumericsCore

public enum BackendPolicy: Sendable, Equatable {
    case auto
    case forceAccelerate
    case forceMLX
    case forceReference
}

public struct DispatchDecision: Sendable, Equatable {
    public let backend: BackendID
    public let reason: String
}

public struct BackendRouter: Sendable {
    public var policy: BackendPolicy
    public var deterministicMode: Bool
    public var gpuAvailable: Bool
    public var gpuGemmElementThreshold: Int

    public init(
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

    public func choose<S: MatrixScalar>(
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
