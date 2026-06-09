import Accelerate
import MxNumericsBackend
import MxNumericsCore

public struct AccelerateBackend: LinearAlgebraBackend {
    public static let id = BackendID.accelerate

    public init() {}

    public static func supports<S: MatrixScalar>(_ op: BackendOperation, scalar: S.Type) -> Bool {
        switch op {
        case .gemm, .reduction, .lu, .qr, .svd, .eig, .cholesky:
            return scalar == Double.self || scalar == Float.self
        case .elementwise:
            return scalar == Double.self || scalar == Float.self || scalar == Float16.self
        }
    }

    public func gemm<S: MatrixScalar>(_ a: BufferRef<S>, _ b: BufferRef<S>) async throws -> BufferRef<S> {
        try await ReferenceBackend().gemm(a, b)
    }
}

public enum AccelerateAvailability {
    public static var isAvailable: Bool { true }
}
