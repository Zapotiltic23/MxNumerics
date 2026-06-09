import MxNumericsBackend
import MxNumericsCore

#if canImport(MLX)
import MLX
#endif

public struct MLXBackend: LinearAlgebraBackend {
    public static let id = BackendID.mlx

    public init() {}

    public static var isCompiledWithMLX: Bool {
        #if canImport(MLX)
        true
        #else
        false
        #endif
    }

    public static var gpuAvailable: Bool {
        #if canImport(MLX)
        true
        #else
        false
        #endif
    }

    public static func supports<S: MatrixScalar>(_ op: BackendOperation, scalar: S.Type) -> Bool {
        guard isCompiledWithMLX else { return false }
        return (op == .gemm || op == .elementwise || op == .reduction)
            && (scalar == Float.self || scalar == Float16.self)
    }

    public func gemm<S: MatrixScalar>(_ a: BufferRef<S>, _ b: BufferRef<S>) async throws -> BufferRef<S> {
        throw LinAlgError.unsupported("MLX GEMM bridge is scaffolded; residency-aware implementation is Phase 7.")
    }
}
