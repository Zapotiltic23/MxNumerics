//
//  MLXBackend.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

#if canImport(MLX)
import MLX
#endif

/// The MLX backend integration point.
///
/// MLX is intended for large Float and Float16 bulk operations where GPU
/// execution and operation fusion can outweigh data transfer and launch costs.
/// The current target compiles conditionally and reports capability without yet
/// implementing residency-aware kernels.
struct MLXBackend: LinearAlgebraBackend {
    /// The backend identifier.
    static let id = BackendID.mlx

    /// Creates an MLX backend.
    init() {}

    /// A Boolean value indicating whether the package was compiled with the MLX module.
    static var isCompiledWithMLX: Bool {
        #if canImport(MLX)
        true
        #else
        false
        #endif
    }

    /// A Boolean value indicating whether the MLX backend can be considered for GPU routing.
    ///
    /// This scaffold returns the compile-time MLX availability. A production
    /// implementation should also check runtime device availability.
    static var gpuAvailable: Bool {
        #if canImport(MLX)
        true
        #else
        false
        #endif
    }

    /// Returns whether MLX is intended to support the operation and scalar type.
    ///
    /// Current policy limits MLX eligibility to Float and Float16 GEMM,
    /// elementwise operations, and reductions. Double precision and complex
    /// operations stay on CPU backends.
    static func supports<S: MatrixScalar>(_ op: BackendOperation, scalar: S.Type) -> Bool {
        guard isCompiledWithMLX else { return false }
        return (op == .gemm || op == .elementwise || op == .reduction)
            && (scalar == Float.self || scalar == Float16.self)
    }

    /// Computes a matrix product with MLX.
    ///
    /// - Throws: ``LinAlgError/unsupported(_:)`` because the residency-aware MLX
    ///   implementation is not yet built.
    func gemm<S: MatrixScalar>(_ a: BufferRef<S>, _ b: BufferRef<S>) async throws -> BufferRef<S> {
        throw LinAlgError.unsupported("MLX GEMM bridge is scaffolded; residency-aware implementation is Phase 7.")
    }
}
