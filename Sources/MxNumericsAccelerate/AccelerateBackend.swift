//
//  AccelerateBackend.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
import Accelerate
import MxNumericsBackend
import MxNumericsCore

/// The Accelerate-backed CPU backend.
///
/// This target is the integration point for Apple's BLAS, LAPACK, and vDSP
/// routines. The current GEMM method delegates to ``ReferenceBackend`` while the
/// typed BLAS/LAPACK shims are being built, so this type presently communicates
/// routing intent and capability rather than performance.
public struct AccelerateBackend: LinearAlgebraBackend {
    /// The backend identifier.
    public static let id = BackendID.accelerate

    /// Creates an Accelerate backend.
    public init() {}

    /// Returns whether Accelerate is intended to support the operation.
    ///
    /// Double and Float matrix products and factorizations are marked as
    /// supported by policy. Half-precision elementwise operations are marked
    /// eligible, but LAPACK-style factorizations for `Float16` are not.
    public static func supports<S: MatrixScalar>(_ op: BackendOperation, scalar: S.Type) -> Bool {
        switch op {
        case .gemm, .reduction, .lu, .qr, .svd, .eig, .cholesky:
            return scalar == Double.self || scalar == Float.self
        case .elementwise:
            return scalar == Double.self || scalar == Float.self || scalar == Float16.self
        }
    }

    /// Computes a matrix product.
    ///
    /// This method currently delegates to ``ReferenceBackend/gemm(_:_:)``. A
    /// future implementation will call the row-major CBLAS GEMM family where the
    /// scalar type is supported.
    public func gemm<S: MatrixScalar>(_ a: BufferRef<S>, _ b: BufferRef<S>) async throws -> BufferRef<S> {
        try await ReferenceBackend().gemm(a, b)
    }
}

/// Runtime availability information for Accelerate.
public enum AccelerateAvailability {
    /// A Boolean value indicating whether the Accelerate framework is linked.
    public static var isAvailable: Bool { true }
}
