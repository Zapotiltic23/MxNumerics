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
/// routines. GEMM is backed by CBLAS for `Double` and `Float`, with the
/// reference backend retained for unsupported scalar types.
public struct AccelerateBackend: LinearAlgebraBackend {
    /// The backend identifier.
    public static let id = BackendID.accelerate

    /// Creates an Accelerate backend.
    public init() {}

    /// Returns whether Accelerate is intended to support the operation.
    ///
    /// Double and Float matrix products and factorizations are native
    /// Accelerate paths. Float16 is eligible through promotion to Float and
    /// demotion back to Float16.
    public static func supports<S: MatrixScalar>(_ op: BackendOperation, scalar: S.Type) -> Bool {
        switch op {
        case .gemm, .elementwise, .reduction, .lu, .qr, .svd, .eig, .cholesky:
            return scalar == Double.self || scalar == Float.self || scalar == Float16.self
        }
    }

    /// Computes a matrix product.
    ///
    /// This method calls the row-major CBLAS GEMM family for `Double` and
    /// `Float`, promotes `Float16` through `Float`, and delegates to
    /// ``ReferenceBackend/gemm(_:_:)`` otherwise.
    public func gemm<S: MatrixScalar>(_ a: BufferRef<S>, _ b: BufferRef<S>) async throws -> BufferRef<S> {
        if S.self == Double.self {
            let left = try Matrix<Double>(rowMajor: a.elements.map { $0 as! Double }, rows: a.rows, columns: a.columns)
            let right = try Matrix<Double>(rowMajor: b.elements.map { $0 as! Double }, rows: b.rows, columns: b.columns)
            let result = try BLASKernels.gemm(left, right)
            return try BufferRef(elements: result.rowMajorElements().map { $0 as! S }, rows: result.rows, columns: result.columns)
        }

        if S.self == Float.self {
            let left = try Matrix<Float>(rowMajor: a.elements.map { $0 as! Float }, rows: a.rows, columns: a.columns)
            let right = try Matrix<Float>(rowMajor: b.elements.map { $0 as! Float }, rows: b.rows, columns: b.columns)
            let result = try BLASKernels.gemm(left, right)
            return try BufferRef(elements: result.rowMajorElements().map { $0 as! S }, rows: result.rows, columns: result.columns)
        }

        if S.self == Float16.self {
            let left = try Matrix<Float>(rowMajor: a.elements.map { Float($0 as! Float16) }, rows: a.rows, columns: a.columns)
            let right = try Matrix<Float>(rowMajor: b.elements.map { Float($0 as! Float16) }, rows: b.rows, columns: b.columns)
            let result = try BLASKernels.gemm(left, right)
            return try BufferRef(elements: result.rowMajorElements().map { Float16($0) as! S }, rows: result.rows, columns: result.columns)
        }

        return try await ReferenceBackend().gemm(a, b)
    }
}

/// Runtime availability information for Accelerate.
public enum AccelerateAvailability {
    /// A Boolean value indicating whether the Accelerate framework is linked.
    public static var isAvailable: Bool { true }
}
