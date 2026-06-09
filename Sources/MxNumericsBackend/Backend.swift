//
//  Backend.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
import MxNumericsCore

/// Identifies a linear algebra backend.
///
/// Backends are implementation choices, not public mathematical semantics. The
/// same operation should return results that agree within documented numerical
/// tolerances even when different backends are selected.
public enum BackendID: String, Sendable {
    /// Apple's Accelerate framework, intended for BLAS and LAPACK-backed CPU routines.
    case accelerate

    /// The MLX backend, intended for GPU-eligible single-precision bulk operations.
    case mlx

    /// The portable Swift reference backend.
    case reference
}

/// A primitive operation known to the backend router.
public enum BackendOperation: Hashable, Sendable {
    /// General matrix-matrix multiplication.
    case gemm

    /// Elementwise scalar operations.
    case elementwise

    /// Reductions such as sums and norms.
    case reduction

    /// LU factorization and LU-based solves.
    case lu

    /// QR factorization.
    case qr

    /// Singular value decomposition.
    case svd

    /// Eigenvalue and eigenvector routines.
    case eig

    /// Cholesky factorization for symmetric or Hermitian positive-definite matrices.
    case cholesky
}

/// A contiguous row-major buffer passed between core values and backends.
///
/// `BufferRef` deliberately contains plain Swift storage rather than backend
/// objects. That keeps the public API independent from Accelerate, MLX, and any
/// future backend-specific types.
public struct BufferRef<Scalar: MatrixScalar>: Sendable {
    /// Row-major scalar storage.
    public let elements: [Scalar]

    /// The row count.
    public let rows: Int

    /// The column count.
    public let columns: Int

    /// Creates a buffer reference from row-major storage.
    ///
    /// - Parameters:
    ///   - elements: The row-major scalar entries.
    ///   - rows: The row count.
    ///   - columns: The column count.
    /// - Throws: ``LinAlgError/invalidShape(rows:columns:)`` if the dimensions
    ///   are negative or do not match the element count.
    public init(elements: [Scalar], rows: Int, columns: Int) throws {
        guard rows >= 0, columns >= 0, rows * columns == elements.count else {
            throw LinAlgError.invalidShape(rows: rows, columns: columns)
        }
        self.elements = elements
        self.rows = rows
        self.columns = columns
    }

    /// Creates a buffer reference by materializing a matrix in logical row-major order.
    public init(_ matrix: Matrix<Scalar>) {
        self.elements = matrix.rowMajorElements()
        self.rows = matrix.rows
        self.columns = matrix.columns
    }

    /// Converts this buffer reference back into a matrix.
    public func matrix() -> Matrix<Scalar> {
        try! Matrix(rowMajor: elements, rows: rows, columns: columns)
    }
}

/// A backend capable of executing primitive dense linear algebra operations.
///
/// Backends operate on contiguous row-major buffers and report support by
/// operation and scalar type. Implementations may dispatch to CPU, GPU, or
/// reference kernels, but must preserve the mathematical contract of each
/// primitive.
public protocol LinearAlgebraBackend: Sendable {
    /// The backend identifier.
    static var id: BackendID { get }

    /// Returns whether the backend supports an operation for a scalar type.
    static func supports<S: MatrixScalar>(_ op: BackendOperation, scalar: S.Type) -> Bool

    /// Computes a general matrix-matrix product.
    ///
    /// For matrices `A` and `B`, GEMM computes `C = A * B` in this package's
    /// no-transpose, `alpha = 1`, `beta = 0` form.
    ///
    /// - Parameters:
    ///   - a: The left matrix buffer.
    ///   - b: The right matrix buffer.
    /// - Returns: A row-major buffer for the product.
    /// - Throws: ``LinAlgError/dimensionMismatch(_:)`` if `a.columns != b.rows`,
    ///   or a backend-specific error if the operation is unsupported.
    func gemm<S: MatrixScalar>(
        _ a: BufferRef<S>,
        _ b: BufferRef<S>
    ) async throws -> BufferRef<S>
}
