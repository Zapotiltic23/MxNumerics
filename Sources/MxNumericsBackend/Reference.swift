//
//  Reference.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
import MxNumericsCore

/// A portable Swift backend used as a correctness reference.
///
/// The reference backend favors clear, deterministic algorithms over peak
/// performance. Its GEMM implementation splits work by output row with Swift
/// task groups, which makes it useful for testing async backend contracts while
/// staying independent of Accelerate and MLX.
public struct ReferenceBackend: LinearAlgebraBackend {
    /// The backend identifier.
    public static let id = BackendID.reference

    /// Creates a reference backend.
    public init() {}

    /// Returns whether the reference backend supports an operation.
    ///
    /// The current implementation supports GEMM.
    public static func supports<S: MatrixScalar>(_ op: BackendOperation, scalar: S.Type) -> Bool {
        op == .gemm
    }

    /// Computes a general matrix-matrix product with a straightforward Swift kernel.
    ///
    /// The result is mathematically equivalent to `C = A * B`. The accumulation
    /// order is fixed by row, inner dimension, and column loops; floating-point
    /// results may still differ from optimized BLAS implementations because BLAS
    /// kernels can use different blocking and fused operations.
    ///
    /// - Parameters:
    ///   - a: The left matrix buffer.
    ///   - b: The right matrix buffer.
    /// - Returns: A row-major buffer containing the matrix product.
    /// - Throws: ``LinAlgError/dimensionMismatch(_:)`` when `a.columns != b.rows`.
    public func gemm<S: MatrixScalar>(
        _ a: BufferRef<S>,
        _ b: BufferRef<S>
    ) async throws -> BufferRef<S> {
        guard a.columns == b.rows else {
            throw LinAlgError.dimensionMismatch("GEMM requires a.columns == b.rows.")
        }

        let resultRows = a.rows
        let resultColumns = b.columns
        let partials = await withTaskGroup(of: (Int, [S]).self, returning: [[S]].self) { group in
            for row in 0..<resultRows {
                group.addTask {
                    var rowValues = Array(repeating: S.zero, count: resultColumns)
                    for k in 0..<a.columns {
                        let aik = a.elements[row * a.columns + k]
                        guard aik != .zero else { continue }
                        for column in 0..<resultColumns {
                            rowValues[column] += aik * b.elements[k * b.columns + column]
                        }
                    }
                    return (row, rowValues)
                }
            }

            var rows = Array(repeating: Array(repeating: S.zero, count: resultColumns), count: resultRows)
            for await (row, values) in group {
                rows[row] = values
            }
            return rows
        }

        return try BufferRef(elements: partials.flatMap { $0 }, rows: resultRows, columns: resultColumns)
    }
}
