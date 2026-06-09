//
//  Vector.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

/// A dense column vector backed by matrix storage.
///
/// `Vector` is a convenience wrapper around an `n x 1` ``Matrix``. It shares the
/// same value semantics and scalar constraints as matrices while exposing
/// one-dimensional indexing.
public struct Vector<Scalar: MatrixScalar>: Sendable, Equatable where Scalar: Equatable {
    /// The vector represented as an `n x 1` matrix.
    public private(set) var matrix: Matrix<Scalar>

    /// The number of vector entries.
    public var count: Int { matrix.rows }

    /// Creates a column vector from scalar entries.
    ///
    /// - Parameter elements: The vector entries in index order.
    public init(_ elements: [Scalar]) {
        self.matrix = try! Matrix(rowMajor: elements, rows: elements.count, columns: 1)
    }

    /// Creates a vector from a column matrix.
    ///
    /// - Parameter matrix: A matrix with exactly one column.
    /// - Precondition: `matrix.columns == 1`.
    public init(column matrix: Matrix<Scalar>) {
        precondition(matrix.columns == 1, "Vector column initializer requires an n x 1 matrix.")
        self.matrix = matrix
    }

    /// Accesses a vector entry by zero-based index.
    ///
    /// - Parameter index: The vector index.
    /// - Precondition: `index` is inside the vector bounds.
    public subscript(_ index: Int) -> Scalar {
        _read {
            yield matrix[index, 0]
        }
        _modify {
            yield &matrix[index, 0]
        }
    }

    /// Returns the vector as an `n x 1` matrix.
    public func asColumnMatrix() -> Matrix<Scalar> {
        matrix
    }

    /// Calls a closure with contiguous vector storage.
    public func withUnsafeValues<R>(
        _ body: (UnsafeBufferPointer<Scalar>) throws -> R
    ) rethrows -> R {
        try matrix.withUnsafeRowMajor { pointer, _, _ in
            try body(pointer)
        }
    }
}

extension Vector where Scalar: FloatingScalar {
    /// Returns the bilinear dot product with another vector.
    ///
    /// This method computes `sum(self[i] * other[i])`. For complex scalars this
    /// is the unconjugated product `x^T y`, not the Hermitian inner product
    /// `x^H y`.
    ///
    /// - Parameter other: The vector on the right side of the product.
    /// - Returns: The scalar dot product.
    /// - Precondition: The vectors have the same length.
    public func dot(_ other: Vector<Scalar>) -> Scalar {
        precondition(count == other.count, "Dot product requires equal vector lengths.")
        var result = Scalar.zero
        for index in 0..<count {
            result += self[index] * other[index]
        }
        return result
    }

    /// The Euclidean 2-norm of the vector.
    ///
    /// The norm is `sqrt(sum(abs(x_i)^2))`. This core-only implementation uses
    /// a direct sum of squared magnitudes. Use the umbrella `norm(_:_:)`
    /// routine when you want the Accelerate-backed `nrm2` path for supported
    /// real floating-point scalars.
    public var norm: Scalar.Magnitude {
        var sum = Scalar.Magnitude.zero
        for index in 0..<count {
            let magnitude = self[index].magnitude
            sum += magnitude * magnitude
        }
        return Scalar.Magnitude.sqrt(sum)
    }
}
