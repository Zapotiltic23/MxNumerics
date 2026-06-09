//
//  MatrixBuffer.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

/// Reference storage for copy-on-write matrix values.
///
/// `MatrixBuffer` is intentionally internal. Public matrix values remain value
/// types, while the buffer lets slices and copies share storage until a mutation
/// requires a private copy.
final class MatrixBuffer<Scalar: Sendable>: @unchecked Sendable {
    var elements: [Scalar]

    init(_ elements: [Scalar]) {
        self.elements = elements
    }

    func copy() -> MatrixBuffer<Scalar> {
        MatrixBuffer(elements)
    }
}
