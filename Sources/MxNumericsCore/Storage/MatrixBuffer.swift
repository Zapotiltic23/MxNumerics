final class MatrixBuffer<Scalar: Sendable>: @unchecked Sendable {
    var elements: [Scalar]

    init(_ elements: [Scalar]) {
        self.elements = elements
    }

    func copy() -> MatrixBuffer<Scalar> {
        MatrixBuffer(elements)
    }
}
