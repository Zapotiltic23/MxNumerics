import MxNumericsCore

public enum BackendID: String, Sendable {
    case accelerate
    case mlx
    case reference
}

public enum BackendOperation: Hashable, Sendable {
    case gemm
    case elementwise
    case reduction
    case lu
    case qr
    case svd
    case eig
    case cholesky
}

public struct BufferRef<Scalar: MatrixScalar>: Sendable {
    public let elements: [Scalar]
    public let rows: Int
    public let columns: Int

    public init(elements: [Scalar], rows: Int, columns: Int) throws {
        guard rows >= 0, columns >= 0, rows * columns == elements.count else {
            throw LinAlgError.invalidShape(rows: rows, columns: columns)
        }
        self.elements = elements
        self.rows = rows
        self.columns = columns
    }

    public init(_ matrix: Matrix<Scalar>) {
        self.elements = matrix.rowMajorElements()
        self.rows = matrix.rows
        self.columns = matrix.columns
    }

    public func matrix() -> Matrix<Scalar> {
        try! Matrix(rowMajor: elements, rows: rows, columns: columns)
    }
}

public protocol LinearAlgebraBackend: Sendable {
    static var id: BackendID { get }
    static func supports<S: MatrixScalar>(_ op: BackendOperation, scalar: S.Type) -> Bool

    func gemm<S: MatrixScalar>(
        _ a: BufferRef<S>,
        _ b: BufferRef<S>
    ) async throws -> BufferRef<S>
}
