import MxNumericsCore

public struct ReferenceBackend: LinearAlgebraBackend {
    public static let id = BackendID.reference

    public init() {}

    public static func supports<S: MatrixScalar>(_ op: BackendOperation, scalar: S.Type) -> Bool {
        op == .gemm
    }

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
