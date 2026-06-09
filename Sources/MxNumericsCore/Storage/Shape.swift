public struct Shape: Equatable, Sendable {
    public let rows: Int
    public let columns: Int

    public init(rows: Int, columns: Int) throws {
        guard rows >= 0, columns >= 0 else {
            throw LinAlgError.invalidShape(rows: rows, columns: columns)
        }
        self.rows = rows
        self.columns = columns
    }

    public var count: Int { rows * columns }
    public var isSquare: Bool { rows == columns }
}

public struct Strides: Equatable, Sendable {
    public let row: Int
    public let column: Int

    public init(row: Int, column: Int) {
        self.row = row
        self.column = column
    }
}
