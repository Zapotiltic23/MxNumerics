//
//  ColumnMajorBridge.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

import MxNumericsCore

public enum ColumnMajorBridge {
    public static func columnMajor<S: MatrixScalar>(from matrix: Matrix<S>) -> [S] {
        var result = Array(repeating: S.zero, count: matrix.count)
        for row in 0..<matrix.rows {
            for column in 0..<matrix.columns {
                result[column * matrix.rows + row] = matrix[row, column]
            }
        }
        return result
    }

    public static func rowMajor<S: MatrixScalar>(
        fromColumnMajor elements: [S],
        rows: Int,
        columns: Int
    ) throws -> Matrix<S> {
        guard elements.count == rows * columns else {
            throw LinAlgError.dimensionMismatch("Column-major buffer count does not match shape.")
        }

        var rowMajor = Array(repeating: S.zero, count: elements.count)
        for row in 0..<rows {
            for column in 0..<columns {
                rowMajor[row * columns + column] = elements[column * rows + row]
            }
        }
        return try Matrix(rowMajor: rowMajor, rows: rows, columns: columns)
    }
}
