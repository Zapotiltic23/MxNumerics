//
//  MatrixSlice.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

/// A matrix slicing token.
///
/// Use ``MatrixSlice/all`` to select every row or every column in a subscript,
/// for example `matrix[0, .all]` for a row or `matrix[.all, 0]` for a column.
public enum MatrixSlice: Sendable {
    /// Selects the complete extent of the indexed dimension.
    case all
}
