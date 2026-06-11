//
//  VDSPKernels.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

internal import Accelerate

enum VDSPKernels {
    static func add(_ a: Matrix<Double>, _ b: Matrix<Double>) throws -> Matrix<Double> {
        try binaryMatrix(a, b, vDSP.add)
    }

    static func add(_ a: Matrix<Float>, _ b: Matrix<Float>) throws -> Matrix<Float> {
        try binaryMatrix(a, b, vDSP.add)
    }

    static func subtract(_ a: Matrix<Double>, _ b: Matrix<Double>) throws -> Matrix<Double> {
        try binaryMatrix(a, b, vDSP.subtract)
    }

    static func subtract(_ a: Matrix<Float>, _ b: Matrix<Float>) throws -> Matrix<Float> {
        try binaryMatrix(a, b, vDSP.subtract)
    }

    static func multiply(_ a: Matrix<Double>, _ b: Matrix<Double>) throws -> Matrix<Double> {
        try binaryMatrix(a, b, vDSP.multiply)
    }

    static func multiply(_ a: Matrix<Float>, _ b: Matrix<Float>) throws -> Matrix<Float> {
        try binaryMatrix(a, b, vDSP.multiply)
    }

    static func divide(_ a: Matrix<Double>, _ b: Matrix<Double>) throws -> Matrix<Double> {
        try binaryMatrix(a, b, vDSP.divide)
    }

    static func divide(_ a: Matrix<Float>, _ b: Matrix<Float>) throws -> Matrix<Float> {
        try binaryMatrix(a, b, vDSP.divide)
    }

    static func multiply(_ a: Matrix<Double>, scalar: Double) throws -> Matrix<Double> {
        var result = Array(repeating: 0.0, count: a.count)
        let elements = a.rowMajorElements()
        vDSP.multiply(scalar, elements, result: &result)
        return try Matrix(rowMajor: result, rows: a.rows, columns: a.columns)
    }

    static func multiply(_ a: Matrix<Float>, scalar: Float) throws -> Matrix<Float> {
        var result = Array(repeating: Float.zero, count: a.count)
        let elements = a.rowMajorElements()
        vDSP.multiply(scalar, elements, result: &result)
        return try Matrix(rowMajor: result, rows: a.rows, columns: a.columns)
    }

    static func add(_ a: Matrix<Double>, scalar: Double) throws -> Matrix<Double> {
        var result = Array(repeating: 0.0, count: a.count)
        let elements = a.rowMajorElements()
        vDSP.add(scalar, elements, result: &result)
        return try Matrix(rowMajor: result, rows: a.rows, columns: a.columns)
    }

    static func add(_ a: Matrix<Float>, scalar: Float) throws -> Matrix<Float> {
        var result = Array(repeating: Float.zero, count: a.count)
        let elements = a.rowMajorElements()
        vDSP.add(scalar, elements, result: &result)
        return try Matrix(rowMajor: result, rows: a.rows, columns: a.columns)
    }

    static func sum(_ a: Matrix<Double>) -> Double {
        vDSP.sum(a.rowMajorElements())
    }

    static func sum(_ a: Matrix<Float>) -> Float {
        vDSP.sum(a.rowMajorElements())
    }

    static func mean(_ x: Vector<Double>) -> Double {
        vDSP.mean(x.asColumnMatrix().rowMajorElements())
    }

    static func mean(_ x: Vector<Float>) -> Float {
        vDSP.mean(x.asColumnMatrix().rowMajorElements())
    }

    private static func binaryMatrix<S: FloatingScalar>(
        _ a: Matrix<S>,
        _ b: Matrix<S>,
        _ op: ([S], [S], inout [S]) -> Void
    ) throws -> Matrix<S> {
        guard a.shape == b.shape else {
            throw LinAlgError.dimensionMismatch("Elementwise operation requires equal shapes.")
        }
        var result = Array(repeating: S.zero, count: a.count)
        op(a.rowMajorElements(), b.rowMajorElements(), &result)
        return try Matrix(rowMajor: result, rows: a.rows, columns: a.columns)
    }
}
