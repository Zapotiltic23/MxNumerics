//
//  BLASKernels.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

import Accelerate
import MxNumericsCore

public enum BLASKernels {
    public static func gemm(_ a: Matrix<Double>, _ b: Matrix<Double>) throws -> Matrix<Double> {
        guard a.columns == b.rows else {
            throw LinAlgError.dimensionMismatch("GEMM requires a.columns == b.rows.")
        }
        let m = checkedBLASInt(a.rows)
        let n = checkedBLASInt(b.columns)
        let k = checkedBLASInt(a.columns)
        var result = Array(repeating: 0.0, count: a.rows * b.columns)

        a.withUnsafeRowMajor { aPointer, _, _ in
            b.withUnsafeRowMajor { bPointer, _, _ in
                cblas_dgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,
                    n,
                    k,
                    1.0,
                    aPointer.baseAddress!,
                    k,
                    bPointer.baseAddress!,
                    n,
                    0.0,
                    &result,
                    n
                )
            }
        }
        return try Matrix(rowMajor: result, rows: a.rows, columns: b.columns)
    }

    public static func gemm(_ a: Matrix<Float>, _ b: Matrix<Float>) throws -> Matrix<Float> {
        guard a.columns == b.rows else {
            throw LinAlgError.dimensionMismatch("GEMM requires a.columns == b.rows.")
        }
        let m = checkedBLASInt(a.rows)
        let n = checkedBLASInt(b.columns)
        let k = checkedBLASInt(a.columns)
        var result = Array(repeating: Float.zero, count: a.rows * b.columns)

        a.withUnsafeRowMajor { aPointer, _, _ in
            b.withUnsafeRowMajor { bPointer, _, _ in
                cblas_sgemm(
                    CblasRowMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    m,
                    n,
                    k,
                    1.0,
                    aPointer.baseAddress!,
                    k,
                    bPointer.baseAddress!,
                    n,
                    0.0,
                    &result,
                    n
                )
            }
        }
        return try Matrix(rowMajor: result, rows: a.rows, columns: b.columns)
    }

    public static func gemv(_ a: Matrix<Double>, _ x: Vector<Double>) throws -> Vector<Double> {
        guard a.columns == x.count else {
            throw LinAlgError.dimensionMismatch("GEMV requires a.columns == x.count.")
        }
        let m = checkedBLASInt(a.rows)
        let n = checkedBLASInt(a.columns)
        var result = Array(repeating: 0.0, count: a.rows)

        a.withUnsafeRowMajor { aPointer, _, _ in
            x.withUnsafeValues { xPointer in
                cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, aPointer.baseAddress!, n, xPointer.baseAddress!, 1, 0.0, &result, 1)
            }
        }
        return Vector(result)
    }

    public static func gemv(_ a: Matrix<Float>, _ x: Vector<Float>) throws -> Vector<Float> {
        guard a.columns == x.count else {
            throw LinAlgError.dimensionMismatch("GEMV requires a.columns == x.count.")
        }
        let m = checkedBLASInt(a.rows)
        let n = checkedBLASInt(a.columns)
        var result = Array(repeating: Float.zero, count: a.rows)

        a.withUnsafeRowMajor { aPointer, _, _ in
            x.withUnsafeValues { xPointer in
                cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1.0, aPointer.baseAddress!, n, xPointer.baseAddress!, 1, 0.0, &result, 1)
            }
        }
        return Vector(result)
    }

    public static func dot(_ x: Vector<Double>, _ y: Vector<Double>) throws -> Double {
        guard x.count == y.count else { throw LinAlgError.dimensionMismatch("Dot product requires equal lengths.") }
        let n = checkedBLASInt(x.count)
        return x.withUnsafeValues { xPointer in
            y.withUnsafeValues { yPointer in
                cblas_ddot(n, xPointer.baseAddress!, 1, yPointer.baseAddress!, 1)
            }
        }
    }

    public static func dot(_ x: Vector<Float>, _ y: Vector<Float>) throws -> Float {
        guard x.count == y.count else { throw LinAlgError.dimensionMismatch("Dot product requires equal lengths.") }
        let n = checkedBLASInt(x.count)
        return x.withUnsafeValues { xPointer in
            y.withUnsafeValues { yPointer in
                cblas_sdot(n, xPointer.baseAddress!, 1, yPointer.baseAddress!, 1)
            }
        }
    }

    public static func nrm2(_ x: Vector<Double>) -> Double {
        x.withUnsafeValues { pointer in
            cblas_dnrm2(checkedBLASInt(x.count), pointer.baseAddress!, 1)
        }
    }

    public static func nrm2(_ x: Vector<Float>) -> Float {
        x.withUnsafeValues { pointer in
            cblas_snrm2(checkedBLASInt(x.count), pointer.baseAddress!, 1)
        }
    }

    public static func nrm2(_ matrix: Matrix<Double>) -> Double {
        matrix.withUnsafeRowMajor { pointer, _, _ in
            cblas_dnrm2(checkedBLASInt(matrix.count), pointer.baseAddress!, 1)
        }
    }

    public static func nrm2(_ matrix: Matrix<Float>) -> Float {
        matrix.withUnsafeRowMajor { pointer, _, _ in
            cblas_snrm2(checkedBLASInt(matrix.count), pointer.baseAddress!, 1)
        }
    }

    public static func asum(_ x: Vector<Double>) -> Double {
        x.withUnsafeValues { pointer in cblas_dasum(checkedBLASInt(x.count), pointer.baseAddress!, 1) }
    }

    public static func asum(_ x: Vector<Float>) -> Float {
        x.withUnsafeValues { pointer in cblas_sasum(checkedBLASInt(x.count), pointer.baseAddress!, 1) }
    }

    public static func axpy(alpha: Double, x: Vector<Double>, y: Vector<Double>) throws -> Vector<Double> {
        guard x.count == y.count else { throw LinAlgError.dimensionMismatch("AXPY requires equal lengths.") }
        var result = y.asColumnMatrix().rowMajorElements()
        x.withUnsafeValues { pointer in
            cblas_daxpy(checkedBLASInt(x.count), alpha, pointer.baseAddress!, 1, &result, 1)
        }
        return Vector(result)
    }

    public static func axpy(alpha: Float, x: Vector<Float>, y: Vector<Float>) throws -> Vector<Float> {
        guard x.count == y.count else { throw LinAlgError.dimensionMismatch("AXPY requires equal lengths.") }
        var result = y.asColumnMatrix().rowMajorElements()
        x.withUnsafeValues { pointer in
            cblas_saxpy(checkedBLASInt(x.count), alpha, pointer.baseAddress!, 1, &result, 1)
        }
        return Vector(result)
    }

    public static func scal(alpha: Double, x: Vector<Double>) -> Vector<Double> {
        var result = x.asColumnMatrix().rowMajorElements()
        cblas_dscal(checkedBLASInt(x.count), alpha, &result, 1)
        return Vector(result)
    }

    public static func scal(alpha: Float, x: Vector<Float>) -> Vector<Float> {
        var result = x.asColumnMatrix().rowMajorElements()
        cblas_sscal(checkedBLASInt(x.count), alpha, &result, 1)
        return Vector(result)
    }
}

func checkedBLASInt(_ value: Int) -> Int32 {
    precondition(value <= Int(Int32.max), "Accelerate BLAS wrapper only supports Int32-sized dimensions.")
    return Int32(value)
}
