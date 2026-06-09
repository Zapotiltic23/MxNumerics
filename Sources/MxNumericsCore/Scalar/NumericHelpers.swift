//
//  NumericHelpers.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
import Foundation

/// Returns the square of a scalar value.
///
/// This helper computes `value * value`. For floating-point values with very
/// large magnitude, the multiplication can overflow in the usual IEEE 754 way.
public func sqr<S: MatrixScalar>(_ value: S) -> S {
    value * value
}

/// Returns the Euclidean length of a two-component real vector.
///
/// This function delegates to the platform `hypot` implementation, which is the
/// numerically preferred way to compute `sqrt(a * a + b * b)` because it avoids
/// many overflow and underflow cases that arise from squaring first.
///
/// - Parameters:
///   - a: The first component.
///   - b: The second component.
/// - Returns: The value `sqrt(a * a + b * b)`, computed by `hypot`.
public func pythag<S: RealFloatingScalar>(_ a: S, _ b: S) -> S {
    S.hypot(a, b)
}

/// Transfers the sign of one real scalar to the magnitude of another.
///
/// This is the Fortran-style `SIGN(a, b)` operation used in many Householder
/// and QR algorithms: it returns `abs(a)` when `b >= 0`, and `-abs(a)` otherwise.
/// It is not the same operation as `signum`.
///
/// - Parameters:
///   - a: The value whose magnitude is used.
///   - b: The value whose sign is used.
/// - Returns: `abs(a)` with the sign of `b`.
public func sign<S: RealFloatingScalar>(_ a: S, _ b: S) -> S {
    b >= .zero ? S.abs(a) : .zero - S.abs(a)
}

/// Estimates the number of decimal digits needed to display the largest entry.
///
/// The value is `ceil(log10(max(abs(a_ij)) + 1))`, using the scalar magnitude for
/// real and complex values. A zero matrix returns `0`.
///
/// - Parameter matrix: The matrix whose entries are inspected.
/// - Returns: A nonnegative decimal magnitude estimate.
public func elementMagnitudeInMatrix<S: FloatingScalar>(_ matrix: Matrix<S>) -> Int
where S.Magnitude: BinaryFloatingPoint {
    let maximum = matrix.rowMajorElements().reduce(S.Magnitude.zero) { partial, value in
        Swift.max(partial, value.magnitude)
    }
    guard maximum > .zero else { return 0 }
    return Int(FoundationBridge.log10(Double(maximum) + 1).rounded(.up))
}

private enum FoundationBridge {
    static func log10(_ value: Double) -> Double {
        Foundation.log10(value)
    }
}
