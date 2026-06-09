//
//  ScalarProtocols.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
import ComplexModule
import Foundation

/// A scalar that can be stored in a matrix.
///
/// Matrix scalars support addition, multiplication, equality, and the additive
/// identity required by dense linear algebra kernels. The protocol also requires
/// a multiplicative identity, ``one``, for constructors such as ``Matrix/eye(_:)``.
public protocol MatrixScalar: Numeric, Equatable, Sendable {
    /// The multiplicative identity.
    static var one: Self { get }
}

/// A real or complex floating-point scalar.
///
/// Floating scalars provide division, an absolute magnitude, and a default
/// tolerance used by numerical predicates. For complex scalars, `Magnitude`
/// denotes the associated real magnitude type.
public protocol FloatingScalar: MatrixScalar where Magnitude: RealFloatingScalar {
    /// The nonnegative absolute magnitude of the scalar.
    var magnitude: Magnitude { get }

    /// Divides one scalar by another.
    static func / (lhs: Self, rhs: Self) -> Self

    /// A type-specific default tolerance derived from floating-point precision.
    static var tolerance: Magnitude { get }
}

/// A real floating-point scalar supported by MxNumerics.
///
/// The protocol gathers the elementary operations that the core algorithms need
/// without exposing backend-specific LAPACK or MLX types.
public protocol RealFloatingScalar: FloatingScalar, BinaryFloatingPoint
where Magnitude == Self {
    /// Returns the principal square root of `x`.
    static func sqrt(_ x: Self) -> Self

    /// Returns `sqrt(a * a + b * b)` using the platform `hypot` routine.
    ///
    /// `hypot` is preferred over forming `a * a + b * b` directly because it is
    /// designed to avoid intermediate overflow and underflow.
    static func hypot(_ a: Self, _ b: Self) -> Self

    /// Returns the absolute value of `x`.
    static func abs(_ x: Self) -> Self
}

/// A complex floating-point scalar supported by MxNumerics.
///
/// The real and imaginary components share the same real scalar type. Linear
/// algebra routines that require Hermitian structure should use ``conjugate``
/// explicitly; a plain transpose and a conjugate transpose are different
/// operations for complex matrices.
public protocol ComplexFloatingScalar: FloatingScalar {
    /// The real component type.
    associatedtype Real: RealFloatingScalar

    /// The real component.
    var real: Real { get }

    /// The imaginary component.
    var imaginary: Real { get }

    /// The complex conjugate.
    var conjugate: Self { get }
}

extension Double: RealFloatingScalar {
    public static var one: Double { 1 }
    public static var tolerance: Double { .ulpOfOne.squareRoot() }

    public static func sqrt(_ x: Double) -> Double { Foundation.sqrt(x) }
    public static func hypot(_ a: Double, _ b: Double) -> Double { Foundation.hypot(a, b) }
    public static func abs(_ x: Double) -> Double { Swift.abs(x) }
}

extension Float: RealFloatingScalar {
    public static var one: Float { 1 }
    public static var tolerance: Float { .ulpOfOne.squareRoot() }

    public static func sqrt(_ x: Float) -> Float { Foundation.sqrt(x) }
    public static func hypot(_ a: Float, _ b: Float) -> Float { Foundation.hypot(a, b) }
    public static func abs(_ x: Float) -> Float { Swift.abs(x) }
}

extension Float16: RealFloatingScalar {
    public static var one: Float16 { 1 }
    public static var tolerance: Float16 { Float16(Float.ulpOfOne.squareRoot()) }

    public static func sqrt(_ x: Float16) -> Float16 { Float16(Float(x).squareRoot()) }
    public static func hypot(_ a: Float16, _ b: Float16) -> Float16 {
        Float16(Foundation.hypot(Float(a), Float(b)))
    }
    public static func abs(_ x: Float16) -> Float16 { Swift.abs(x) }
}

extension Int: MatrixScalar {
    public static var one: Int { 1 }
}

extension Complex: MatrixScalar where RealType: RealFloatingScalar {
    public static var one: Complex<RealType> { Complex(1) }
}

extension Complex: FloatingScalar where RealType: RealFloatingScalar {
    public typealias Magnitude = RealType

    public static var tolerance: RealType { RealType.tolerance }
}

extension Complex: ComplexFloatingScalar where RealType: RealFloatingScalar {
    public typealias Real = RealType

    public var conjugate: Complex<RealType> {
        Complex(real, -imaginary)
    }
}
