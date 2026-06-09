import ComplexModule
import Foundation

public protocol MatrixScalar: Numeric, Equatable, Sendable {
    static var one: Self { get }
}

public protocol FloatingScalar: MatrixScalar where Magnitude: RealFloatingScalar {
    var magnitude: Magnitude { get }
    static func / (lhs: Self, rhs: Self) -> Self
    static var tolerance: Magnitude { get }
}

public protocol RealFloatingScalar: FloatingScalar, BinaryFloatingPoint
where Magnitude == Self {
    static func sqrt(_ x: Self) -> Self
    static func hypot(_ a: Self, _ b: Self) -> Self
    static func abs(_ x: Self) -> Self
}

public protocol ComplexFloatingScalar: FloatingScalar {
    associatedtype Real: RealFloatingScalar

    var real: Real { get }
    var imaginary: Real { get }
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
