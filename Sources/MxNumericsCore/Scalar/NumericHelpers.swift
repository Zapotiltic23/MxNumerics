import Foundation

public func sqr<S: MatrixScalar>(_ value: S) -> S {
    value * value
}

public func pythag<S: RealFloatingScalar>(_ a: S, _ b: S) -> S {
    S.hypot(a, b)
}

public func sign<S: RealFloatingScalar>(_ a: S, _ b: S) -> S {
    b >= .zero ? S.abs(a) : .zero - S.abs(a)
}

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
