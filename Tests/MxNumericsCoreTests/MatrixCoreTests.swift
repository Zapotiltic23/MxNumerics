//
//  MatrixCoreTests.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//
import MxNumericsCore
import Testing

@Suite
struct MatrixCoreTests {
    @Test
    func tupleAndChainedSubscriptsMutateInPlace() throws {
        var matrix = try Matrix<Double>([
            [1, 2],
            [3, 4],
        ])

        matrix[0][1] = 9
        matrix[1, 0] = 6

        #expect(matrix.rowMajorElements() == [1, 9, 6, 4])
    }

    @Test
    func copyOnWriteKeepsCopiesIndependent() throws {
        var original = try Matrix<Double>([
            [1, 2],
            [3, 4],
        ])
        var copy = original

        copy[0, 0] = 99
        original[1][1] = 7

        #expect(original.rowMajorElements() == [1, 2, 3, 7])
        #expect(copy.rowMajorElements() == [99, 2, 3, 4])
    }

    @Test
    func transposeIsAViewAndSlicesCopyExpectedBlock() throws {
        let matrix = try Matrix<Int>([
            [1, 2, 3],
            [4, 5, 6],
        ])

        #expect(matrix.t.rowMajorElements() == [1, 4, 2, 5, 3, 6])
        #expect(matrix[0...0, .all].rowMajorElements() == [1, 2, 3])
        #expect(matrix[.all, 1].rowMajorElements() == [2, 5])
    }

    @Test
    func matrixMultiplicationAndPowerUseReferenceMath() throws {
        let a = try Matrix<Double>([
            [1, 2],
            [3, 4],
        ])
        let b = try Matrix<Double>([
            [2, 0],
            [1, 2],
        ])

        #expect((a * b).rowMajorElements() == [4, 4, 10, 8])
        #expect((a ^^ 2).rowMajorElements() == [7, 10, 15, 22])
    }

    @Test
    func vectorDotAndNorm() {
        let a = Vector<Double>([3, 4])
        let b = Vector<Double>([2, 1])

        #expect(a.dot(b) == 10)
        #expect(a.norm == 5)
    }
}
