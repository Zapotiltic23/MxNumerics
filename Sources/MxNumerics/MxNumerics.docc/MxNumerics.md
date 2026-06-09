# ``MxNumerics``

A Swift 6 numerical linear algebra framework for dense matrices and vectors.

## Overview

MxNumerics provides value-type dense matrices, dense vectors, numerical routine
families, and backend routing infrastructure for Apple platforms. The package
implements row-major matrix storage, copy-on-write value semantics, zero-based
indexing, matrix construction, slicing, transposition, arithmetic, integer
powers, vector dot products, Euclidean norms, an async reference GEMM backend,
and the numerical routine catalog from the development plan.

The framework uses the following numerical conventions:

- Matrix entries are indexed from zero.
- Dense matrix buffers are row-major in the public core layer.
- The standard matrix product `A * B` computes `C[i, j] = sum(A[i, k] * B[k, j])`.
- Vector ``Vector/dot(_:)`` currently computes the unconjugated bilinear product
  `x^T y`. Complex Hermitian products should be implemented explicitly as `x^H y`.
- ``Matrix/frobeniusNorm`` and ``Vector/norm`` currently use direct sums of
  squared magnitudes. This is mathematically correct for ordinary inputs but is
  not yet the scaled BLAS `nrm2` algorithm used to reduce overflow and underflow
  risk for extreme data.
- ``MxNumerics/deterministicMode`` forces CPU routing decisions for regression
  stability.

The backend layer is intentionally separated from the public matrix API.
``ReferenceBackend`` is a portable correctness backend. ``AccelerateBackend`` is
the CPU integration point for BLAS/LAPACK shims. ``MLXBackend`` is the GPU
integration point for Float and Float16 bulk kernels. The current routine
catalog uses Swift reference algorithms for broad correctness coverage while
leaving backend-specialized kernels isolated behind the backend targets.

## Topics

### Core Types

- ``Matrix``
- ``Vector``
- ``Shape``
- ``Strides``
- ``MatrixSlice``
- ``LinAlgError``

### Scalars

- ``MatrixScalar``
- ``FloatingScalar``
- ``RealFloatingScalar``
- ``ComplexFloatingScalar``

### Numeric Helpers

- ``sqr(_:)``
- ``pythag(_:_:)``
- ``sign(_:_:)``
- ``elementMagnitudeInMatrix(_:)``

### Factorizations and Solvers

- ``qr(_:)``
- ``gramSchmidtFactorization(_:mode:)``
- ``gramSmchmidtFactorization(_:)``
- ``householderVector(_:)``
- ``luDecompositionDoolittle(_:)``
- ``luWithScaledRowPivoting(_:)``
- ``croutsLUwithPartialImplicitPivoting(_:)``
- ``solveSystemPLU(_:_:)``
- ``determinant(_:)``
- ``inverse(_:)``
- ``inverseLU(_:)``
- ``choleskyDecomposition(_:)``
- ``singularValueDecomposition(_:)``
- ``pseudoInverseMoorePenrose(_:tolerance:)``
- ``solveSystemPseudoInverse(_:_:)``
- ``solveSystemOrdinaryLeastSquares(_:_:)``
- ``solveSystemIterativelyReweightedLeastSquares(_:_:p:delta:maxIterations:tolerance:)``

### Eigenvalues and Matrix Polynomials

- ``balanceMatrix(_:radix:)``
- ``upperHessenberg(_:)``
- ``francisQRStep(_:)``
- ``qrAlgorithm(_:iterations:)``
- ``qrAlgorithmBasic(_:iterations:)``
- ``realSchurFormDecomposition(_:iterations:)``
- ``eigenValuesVectors(_:iterations:)``
- ``jacobiEigen(_:maxIterations:tolerance:)``
- ``powerMethod(_:maxIterations:tolerance:)``
- ``shiftedInversePM(_:shift:maxIterations:tolerance:)``
- ``spectralRadius(_:)``
- ``charPolyCoefficientsFaddeevLeverrier(_:)``
- ``characteristicPolynomialRoots(_:)``

### Predicates, Structure, and Geometry

- ``norm(_:_:)``
- ``conditionNumber(_:)``
- ``rank(_:tolerance:)``
- ``nullity(_:tolerance:)``
- ``fundamentalSubspaces(_:tolerance:)``
- ``repmat(_:rows:columns:)``
- ``reshape(_:rows:columns:)``
- ``embedMatrix(_:into:row:column:)``
- ``dropRow(_:_:)``
- ``dropColumn(_:_:)``
- ``isSymmetric(_:tolerance:)``
- ``isPositiveDefinite(_:)``
- ``isSingular(_:tolerance:)``
- ``innerProduct(_:_:)``
- ``crossProductVector(_:_:)``
- ``angleBetweenVectors(_:_:)``
- ``orthogonalProjectionUontoV(_:_:)``

### Polynomial, Complex, and Miscellaneous Routines

- ``polynomialEvaluation(_:at:)``
- ``polynomialDivision(_:by:)``
- ``mullerRoots(_:maxIterations:tolerance:)``
- ``cheb(_:)``
- ``rationalApproximation(_:tolerance:maxDenominator:)``
- ``elementStringtoArray(_:)``
- ``conjugateTranspose(_:)``
- ``complexAbsoluteValue(_:)``
- ``sqrtComplex(_:)``
- ``realToComplex(_:)``

### Backend Routing

- ``MxNumerics``
- ``BackendID``
- ``BackendOperation``
- ``BackendPolicy``
- ``BackendRouter``
- ``DispatchDecision``
- ``LinearAlgebraBackend``
- ``BufferRef``
- ``ReferenceBackend``
- ``AccelerateBackend``
- ``MLXBackend``

## Numerical References

The current API documentation follows the standard BLAS/LAPACK terminology for
GEMM, vector 2-norms, and matrix factorizations, and uses the platform `hypot`
routine for stable two-component Euclidean lengths.

- BLAS quick reference, Netlib LAPACK Users' Guide.
- LAPACK QR and orthogonal/unitary-factor routine groups, Netlib LAPACK.
- LAPACK Users' Guide, especially sections on computational routines, accuracy,
  and stability.
