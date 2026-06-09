# ``MxNumerics``

A Swift 6 numerical linear algebra framework for dense matrices and vectors.

## Overview

MxNumerics provides value-type dense matrices, dense vectors, and backend routing
infrastructure for Apple platforms. The current package implements the
foundational API: row-major matrix storage, copy-on-write value semantics,
zero-based indexing, matrix construction, slicing, transposition, arithmetic,
integer powers, vector dot products, Euclidean norms, and an async reference
GEMM backend.

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
the CPU integration point for future BLAS/LAPACK shims. ``MLXBackend`` is the GPU
integration point for future Float and Float16 bulk kernels.

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
