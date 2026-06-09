# MxNumerics

MxNumerics is a Swift 6 numerical linear algebra framework for Apple platforms.
It provides MATLAB-style dense matrix and vector ergonomics on top of a layered
architecture designed for Accelerate, MLX, and portable Swift reference kernels.

The package currently implements the full public routine catalog described in
the development plan as Swift reference algorithms, plus backend routing hooks
for future tuned Accelerate and MLX kernels. In other words: the API surface and
mathematical routines are present and tested, while backend-specialized
performance work remains isolated behind the backend targets.

## Package Status

- **Language:** Swift 6
- **Package manager:** Swift Package Manager
- **Platforms:** macOS 14+, iOS 17+, visionOS 1+, tvOS 17+
- **Primary product:** `MxNumerics`
- **Scalar support:** `Double`, `Float`, `Float16`, `Complex<Double>`, `Complex<Float>` where routines permit them
- **Indexing:** zero-based, with both `A[i, j]` and `A[i][j]`
- **Storage:** dense row-major storage with copy-on-write value semantics
- **Current kernels:** Swift reference implementations with backend dispatch infrastructure

## Installation

Add the package to an app or framework with Swift Package Manager:

```swift
.package(url: "https://github.com/Zapotiltic23/MxNumerics.git", branch: "develop")
```

Then add the product to your target:

```swift
.product(name: "MxNumerics", package: "MxNumerics")
```

Use it from Swift:

```swift
import MxNumerics

let A = try Matrix<Double>([
    [4, 3],
    [6, 3],
])

let b = Vector<Double>([10, 12])
let x = try A.solve(b)
```

## Design Overview

MxNumerics is split into narrow modules:

| Target | Role |
|---|---|
| `MxNumericsCore` | Scalar protocols, dense matrix/vector types, COW storage, shape/stride metadata, slicing tokens, basic arithmetic |
| `MxNumericsBackend` | Backend identifiers, operation routing, plain row-major `BufferRef`, reference backend protocol |
| `MxNumericsAccelerate` | Accelerate integration point for BLAS/LAPACK/vDSP shims |
| `MxNumericsMLX` | MLX integration point for future GPU-resident Float/Float16 kernels |
| `MxNumerics` | Umbrella API, runtime controls, numerical routine catalog |

The public API does not expose Accelerate or MLX types. Backends communicate
through plain row-major buffers, which keeps the API stable while allowing the
implementation to move work between CPU, GPU, and reference kernels.

## Core Data Model

### Matrix

`Matrix<Scalar>` is a value type with internal copy-on-write storage.

```swift
var A: Matrix<Double> = [
    [1, 2, 3],
    [4, 5, 6],
]

A[0, 1] = 9
A[1][2] = 7
let row = A[0, .all]
let column = A[.all, 1]
let transpose = A.t
```

MxNumerics stores dense matrices in row-major logical order. A transpose is a
stride view where possible, and `rowMajorElements()` materializes the logical
row-major order.

### Vector

`Vector<Scalar>` is a thin wrapper around an \(n \times 1\) matrix:

```swift
let v = Vector<Double>([3, 4])
let length = v.norm
```

For real scalars, `dot(_:)` computes \(x^T y\). For complex scalars,
`innerProduct(_:_:)` uses the Hermitian convention \(x^H y\).

### Scalars

The scalar hierarchy separates storage-capable values from floating-point
values and complex values:

| Protocol | Purpose |
|---|---|
| `MatrixScalar` | Anything that can be stored in a matrix |
| `FloatingScalar` | Real or complex floating values with magnitude and tolerance |
| `RealFloatingScalar` | Real floating fields such as `Double`, `Float`, `Float16` |
| `ComplexFloatingScalar` | Complex fields backed by a real floating scalar |

## Backend Policy

The runtime router chooses a backend based on operation, scalar type, size, and
availability:

```swift
MxNumerics.deterministicMode = true
MxNumerics.backendPolicy = .auto
let decision = A.backendDecision(for: .gemm)
```

Current policy:

- `Double` and factorization-class routines prefer the Accelerate path.
- Large `Float` GEMM is eligible for MLX when MLX is available.
- The portable `ReferenceBackend` is used as a correctness path.
- `deterministicMode` forces CPU-oriented routing decisions for reproducibility.

The implemented high-level numerical catalog currently uses Swift reference
algorithms. This makes the API usable and testable today, while preserving the
backend boundary for future tuned BLAS/LAPACK/MLX kernels.

## Numerical Accuracy Notes

MxNumerics favors stable routines where the API name implies production use, but
some educational or compatibility routines are intentionally exposed with
caveats:

- `solveSystemPLU` uses pivoted LU for square systems.
- `qr(_:)` uses Householder reflectors.
- `gramSchmidtFactorization(_:mode:)` defaults to modified Gram-Schmidt.
- `solveSystemOrdinaryLeastSquares` intentionally implements normal equations:
  \[
  x = (A^T A)^{-1}A^T b
  \]
  This squares the condition number and is less stable than QR/SVD least squares.
- `characteristicPolynomialRoots(_:)` is included for completeness, but matrix
  \(\rightarrow\) polynomial \(\rightarrow\) roots is generally ill-conditioned.
  Prefer `eigenValuesVectors(_:)` for eigenvalues.
- `singularValueDecomposition(_:)` currently computes a reference SVD through
  an eigen decomposition of \(A^T A\). This is useful for coverage, but a tuned
  LAPACK-backed SVD is the desired production kernel.
- `Matrix.frobeniusNorm` and `Vector.norm` currently use direct sums of squared
  magnitudes. Extreme values can overflow or underflow before the final square
  root. A scaled BLAS `nrm2` path is planned.

## Examples

### Linear Solve

```swift
let A = try Matrix<Double>([
    [4, 3],
    [6, 3],
])
let b = Vector<Double>([10, 12])
let x = try solveSystemPLU(A, b)
```

Solves \(Ax=b\) using pivoted LU.

### QR Factorization

```swift
let qrA = qr(A)
let residual = norm(qrA.q * qrA.r - A, .frobenius)
```

Householder QR returns matrices \(Q\) and \(R\) such that:

\[
A = QR,\quad Q^TQ \approx I
\]

### Cholesky Factorization

```swift
let spd = try Matrix<Double>([
    [25, 15, -5],
    [15, 18, 0],
    [-5, 0, 11],
])

let L = try choleskyDecomposition(spd).lower
```

For symmetric positive-definite \(A\):

\[
A = LL^T
\]

### SVD and Pseudoinverse

```swift
let svd = singularValueDecomposition(A)
let pinv = pseudoInverseMoorePenrose(A)
```

The singular value decomposition is:

\[
A = U\Sigma V^T
\]

The Moore-Penrose pseudoinverse is:

\[
A^+ = V\Sigma^+U^T
\]

### Eigenvalues

```swift
let eig = eigenValuesVectors(A)
let radius = spectralRadius(A)
```

The spectral radius is:

\[
\rho(A) = \max_i |\lambda_i|
\]

## Implemented Routine Table

The table below lists the public routines implemented in the package. “Reference”
means the routine is implemented in Swift today, not yet as a tuned backend
kernel.

### Matrix Construction, Access, and Structure

| API | Formula / behavior | Notes |
|---|---|---|
| `Matrix(rows:columns:repeating:)` | \(A_{ij}=c\) | Dense constant matrix |
| `Matrix(rows:cols:repeating:)` | \(A_{ij}=c\) | MATLAB-style spelling |
| `Matrix(rowMajor:rows:columns:)` | row-major buffer to matrix | Validates element count |
| `Matrix(rowMajor:shape:)` | row-major buffer to shape | Shape-based initializer |
| `Matrix(_:)` | nested row arrays | Rejects empty/ragged input |
| `Matrix.zeros(_:_:)` | \(A_{ij}=0\) | Dense zero matrix |
| `Matrix.ones(_:_:)` | \(A_{ij}=1\) | Dense ones matrix |
| `Matrix.eye(_:)` | \(I_{ij}=1\) if \(i=j\) else \(0\) | Identity matrix |
| `Matrix.diag(_:)` / `diagonal(_:)` | \(A_{ii}=v_i\) | Builds diagonal matrix |
| `getDiagonal(_:)` | \(v_i=A_{ii}\) | Extracts main diagonal |
| `getMatrixRow(_:_:)` | row \(i\) | Returns `Vector` |
| `getMatrixColumn(_:_:)` | column \(j\) | Returns `Vector` |
| `transpose(_:)`, `Matrix.t`, `Matrix.transposed` | \(A^T_{ij}=A_{ji}\) | Stride view where possible |
| `Matrix.h` | \(A^H=\overline{A}^T\) | Complex conjugate transpose |
| `rowMajorElements()` | logical row-major materialization | Copies values |
| `repmat(_:rows:columns:)` | tile matrix by row/column repeats | MATLAB-like |
| `reshape(_:rows:columns:)` | reshape preserving row-major order | Element count must match |
| `reshapeV2(_:rowRange:columnRange:)` | block extraction | Alias-style routine |
| `embedMatrix(_:into:row:column:)` | writes block into a matrix | Returns a new matrix |
| `dropRow(_:_:)` | remove one row | Structural copy |
| `dropColumn(_:_:)` | remove one column | Structural copy |
| `obtainSubMatrix(_:droppingRow:column:)` | delete row and column | Minor/submatrix helper |
| `principalSubmatrices(_:)` | leading \(A_{1:k,1:k}\) blocks | Square matrices |
| `principalMinors(_:)` | \(\det(A_{1:k,1:k})\) | Uses determinant |
| `elementStringtoArray(_:)` | parses `"[1, 2, 3]"` | Strict comma parser |

### Arithmetic and Elementwise Operations

| API | Formula / behavior | Notes |
|---|---|---|
| `+`, `-` for matrices | \(C_{ij}=A_{ij}\pm B_{ij}\) | Equal shapes |
| `+`, `-` with scalar | \(C_{ij}=A_{ij}\pm c\) | Broadcast scalar |
| `*` matrix-matrix | \(C_{ij}=\sum_k A_{ik}B_{kj}\) | Reference GEMM |
| `*` scalar-matrix | \(C_{ij}=cA_{ij}\) | Scalar broadcast |
| `/` matrix-scalar | \(C_{ij}=A_{ij}/c\) | Floating scalars |
| `/` matrix-matrix | \(A/B=(B^T\backslash A^T)^T\) | Right solve |
| `hadamard(_:_:)` | \(C_{ij}=A_{ij}B_{ij}\) | Elementwise product |
| `elementwiseDivide(_:_:)` | \(C_{ij}=A_{ij}/B_{ij}\) | Elementwise quotient |
| `dividePwMatrix(_:_:)` | \(A ./ B\) | Alias |
| `multiply(_:_:)` | \(AB\) | Function spelling |
| `constantMatrixMultiplication(_:_:)` | \(cA\) | Alias |
| `multiplyConstantMatrix(_:_:)` | \(Ac\) | Alias |
| `divideConstantMatrix(_:_:)` | \(A/c\) | Alias |
| `multiplyAddInPlace(_:_:into:)` | \(C \leftarrow C + AB\) | Reference update |
| `powerMatrix(_:_:)`, `^`, `^^` | \(A^k\) | Repeated squaring, \(k\ge0\) |
| `sum(_:)` | \(\sum_{ij} A_{ij}\) | Generic scalar sum |
| `trace(_:)`, `Matrix.trace` | \(\operatorname{tr}(A)=\sum_i A_{ii}\) | Square matrix |
| `absMatrix(_:)` | \(B_{ij}=|A_{ij}|\) | Uses scalar magnitude |
| `swapElements(_:_:_:)` | swaps two array entries | Utility |
| `sqr(_:)` | \(x^2\) | Utility |
| `pythag(_:_:)` | \(\sqrt{a^2+b^2}\) | Uses `hypot` |
| `sign(_:_:)` | \(|a|\operatorname{sign}(b)\) | Fortran SIGN behavior |
| `elementMagnitudeInMatrix(_:)` | \(\lceil\log_{10}(\max |a_{ij}|+1)\rceil\) | Display helper |

### Norms, Rank, Conditioning, and Subspaces

| API | Formula / behavior | Notes |
|---|---|---|
| `norm(_: .one)` | \(\max_j\sum_i |a_{ij}|\) | Matrix 1-norm |
| `norm(_: .infinity)` | \(\max_i\sum_j |a_{ij}|\) | Matrix infinity norm |
| `norm(_: .frobenius)` | \(\sqrt{\sum_{ij}|a_{ij}|^2}\) | Direct sum |
| `norm(_: .two)` | \(\sigma_{\max}(A)\) | Via SVD |
| `norm(_: VectorNorm.two)` | \(\sqrt{\sum_i |x_i|^2}\) | Direct sum |
| `conditionNumber(_:)` | \(\kappa_2(A)=\sigma_{\max}/\sigma_{\min}\) | Via SVD |
| `rank(_:tolerance:)` | \(\#\{\sigma_i>\tau\}\) | SVD threshold |
| `nullity(_:tolerance:)` | \(n-\operatorname{rank}(A)\) | Rank-nullity |
| `fundamentalSubspaces(_:tolerance:)` | column, row, null, left-null bases | SVD-based |
| `fundemantalSubspaces(_:tolerance:)` | same as above | Deprecated typo-compatible alias |

### QR and Orthogonalization

| API | Formula / behavior | Notes |
|---|---|---|
| `householderVector(_:)` | \(H=I-\beta vv^T\) | Stable sign choice |
| `qr(_:)`, `Matrix.qr()` | \(A=QR\) | Householder reference QR |
| `gramSchmidtFactorization(_:mode:)` | \(A=QR\) | Modified or classical GS |
| `gramSmchmidtFactorization(_:)` | \(A=QR\) | Typo-compatible alias |
| `isOrthogonal(_:)` | \(\|Q^TQ-I\|\le\tau\) | Tolerance-based |
| `isGramSchmidt(_:of:)` | checks \(QR\approx A\), \(Q^TQ\approx I\) | Property check |

### LU, Direct Solves, Determinant, and Inverse

| API | Formula / behavior | Notes |
|---|---|---|
| `luDecompositionDoolittle(_:)` | \(A=LU\) | Unpivoted; educational |
| `luWithScaledRowPivoting(_:)` | \(PA=LU\) | Scaled partial pivoting |
| `croutsLUwithPartialImplicitPivoting(_:)` | \(PA=LU\) | Delegates to scaled pivoting |
| `solveSystemPLU(_:_:)`, `Matrix.solve(_:)` | solves \(Ax=b\) | Pivoted LU |
| `solveVectorInverseLU(_:_:)` | solves \(Ax=b\) | Alias |
| `solveMatrixInverseLU(_:_:)` | solves \(AX=B\) | Multiple RHS |
| `determinant(_:)`, `Matrix.determinant()` | \(\det(A)=\operatorname{sgn}(P)\prod_i U_{ii}\) | LU-based |
| `inverse(_:)`, `Matrix.inverse` | \(A^{-1}\) | Solves against identity |
| `inverseLU(_:)` | \(A^{-1}\) | Alias |
| `adjoint(_:)` | \(\operatorname{adj}(A)=\det(A)A^{-1}\) | Requires nonsingular matrix |

### Cholesky

| API | Formula / behavior | Notes |
|---|---|---|
| `choleskyDecomposition(_:)`, `Matrix.cholesky()` | \(A=LL^T\) | SPD matrices |
| `isPositiveDefinite(_:)` | Cholesky succeeds | Numerical predicate |
| `isCholesky(_:of:)` | \(\|LL^T-A\|\le\tau\) | Tolerance-based |

### SVD, Pseudoinverse, and Least Squares

| API | Formula / behavior | Notes |
|---|---|---|
| `singularValueDecomposition(_:)`, `Matrix.svd()` | \(A=U\Sigma V^T\) | Reference via \(A^TA\) |
| `reorderSVD(_:)` | \(\sigma_1\ge\sigma_2\ge\cdots\) | Reorders columns |
| `pseudoInverseMoorePenrose(_:tolerance:)` | \(A^+=V\Sigma^+U^T\) | SVD-based |
| `solveSystemPseudoInverse(_:_:)` | \(x=A^+b\) | Minimum-norm LS style solve |
| `solveSystemOrdinaryLeastSquares(_:_:)` | \(x=(A^TA)^{-1}A^Tb\) | Normal equations caveat |
| `solveSystemIterativelyReweightedLeastSquares` | repeated weighted normal equations | Robust/weighted LS reference |

### Eigenvalues, Schur, Hessenberg, and Characteristic Polynomial

| API | Formula / behavior | Notes |
|---|---|---|
| `jacobiEigen(_:maxIterations:tolerance:)` | \(A=VDV^T\) for symmetric \(A\) | Jacobi rotations |
| `balanceMatrix(_:radix:)` | \(B=D^{-1}AD\) | Power-of-radix scaling |
| `upperHessenberg(_:)` | \(H=Q^TAQ\) | Householder reduction |
| `francisQRStep(_:)` | one shifted QR-style step | Educational reference |
| `qrAlgorithmBasic(_:iterations:)` | \(A_{k+1}=R_kQ_k\) | Unshifted QR iteration |
| `qrAlgorithm(_:iterations:)` | shifted QR iteration | Hessenberg start |
| `hessRealSchurForm(_:iterations:)` | quasi-Schur approximation | Reference QR |
| `realSchurFormDecomposition(_:iterations:)` | \(A\approx QTQ^T\) | Accumulates Q |
| `similarityTransformsRealSchurForm(_:iterations:)` | same as Schur decomposition | Alias |
| `eigenPairsRealSchurWithExceptionalShift` | Schur/eigen driver | Reference alias |
| `eigenValuesVectors(_:iterations:)`, `Matrix.eig()` | eigenvalues, symmetric vectors when available | Jacobi for symmetric |
| `powerMethod(_:maxIterations:tolerance:)` | dominant eigenpair | \(v_{k+1}=Av_k/\|Av_k\|\) |
| `shiftedInversePM(_:shift:maxIterations:tolerance:)` | inverse iteration near shift | Solves \((A-\mu I)y=x\) |
| `spectralRadius(_:)` | \(\rho(A)=\max_i|\lambda_i|\) | Eigenvalue-based |
| `sortEigenpairs(_:key:)` | sort by real part or magnitude | Reorders vectors too |
| `normalizeEigenVectors(_:)` | column normalization | Unit 2-norm columns |
| `scaleVectors(_:scales:)` | \(V_s=VS\) | Column scaling |
| `charPolyCoefficientsFaddeevLeverrier(_:)` | Faddeev-LeVerrier coefficients | \(p_A(\lambda)\) |
| `characteristicPolynomialRoots(_:)` | roots via companion matrix eigenvalues | Ill-conditioned path |
| `sortPolyCoefficients(_:key:)` | sort complex roots | Name preserved from plan |

### Iterative Linear Solvers

| API | Formula / behavior | Notes |
|---|---|---|
| `solveSystemJacobi(_:_:)` | \(x^{k+1}=D^{-1}(b-(L+U)x^k)\) | Stationary iteration |
| `solveSystemGaussSeidel(_:_:)` | forward in-place sweep | Stationary iteration |
| `solveSystemSuccessiveOverRelaxation` | \(x\leftarrow(1-\omega)x+\omega x_{GS}\) | \(0<\omega<2\) |
| `solveSystemConjugateGradient(_:_:)` | CG recurrences | SPD systems |
| `solveSystemKaczmarz(_:_:)` | row-action projection | Cyclic rows |

### Complex Routines

| API | Formula / behavior | Notes |
|---|---|---|
| `complexAbsoluteValue(_:)`, `absComplex(_:)` | \(|z|=\sqrt{x^2+y^2}\) | Uses `hypot` |
| `sqrtComplex(_:)` | principal square root | Stable branch formula |
| `conjugateMatrix(_:)` | \(\overline{A}\) | Elementwise conjugate |
| `complexTranspose(_:)` | \(A^T\) | Plain transpose |
| `conjugateTranspose(_:)` | \(A^H=\overline{A}^T\) | Hermitian transpose |
| `conjugateTransposeVector(_:)` | \(x^H\) | Row matrix |
| `multiplyComplexMatrix(_:_:)` | \(AB\) | Complex matrix product |
| `subtractComplexMatrix(_:_:)` | \(A-B\) | Complex subtraction |
| `diagonalComplexMatrix(_:)` | complex diagonal matrix | Diagonal builder |
| `getComplexMatrixColumn(_:_:)` | complex column vector | Extractor |
| `complexVectorNorm(_:)` | \(\sqrt{\sum_i |z_i|^2}\) | Direct sum |
| `realMatrix(_:)` | \(\Re(A)\) | Part extraction |
| `imaginaryMatrix(_:)` | \(\Im(A)\) | Part extraction |
| `realToComplex(_:)` | \(A+0i\) | Promotion |
| `innerProduct(_:_:)` for complex vectors | \(x^Hy\) | Hermitian convention |

### Vector Geometry and Statistics

| API | Formula / behavior | Notes |
|---|---|---|
| `+`, `-` for vectors | \(x\pm y\) | Equal lengths |
| `*` scalar-vector | \(cx\) | Scalar broadcast |
| `innerProduct(_:_:)` for real vectors | \(x^Ty\) | Dot product |
| `crossProductVector(_:_:)` | \(x\times y\) | 3D only |
| `angleBetweenVectors(_:_:)` | \(\arccos\frac{x^Ty}{\|x\|\|y\|}\) | Clamped cosine |
| `orthogonalProjectionUontoV(_:_:)` | \(\frac{u^Tv}{v^Tv}v\) | Projection onto \(v\) |
| `meanOfVector(_:)` | \(\frac1n\sum_i x_i\) | Nonempty vector |
| `vectorCorrelation(_:_:)` | Pearson correlation | Centered normalized dot |

### Polynomial and Miscellaneous Routines

| API | Formula / behavior | Notes |
|---|---|---|
| `polynomialEvaluation(_:at:)` | Horner evaluation | Real and complex overloads |
| `polynomialDivision(_:by:)` | \(u=qv+r\) | Long division |
| `mullerRoots(_:maxIterations:tolerance:)` | Muller's method | Complex roots, deflation |
| `cheb(_:)` | Chebyshev differentiation matrix | Trefethen-style nodes |
| `rationalApproximation(_:tolerance:maxDenominator:)` | rational \(p/q\approx x\) | Stern-Brocot search |

### Matrix Predicates

| API | Formula / behavior | Notes |
|---|---|---|
| `isUpperTriangular(_:)` | \(a_{ij}=0\) for \(i>j\) | Tolerance-based |
| `isLowerTriangular(_:)` | \(a_{ij}=0\) for \(i<j\) | Tolerance-based |
| `isDiagonal(_:)` | upper and lower triangular | Tolerance-based |
| `isSymmetric(_:)` | \(A=A^T\) | Real matrices |
| `isUpperHessenberg(_:)` | \(a_{ij}=0\) for \(i>j+1\) | Tolerance-based |
| `isZeroMatrix(_:)` | all entries near zero | Tolerance-based |
| `isToeplitz(_:)` | constant diagonals | Tolerance-based |
| `isConvergentMatrix(_:)` | \(\|A\|_\infty<1\) | Sufficient check |
| `isRowDiagonallyDominant(_:)` | \(|a_{ii}|\ge\sum_{j\ne i}|a_{ij}|\) | Optional strictness |
| `isColumnDiagonallyDominant(_:)` | \(|a_{jj}|\ge\sum_{i\ne j}|a_{ij}|\) | Optional strictness |
| `isAbsoluteDiagonallyDominant(_:)` | row and column dominance | Optional strictness |
| `isSingular(_:)` | \(\operatorname{rank}(A)<n\) | SVD-based |

## Development and Verification

Run the test suite:

```bash
swift test
```

Generate symbol graphs for documentation validation:

```bash
swift package dump-symbol-graph
```

Current test coverage includes construction, COW behavior, indexing, matrix
multiplication, backend routing, QR reconstruction, LU solve/determinant/inverse,
Cholesky, SVD reconstruction, eigenvalues, iterative solvers, polynomial
routines, complex helpers, and a catalog smoke test that touches the Section 6
API surface.

## Roadmap

The next engineering step is backend specialization:

1. Replace reference GEMM with typed CBLAS row-major calls for `Float` and `Double`.
2. Add LAPACK shims for LU, QR, Cholesky, SVD, and eigen drivers.
3. Add MLX residency-backed Float/Float16 bulk kernels.
4. Expand golden fixtures against NumPy/SciPy, MATLAB, and Accelerate.
5. Add performance benchmarks and backend crossing diagnostics.
