# MxNumerics Binary Framework Release Checklist

MxNumerics ships source for development and a binary Swift package for app
consumption. App targets should depend on the binary package so Xcode shows the
generated public interface instead of implementation source.

## Public API Gate

- `import MxNumerics` is the only app-facing import.
- Backend and kernel implementation types stay internal.
- Do not add `@inlinable`, `@usableFromInline`, or `@frozen` without profiling
  and an explicit ABI review.
- Keep Swift Numerics exposure intentional. `ComplexModule.Complex` remains part
  of the v1 API until MxNumerics owns a replacement complex type.
- The current archive project excludes macOS `x86_64` because the v1 public API
  exposes `Float16`, which is unavailable for Intel macOS. Revisit this by
  adding availability-gated `Float16` APIs or dropping `Float16` from the public
  macOS surface if Intel macOS support becomes a release requirement.

## Release Steps

1. Run source tests in Swift 6 mode.
2. Archive `MxNumerics.framework` for every supported Apple destination with
   `BUILD_LIBRARY_FOR_DISTRIBUTION=YES` and `SKIP_INSTALL=NO`.
3. Create `MxNumerics.xcframework` with `xcodebuild -create-xcframework`.
4. Verify every variant contains a `.swiftinterface`.
5. Verify the public interface does not expose internal backend or kernel names.
6. Zip the XCFramework and compute `swift package compute-checksum`.
7. Publish the zip, update `Distribution/BinaryPackage/Package.swift`, tag the
   source release, and test a sample iOS app against the binary package.

## Required Framework Build Settings

- `BUILD_LIBRARY_FOR_DISTRIBUTION = YES`
- `DEFINES_MODULE = YES`
- `SKIP_INSTALL = NO`
- `SWIFT_VERSION = 6.0`
- `SWIFT_STRICT_CONCURRENCY = complete`
- `MACH_O_TYPE = mh_dylib`
- `VERSIONING_SYSTEM = apple-generic`
- `MARKETING_VERSION = 1.0.0`
- `CURRENT_PROJECT_VERSION = 1`
- `EXCLUDED_ARCHS[sdk=macosx*] = x86_64`
