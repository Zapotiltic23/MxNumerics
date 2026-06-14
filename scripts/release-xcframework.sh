#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-1.0.0}"
ARTIFACT_BASE_URL="${2:-https://example.com/MxNumerics/releases/download/${VERSION}}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/.build/binary-release/${VERSION}"
ARCHIVE_DIR="${BUILD_DIR}/archives"
PRODUCT_DIR="${BUILD_DIR}/products"
XCFRAMEWORK_PATH="${PRODUCT_DIR}/MxNumerics.xcframework"
ZIP_PATH="${PRODUCT_DIR}/MxNumerics-${VERSION}.xcframework.zip"
MANIFEST_PATH="${ROOT_DIR}/Distribution/BinaryPackage/Package.swift"
SCHEME="${MXNUMERICS_SCHEME:-MxNumerics}"
PROJECT="${MXNUMERICS_PROJECT:-${ROOT_DIR}/MxNumerics.xcodeproj}"
WORKSPACE="${MXNUMERICS_WORKSPACE:-}"

if [[ -n "${WORKSPACE}" ]]; then
    BUILD_CONTAINER=(-workspace "${WORKSPACE}")
elif [[ -d "${PROJECT}" ]]; then
    BUILD_CONTAINER=(-project "${PROJECT}")
else
    cat >&2 <<MSG
Missing archive-capable framework project.

Create MxNumerics.xcodeproj or set MXNUMERICS_PROJECT / MXNUMERICS_WORKSPACE.
The framework target must use:
  BUILD_LIBRARY_FOR_DISTRIBUTION=YES
  SKIP_INSTALL=NO
  DEFINES_MODULE=YES
  SWIFT_VERSION=6.0
  SWIFT_STRICT_CONCURRENCY=complete
MSG
    exit 2
fi

archive_framework() {
    local name="$1"
    local destination="$2"
    local archive_path="${ARCHIVE_DIR}/${name}.xcarchive"

    xcodebuild archive \
        "${BUILD_CONTAINER[@]}" \
        -scheme "${SCHEME}" \
        -configuration Release \
        -destination "${destination}" \
        -archivePath "${archive_path}" \
        SKIP_INSTALL=NO \
        BUILD_LIBRARY_FOR_DISTRIBUTION=YES \
        DEFINES_MODULE=YES \
        SWIFT_VERSION=6.0 \
        SWIFT_STRICT_CONCURRENCY=complete \
        MARKETING_VERSION="${VERSION}" \
        CURRENT_PROJECT_VERSION="${VERSION}" \
        CODE_SIGNING_ALLOWED=NO

    local framework="${archive_path}/Products/Library/Frameworks/MxNumerics.framework"
    if [[ ! -d "${framework}" ]]; then
        echo "Archive did not produce ${framework}" >&2
        exit 3
    fi

    FRAMEWORK_ARGS+=(-framework "${framework}")

    local dsym="${archive_path}/dSYMs/MxNumerics.framework.dSYM"
    if [[ ! -d "${dsym}" ]]; then
        echo "Archive did not produce ${dsym}" >&2
        exit 4
    fi
    FRAMEWORK_ARGS+=(-debug-symbols "${dsym}")

    local symbol_maps_dir="${archive_path}/BCSymbolMaps"
    if [[ -d "${symbol_maps_dir}" ]]; then
        while IFS= read -r -d '' symbol_map; do
            FRAMEWORK_ARGS+=(-debug-symbols "${symbol_map}")
        done < <(find "${symbol_maps_dir}" -name '*.bcsymbolmap' -print0)
    fi
}

verify_interface() {
    local forbidden='MxNumericsBackend|MxNumericsAccelerate|BLASKernels|LAPACKKernels|VDSPKernels|MLXBackend|ReferenceBackend|BufferRef|LinearAlgebraBackend'
    local interfaces
    interfaces="$(find "${XCFRAMEWORK_PATH}" -name '*.swiftinterface' -print)"

    if [[ -z "${interfaces}" ]]; then
        echo "No .swiftinterface files were found in ${XCFRAMEWORK_PATH}" >&2
        exit 5
    fi

    if find "${XCFRAMEWORK_PATH}" -name '*.swiftinterface' -print0 | xargs -0 grep -E "${forbidden}"; then
        echo "Public interface exposes internal implementation symbols." >&2
        exit 6
    fi
}

mkdir -p "${ARCHIVE_DIR}" "${PRODUCT_DIR}"

swift test

FRAMEWORK_ARGS=()
archive_framework "MxNumerics-iOS" "generic/platform=iOS"
archive_framework "MxNumerics-iOSSimulator" "generic/platform=iOS Simulator"
archive_framework "MxNumerics-macOS" "generic/platform=macOS"
archive_framework "MxNumerics-tvOS" "generic/platform=tvOS"
archive_framework "MxNumerics-tvOSSimulator" "generic/platform=tvOS Simulator"
archive_framework "MxNumerics-visionOS" "generic/platform=visionOS"
archive_framework "MxNumerics-visionOSSimulator" "generic/platform=visionOS Simulator"

rm -rf "${XCFRAMEWORK_PATH}" "${ZIP_PATH}"
xcodebuild -create-xcframework "${FRAMEWORK_ARGS[@]}" -output "${XCFRAMEWORK_PATH}"
verify_interface

(
    cd "${PRODUCT_DIR}"
    /usr/bin/zip -qry "$(basename "${ZIP_PATH}")" "$(basename "${XCFRAMEWORK_PATH}")"
)

CHECKSUM="$(swift package compute-checksum "${ZIP_PATH}")"
cat > "${MANIFEST_PATH}" <<MANIFEST
// swift-tools-version: 6.2

import PackageDescription

let package = Package(
    name: "MxNumericsBinary",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
        .tvOS(.v17),
    ],
    products: [
        .library(
            name: "MxNumerics",
            targets: [
                "MxNumerics",
                "MxNumericsDependencyShim",
            ]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics.git", from: "1.1.1"),
    ],
    targets: [
        .binaryTarget(
            name: "MxNumerics",
            url: "${ARTIFACT_BASE_URL}/MxNumerics-${VERSION}.xcframework.zip",
            checksum: "${CHECKSUM}"
        ),
        .target(
            name: "MxNumericsDependencyShim",
            dependencies: [
                .product(name: "ComplexModule", package: "swift-numerics"),
                .product(name: "RealModule", package: "swift-numerics"),
            ],
            path: "Sources/MxNumericsDependencyShim"
        ),
    ]
)
MANIFEST

cat > "${PRODUCT_DIR}/release.json" <<JSON
{
  "version": "${VERSION}",
  "artifact": "${ZIP_PATH}",
  "url": "${ARTIFACT_BASE_URL}/MxNumerics-${VERSION}.xcframework.zip",
  "checksum": "${CHECKSUM}",
  "binaryPackageManifest": "${MANIFEST_PATH}"
}
JSON

echo "Created ${ZIP_PATH}"
echo "Checksum ${CHECKSUM}"
