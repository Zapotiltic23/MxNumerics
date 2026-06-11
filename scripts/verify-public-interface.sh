#!/usr/bin/env bash
set -euo pipefail

XCFRAMEWORK_PATH="${1:?usage: scripts/verify-public-interface.sh path/to/MxNumerics.xcframework}"
FORBIDDEN='MxNumericsBackend|MxNumericsAccelerate|BLASKernels|LAPACKKernels|VDSPKernels|MLXBackend|ReferenceBackend|BufferRef|LinearAlgebraBackend'
interfaces="$(find "${XCFRAMEWORK_PATH}" -name '*.swiftinterface' -print)"

if [[ -z "${interfaces}" ]]; then
    echo "No .swiftinterface files found under ${XCFRAMEWORK_PATH}" >&2
    exit 2
fi

if grep -E "${FORBIDDEN}" ${interfaces}; then
    echo "Public interface exposes internal implementation symbols." >&2
    exit 3
fi

echo "Public interface check passed."
