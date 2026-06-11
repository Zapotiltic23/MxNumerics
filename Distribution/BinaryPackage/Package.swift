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
        .library(name: "MxNumerics", targets: ["MxNumerics"]),
    ],
    targets: [
        .binaryTarget(
            name: "MxNumerics",
            url: "https://example.com/MxNumerics/releases/download/1.0.0/MxNumerics-1.0.0.xcframework.zip",
            checksum: "0000000000000000000000000000000000000000000000000000000000000000"
        ),
    ]
)
