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
            url: "https://example.com/MxNumerics/releases/download/1.0.0/MxNumerics-1.0.0.xcframework.zip",
            checksum: "0000000000000000000000000000000000000000000000000000000000000000"
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
