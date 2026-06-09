// swift-tools-version: 6.2

//
//  Package.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

import PackageDescription

let package = Package(
    name: "MxNumerics",
    platforms: [
        .macOS(.v14),
        .iOS(.v17),
        .visionOS(.v1),
        .tvOS(.v17),
    ],
    products: [
        .library(name: "MxNumerics", targets: ["MxNumerics"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics.git", from: "1.1.1"),
    ],
    targets: [
        .target(
            name: "MxNumericsCore",
            dependencies: [
                .product(name: "ComplexModule", package: "swift-numerics"),
                .product(name: "RealModule", package: "swift-numerics"),
            ],
            swiftSettings: swift6Settings
        ),
        .target(
            name: "MxNumericsBackend",
            dependencies: ["MxNumericsCore"],
            swiftSettings: swift6Settings
        ),
        .target(
            name: "MxNumericsAccelerate",
            dependencies: ["MxNumericsCore", "MxNumericsBackend"],
            swiftSettings: swift6Settings,
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .target(
            name: "MxNumericsMLX",
            dependencies: ["MxNumericsCore", "MxNumericsBackend"],
            swiftSettings: swift6Settings
        ),
        .target(
            name: "MxNumerics",
            dependencies: [
                "MxNumericsCore",
                "MxNumericsBackend",
                "MxNumericsAccelerate",
                "MxNumericsMLX",
                .product(name: "ComplexModule", package: "swift-numerics"),
            ],
            swiftSettings: swift6Settings
        ),
        .testTarget(
            name: "MxNumericsCoreTests",
            dependencies: ["MxNumericsCore"],
            swiftSettings: swift6Settings
        ),
        .testTarget(
            name: "MxNumericsBackendTests",
            dependencies: ["MxNumericsBackend"],
            swiftSettings: swift6Settings
        ),
        .testTarget(
            name: "MxNumericsTests",
            dependencies: ["MxNumerics"],
            swiftSettings: swift6Settings
        ),
    ],
    swiftLanguageModes: [.v6]
)

let swift6Settings: [SwiftSetting] = [
    .enableUpcomingFeature("ExistentialAny"),
    .enableUpcomingFeature("InternalImportsByDefault"),
    .enableUpcomingFeature("MemberImportVisibility"),
    .enableUpcomingFeature("InferSendableFromCaptures"),
    .enableExperimentalFeature("StrictConcurrency"),
]
