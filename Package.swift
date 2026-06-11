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
            name: "MxNumerics",
            dependencies: [
                .product(name: "ComplexModule", package: "swift-numerics"),
                .product(name: "RealModule", package: "swift-numerics"),
            ],
            path: "Sources",
            cSettings: accelerateCSettings,
            swiftSettings: swift6Settings,
            linkerSettings: [.linkedFramework("Accelerate")]
        ),
        .testTarget(
            name: "MxNumericsTests",
            dependencies: [
                "MxNumerics",
                .product(name: "ComplexModule", package: "swift-numerics"),
            ],
            path: "Tests",
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

let accelerateCSettings: [CSetting] = [
    .define("ACCELERATE_NEW_LAPACK"),
]
