//
//  StorageResidency.swift
//  MxNumerics
//
//  Created by Alexandro Sanchez on 6/9/26.
//

/// The location of the authoritative matrix storage.
///
/// The core package is backend-neutral, so this enum intentionally does not
/// store backend-specific objects such as MLX arrays. It records residency state
/// so backend bridges can avoid accidental host-device ping-pong once device
/// storage is introduced.
public enum StorageResidency: Equatable, Sendable {
    /// The host buffer is authoritative.
    case host

    /// A device backend buffer is authoritative.
    case device

    /// Host and device buffers are both current.
    case both
}
