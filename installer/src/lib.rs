//! Self-contained, dependency-free setup payload support for Argus Next.
//!
//! The setup executable carries its files appended to the stub binary. Every
//! entry records its own SHA-256, so the installer can refuse a truncated or
//! tampered download before it writes anything to disk.

pub mod container;
pub mod sha256;
