//! Shannon entropy and statistical flow dynamics calculators.
//!
//! Provides deterministic calculation of byte entropy, packet jitter, and throughput rates
//! for traffic profiling and encrypted / tunneling payload detection.

/// Computes Shannon entropy of an arbitrary byte buffer in bits per byte [0.0, 8.0].
///
/// An entropy near 8.0 indicates high randomness (e.g. TLS, encrypted payloads, compressed files).
/// Low entropy (< 4.5) indicates structured ASCII, repeated headers, or uncompressed plaintext.
#[must_use]
pub fn calculate_byte_entropy(bytes: &[u8]) -> f64 {
    if bytes.is_empty() {
        return 0.0;
    }

    let mut frequency = [0u64; 256];
    for &byte in bytes {
        frequency[byte as usize] += 1;
    }

    calculate_entropy_from_counts(&frequency, bytes.len() as u64)
}

/// Computes Shannon entropy from pre-aggregated byte frequency counts.
///
/// Byte counts are bounded by the analyzed buffer size, so converting them to
/// `f64` for the Shannon ratio cannot lose precision in practice.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn calculate_entropy_from_counts(counts: &[u64; 256], total: u64) -> f64 {
    if total == 0 {
        return 0.0;
    }

    let total_f = total as f64;
    let mut entropy = 0.0;

    for &count in counts {
        if count > 0 {
            let p = count as f64 / total_f;
            entropy -= p * p.log2();
        }
    }

    entropy.clamp(0.0, 8.0)
}

/// Computes packet inter-arrival jitter (mean absolute deviation of consecutive inter-arrival intervals).
///
/// Interval counts and accumulated nanoseconds are bounded by the analyzed
/// window, so converting them to `f64` cannot lose precision in practice.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn calculate_jitter_ms(interarrival_intervals_ns: &[u64]) -> f64 {
    if interarrival_intervals_ns.len() < 2 {
        return 0.0;
    }

    let mut total_diff_ns = 0u64;
    for i in 1..interarrival_intervals_ns.len() {
        let diff = interarrival_intervals_ns[i].abs_diff(interarrival_intervals_ns[i - 1]);
        total_diff_ns = total_diff_ns.saturating_add(diff);
    }

    let count = (interarrival_intervals_ns.len() - 1) as f64;
    (total_diff_ns as f64 / count) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_distribution_yields_maximum_entropy() {
        // All 256 byte values equally represented -> H = 8.0 bits
        let mut uniform = Vec::with_capacity(256 * 10);
        for _ in 0..10 {
            for byte in 0..=255u8 {
                uniform.push(byte);
            }
        }

        let entropy = calculate_byte_entropy(&uniform);
        assert!((entropy - 8.0).abs() < 1e-6);
    }

    #[test]
    fn constant_buffer_yields_zero_entropy() {
        let zeros = [0u8; 1000];
        let entropy = calculate_byte_entropy(&zeros);
        assert!((entropy - 0.0).abs() < 1e-6);
    }

    #[test]
    fn calculates_jitter_correctly() {
        let iats = [1_000_000, 2_000_000, 3_000_000]; // 1ms step diffs
        let jitter = calculate_jitter_ms(&iats);
        assert!((jitter - 1.0).abs() < 1e-6);
    }
}
