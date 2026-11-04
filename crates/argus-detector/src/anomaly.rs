//! Calibrated Statistical Anomaly Engine and Security Alert Evaluator.
//!
//! Provides mathematically sound, bounded rolling statistics (Welford's algorithm)
//! for tracking temporal forecast residuals, protocol flag anomalies, burst rates,
//! and causal quality excursions without external dependencies.

use argus_core::FlowInstanceId;

/// Severity classification for security and operational telemetry events.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum AnomalySeverity {
    /// Expected variations within statistical baseline.
    Nominal,
    /// Statistically elevated residual or abnormal pattern warranting investigation.
    Suspicious,
    /// Severe violation, attack signature, or extreme baseline departure.
    Critical,
}

impl AnomalySeverity {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Suspicious => "suspicious",
            Self::Critical => "critical",
        }
    }
}

/// Specific category of anomaly detected.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlertCategory {
    /// Neural temporal forecasting residual excursion (z-score >= threshold).
    ForecastResidual,
    /// Evasive or malformed TCP control flags (e.g. SYN+FIN, Xmas, Null scan).
    TcpEvasion,
    /// Abnormal traffic volume or packet rate spike.
    TrafficBurst,
    /// Heavy packet loss or frame truncation indicating network degradation or evasion.
    QualityDegradation,
    /// Application protocol anomaly or high-entropy clandestine exfiltration.
    ProtocolViolation,
}

impl AlertCategory {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ForecastResidual => "forecast_residual",
            Self::TcpEvasion => "tcp_evasion",
            Self::TrafficBurst => "traffic_burst",
            Self::QualityDegradation => "quality_degradation",
            Self::ProtocolViolation => "protocol_violation",
        }
    }
}

/// Structured anomaly alert event emitted by the detector.
#[derive(Clone, Debug, PartialEq)]
pub struct AnomalyRecord {
    pub timestamp_ns: u64,
    pub flow_id: Option<FlowInstanceId>,
    pub category: AlertCategory,
    pub severity: AnomalySeverity,
    pub score: f64,
    pub threshold: f64,
    pub summary: String,
    pub detail: String,
}

/// Numerically stable online running mean and variance estimator (Welford's algorithm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnlineStats {
    count: u64,
    mean: f64,
    m2: f64,
}

impl Default for OnlineStats {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineStats {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Observes a new value and updates running moments.
    ///
    /// The observation count is bounded by the analyzed window, so converting
    /// it to `f64` for the running mean cannot lose precision in practice.
    #[allow(clippy::cast_precision_loss)]
    pub fn update(&mut self, value: f64) {
        if !value.is_finite() {
            return;
        }
        self.count = self.count.saturating_add(1);
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    #[must_use]
    // Observation count is bounded by the analyzed window; `f64` conversion is exact in practice.
    #[allow(clippy::cast_precision_loss)]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Computes the Z-Score (standard deviations from mean) for an observed value.
    #[must_use]
    pub fn z_score(&self, value: f64) -> f64 {
        let std = self.std_dev();
        if std <= 1e-7 || self.count < 5 {
            0.0
        } else {
            (value - self.mean) / std
        }
    }
}

/// Configuration parameters for anomaly evaluation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnomalyConfig {
    pub min_warmup_samples: u64,
    pub suspicious_z_score: f64,
    pub critical_z_score: f64,
    pub max_packet_loss_allowed: u64,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            min_warmup_samples: 10,
            suspicious_z_score: 2.5,
            critical_z_score: 4.0,
            max_packet_loss_allowed: 5,
        }
    }
}

/// Stateful anomaly tracker maintaining rolling baselines and evaluating live excursions.
#[derive(Clone, Debug)]
pub struct AnomalyTracker {
    config: AnomalyConfig,
    forecast_stats: OnlineStats,
    packet_rate_stats: OnlineStats,
    byte_rate_stats: OnlineStats,
}

impl AnomalyTracker {
    #[must_use]
    pub fn new(config: AnomalyConfig) -> Self {
        Self {
            config,
            forecast_stats: OnlineStats::new(),
            packet_rate_stats: OnlineStats::new(),
            byte_rate_stats: OnlineStats::new(),
        }
    }

    /// Evaluates a temporal forecast error against rolling baseline distributions.
    pub fn evaluate_forecast_error(
        &mut self,
        timestamp_ns: u64,
        flow_id: Option<FlowInstanceId>,
        forecast_error: f64,
        model_name: &str,
    ) -> Option<AnomalyRecord> {
        let z = self.forecast_stats.z_score(forecast_error);
        self.forecast_stats.update(forecast_error);

        if self.forecast_stats.count() < self.config.min_warmup_samples {
            return None;
        }

        if z >= self.config.critical_z_score {
            Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::ForecastResidual,
                severity: AnomalySeverity::Critical,
                score: z,
                threshold: self.config.critical_z_score,
                summary: format!("High forecast residual diagnostic from {model_name}"),
                detail: format!(
                    "Forecast error {:.5} is {:.2}σ above baseline mean {:.5} (σ={:.5})",
                    forecast_error,
                    z,
                    self.forecast_stats.mean(),
                    self.forecast_stats.std_dev()
                ),
            })
        } else if z >= self.config.suspicious_z_score {
            Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::ForecastResidual,
                severity: AnomalySeverity::Suspicious,
                score: z,
                threshold: self.config.suspicious_z_score,
                summary: format!("Elevated forecast residual diagnostic from {model_name}"),
                detail: format!(
                    "Forecast error {:.5} is {:.2}σ above baseline mean {:.5}",
                    forecast_error,
                    z,
                    self.forecast_stats.mean()
                ),
            })
        } else {
            None
        }
    }

    /// Evaluates TCP control flags for evasion techniques (SYN+FIN, Null Scan, Xmas Tree).
    #[must_use]
    pub fn evaluate_tcp_flags(
        timestamp_ns: u64,
        flow_id: Option<FlowInstanceId>,
        tcp_flags: u8,
    ) -> Option<AnomalyRecord> {
        let syn = tcp_flags & 0x02 != 0;
        let fin = tcp_flags & 0x01 != 0;
        let rst = tcp_flags & 0x04 != 0;
        let psh = tcp_flags & 0x08 != 0;
        let urg = tcp_flags & 0x20 != 0;

        // SYN + FIN collision (classic scan / evasion indicator)
        if syn && fin {
            return Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::TcpEvasion,
                severity: AnomalySeverity::Critical,
                score: 1.0,
                threshold: 1.0,
                summary: "Evasive TCP SYN+FIN flag collision".into(),
                detail: format!(
                    "Malformed TCP flag combo: 0x{tcp_flags:02X} (SYN and FIN set simultaneously)"
                ),
            });
        }

        // Xmas Tree scan (FIN + PSH + URG)
        if fin && psh && urg {
            return Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::TcpEvasion,
                severity: AnomalySeverity::Critical,
                score: 1.0,
                threshold: 1.0,
                summary: "TCP Xmas Tree scan detected".into(),
                detail: format!(
                    "Probing pattern 0x{tcp_flags:02X} with FIN, PSH, and URG flags set"
                ),
            });
        }

        // NULL scan (No flags set on active frame)
        if tcp_flags == 0 {
            return Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::TcpEvasion,
                severity: AnomalySeverity::Suspicious,
                score: 0.8,
                threshold: 0.5,
                summary: "TCP NULL flag signature detected".into(),
                detail: "TCP frame observed with 0x00 control flags".into(),
            });
        }

        // SYN + RST collision
        if syn && rst {
            return Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::TcpEvasion,
                severity: AnomalySeverity::Suspicious,
                score: 0.75,
                threshold: 0.5,
                summary: "TCP SYN+RST anomaly detected".into(),
                detail: format!("Anomalous control combination 0x{tcp_flags:02X}"),
            });
        }

        None
    }

    /// Evaluates packet and wire byte throughput rate against rolling baseline.
    pub fn evaluate_traffic_rate(
        &mut self,
        timestamp_ns: u64,
        flow_id: Option<FlowInstanceId>,
        packets_per_sec: f64,
        bytes_per_sec: f64,
    ) -> Option<AnomalyRecord> {
        let p_z = self.packet_rate_stats.z_score(packets_per_sec);
        let b_z = self.byte_rate_stats.z_score(bytes_per_sec);
        self.packet_rate_stats.update(packets_per_sec);
        self.byte_rate_stats.update(bytes_per_sec);

        if self.packet_rate_stats.count() < self.config.min_warmup_samples {
            return None;
        }

        let max_z = p_z.max(b_z);
        if max_z >= self.config.critical_z_score {
            Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::TrafficBurst,
                severity: AnomalySeverity::Critical,
                score: max_z,
                threshold: self.config.critical_z_score,
                summary: "Volumetric traffic burst anomaly detected".into(),
                detail: format!(
                    "Throughput rate ({packets_per_sec:.0} pkts/s, {bytes_per_sec:.0} B/s) is {max_z:.2}σ above baseline"
                ),
            })
        } else if max_z >= self.config.suspicious_z_score {
            Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::TrafficBurst,
                severity: AnomalySeverity::Suspicious,
                score: max_z,
                threshold: self.config.suspicious_z_score,
                summary: "Elevated traffic rate burst detected".into(),
                detail: format!(
                    "Throughput rate ({packets_per_sec:.0} pkts/s, {bytes_per_sec:.0} B/s) is {max_z:.2}σ above baseline"
                ),
            })
        } else {
            None
        }
    }

    /// Evaluates flow quality metrics for anomalies (unusual loss or truncation).
    ///
    /// Packet counters are bounded by the analyzed window, so converting them
    /// to `f64` for the anomaly score cannot lose precision in practice.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn evaluate_flow_quality(
        &self,
        timestamp_ns: u64,
        flow_id: Option<FlowInstanceId>,
        lost_packets: u64,
        truncated_packets: u64,
    ) -> Option<AnomalyRecord> {
        if lost_packets > self.config.max_packet_loss_allowed {
            Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::QualityDegradation,
                severity: AnomalySeverity::Suspicious,
                score: lost_packets as f64,
                threshold: self.config.max_packet_loss_allowed as f64,
                summary: "High packet loss observed on flow".into(),
                detail: format!(
                    "{lost_packets} missing packets detected before sequence resumption"
                ),
            })
        } else if truncated_packets > 0 {
            Some(AnomalyRecord {
                timestamp_ns,
                flow_id,
                category: AlertCategory::QualityDegradation,
                severity: AnomalySeverity::Suspicious,
                score: truncated_packets as f64,
                threshold: 0.0,
                summary: "Frame truncation detected on flow".into(),
                detail: format!("{truncated_packets} truncated packet frames observed"),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn online_stats_calculates_exact_mean_and_variance() {
        let mut stats = OnlineStats::new();
        let samples = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0];
        for &s in &samples {
            stats.update(s);
        }
        assert_eq!(stats.count(), 8);
        assert!((stats.mean() - 18.0).abs() < 1e-6);
        assert!((stats.variance() - 27.428_571).abs() < 1e-4);
        assert!((stats.std_dev() - 5.237_229).abs() < 1e-4);
        // Z-score of 33.71 should be ~3.0
        assert!((stats.z_score(33.711_688) - 3.0).abs() < 1e-3);
    }

    #[test]
    fn detects_tcp_evasion_signatures() {
        // SYN + FIN
        let alert = AnomalyTracker::evaluate_tcp_flags(1000, None, 0x03).expect("SYN+FIN alert");
        assert_eq!(alert.severity, AnomalySeverity::Critical);
        assert_eq!(alert.category, AlertCategory::TcpEvasion);

        // Xmas (FIN+PSH+URG = 0x01 | 0x08 | 0x20 = 0x29)
        let alert = AnomalyTracker::evaluate_tcp_flags(1000, None, 0x29).expect("Xmas alert");
        assert_eq!(alert.severity, AnomalySeverity::Critical);

        // NULL scan (0x00)
        let alert = AnomalyTracker::evaluate_tcp_flags(1000, None, 0x00).expect("NULL scan alert");
        assert_eq!(alert.severity, AnomalySeverity::Suspicious);

        // Normal SYN-ACK (0x12) -> None
        assert!(AnomalyTracker::evaluate_tcp_flags(1000, None, 0x12).is_none());
        // Normal ACK (0x10) -> None
        assert!(AnomalyTracker::evaluate_tcp_flags(1000, None, 0x10).is_none());
    }

    #[test]
    fn detects_forecast_residual_spike_after_warmup() {
        let mut tracker = AnomalyTracker::new(AnomalyConfig {
            min_warmup_samples: 5,
            suspicious_z_score: 2.0,
            critical_z_score: 3.5,
            max_packet_loss_allowed: 5,
        });

        // Feed benign baseline errors ~0.04
        for _ in 0..10 {
            tracker.evaluate_forecast_error(100, None, 0.04, "cnn_local");
        }

        // Slight fluctuation
        assert!(
            tracker
                .evaluate_forecast_error(200, None, 0.045, "cnn_local")
                .is_none()
        );

        // Massive anomaly spike (error 0.50) -> Critical alert
        let alert = tracker
            .evaluate_forecast_error(300, None, 0.50, "cnn_local")
            .expect("critical residual alert");
        assert_eq!(alert.severity, AnomalySeverity::Critical);
        assert_eq!(alert.category, AlertCategory::ForecastResidual);
    }

    #[test]
    fn detects_traffic_rate_anomaly() {
        let mut tracker = AnomalyTracker::new(AnomalyConfig {
            min_warmup_samples: 5,
            suspicious_z_score: 2.0,
            critical_z_score: 3.5,
            max_packet_loss_allowed: 5,
        });

        // Warmup with ~100 pkts/s, ~50KB/s
        for _ in 0..10 {
            tracker.evaluate_traffic_rate(100, None, 100.0, 50_000.0);
        }

        // Normal fluctuation
        assert!(
            tracker
                .evaluate_traffic_rate(200, None, 120.0, 55_000.0)
                .is_none()
        );

        // Massive 10,000 pkts/s flood
        let alert = tracker
            .evaluate_traffic_rate(300, None, 10_000.0, 5_000_000.0)
            .expect("traffic burst alert");
        assert_eq!(alert.severity, AnomalySeverity::Critical);
        assert_eq!(alert.category, AlertCategory::TrafficBurst);
    }
}
