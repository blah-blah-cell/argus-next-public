pub mod anomaly;
pub use anomaly::{
    AlertCategory, AnomalyConfig, AnomalyRecord, AnomalySeverity, AnomalyTracker, OnlineStats,
};

use argus_core::FlowInstanceId;
use argus_features::{FEATURE_SCHEMA_ID, FEATURE_SCHEMA_VERSION, FeatureIndex, FeatureVector};

pub const BASELINE_ID: &str = "argus.deterministic-threshold-baseline";
pub const BASELINE_VERSION: u32 = 1;
pub const THRESHOLD_CONFIG_ID: &str = "argus.deterministic-thresholds";
pub const THRESHOLD_CONFIG_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BaselineConfig {
    pub readiness_packets: u64,
    pub maximum_packet_bytes: u64,
    pub maximum_loss_before: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlgorithmKind {
    DeterministicThresholdRules,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdConfigIdentity {
    pub config_id: &'static str,
    pub config_version: u32,
    pub readiness_packets: u64,
    pub maximum_packet_bytes: u64,
    pub maximum_loss_before: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FeatureSchemaIdentity {
    pub schema_id: &'static str,
    pub schema_version: u32,
}

/// Facts that identify the detector implementation and its exact threshold configuration.
///
/// Artifact fields are optional because this baseline is implemented as deterministic code and
/// neither loads nor claims a trained artifact.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetectorProvenance {
    pub detector_id: &'static str,
    pub detector_version: u32,
    pub algorithm_kind: AlgorithmKind,
    pub threshold_config: ThresholdConfigIdentity,
    pub trained_model: bool,
    pub artifact_id: Option<&'static str>,
    pub artifact_checksum: Option<&'static str>,
    pub feature_schema: FeatureSchemaIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Readiness {
    WarmingUp {
        observed_packets: u64,
        required_packets: u64,
    },
    Ready,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationHealth {
    Nominal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScoreKind {
    /// A unitless detector-specific score, not a probability or calibrated risk value.
    DetectorSpecific,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectorScore {
    pub kind: ScoreKind,
    pub value: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationReason {
    InsufficientPackets,
    NoThresholdExceeded,
    ThresholdExceeded,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AlertDecision {
    SuppressedNotReady,
    DoNotAlert,
    Alert,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Comparison {
    GreaterThan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Evidence {
    pub feature: &'static str,
    pub observed: u64,
    pub comparison: Comparison,
    pub threshold: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Alert {
    pub flow: FlowInstanceId,
    pub monotonic_timestamp_ns: u64,
    pub provenance: DetectorProvenance,
    pub health: EvaluationHealth,
    pub score: Option<DetectorScore>,
    pub reason: EvaluationReason,
    pub evidence: Vec<Evidence>,
}

/// Complete result of a successful evaluation. Each concern is explicit rather than inferred
/// from the presence or absence of an alert object.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    pub readiness: Readiness,
    pub health: EvaluationHealth,
    pub score: Option<DetectorScore>,
    pub reason: EvaluationReason,
    pub evidence: Vec<Evidence>,
    pub alert_decision: AlertDecision,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationError {
    IncompatibleFeatureSchema,
}

/// Fixed-rule baseline useful for validating feature-to-alert plumbing.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ThresholdBaseline {
    config: BaselineConfig,
}

impl ThresholdBaseline {
    #[must_use]
    pub const fn new(config: BaselineConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub const fn provenance(&self) -> DetectorProvenance {
        DetectorProvenance {
            detector_id: BASELINE_ID,
            detector_version: BASELINE_VERSION,
            algorithm_kind: AlgorithmKind::DeterministicThresholdRules,
            threshold_config: ThresholdConfigIdentity {
                config_id: THRESHOLD_CONFIG_ID,
                config_version: THRESHOLD_CONFIG_VERSION,
                readiness_packets: self.config.readiness_packets,
                maximum_packet_bytes: self.config.maximum_packet_bytes,
                maximum_loss_before: self.config.maximum_loss_before,
            },
            trained_model: false,
            artifact_id: None,
            artifact_checksum: None,
            feature_schema: FeatureSchemaIdentity {
                schema_id: FEATURE_SCHEMA_ID,
                schema_version: FEATURE_SCHEMA_VERSION,
            },
        }
    }

    /// Evaluates fixed thresholds once the configured packet warmup is complete.
    ///
    /// This baseline intentionally emits no score: its output is an exact rule decision with
    /// threshold evidence, not a probability or calibrated risk estimate.
    ///
    /// # Errors
    /// Rejects feature vectors from any other schema ID or version.
    pub fn evaluate(&self, features: &FeatureVector) -> Result<Evaluation, EvaluationError> {
        if features.schema_id != FEATURE_SCHEMA_ID
            || features.schema_version != FEATURE_SCHEMA_VERSION
        {
            return Err(EvaluationError::IncompatibleFeatureSchema);
        }
        let observed_packets = features.value(FeatureIndex::FlowPacketCount);
        if observed_packets < self.config.readiness_packets {
            return Ok(Evaluation {
                readiness: Readiness::WarmingUp {
                    observed_packets,
                    required_packets: self.config.readiness_packets,
                },
                health: EvaluationHealth::Nominal,
                score: None,
                reason: EvaluationReason::InsufficientPackets,
                evidence: Vec::new(),
                alert_decision: AlertDecision::SuppressedNotReady,
            });
        }

        let mut evidence = Vec::with_capacity(2);
        let packet_bytes = features.value(FeatureIndex::PacketLengthBytes);
        if packet_bytes > self.config.maximum_packet_bytes {
            evidence.push(Evidence {
                feature: "packet_length_bytes",
                observed: packet_bytes,
                comparison: Comparison::GreaterThan,
                threshold: self.config.maximum_packet_bytes,
            });
        }
        let loss = features.value(FeatureIndex::PacketsLostBefore);
        if loss > self.config.maximum_loss_before {
            evidence.push(Evidence {
                feature: "packets_lost_before",
                observed: loss,
                comparison: Comparison::GreaterThan,
                threshold: self.config.maximum_loss_before,
            });
        }
        let (reason, alert_decision) = if evidence.is_empty() {
            (
                EvaluationReason::NoThresholdExceeded,
                AlertDecision::DoNotAlert,
            )
        } else {
            (EvaluationReason::ThresholdExceeded, AlertDecision::Alert)
        };
        Ok(Evaluation {
            readiness: Readiness::Ready,
            health: EvaluationHealth::Nominal,
            score: None,
            reason,
            evidence,
            alert_decision,
        })
    }

    #[must_use]
    pub fn alert(&self, features: &FeatureVector, evaluation: &Evaluation) -> Option<Alert> {
        (evaluation.alert_decision == AlertDecision::Alert).then(|| Alert {
            flow: features.flow.clone(),
            monotonic_timestamp_ns: features.monotonic_timestamp_ns,
            provenance: self.provenance(),
            health: evaluation.health,
            score: evaluation.score,
            reason: evaluation.reason,
            evidence: evaluation.evidence.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argus_core::{Endpoint, FlowKey, TransportProtocol};
    use argus_features::PacketProtocolMetadata;
    use std::net::{IpAddr, Ipv4Addr};

    fn vector(packet_count: u64, packet_bytes: u64, loss: u64) -> FeatureVector {
        let mut values = [0; 12];
        values[FeatureIndex::FlowPacketCount as usize] = packet_count;
        values[FeatureIndex::PacketLengthBytes as usize] = packet_bytes;
        values[FeatureIndex::PacketsLostBefore as usize] = loss;
        FeatureVector {
            schema_id: FEATURE_SCHEMA_ID,
            schema_version: FEATURE_SCHEMA_VERSION,
            flow: FlowInstanceId {
                key: FlowKey {
                    initiator: Endpoint {
                        address: IpAddr::V4(Ipv4Addr::LOCALHOST),
                        port: 1234,
                    },
                    responder: Endpoint {
                        address: IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)),
                        port: 443,
                    },
                    protocol: TransportProtocol::Tcp,
                },
                generation: 0,
            },
            monotonic_timestamp_ns: 99,
            protocol: PacketProtocolMetadata::default(),
            values,
        }
    }

    fn detector() -> ThresholdBaseline {
        ThresholdBaseline::new(BaselineConfig {
            readiness_packets: 3,
            maximum_packet_bytes: 1_500,
            maximum_loss_before: 0,
        })
    }

    #[test]
    fn provenance_identifies_algorithm_thresholds_and_absent_artifact() {
        let provenance = detector().provenance();
        assert_eq!(
            provenance.algorithm_kind,
            AlgorithmKind::DeterministicThresholdRules
        );
        assert_eq!(provenance.threshold_config.maximum_packet_bytes, 1_500);
        assert!(!provenance.trained_model);
        assert_eq!(provenance.artifact_id, None);
        assert_eq!(provenance.artifact_checksum, None);
        assert_eq!(provenance.feature_schema.schema_version, 1);
    }

    #[test]
    fn provenance_changes_with_exact_threshold_configuration() {
        let mut changed = detector();
        changed.config.maximum_packet_bytes = 900;
        assert_ne!(detector().provenance(), changed.provenance());
    }

    #[test]
    fn warmup_has_explicit_non_alert_result() {
        let evaluation = detector()
            .evaluate(&vector(2, 9_000, 4))
            .expect("compatible");
        assert_eq!(evaluation.health, EvaluationHealth::Nominal);
        assert_eq!(evaluation.score, None);
        assert_eq!(evaluation.reason, EvaluationReason::InsufficientPackets);
        assert_eq!(evaluation.evidence, Vec::new());
        assert_eq!(evaluation.alert_decision, AlertDecision::SuppressedNotReady);
        assert_eq!(detector().alert(&vector(2, 9_000, 4), &evaluation), None);
    }

    #[test]
    fn ready_alert_carries_exact_threshold_evidence() {
        let features = vector(3, 1_501, 2);
        let evaluation = detector().evaluate(&features).expect("compatible");
        assert_eq!(evaluation.readiness, Readiness::Ready);
        assert_eq!(evaluation.reason, EvaluationReason::ThresholdExceeded);
        assert_eq!(evaluation.alert_decision, AlertDecision::Alert);
        assert_eq!(evaluation.score, None);
        let alert = detector().alert(&features, &evaluation).expect("alert");
        assert_eq!(alert.provenance, detector().provenance());
        assert_eq!(alert.reason, EvaluationReason::ThresholdExceeded);
        assert_eq!(
            alert.evidence,
            vec![
                Evidence {
                    feature: "packet_length_bytes",
                    observed: 1_501,
                    comparison: Comparison::GreaterThan,
                    threshold: 1_500
                },
                Evidence {
                    feature: "packets_lost_before",
                    observed: 2,
                    comparison: Comparison::GreaterThan,
                    threshold: 0
                },
            ]
        );
    }

    #[test]
    fn boundary_values_have_explicit_no_alert_decision() {
        let evaluation = detector()
            .evaluate(&vector(3, 1_500, 0))
            .expect("compatible");
        assert_eq!(evaluation.readiness, Readiness::Ready);
        assert_eq!(evaluation.health, EvaluationHealth::Nominal);
        assert_eq!(evaluation.reason, EvaluationReason::NoThresholdExceeded);
        assert_eq!(evaluation.evidence, Vec::new());
        assert_eq!(evaluation.alert_decision, AlertDecision::DoNotAlert);
    }

    #[test]
    fn incompatible_schema_is_rejected() {
        let mut features = vector(3, 100, 0);
        features.schema_version += 1;
        assert_eq!(
            detector().evaluate(&features),
            Err(EvaluationError::IncompatibleFeatureSchema)
        );
    }
}
