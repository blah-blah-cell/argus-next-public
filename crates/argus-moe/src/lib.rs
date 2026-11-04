//! Small fail-closed mixture-of-experts execution core.

#![forbid(unsafe_code)]

use std::{collections::HashSet, fmt, num::NonZeroUsize};

#[derive(Clone, Debug, PartialEq)]
pub struct ExpertOutput {
    /// Calibrated attack probability in `[0, 1]`.
    pub attack_probability: f32,
}

pub trait Expert: Send + Sync {
    fn id(&self) -> &str;
    fn feature_schema(&self) -> (&str, u32);
    fn calibrated(&self) -> bool;
    /// Runs one admitted expert.
    ///
    /// # Errors
    /// Returns a sanitized backend error; the engine propagates it fail-closed.
    fn infer(&self, features: &[f32]) -> Result<ExpertOutput, String>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub expert_index: usize,
    pub weight: f32,
}

pub trait Router: Send + Sync {
    fn feature_schema(&self) -> (&str, u32);
    /// Selects a bounded set of experts using causal pre-expert features.
    ///
    /// # Errors
    /// Returns a sanitized routing error; the engine will not select a fallback route.
    fn route(&self, features: &[f32], expert_count: usize) -> Result<Vec<Route>, String>;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Contribution {
    pub expert_id: String,
    pub weight: f32,
    pub attack_probability: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MoeOutput {
    pub attack_probability: f32,
    pub contributions: Vec<Contribution>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MoeError {
    NoExperts,
    InvalidTopK,
    EmptyIdentifier,
    DuplicateExpert(String),
    SchemaMismatch,
    UncalibratedExpert(String),
    RouterFailure(String),
    InvalidRoute,
    ExpertFailure { expert: String, detail: String },
    InvalidExpertOutput(String),
}

impl fmt::Display for MoeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "MoE execution failed: {self:?}")
    }
}

impl std::error::Error for MoeError {}

pub struct MoeEngine {
    experts: Vec<Box<dyn Expert>>,
    router: Box<dyn Router>,
    top_k: NonZeroUsize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpertAdmissionState {
    Admitted,
    Unavailable,
    Rejected,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertAdmission {
    pub expert_id: String,
    pub state: ExpertAdmissionState,
    pub detail: String,
}

pub enum ExpertCandidate {
    Available(Box<dyn Expert>),
    Unavailable { expert_id: String, detail: String },
}

pub struct MoeRuntime {
    engine: Option<MoeEngine>,
    admissions: Vec<ExpertAdmission>,
}

impl MoeRuntime {
    /// Discovers experts independently. Missing or invalid optional experts are
    /// reported and quarantined instead of preventing the process from starting.
    /// The resulting engine remains strict and exists only when at least one
    /// calibrated, schema-compatible expert and a router are available.
    #[must_use]
    pub fn discover(
        candidates: Vec<ExpertCandidate>,
        router: Option<Box<dyn Router>>,
        requested_top_k: usize,
    ) -> Self {
        let mut admissions = Vec::with_capacity(candidates.len());
        let mut admitted = Vec::new();
        let mut identifiers = HashSet::new();
        let router_schema = router.as_ref().map(|router| router.feature_schema());

        for candidate in candidates {
            match candidate {
                ExpertCandidate::Unavailable { expert_id, detail } => {
                    admissions.push(ExpertAdmission {
                        expert_id,
                        state: ExpertAdmissionState::Unavailable,
                        detail,
                    });
                }
                ExpertCandidate::Available(expert) => {
                    let id = expert.id().to_owned();
                    let rejection = if id.is_empty() {
                        Some("empty expert identifier")
                    } else if !identifiers.insert(id.clone()) {
                        Some("duplicate expert identifier")
                    } else if !expert.calibrated() {
                        Some("expert output is not calibrated")
                    } else if router_schema.is_none() {
                        Some("router is unavailable")
                    } else if Some(expert.feature_schema()) != router_schema {
                        Some("expert/router feature schema mismatch")
                    } else {
                        None
                    };
                    if let Some(detail) = rejection {
                        admissions.push(ExpertAdmission {
                            expert_id: id,
                            state: ExpertAdmissionState::Rejected,
                            detail: detail.to_owned(),
                        });
                    } else {
                        admissions.push(ExpertAdmission {
                            expert_id: id,
                            state: ExpertAdmissionState::Admitted,
                            detail: "calibrated and schema-compatible".to_owned(),
                        });
                        admitted.push(expert);
                    }
                }
            }
        }

        let engine = router.and_then(|router| {
            let top_k = requested_top_k.min(admitted.len());
            MoeEngine::new(admitted, router, top_k).ok()
        });
        Self { engine, admissions }
    }

    #[must_use]
    pub const fn engine(&self) -> Option<&MoeEngine> {
        self.engine.as_ref()
    }

    #[must_use]
    pub fn admissions(&self) -> &[ExpertAdmission] {
        &self.admissions
    }

    #[must_use]
    pub fn admitted_count(&self) -> usize {
        self.admissions
            .iter()
            .filter(|entry| entry.state == ExpertAdmissionState::Admitted)
            .count()
    }
}

impl MoeEngine {
    /// Admits a complete calibrated expert set and one schema-compatible router.
    ///
    /// # Errors
    /// Returns a precise admission error. Partial or incompatible sets are never accepted.
    pub fn new(
        experts: Vec<Box<dyn Expert>>,
        router: Box<dyn Router>,
        top_k: usize,
    ) -> Result<Self, MoeError> {
        if experts.is_empty() {
            return Err(MoeError::NoExperts);
        }
        let top_k = NonZeroUsize::new(top_k).ok_or(MoeError::InvalidTopK)?;
        if top_k.get() > experts.len() {
            return Err(MoeError::InvalidTopK);
        }
        let router_schema = router.feature_schema();
        let mut identifiers = HashSet::new();
        for expert in &experts {
            if expert.id().is_empty() {
                return Err(MoeError::EmptyIdentifier);
            }
            if !identifiers.insert(expert.id().to_owned()) {
                return Err(MoeError::DuplicateExpert(expert.id().to_owned()));
            }
            if expert.feature_schema() != router_schema {
                return Err(MoeError::SchemaMismatch);
            }
            if !expert.calibrated() {
                return Err(MoeError::UncalibratedExpert(expert.id().to_owned()));
            }
        }
        Ok(Self {
            experts,
            router,
            top_k,
        })
    }

    #[must_use]
    pub fn expert_count(&self) -> usize {
        self.experts.len()
    }

    /// Routes once, runs only the selected experts, and returns an auditable mixture.
    ///
    /// # Errors
    /// Fails closed on router, route, expert, calibration, or numerical errors.
    pub fn infer(&self, features: &[f32]) -> Result<MoeOutput, MoeError> {
        if features.is_empty() || features.iter().any(|value| !value.is_finite()) {
            return Err(MoeError::InvalidRoute);
        }
        let routes = self
            .router
            .route(features, self.experts.len())
            .map_err(MoeError::RouterFailure)?;
        validate_routes(&routes, self.experts.len(), self.top_k.get())?;

        let total_weight: f32 = routes.iter().map(|route| route.weight).sum();
        let mut probability = 0.0_f32;
        let mut contributions = Vec::with_capacity(routes.len());
        for route in routes {
            let expert = &self.experts[route.expert_index];
            let output = expert
                .infer(features)
                .map_err(|detail| MoeError::ExpertFailure {
                    expert: expert.id().to_owned(),
                    detail,
                })?;
            if !output.attack_probability.is_finite()
                || !(0.0..=1.0).contains(&output.attack_probability)
            {
                return Err(MoeError::InvalidExpertOutput(expert.id().to_owned()));
            }
            let weight = route.weight / total_weight;
            probability += weight * output.attack_probability;
            contributions.push(Contribution {
                expert_id: expert.id().to_owned(),
                weight,
                attack_probability: output.attack_probability,
            });
        }
        Ok(MoeOutput {
            attack_probability: probability,
            contributions,
        })
    }
}

fn validate_routes(routes: &[Route], expert_count: usize, top_k: usize) -> Result<(), MoeError> {
    if routes.is_empty() || routes.len() > top_k {
        return Err(MoeError::InvalidRoute);
    }
    let mut selected = HashSet::new();
    let mut weight_sum = 0.0_f32;
    for route in routes {
        if route.expert_index >= expert_count
            || !route.weight.is_finite()
            || route.weight <= 0.0
            || !selected.insert(route.expert_index)
        {
            return Err(MoeError::InvalidRoute);
        }
        weight_sum += route.weight;
    }
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(MoeError::InvalidRoute);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedExpert {
        id: &'static str,
        probability: f32,
        calibrated: bool,
    }

    impl Expert for FixedExpert {
        fn id(&self) -> &str {
            self.id
        }
        fn feature_schema(&self) -> (&str, u32) {
            ("argus.causal-flow", 1)
        }
        fn calibrated(&self) -> bool {
            self.calibrated
        }
        fn infer(&self, _: &[f32]) -> Result<ExpertOutput, String> {
            Ok(ExpertOutput {
                attack_probability: self.probability,
            })
        }
    }

    struct FixedRouter(Vec<Route>);

    impl Router for FixedRouter {
        fn feature_schema(&self) -> (&str, u32) {
            ("argus.causal-flow", 1)
        }
        fn route(&self, _: &[f32], _: usize) -> Result<Vec<Route>, String> {
            Ok(self.0.clone())
        }
    }

    fn expert(id: &'static str, probability: f32) -> Box<dyn Expert> {
        Box::new(FixedExpert {
            id,
            probability,
            calibrated: true,
        })
    }

    #[test]
    fn routes_bounded_experts_and_reports_exact_contributions() {
        let engine = MoeEngine::new(
            vec![expert("timing", 0.2), expert("burst", 0.8)],
            Box::new(FixedRouter(vec![
                Route {
                    expert_index: 0,
                    weight: 1.0,
                },
                Route {
                    expert_index: 1,
                    weight: 3.0,
                },
            ])),
            2,
        )
        .expect("engine");
        let output = engine.infer(&[1.0, 2.0]).expect("inference");
        assert!((output.attack_probability - 0.65).abs() < f32::EPSILON);
        assert_eq!(output.contributions.len(), 2);
        assert!((output.contributions[1].weight - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn rejects_missing_uncalibrated_or_duplicate_experts() {
        assert!(matches!(
            MoeEngine::new(Vec::new(), Box::new(FixedRouter(Vec::new())), 1),
            Err(MoeError::NoExperts)
        ));
        let uncalibrated: Box<dyn Expert> = Box::new(FixedExpert {
            id: "raw",
            probability: 0.5,
            calibrated: false,
        });
        assert!(matches!(
            MoeEngine::new(vec![uncalibrated], Box::new(FixedRouter(Vec::new())), 1),
            Err(MoeError::UncalibratedExpert(_))
        ));
        assert!(matches!(
            MoeEngine::new(
                vec![expert("same", 0.1), expert("same", 0.2)],
                Box::new(FixedRouter(Vec::new())),
                1
            ),
            Err(MoeError::DuplicateExpert(_))
        ));
    }

    #[test]
    fn rejects_duplicate_out_of_range_and_nonfinite_routes() {
        for routes in [
            vec![Route {
                expert_index: 2,
                weight: 1.0,
            }],
            vec![Route {
                expert_index: 0,
                weight: f32::NAN,
            }],
            vec![
                Route {
                    expert_index: 0,
                    weight: 1.0,
                },
                Route {
                    expert_index: 0,
                    weight: 1.0,
                },
            ],
        ] {
            let engine = MoeEngine::new(
                vec![expert("one", 0.5), expert("two", 0.5)],
                Box::new(FixedRouter(routes)),
                2,
            )
            .expect("engine admission");
            assert_eq!(engine.infer(&[1.0]), Err(MoeError::InvalidRoute));
        }
    }

    #[test]
    fn discovery_quarantines_missing_and_invalid_experts_without_crashing() {
        let raw: Box<dyn Expert> = Box::new(FixedExpert {
            id: "raw-cnn",
            probability: 0.5,
            calibrated: false,
        });
        let runtime = MoeRuntime::discover(
            vec![
                ExpertCandidate::Unavailable {
                    expert_id: "missing-lstm".to_owned(),
                    detail: "artifact not found".to_owned(),
                },
                ExpertCandidate::Available(raw),
                ExpertCandidate::Available(expert("ready-cnn", 0.7)),
            ],
            Some(Box::new(FixedRouter(vec![Route {
                expert_index: 0,
                weight: 1.0,
            }]))),
            2,
        );
        assert_eq!(runtime.admitted_count(), 1);
        assert!(runtime.engine().is_some());
        let probability = runtime
            .engine()
            .expect("one valid expert")
            .infer(&[1.0])
            .expect("inference")
            .attack_probability;
        assert!((probability - 0.7).abs() < f32::EPSILON);
        assert_eq!(
            runtime.admissions()[0].state,
            ExpertAdmissionState::Unavailable
        );
        assert_eq!(
            runtime.admissions()[1].state,
            ExpertAdmissionState::Rejected
        );
    }

    #[test]
    fn discovery_with_no_usable_experts_keeps_runtime_available() {
        let runtime = MoeRuntime::discover(
            vec![ExpertCandidate::Unavailable {
                expert_id: "cnn".to_owned(),
                detail: "artifact absent".to_owned(),
            }],
            None,
            2,
        );
        assert!(runtime.engine().is_none());
        assert_eq!(runtime.admitted_count(), 0);
    }
}
