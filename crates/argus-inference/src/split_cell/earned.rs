//! Research-only size correction on top of the frozen Split Cell proposal.

use super::{InvalidFeatures, SplitCell};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarnedForecast {
    pub frozen_log_bytes: f64,
    pub corrected_log_bytes: f64,
    pub log_delay: f64,
}

/// Each generation starts with zero correction weight. No state is shared
/// across generations and this wrapper does not change the frozen cell.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EarnedCorrection {
    proposal: SplitCell,
    weight: f64,
    previous_level: f64,
    previous_delta: f64,
}

impl EarnedCorrection {
    /// Score the correction made before this arrival, then forecast the next.
    ///
    /// # Errors
    /// The frozen proposal rejects invalid conditioned inputs without changing
    /// its state. Rejected arrivals also leave the correction state untouched.
    pub fn observe_conditioned(
        &mut self,
        log_bytes: f64,
        log_delay: f64,
    ) -> Result<EarnedForecast, InvalidFeatures> {
        let forecast = self.proposal.observe_conditioned(log_bytes, log_delay)?;

        // A cold or abstaining proposal (delta == 0) supplies no gradient.
        if self.previous_delta != 0. {
            let previous_prediction = self.previous_level + self.weight * self.previous_delta;
            let error = (previous_prediction - log_bytes).clamp(-1., 1.);
            self.weight = (self.weight - 0.125 * error * self.previous_delta).clamp(0., 1.);
        }

        let level = forecast.smooth_log_bytes;
        let delta = (forecast.log_bytes - level).clamp(-1., 1.);
        self.previous_level = level;
        self.previous_delta = delta;
        Ok(EarnedForecast {
            frozen_log_bytes: forecast.log_bytes,
            corrected_log_bytes: level + self.weight * delta,
            log_delay: forecast.log_delay,
        })
    }

    /// Condition frame size and same-generation gap, then update and forecast.
    /// A cold lane ignores the supplied gap, matching `SplitCell::observe_packet`.
    #[must_use]
    pub fn observe_packet(&mut self, frame_bytes: u32, gap_ns: u64) -> EarnedForecast {
        let log_bytes = f64::from(frame_bytes).ln_1p();
        let forecast = self
            .proposal
            .observe_packet_with_log_bytes(log_bytes, gap_ns);

        if self.previous_delta != 0. {
            let previous_prediction = self.previous_level + self.weight * self.previous_delta;
            let error = (previous_prediction - log_bytes).clamp(-1., 1.);
            self.weight = (self.weight - 0.125 * error * self.previous_delta).clamp(0., 1.);
        }

        let level = forecast.smooth_log_bytes;
        let delta = (forecast.log_bytes - level).clamp(-1., 1.);
        self.previous_level = level;
        self.previous_delta = delta;
        EarnedForecast {
            frozen_log_bytes: forecast.log_bytes,
            corrected_log_bytes: level + self.weight * delta,
            log_delay: forecast.log_delay,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_only_from_the_previous_forecast_and_matches_recurrence() {
        let mut cell = EarnedCorrection::default();
        let mut frozen = SplitCell::default();
        let mut history: Option<(f64, f64)> = None;
        let mut weight = 0.;
        let mut changed = 0;
        let mut clipped = false;
        let mut reached_ceiling = false;
        let mut reached_floor_after_learning = false;
        for i in 0..1024 {
            let x = match i {
                // Break the alternation immediately after the first time it
                // earns weight, forcing a counterexample back to zero.
                5 => 0.,
                0..128 => f64::from(i % 2) * 8.,
                _ => f64::from((i * 1111 + i / 53) % 37),
            };
            let y = if i == 0 { 0. } else { f64::from(i % 13) / 4. };
            if let Some((level, delta)) = history {
                let residual: f64 = level + weight * delta - x;
                let next_weight = (weight - residual.clamp(-1., 1.) * delta / 8.).clamp(0., 1.);
                reached_floor_after_learning |= weight > 0. && next_weight == 0.;
                weight = next_weight;
            }
            let split = frozen.observe_conditioned(x, y).unwrap();
            let level = split.smooth_log_bytes;
            let raw_delta = split.log_bytes - level;
            let delta = raw_delta.clamp(-1., 1.);
            history = Some((level, delta));

            let actual = cell.observe_conditioned(x, y).unwrap();
            assert_eq!(actual.frozen_log_bytes.to_bits(), split.log_bytes.to_bits());
            assert_eq!(actual.log_delay.to_bits(), split.log_delay.to_bits());
            assert_eq!(
                actual.corrected_log_bytes.to_bits(),
                (level + weight * delta).to_bits()
            );
            assert_eq!(cell.weight.to_bits(), weight.to_bits());
            assert!((0. ..=1.).contains(&cell.weight));
            assert!(((level - 1.)..=(level + 1.)).contains(&actual.corrected_log_bytes));
            changed += usize::from(actual.corrected_log_bytes.to_bits() != level.to_bits());
            clipped |= raw_delta.abs() > 1.;
            reached_ceiling |= cell.weight.to_bits() == 1_f64.to_bits();
            if i < 4 {
                assert_eq!(cell.weight.to_bits(), 0_f64.to_bits());
            }
        }
        assert!(changed > 100);
        assert!(clipped, "fixture must exercise proposal clipping");
        assert!(reached_ceiling, "fixture must exercise weight ceiling");
        assert!(
            reached_floor_after_learning,
            "fixture must exercise weight floor"
        );
    }

    #[test]
    fn invalid_arrivals_are_atomic_and_reset_clears_earned_weight() {
        let mut cell = EarnedCorrection::default();
        assert_eq!(cell.observe_conditioned(4., 1.), Err(InvalidFeatures));
        assert_eq!(cell, EarnedCorrection::default());
        for i in 0..32 {
            let _ = cell.observe_conditioned(f64::from(i % 2) * 8., 0.).unwrap();
        }
        assert!(cell.weight > 0.);
        let before = cell;
        for invalid in [(f64::NAN, 0.), (0., -1.), (0., 5.)] {
            assert_eq!(
                cell.observe_conditioned(invalid.0, invalid.1),
                Err(InvalidFeatures)
            );
            assert_eq!(cell, before);
        }
        cell = EarnedCorrection::default();
        assert_eq!(cell.observe_conditioned(4., 1.), Err(InvalidFeatures));
        assert_eq!(
            cell.observe_conditioned(4., 0.).unwrap(),
            EarnedForecast {
                frozen_log_bytes: 4.,
                corrected_log_bytes: 4.,
                log_delay: 0.,
            }
        );
        assert_eq!(cell.weight.to_bits(), 0_f64.to_bits());
    }

    #[test]
    fn observe_packet_matches_observe_conditioned_parity() {
        let mut pkt_cell = EarnedCorrection::default();
        let mut cond_cell = EarnedCorrection::default();
        let mut rng = 91_u32;
        for i in 0..512_u32 {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let frame_bytes = 64 + (rng % 1437);
            let gap_u32 = rng % 50_000_000;
            let gap_ns = u64::from(gap_u32);
            let log_bytes = f64::from(frame_bytes).ln_1p();
            let log_delay = if i == 0 {
                0.
            } else {
                (f64::from(gap_u32) / 1_000_000_000.).ln_1p()
            };
            let a = pkt_cell.observe_packet(frame_bytes, gap_ns);
            let b = cond_cell.observe_conditioned(log_bytes, log_delay).unwrap();
            assert_eq!(a.frozen_log_bytes.to_bits(), b.frozen_log_bytes.to_bits());
            assert_eq!(
                a.corrected_log_bytes.to_bits(),
                b.corrected_log_bytes.to_bits()
            );
            assert_eq!(a.log_delay.to_bits(), b.log_delay.to_bits());
        }
    }
}
