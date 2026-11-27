//! Frozen AHEM Split Cell v1. The capture host owns flow identity, ordering
//! and capacity. Store one cell directly in each bounded generation lane.

use std::fmt;

pub mod earned;
pub mod ewma;

const MAX_DELAY_NS: u64 = 60_000_000_000;
const MAX_LOG_DELAY: f64 = 4.110_873_864_173_311;

/// The three size experts, in deterministic tie-break order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SizeExpert {
    Level,
    Last,
    Lag,
}

/// A forecast in conditioned log space, for the next same-generation packet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Forecast {
    pub log_bytes: f64,
    pub log_delay: f64,
    pub size_expert: SizeExpert,
    pub smooth_log_bytes: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidFeatures;

impl fmt::Display for InvalidFeatures {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid Split Cell features")
    }
}

impl std::error::Error for InvalidFeatures {}

/// An empty cell is cold. Replace it with the default at generation changes,
/// lane reuse, or record/capture reset. This object never retains packet data.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SplitCell {
    level: f64,
    last: f64,
    lag: f64,
    delay: f64,
    risk: [f64; 3],
    count: u8,
}

impl SplitCell {
    /// Produce the next forecast from already-conditioned input.
    ///
    /// # Errors
    /// Rejects non-finite or negative features, delay above the clipping
    /// bound, or nonzero delay at cold start. Leaves the previous state intact.
    pub fn observe_conditioned(
        &mut self,
        log_bytes: f64,
        log_delay: f64,
    ) -> Result<Forecast, InvalidFeatures> {
        if !log_bytes.is_finite()
            || log_bytes < 0.
            || !(0. ..=MAX_LOG_DELAY).contains(&log_delay)
            || (self.count == 0 && log_delay != 0.)
        {
            return Err(InvalidFeatures);
        }
        Ok(self.update(log_bytes, log_delay))
    }

    /// Condition a validated frame size and same-generation gap, then update
    /// and forecast in one call. The caller has already checked ordering; an
    /// evicted or newly allocated lane ignores the supplied gap and starts cold.
    #[must_use]
    pub fn observe_packet(&mut self, frame_bytes: u32, gap_ns: u64) -> Forecast {
        self.observe_packet_with_log_bytes(f64::from(frame_bytes).ln_1p(), gap_ns)
    }

    pub(crate) fn observe_packet_with_log_bytes(
        &mut self,
        log_bytes: f64,
        gap_ns: u64,
    ) -> Forecast {
        // The gap is clipped below 2^53, so this conversion is exact.
        #[allow(clippy::cast_precision_loss)]
        let seconds = gap_ns.min(MAX_DELAY_NS) as f64 / 1_000_000_000.;
        let delay = if self.count == 0 { 0. } else { seconds.ln_1p() };
        self.update(log_bytes, delay)
    }

    fn update(&mut self, x: f64, y: f64) -> Forecast {
        if self.count == 0 {
            self.level = x;
            self.last = x;
            self.lag = x;
            self.delay = y;
        } else {
            for (risk, previous) in self.risk.iter_mut().zip([self.level, self.last, self.lag]) {
                *risk += ((x - previous).abs() - *risk) / 8.;
            }
            self.level += (x - self.level) / 4.;
            self.delay += (y - self.delay) / 4.;
            self.lag = self.last;
            self.last = x;
        }
        self.count = (self.count + 1).min(4);
        let expert = if self.count < 4 {
            SizeExpert::Level
        } else {
            let [level, last, lag] = self.risk;
            if last < level && last <= lag {
                SizeExpert::Last
            } else if lag < level && lag < last {
                SizeExpert::Lag
            } else {
                SizeExpert::Level
            }
        };
        Forecast {
            log_bytes: match expert {
                SizeExpert::Level => self.level,
                SizeExpert::Last => self.last,
                SizeExpert::Lag => self.lag,
            },
            log_delay: self.delay,
            size_expert: expert,
            smooth_log_bytes: self.level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_pre_arrival_and_tracks_the_oracle_for_every_event() {
        let mut state = SplitCell::default();
        let mut previous = [0.; 3];
        let mut risk = [0.; 3];
        let mut delay = 0.;
        let mut rng = 37_u32;
        for i in 0..10_000 {
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let x = match i % 257 {
                0..70 => 1.,
                70..150 => f64::from(i % 2),
                _ => f64::from(rng % 256) / 32.,
            };
            let y = if i == 0 {
                0.
            } else {
                f64::from(rng % 64) / 16.
            };
            if i == 0 {
                previous = [x; 3];
            } else {
                for (r, p) in risk.iter_mut().zip(previous) {
                    *r += 0.125 * ((x - p).abs() - *r);
                }
                previous = [previous[0] + 0.25 * (x - previous[0]), x, previous[1]];
                delay += 0.25 * (y - delay);
            }
            let choice = if i < 3 {
                0
            } else {
                (0..3).min_by(|&a, &b| risk[a].total_cmp(&risk[b])).unwrap()
            };
            let actual = state.observe_conditioned(x, y).unwrap();
            assert_eq!(
                actual.size_expert,
                [SizeExpert::Level, SizeExpert::Last, SizeExpert::Lag][choice]
            );
            assert_eq!(actual.log_bytes.to_bits(), previous[choice].to_bits());
            assert_eq!(actual.log_delay.to_bits(), delay.to_bits());
        }
        assert_eq!(state.count, 4);
    }

    #[test]
    fn cold_ties_and_period_two_have_exact_routes() {
        let mut state = SplitCell::default();
        for _ in 0..8 {
            assert_eq!(
                state.observe_conditioned(1., 0.).unwrap().size_expert,
                SizeExpert::Level
            );
        }
        let mut state = SplitCell::default();
        for i in 0..50 {
            let x = f64::from(i % 2);
            let pred = state.observe_conditioned(x, 0.).unwrap();
            if i < 3 {
                assert_eq!(pred.size_expert, SizeExpert::Level);
            } else {
                assert_eq!(pred.size_expert, SizeExpert::Lag);
                assert_eq!(pred.log_bytes.to_bits(), (1. - x).to_bits());
            }
        }
    }

    #[test]
    fn invalid_features_do_not_poison_or_advance_state() {
        let mut state = SplitCell::default();
        assert_eq!(state.observe_conditioned(1., 0.1), Err(InvalidFeatures));
        assert_eq!(state, SplitCell::default());
        let _ = state.observe_conditioned(1., 0.).unwrap();
        let before = state;
        for (x, y) in [
            (f64::NAN, 0.),
            (1., f64::INFINITY),
            (1., 4.2),
            (-1., 0.),
            (1., -1.),
        ] {
            assert_eq!(state.observe_conditioned(x, y), Err(InvalidFeatures));
            assert_eq!(state, before);
        }
    }

    #[test]
    fn interleaving_reset_and_eviction_behave_like_independent_lanes() {
        let mut lanes = [SplitCell::default(); 2];
        let mut standalone = SplitCell::default();
        for i in 0..100_u32 {
            let x = i * 10;
            assert_eq!(
                lanes[0].observe_packet(x, 1000),
                standalone.observe_packet(x, 1000)
            );
            let _ = lanes[1].observe_packet(1000 - x, 7000);
        }
        lanes[0] = SplitCell::default();
        assert_eq!(
            lanes[0].observe_packet(75, u64::MAX),
            SplitCell::default().observe_packet(75, 0)
        );
        assert_eq!(
            lanes[0].observe_packet(80, u64::MAX).log_delay.to_bits(),
            (4.110_873_864_173_311_f64 / 4.).to_bits()
        );
        lanes.fill(SplitCell::default());
        assert_eq!(lanes, [SplitCell::default(); 2]);
    }
}
