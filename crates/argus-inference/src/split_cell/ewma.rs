//! Standalone size and delay moving-average baseline for a single generation.

use super::{InvalidFeatures, MAX_DELAY_NS, MAX_LOG_DELAY};

/// Next same-generation forecast, in conditioned log space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EwmaForecast {
    pub log_bytes: f64,
    pub log_delay: f64,
}

/// Exactly two scalars of state. Reset with `Default` at a generation boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ewma {
    level: f64,
    delay: f64,
}

impl Default for Ewma {
    fn default() -> Self {
        // A negative level is outside the accepted feature domain.
        Self {
            level: -1.,
            delay: 0.,
        }
    }
}

impl Ewma {
    /// Update from already-conditioned features and forecast the next packet.
    ///
    /// # Errors
    /// Non-finite or negative inputs, delay above the clipping bound, and
    /// nonzero cold delay are rejected without changing the state.
    pub fn observe_conditioned(
        &mut self,
        log_bytes: f64,
        log_delay: f64,
    ) -> Result<EwmaForecast, InvalidFeatures> {
        if !log_bytes.is_finite()
            || log_bytes < 0.
            || !(0. ..=MAX_LOG_DELAY).contains(&log_delay)
            || (self.level < 0. && log_delay != 0.)
        {
            return Err(InvalidFeatures);
        }
        Ok(self.update(log_bytes, log_delay))
    }

    /// Condition frame size and gap, then update. The caller owns ordering
    /// and lane reuse. A cold lane ignores the supplied gap.
    #[must_use]
    pub fn observe_packet(&mut self, frame_bytes: u32, gap_ns: u64) -> EwmaForecast {
        // Match the frozen cell's exact conditioning, including gap clipping.
        #[allow(clippy::cast_precision_loss)]
        let seconds = gap_ns.min(MAX_DELAY_NS) as f64 / 1_000_000_000.;
        let delay = if self.level < 0. { 0. } else { seconds.ln_1p() };
        self.update(f64::from(frame_bytes).ln_1p(), delay)
    }

    fn update(&mut self, x: f64, y: f64) -> EwmaForecast {
        if self.level < 0. {
            self.level = x;
            self.delay = y;
        } else {
            self.level += (x - self.level) / 4.;
            self.delay += (y - self.delay) / 4.;
        }
        EwmaForecast {
            log_bytes: self.level,
            log_delay: self.delay,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::split_cell::{Forecast, SplitCell};

    fn assert_parity(ewma: EwmaForecast, split: Forecast) {
        assert_eq!(ewma.log_bytes.to_bits(), split.smooth_log_bytes.to_bits());
        assert_eq!(ewma.log_delay.to_bits(), split.log_delay.to_bits());
    }

    #[test]
    fn exact_two_scalars_and_packet_parity_across_resets() {
        assert_eq!(size_of::<Ewma>(), 2 * size_of::<f64>());
        let mut ewma = Ewma::default();
        let mut split = SplitCell::default();
        let mut rng = 37_u32;
        for i in 0..10_000 {
            if i % 257 == 0 {
                ewma = Ewma::default();
                split = SplitCell::default();
            }
            rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let (bytes, gap) = match i % 7 {
                0 => (0, u64::MAX),
                1 => (u32::MAX, 0),
                2 => (rng, MAX_DELAY_NS),
                _ => (rng % 1500, u64::from(rng)),
            };
            assert_parity(
                ewma.observe_packet(bytes, gap),
                split.observe_packet(bytes, gap),
            );
        }
    }

    #[test]
    fn conditioned_invalid_and_cold_parity() {
        let mut ewma = Ewma::default();
        let mut split = SplitCell::default();
        for (x, y) in [
            (1., MAX_LOG_DELAY),
            (-0., -0.),
            (f64::INFINITY, 0.),
            (f64::NAN, 0.),
            (1., f64::NAN),
            (1., f64::INFINITY),
            (-1., 0.),
            (1., -1.),
            (1., MAX_LOG_DELAY.next_up()),
            (1., MAX_LOG_DELAY),
            (f64::MAX, 0.),
            (0., 0.),
        ] {
            let before = ewma;
            match split.observe_conditioned(x, y) {
                Ok(forecast) => assert_parity(ewma.observe_conditioned(x, y).unwrap(), forecast),
                Err(error) => {
                    assert_eq!(ewma.observe_conditioned(x, y), Err(error));
                    assert_eq!(ewma, before);
                }
            }
        }
    }
}
