//! Time-axis bookkeeping for batch sampling.
//!
//! Port of the subset of DDR's `Dates` class (`geodatazoo/dataclasses.py`)
//! that the loader actually uses:
//!   - construct from daily start/end
//!   - random `rho`-day window sampler (mirrors `calculate_time_period`)
//!   - daily ↔ hourly index conversion with the "exclude trailing day boundary"
//!     trim (mirrors `set_batch_time`'s `inclusive="left"` semantics on the
//!     hourly range)
//!
//! Key invariant: when `rho` daily steps are selected, the corresponding
//! hourly range has `(rho - 1) * 24` entries — DDR's `StreamflowReader.forward`
//! relies on this when it does `np.repeat(daily, 24)[:, :n_hourly]`.
use chrono::{Duration, NaiveDate};
use rand::Rng;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Frequency {
    Daily,
    Hourly,
}

impl Frequency {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "daily" => Some(Frequency::Daily),
            "hourly" => Some(Frequency::Hourly),
            _ => None,
        }
    }
}

/// The full daily time axis of an experiment (e.g. 1981-10-01..1995-09-30).
/// Per-batch windows are sampled from this axis.
#[derive(Clone, Debug)]
pub struct TimeAxis {
    pub start: NaiveDate,
    pub end: NaiveDate, // inclusive
    pub num_days: usize,
}

impl TimeAxis {
    /// Build a daily axis inclusive of both endpoints.
    pub fn new(start: NaiveDate, end: NaiveDate) -> Self {
        assert!(end >= start, "TimeAxis: end {end} precedes start {start}");
        let num_days = (end - start).num_days() as usize + 1;
        Self {
            start,
            end,
            num_days,
        }
    }

    /// Sample a `rho`-day window uniformly at random.
    ///
    /// Mirrors `Dates.calculate_time_period` (`dataclasses.py:160-167`).
    pub fn sample_rho_window<R: Rng + ?Sized>(&self, rng: &mut R, rho_days: usize) -> RhoWindow {
        assert!(
            rho_days <= self.num_days,
            "rho={rho_days} > num_days={}",
            self.num_days
        );
        // DDR: random_start ~ U[0, sample_size - rho).
        let max_start_exclusive = self.num_days - rho_days;
        let start_day_idx = if max_start_exclusive == 0 {
            0
        } else {
            rng.gen_range(0..max_start_exclusive)
        };
        RhoWindow {
            start_day_idx,
            rho_days,
            window_start: self.start + Duration::days(start_day_idx as i64),
        }
    }

    /// Convert a calendar date to a 0-based day index. Returns `None` if the
    /// date is outside `[start, end]`.
    pub fn day_index(&self, date: NaiveDate) -> Option<usize> {
        if date < self.start || date > self.end {
            return None;
        }
        Some((date - self.start).num_days() as usize)
    }
}

/// A single sampled window. Holds enough state to slice the streamflow /
/// observations arrays along both daily and hourly axes.
#[derive(Copy, Clone, Debug)]
pub struct RhoWindow {
    /// 0-based index into the parent `TimeAxis` (daily resolution).
    pub start_day_idx: usize,
    /// Number of daily entries in this window.
    pub rho_days: usize,
    /// Calendar date of the first day in the window (`TimeAxis.start +
    /// start_day_idx days`). Kept for logging / cross-store alignment.
    pub window_start: NaiveDate,
}

impl RhoWindow {
    /// Half-open daily index range `[start, end)` into the parent axis.
    pub fn daily_range(&self) -> std::ops::Range<usize> {
        self.start_day_idx..self.start_day_idx + self.rho_days
    }

    /// Number of hourly steps consumed by the MC engine for this window.
    ///
    /// DDR sets `batch_hourly_time_range = pd.date_range(daily[0], daily[-1],
    /// freq='h', inclusive='left')`, which yields `(rho - 1) * 24` hours.
    /// `StreamflowReader.forward` then trims `np.repeat(daily, 24)` to this
    /// length. We mirror the trim here.
    pub fn n_hourly(&self) -> usize {
        (self.rho_days.saturating_sub(1)) * 24
    }

    /// Half-open hourly index range into the parent axis (only meaningful
    /// when the store is hourly-native). For daily-native stores, the loader
    /// reads `daily_range()` and then `repeat_24` + trims to `n_hourly()`.
    pub fn hourly_range(&self) -> std::ops::Range<usize> {
        let h0 = self.start_day_idx * 24;
        h0..h0 + self.n_hourly()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn axis_length() {
        let axis = TimeAxis::new(
            NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
            NaiveDate::from_ymd_opt(1995, 9, 30).unwrap(),
        );
        // 14 water-year span. Inclusive both ends.
        assert_eq!(axis.num_days, 5113);
    }

    #[test]
    fn rho_window_n_hourly_is_rho_minus_1_times_24() {
        // Mirrors DDR's inclusive='left' semantics on the hourly range.
        let w = RhoWindow {
            start_day_idx: 0,
            rho_days: 90,
            window_start: NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
        };
        assert_eq!(w.n_hourly(), 89 * 24);
        assert_eq!(w.daily_range(), 0..90);
    }

    #[test]
    fn seeded_sampling_is_reproducible() {
        let axis = TimeAxis::new(
            NaiveDate::from_ymd_opt(1981, 10, 1).unwrap(),
            NaiveDate::from_ymd_opt(1995, 9, 30).unwrap(),
        );
        let mut r1 = StdRng::seed_from_u64(42);
        let mut r2 = StdRng::seed_from_u64(42);
        let w1 = axis.sample_rho_window(&mut r1, 90);
        let w2 = axis.sample_rho_window(&mut r2, 90);
        assert_eq!(w1.start_day_idx, w2.start_day_idx);
        assert!(w1.start_day_idx < axis.num_days - 90);
    }
}
