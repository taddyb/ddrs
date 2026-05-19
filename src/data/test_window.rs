//! Contiguous-hourly time window for test-mode chunking.
//!
//! Unlike `RhoWindow` (which drops the trailing day to mirror DDR pandas
//! `inclusive='left'` for training-mode random rho-window sampling), this
//! `TestWindow` exposes the full `n_days * 24` hours so that consecutive
//! chunks tile the hourly axis without gap or overlap.
//!
//! Used by `MeritGagesDataset::collate_window` and `evaluate()`.

use chrono::{Duration, NaiveDate};

use crate::data::dates::TimeAxis;

#[derive(Copy, Clone, Debug)]
pub struct TestWindow {
    /// 0-based index into the parent `TimeAxis` (daily resolution).
    pub start_day_idx: usize,
    /// Number of daily entries in this window.
    pub n_days: usize,
    /// Calendar date of the first day in the window.
    pub window_start: NaiveDate,
}

impl TestWindow {
    pub fn new(axis: &TimeAxis, start_day_idx: usize, n_days: usize) -> Self {
        assert!(
            start_day_idx + n_days <= axis.num_days,
            "TestWindow exceeds axis: start={start_day_idx} + n_days={n_days} > num_days={}",
            axis.num_days
        );
        Self {
            start_day_idx,
            n_days,
            window_start: axis.start + Duration::days(start_day_idx as i64),
        }
    }

    /// Half-open daily index range `[start, end)` into the parent axis.
    pub fn daily_range(&self) -> std::ops::Range<usize> {
        self.start_day_idx..self.start_day_idx + self.n_days
    }

    /// Contiguous hourly length: `n_days * 24`. No trailing-day trim.
    pub fn n_hourly(&self) -> usize {
        self.n_days * 24
    }

    /// Half-open hourly index range into the parent axis (assumes hourly-native
    /// store; daily-native stores use `daily_range()` + repeat-24).
    pub fn hourly_range(&self) -> std::ops::Range<usize> {
        let h0 = self.start_day_idx * 24;
        h0..h0 + self.n_hourly()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_axis(num_days: usize) -> TimeAxis {
        TimeAxis::new(
            NaiveDate::from_ymd_opt(1995, 10, 1).unwrap(),
            NaiveDate::from_ymd_opt(1995, 10, 1).unwrap() + Duration::days(num_days as i64 - 1),
        )
    }

    #[test]
    fn test_window_is_contiguous_unlike_rho_window() {
        let axis = fake_axis(30);
        let w = TestWindow::new(&axis, 0, 15);
        // RhoWindow with rho_days=15 → n_hourly = 14*24 = 336.
        // TestWindow with n_days=15 → n_hourly = 15*24 = 360. No trim.
        assert_eq!(w.n_hourly(), 15 * 24);
        assert_eq!(w.daily_range(), 0..15);
        assert_eq!(w.hourly_range(), 0..360);
    }

    #[test]
    fn consecutive_test_windows_tile_with_no_gap() {
        let axis = fake_axis(45);
        let w0 = TestWindow::new(&axis, 0, 15);
        let w1 = TestWindow::new(&axis, 15, 15);
        assert_eq!(w0.hourly_range().end, w1.hourly_range().start);
        assert_eq!(w0.daily_range().end, w1.daily_range().start);
    }
}
