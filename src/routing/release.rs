//! Seasonal residence time of a dam routed as a linear reservoir.
//!
//! The learned dam release (and the seasonal `fixed` table) route each dam row
//! as `S = T_d(t)·Q` with
//!
//! ```text
//! T_d(t) = max(T0_d · exp(a_d · sin ω_t + b_d · cos ω_t), 1/24 d)
//! ω_t    = 2π · doy(t) / 365.25          doy(t) fractional, 1-based
//! ```
//!
//! `doy` is 1-based like pandas `DatetimeIndex.dayofyear`, which the offline
//! release fit (`experiments/reservoir/smoke/expected_release_fit.py`) uses, so
//! hour 0 of a day carries exactly the fit's daily `ω`. The engine evaluates
//! `T` at the END hour of each step: the step producing routed column `t`
//! (from lateral-inflow row `t − 1`) reads phase row `t`, matching the offline
//! law's implicit-Euler use of `T_t` to compute `Q_t`.
//!
//! The clamp keeps `T >= dt = 1 h`, above the `dt/2` that the dam row's
//! `c3 = (2T − dt)/(2T + dt) >= 0` needs. Where it binds the gradient into
//! `T0`, `a` and `b` is exactly zero for that step (Burn's `clamp_min`
//! backward), which is the correct subgradient of a flat function.

use chrono::{Datelike, NaiveDate};

/// Seconds per day, the `T0` (days) to `K` (seconds) conversion.
pub const SECONDS_PER_DAY: f32 = 86_400.0;

/// Floor on the residence time, days: one hour, the routing `dt`.
pub const MIN_T_DAYS: f32 = 1.0 / 24.0;

/// Days per seasonal cycle.
pub const DAYS_PER_YEAR: f64 = 365.25;

/// `(sin ω, cos ω)` for each hour of a window starting at midnight of
/// `window_start`, `n_hours` rows. Row `h` is `window_start + h` hours.
pub fn seasonal_phase(window_start: NaiveDate, n_hours: usize) -> Vec<[f32; 2]> {
    (0..n_hours)
        .map(|h| {
            let date = window_start + chrono::Duration::days((h / 24) as i64);
            let doy = date.ordinal() as f64 + (h % 24) as f64 / 24.0;
            let w = 2.0 * std::f64::consts::PI * doy / DAYS_PER_YEAR;
            [w.sin() as f32, w.cos() as f32]
        })
        .collect()
}

/// Smallest and largest `T` (days) the seasonal law reaches over a year,
/// after the clamp: `T0·exp(∓√(a² + b²))`.
pub fn seasonal_t_range(t0_days: f32, a: f32, b: f32) -> (f32, f32) {
    let amp = (a * a + b * b).sqrt();
    (
        (t0_days * (-amp).exp()).max(MIN_T_DAYS),
        (t0_days * amp.exp()).max(MIN_T_DAYS),
    )
}
