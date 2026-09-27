//! Seasonal residence time of a dam routed as a linear reservoir.
//!
//! The learned dam release (and the seasonal `fixed` table) route each dam row
//! with the residence time
//!
//! ```text
//! T_d(t) = T0_d · exp(a_d · sin ω_t + b_d · cos ω_t)      (floored, see below)
//! ω_t    = 2π · doy(t) / 365.25          doy(t) fractional, 1-based
//! ```
//!
//! `doy` is 1-based like pandas `DatetimeIndex.dayofyear`, which the offline
//! release fit (`experiments/reservoir/smoke/expected_release_fit.py`) uses, so
//! hour 0 of a day carries exactly the fit's daily `ω`. The step producing
//! routed column `t` (from lateral-inflow row `t − 1`) reads `T` at BOTH ends:
//! `T_{t+1}` at phase row `t` and `T_t` at row `t − 1`. Both ends come from
//! the closed form, so a window or test-phase chunk start needs no state.
//!
//! # The two dam rows (`release_head.dam_row`)
//!
//! **`replace`** (the default; S19'' / S19''' in `mmc_op`). The dam row IS the
//! reservoir: its storage is `S = T·Q` alone, which replaces the reach's own
//! channel routing. The trapezoid on it,
//!
//! ```text
//! T_{t+1}·Q_{t+1} − T_t·Q_t = dt·[(I_t + I_{t+1})/2 + q' − (Q_t + Q_{t+1})/2]
//! c1 = c2 = dt/(2·T_{t+1} + dt),  c3 = (2·T_t − dt)/(2·T_{t+1} + dt),  c4 = 2·dt/(2·T_{t+1} + dt)
//! ```
//!
//! conserves storage exactly for any `T(t)` and is the Muskingum row at
//! `K = T`, `X = 0` when `T` is constant. (The first build set only
//! `K := T_{t+1}`, which carries `Q` rather than `S` across a change in `T`;
//! see `research/findings/2026-09-27-learned-dam-release-findings.md`, check 2.)
//! Because the reach's channel storage is dropped, `T = 1 h` is FASTER than no
//! dam on a reach whose `K_r` is several hours, and the head cannot opt out.
//! `T` is floored at one hour ([`MIN_T_DAYS`]), above the `dt/2` that
//! `c3 >= 0` needs.
//!
//! **`additive`** (S19'''' in `mmc_op`, added 2026-09-27). The reservoir's
//! storage is ADDED to the reach's channel storage, with the reach's own
//! Muskingum `K_r`, `X_r` at this step (as the channel row computes them,
//! including the `enforce_positivity` clamps):
//!
//! ```text
//! S  = K_r·[X_r·I + (1 − X_r)·Q] + T·Q
//! D  = K_r·(1 − X_r) + T_{t+1} + dt/2
//! c1 = (dt/2 − K_r·X_r)/D      on I_{t+1}
//! c2 = (dt/2 + K_r·X_r)/D      on I_t
//! c3 = (K_r·(1 − X_r) + T_t − dt/2)/D      on Q_t
//! c4 = dt/D                    on q'
//! ```
//!
//! the trapezoid on `S` with `K_r`, `X_r` held over the step (the Muskingum
//! channel row's own convention), so the `T` part conserves storage exactly
//! for a time-varying `T`. `T = 0` is the channel row bit for bit (the code
//! adds `2·T` to the channel's `2K(1−X)` in the denominator and in c3's
//! numerator); `K_r = 0` is the replace row. `T` is not floored: `T >= 0`,
//! so the head can take a dam out. `c1 >= 0` exactly when the channel's does
//! (`K_r·X_r <= dt/2`); `c3 >= 0` needs `K_r·(1 − X_r) + T_t >= dt/2`, which
//! the channel part meets whenever its own `c3 >= 0` and a positive `T` only
//! helps. Where it fails, the dam row does what the channel row does (no
//! extra clamp: with `enforce_positivity` the S18'/S19' clamps already hold
//! the channel part's `c1, c3 >= 0`; without it the S28 discharge clamp is
//! the only guard, as on every reach).
//!
//! Where the replace row's one-hour clamp binds, the gradient into `T0`, `a`
//! and `b` is exactly zero for that step (Burn's `clamp_min` backward), which
//! is the correct subgradient of a flat function.
//!
//! # The harmonic rule curve (`release_head.rule_curve`, 2026-09-27)
//!
//! The storage law gains a periodic target, `S = T·Q + S0_d(t)` (on either
//! row), whose derivative is the redistribution flux
//!
//! ```text
//! r_d(t)  = Ibar_d · Σ_{k=1,2} (c_{k,s} sin kω_t + c_{k,c} cos kω_t)          m³/s
//! S0_d(t) = Ibar_d · Σ_{k=1,2} (−c_{k,s} cos kω_t + c_{k,c} sin kω_t) / (k·Ω)  m³
//! Ω       = 2π / (365.25 · 86400 s)
//! ```
//!
//! with `ω_t` the CONTINUOUS seasonal phase ([`rule_curve_phase_start`]):
//! `ω = Ω·(seconds since 1970-01-01 00:00) + 2π/365.25`, so every hourly
//! step advances it by exactly `Ω·dt` (`sin 2ω`, `cos 2ω` by the double
//! angle). The trapezoid on `S` puts `−(S0_{t+1} − S0_t)` on the dam row's
//! right-hand side, i.e. the dam row's lateral inflow becomes
//! `q'_eff = q'_d − (S0_{t+1} − S0_t)/dt` for the step: the reservoir stores
//! (`r > 0`) or releases (`r < 0`) on top of the bucket. The system matrix,
//! the CSR pattern and the hand-written backward are unchanged; the flux
//! reaches the per-dam coefficients by ordinary autodiff through the
//! timestep op's `q'` parent (whose gradient the op already returns, B25).
//! `S0` is periodic in `ω`, so over whole periods (365.25 d) its increments
//! telescope to zero and the rule curve moves no volume; over any span they
//! telescope to `S0(end) − S0(start)`. [`rule_curve_increments`] tabulates
//! the per-step phase increments in f64 so the flux carries no large-number
//! cancellation. `Ibar_d` is the table's `inflow_mean_m3s` (training-period
//! mean of the upstream-summed Q'); `c = rule_curve_max·tanh(θ)` per dam
//! (`crate::nn::dam_params`), so `θ = 0` is no rule curve bit for bit.
//!
//! **Why a continuous phase, not [`seasonal_phase`].** The seasonal `T` law
//! reads the calendar day of year, pandas-style, to match its offline fit.
//! Used for the rule curve (the first build, 2026-09-27), that phase restarts
//! at 1 January, so the Dec 31 23:00 → Jan 1 00:00 step spanned +7 h of phase
//! in a non-leap year and −17 h in a leap year: 7× (or −17×) the hourly flux
//! in one step, enough to force the discharge clamp (review v2, finding 3).
//! The continuous phase advances one hour of phase per hourly step always. It
//! stays within about ±0.75 d of the day-of-year phase (the leap cycle), so
//! the coefficients keep their meaning; it is 2π/365.25 at 1970-01-01 00:00,
//! i.e. day-of-year 1 there. The `T` law keeps [`seasonal_phase`] unchanged.

use chrono::{Datelike, NaiveDate};

use crate::config::DamRow;

/// Seconds per day, the `T0` (days) to `K` (seconds) conversion.
pub const SECONDS_PER_DAY: f32 = 86_400.0;

/// Floor on the replace row's residence time, days: one hour, the routing `dt`.
pub const MIN_T_DAYS: f32 = 1.0 / 24.0;

/// Days per seasonal cycle.
pub const DAYS_PER_YEAR: f64 = 365.25;

/// Floor on `T` (days) for a dam-row form: one hour for `replace`, none (0)
/// for `additive`.
pub fn t_floor_days(dam_row: DamRow) -> f32 {
    match dam_row {
        DamRow::Replace => MIN_T_DAYS,
        DamRow::Additive => 0.0,
    }
}

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

/// Angular frequency of the seasonal cycle, rad/s: `2π / (365.25 d)`.
pub const OMEGA_RAD_PER_S: f64 = 2.0 * std::f64::consts::PI / (DAYS_PER_YEAR * 86_400.0);

/// The rule curve's continuous seasonal phase `ω` (radians, in `[0, 2π)`) at
/// hour 0 of `window_start`: `Ω·(seconds since 1970-01-01 00:00) + 2π/365.25`,
/// reduced mod 2π. Row `h` of a window starting there has phase
/// `ω_0 + Ω·3600·h`, so consecutive windows, test-phase chunks and year
/// boundaries all line up with one hour of phase per hourly step. At
/// 1970-01-01 00:00 it equals [`seasonal_phase`]'s day-of-year-1 phase, and it
/// stays within about ±0.75 d of it (the leap cycle).
pub fn rule_curve_phase_start(window_start: NaiveDate) -> f64 {
    let epoch = NaiveDate::from_ymd_opt(1970, 1, 1).expect("valid date");
    let days = (window_start - epoch).num_days() as f64 + 1.0;
    let cycles = days / DAYS_PER_YEAR;
    2.0 * std::f64::consts::PI * (cycles - cycles.floor())
}

/// `H(ω) = [−cos ω, sin ω, −cos 2ω / 2, sin 2ω / 2] / Ω` (seconds), so that
/// `S0(ω) = Ibar · Σ_j H_j(ω)·c_j` for `c = (c1s, c1c, c2s, c2c)`.
pub fn rule_curve_h(w: f64) -> [f64; 4] {
    let o = OMEGA_RAD_PER_S;
    [-w.cos() / o, w.sin() / o, -(2.0 * w).cos() / (2.0 * o), (2.0 * w).sin() / (2.0 * o)]
}

/// The rule curve's per-step phase increments for a window whose row 0 has
/// continuous phase `phase0` ([`rule_curve_phase_start`]): row `s` (the step
/// from row `s` to `s + 1`, one hour) is `H(ω_{s+1}) − H(ω_s)` ([`rule_curve_h`])
/// with `ω_s = phase0 + Ω·dt·s`, so that
/// `S0_{s+1} − S0_s = Ibar · Σ_j ΔH[s, j]·c_j`. Computed in f64, `n_rows − 1`
/// rows, row-major `[n_rows − 1, 4]`; they telescope to
/// `H(ω_{n_rows−1}) − H(ω_0)`, which is zero over a whole period.
pub fn rule_curve_increments(phase0: f64, n_rows: usize) -> Vec<f32> {
    let dw = OMEGA_RAD_PER_S * crate::routing::mmc::DT_SECONDS as f64;
    let mut out = Vec::with_capacity(n_rows.saturating_sub(1) * 4);
    let mut a = rule_curve_h(phase0);
    for s in 0..n_rows.saturating_sub(1) {
        let b = rule_curve_h(phase0 + dw * (s + 1) as f64);
        out.extend((0..4).map(|j| (b[j] - a[j]) as f32));
        a = b;
    }
    out
}

/// Smallest and largest `T` (days) the seasonal law reaches over a year,
/// after the dam row's floor ([`t_floor_days`]): `T0·exp(∓√(a² + b²))`.
pub fn seasonal_t_range(t0_days: f32, a: f32, b: f32, dam_row: DamRow) -> (f32, f32) {
    let amp = (a * a + b * b).sqrt();
    let floor = t_floor_days(dam_row);
    (
        (t0_days * (-amp).exp()).max(floor),
        (t0_days * amp.exp()).max(floor),
    )
}
