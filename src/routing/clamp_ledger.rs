//! Records of where the Muskingum-Cunge clamps are active.
//!
//! Two optional sinks. The forward fills a [`ClampLedger`]: per step and per
//! reach, whether each clamp bound is active (`clamp_rule`: a tie is
//! inactive). The backward fills a [`ClampProbe`]: per step and per site, the
//! gradient right before and right after the clamp mask.
//!
//! Both are explicit arguments. With every sink `None`, the forward and the
//! backward run as if this module did not exist: no extra kernel and no host
//! copy. Later checks compare the gradient zero pattern with the ledger.

use std::collections::BTreeMap;

/// One clamp on the forward path. `OutsideCrWindow` is not a clamp: it marks
/// reaches where a Muskingum coefficient `c1` or `c3` is negative.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClampSite {
    DepthFloor,
    SideSlopeLow,
    SideSlopeHigh,
    BottomWidthFloor,
    VelocityLow,
    VelocityHigh,
    KFloor,
    XBandLow,
    XBandHigh,
    XCap,
    DischargeFloor,
    OutsideCrWindow,
}

/// One place where the backward applies a clamp mask to a gradient.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProbeSite {
    Depth,
    SideSlope,
    BottomWidth,
    Velocity,
    KFloor,
    XBand,
    XCap,
    Discharge,
}

impl ClampSite {
    pub const ALL: [ClampSite; 12] = [
        ClampSite::DepthFloor,
        ClampSite::SideSlopeLow,
        ClampSite::SideSlopeHigh,
        ClampSite::BottomWidthFloor,
        ClampSite::VelocityLow,
        ClampSite::VelocityHigh,
        ClampSite::KFloor,
        ClampSite::XBandLow,
        ClampSite::XBandHigh,
        ClampSite::XCap,
        ClampSite::DischargeFloor,
        ClampSite::OutsideCrWindow,
    ];

    /// snake_case name, for tables and logs.
    pub fn name(self) -> &'static str {
        match self {
            ClampSite::DepthFloor => "depth_floor",
            ClampSite::SideSlopeLow => "side_slope_low",
            ClampSite::SideSlopeHigh => "side_slope_high",
            ClampSite::BottomWidthFloor => "bottom_width_floor",
            ClampSite::VelocityLow => "velocity_low",
            ClampSite::VelocityHigh => "velocity_high",
            ClampSite::KFloor => "k_floor",
            ClampSite::XBandLow => "x_band_low",
            ClampSite::XBandHigh => "x_band_high",
            ClampSite::XCap => "x_cap",
            ClampSite::DischargeFloor => "discharge_floor",
            ClampSite::OutsideCrWindow => "outside_cr_window",
        }
    }

    /// The backward mask that cuts the gradient for this clamp. `None` for
    /// `OutsideCrWindow`, which has no mask.
    pub fn probe_site(self) -> Option<ProbeSite> {
        match self {
            ClampSite::DepthFloor => Some(ProbeSite::Depth),
            ClampSite::SideSlopeLow | ClampSite::SideSlopeHigh => Some(ProbeSite::SideSlope),
            ClampSite::BottomWidthFloor => Some(ProbeSite::BottomWidth),
            ClampSite::VelocityLow | ClampSite::VelocityHigh => Some(ProbeSite::Velocity),
            ClampSite::KFloor => Some(ProbeSite::KFloor),
            ClampSite::XBandLow | ClampSite::XBandHigh => Some(ProbeSite::XBand),
            ClampSite::XCap => Some(ProbeSite::XCap),
            ClampSite::DischargeFloor => Some(ProbeSite::Discharge),
            ClampSite::OutsideCrWindow => None,
        }
    }
}

/// Clamp activity of one routed step. `active[site][i]` is true when the
/// bound is active on reach `i`. A site that does not run on this step's path
/// (for example side slope on the power law) has no key.
#[derive(Clone, Debug, Default)]
pub struct StepLedger {
    pub n: usize,
    pub active: BTreeMap<ClampSite, Vec<bool>>,
}

/// One [`StepLedger`] per recorded step, in forward order.
#[derive(Clone, Debug, Default)]
pub struct ClampLedger {
    pub steps: Vec<StepLedger>,
}

impl ClampLedger {
    /// Number of (step, reach) entries where `s` is active.
    pub fn count(&self, s: ClampSite) -> u64 {
        self.steps
            .iter()
            .filter_map(|st| st.active.get(&s))
            .map(|v| v.iter().filter(|&&a| a).count() as u64)
            .sum()
    }

    /// Number of (step, reach) entries where `s` differs between the two
    /// ledgers. Panics if the ledgers differ in shape: step count, reach
    /// count, or a site present in only one of them.
    pub fn switches(&self, other: &ClampLedger, s: ClampSite) -> u64 {
        assert_eq!(
            self.steps.len(),
            other.steps.len(),
            "ledgers have different step counts"
        );
        let mut total = 0u64;
        for (k, (a, b)) in self.steps.iter().zip(&other.steps).enumerate() {
            assert_eq!(a.n, b.n, "ledgers differ in reach count at step {k}");
            match (a.active.get(&s), b.active.get(&s)) {
                (None, None) => {}
                (Some(x), Some(y)) => {
                    assert_eq!(x.len(), y.len(), "site {} has different lengths at step {k}", s.name());
                    total += x.iter().zip(y).filter(|(p, q)| p != q).count() as u64;
                }
                _ => panic!("site {} is present in only one ledger at step {k}", s.name()),
            }
        }
        total
    }
}

/// The gradient at one probe site: right before the mask and right after it.
#[derive(Clone, Debug)]
pub struct SiteGrad {
    pub g_in: Vec<f32>,
    pub g_out: Vec<f32>,
}

/// Gradients recorded by one backward call.
#[derive(Clone, Debug, Default)]
pub struct StepProbe {
    pub sites: BTreeMap<ProbeSite, SiteGrad>,
}

/// One [`StepProbe`] per backward call, in backward call order (the reverse
/// of the forward order).
#[derive(Clone, Debug, Default)]
pub struct ClampProbe {
    pub steps: Vec<StepProbe>,
}
