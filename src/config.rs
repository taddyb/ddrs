//! Subset of the DDR `Config` schema needed by the routing core.
//!
//! Mirrors `~/projects/ddr/src/ddr/validation/configs.py` for the fields the
//! Muskingum-Cunge solver actually reads: `parameter_ranges`, `log_space_parameters`,
//! `defaults`, and `attribute_minimums`. Higher-level fields (data sources, KAN,
//! experiment) are intentionally not modeled here — the solver does not need them.

use std::collections::HashMap;

/// Physical lower bounds applied during routing to keep the math stable.
#[derive(Debug, Clone)]
pub struct AttributeMinimums {
    pub discharge: f32,
    pub slope: f32,
    pub velocity: f32,
    pub depth: f32,
    pub bottom_width: f32,
}

impl Default for AttributeMinimums {
    fn default() -> Self {
        // Matches `Params.attribute_minimums` defaults in DDR.
        Self {
            discharge: 1e-4,
            slope: 1e-3,
            velocity: 0.01,
            depth: 0.01,
            bottom_width: 0.01,
        }
    }
}

/// Physical bounds `[min, max]` used to denormalize NN [0,1] outputs.
#[derive(Debug, Clone)]
pub struct ParameterRanges {
    pub n: [f32; 2],
    pub q_spatial: [f32; 2],
    pub p_spatial: [f32; 2],
}

impl Default for ParameterRanges {
    fn default() -> Self {
        Self {
            n: [0.015, 0.25],
            q_spatial: [0.0, 1.0],
            p_spatial: [1.0, 200.0],
        }
    }
}

/// Routing parameter configuration.
#[derive(Debug, Clone)]
pub struct Params {
    pub parameter_ranges: ParameterRanges,
    pub log_space_parameters: Vec<String>,
    pub defaults: HashMap<String, f32>,
    pub attribute_minimums: AttributeMinimums,
}

impl Default for Params {
    fn default() -> Self {
        let mut defaults = HashMap::new();
        defaults.insert("p_spatial".to_string(), 21.0);
        Self {
            parameter_ranges: ParameterRanges::default(),
            log_space_parameters: vec!["p_spatial".to_string()],
            defaults,
            attribute_minimums: AttributeMinimums::default(),
        }
    }
}

/// Root config — currently just `params`, since that's all the solver consumes.
#[derive(Debug, Clone, Default)]
pub struct Config {
    pub params: Params,
}
