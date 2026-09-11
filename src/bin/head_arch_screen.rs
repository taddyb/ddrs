//! Screen KAN head topologies for output collapse, without training the router.
//!
//! # Why this exists
//!
//! §31 of `docs/2026-09-08-landscape-hypothesis-tests-findings.md` found that
//! the trained head's two learnable outputs are the same latent direction
//! relabelled: over all 346,321 CONUS reaches,
//! `logit(q) = 3.979 · logit(n) + 2.752` with R² = 0.987. That gain of ~4 is
//! also why `q` saturates both of its bounds while `n` never touches its own.
//!
//! Three candidate causes were proposed, and they need different fixes:
//!
//!   1. the `Linear(H, P)` read-out, which makes every output an affine
//!      functional of one shared latent `h`;
//!   2. the trunk, if `h` itself carries only one usable direction;
//!   3. the inputs, if the 10 attributes carry only one usable direction
//!      (their GBM ceiling is R² = 0.160).
//!
//! Training a CONUS arm per topology costs ~1.8 h each. This binary answers
//! most of the question in minutes instead, because the head is a pure
//! per-reach function of the attributes: no routing, no observations, no
//! optimizer schedule. It writes fields; `experiments/head_arch/analyze.py`
//! computes the statistics.
//!
//! # What each arm is asked
//!
//! * **init** — forward the untrained head over every CONUS reach. Gives the
//!   prior each topology starts from. §31 measured ρ(n, q) = 0.727 at epoch 1
//!   for the shared head, so collapse is partly inherited rather than learned.
//! * **trunk** — the penultimate activations `h`. §31 inferred the trunk's
//!   rank from its outputs and flagged that as a caveat; this measures it
//!   directly, which is the check that separates cause 2 from cause 1.
//! * **capacity** — a positive control. Fit the head, supervised, to two
//!   synthetic targets built from the attributes and orthogonalised against
//!   each other so their true correlation is zero. A topology that cannot
//!   decorrelate two outputs when it is *told* the answer will certainly not
//!   do it under a weak routing gradient. This isolates architecture from the
//!   objective, which is the confound every previous attempt ran into.
//!
//! # Reading the result
//!
//! If the capacity control decorrelates for every arm, the architecture is not
//! the binding constraint and the answer is cause 3 (the inputs) — in which
//! case a separate KAN buys nothing and the fix is more informative
//! attributes, or an observation that sees channel width directly.
//! If some arms fail it, those are structurally incapable and no objective
//! will rescue them.
//!
//! ```bash
//! cargo run --release --bin head_arch_screen -- \
//!   --config .ddrs/runs/<id>/config.yaml \
//!   --out-dir .ddrs/experiments/head-arch/<ts>
//! ```

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use burn::module::AutodiffModule;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::backend::{AutodiffBackend, Backend, BackendTypes};
use burn::tensor::{ElementConversion, Tensor, TensorData};
use clap::Parser;
use ndarray::Array2;

use ddrs::config::Config;
use ddrs::dump_parameters::load_conus_attributes;
use ddrs::nn::kan_head::{KanHead, KanHeadConfig};

type R<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Parser, Debug)]
#[command(
    name = "head_arch_screen",
    about = "Compare KAN head topologies for output collapse, forward passes only"
)]
struct Cli {
    /// Run config snapshot (`.ddrs/runs/<id>/config.yaml`) — supplies the
    /// `kan_head` section, the attribute sources and the resolved adjacency.
    #[arg(long)]
    config: PathBuf,

    /// Directory to write per-arm fields into.
    #[arg(long)]
    out_dir: PathBuf,

    /// Parameters each arm emits. Defaults to the three the routing solver can
    /// learn together.
    #[arg(long, default_value = "n,p_spatial,q_spatial", value_delimiter = ',')]
    params: Vec<String>,

    /// Adam steps for the capacity positive control.
    #[arg(long, default_value_t = 2000)]
    fit_steps: usize,

    /// Reaches sampled (evenly, by stride) for the capacity control and the
    /// trunk-activation dump. The full CONUS forward is always used for the
    /// init fields.
    #[arg(long, default_value_t = 20000)]
    sample: usize,

    /// Learning rate for the capacity control.
    #[arg(long, default_value_t = 0.01)]
    lr: f64,

    /// Restrict to these arm names (default: all).
    #[arg(long, value_delimiter = ',')]
    arms: Vec<String>,
}

/// One head topology under test.
struct Arm {
    name: &'static str,
    /// One-line statement of what this arm changes and what it would prove.
    rationale: &'static str,
    apply: fn(KanHeadConfig, &[String]) -> KanHeadConfig,
}

fn arms() -> Vec<Arm> {
    vec![
        Arm {
            name: "shared-linear",
            rationale: "control: current architecture, Linear read-out over one shared trunk",
            apply: |c, _| c,
        },
        Arm {
            name: "split-trunk",
            rationale: "separate trunk for {n} and for {p,q}: shares no weight at all",
            apply: |c, params| {
                let (geom, rough): (Vec<String>, Vec<String>) = params
                    .iter()
                    .cloned()
                    .partition(|p| p == "p_spatial" || p == "q_spatial");
                let mut groups = Vec::new();
                if !rough.is_empty() {
                    groups.push(rough);
                }
                if !geom.is_empty() {
                    groups.push(geom);
                }
                c.with_parameter_groups(groups)
            },
        },
        Arm {
            name: "split-trunk-per-param",
            rationale: "one trunk per parameter: the maximal separation, bounds what splitting can buy",
            apply: |c, params| {
                c.with_parameter_groups(params.iter().map(|p| vec![p.clone()]).collect())
            },
        },
        Arm {
            name: "kan-readout",
            rationale: "shared trunk, KanLayer(H,P) read-out: removes the affine tie, keeps the trunk",
            apply: |c, _| c.with_output_layer_kan(true),
        },
        Arm {
            name: "kan-embed-readout",
            rationale: "KanLayer both ends, no Linear anywhere: per-attribute splines on the raw inputs",
            apply: |c, _| c.with_input_layer_kan(true).with_output_layer_kan(true),
        },
        Arm {
            name: "deeper",
            rationale: "control for depth alone: same topology, 4 trunk layers instead of 2",
            apply: |c, _| c.with_num_hidden_layers(4),
        },
        Arm {
            name: "wider",
            rationale: "control for width alone: same topology, H=64 instead of 21",
            apply: |c, _| c.with_hidden_size(64),
        },
    ]
}

fn main() -> R<()> {
    let cli = Cli::parse();
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
    let device = <burn::backend::NdArray<f32> as BackendTypes>::Device::default();
    run::<B>(cli, device)
}

fn run<B: AutodiffBackend>(cli: Cli, device: B::Device) -> R<()>
where
    B::InnerBackend: Backend<Device = B::Device>,
{
    let cfg = Config::from_yaml_file(&cli.config)?;
    let section = cfg
        .kan_head
        .as_ref()
        .ok_or("config has no kan_head section")?;

    let (attrs, _comids) = load_conus_attributes(&cfg, &section.input_var_names)?;
    let (n_reaches, n_feat) = (attrs.shape()[0], attrs.shape()[1]);
    eprintln!("attributes: {n_reaches} reaches x {n_feat} features");

    std::fs::create_dir_all(&cli.out_dir)?;

    // Even stride keeps the subsample representative of the CONUS attribute
    // distribution without needing an RNG.
    let stride = (n_reaches / cli.sample).max(1);
    let idx: Vec<usize> = (0..n_reaches).step_by(stride).collect();
    eprintln!("subsample: {} reaches (stride {stride})", idx.len());

    let x_full = to_tensor::<B>(&attrs, &(0..n_reaches).collect::<Vec<_>>(), &device);
    let x_sub = to_tensor::<B>(&attrs, &idx, &device);

    // Two targets that are exactly recoverable from the inputs and exactly
    // uncorrelated with each other. If a topology cannot separate these, the
    // routing objective never had a chance.
    let (t1, t2, target_corr) = synthetic_targets(&attrs, &idx, &section.input_var_names);
    eprintln!("synthetic target correlation: {target_corr:+.6} (0 by construction)");
    let targets = Tensor::<B, 2>::from_data(
        TensorData::new(
            t1.iter().zip(&t2).flat_map(|(a, b)| [*a, *b]).collect::<Vec<f32>>(),
            [idx.len(), 2],
        ),
        &device,
    );

    // Built from the section rather than via `kan_config` because this binary
    // overrides `learnable_parameters` (a required Config field, so there is no
    // `with_` setter) and has no use for the disaggregation head: the question
    // is about the per-reach parameter map, not the forcing.
    let base = KanHeadConfig::new(
        section.input_var_names.clone(),
        cli.params.clone(),
        cfg.seed,
    )
    .with_hidden_size(section.hidden_size)
    .with_num_hidden_layers(section.num_hidden_layers)
    .with_grid(section.grid)
    .with_k(section.k)
    .with_kan_grid_range(section.kan_grid_range);

    let mut manifest = serde_json::Map::new();
    for arm in arms() {
        if !cli.arms.is_empty() && !cli.arms.iter().any(|a| a == arm.name) {
            continue;
        }
        eprintln!("\n=== {} — {}", arm.name, arm.rationale);
        let head_cfg = (arm.apply)(base.clone(), &cli.params);
        let head: KanHead<B> = head_cfg.init(&device);

        // 1. init fields over all of CONUS.
        let init = head.valid().forward(x_full.clone().inner());
        write_fields(
            &cli.out_dir.join(format!("{}.init.bin", arm.name)),
            &init,
            &cli.params,
            n_reaches,
        )?;

        // 2. trunk activations on the subsample.
        let h = head.valid().trunk_activations(x_sub.clone().inner());
        let hd = h.dims();
        write_f32(
            &cli.out_dir.join(format!("{}.trunk.bin", arm.name)),
            &h.into_data().to_vec::<f32>().map_err(|e| format!("{e:?}"))?,
        )?;

        // 3. capacity positive control.
        let (fitted, loss_trace) = fit_capacity::<B>(head, x_sub.clone(), targets.clone(), &cli);
        write_fields(
            &cli.out_dir.join(format!("{}.fitted.bin", arm.name)),
            &fitted,
            &cli.params,
            idx.len(),
        )?;

        manifest.insert(
            arm.name.to_string(),
            serde_json::json!({
                "rationale": arm.rationale,
                "hidden_size": head_cfg.hidden_size,
                "num_hidden_layers": head_cfg.num_hidden_layers,
                "parameter_groups": head_cfg.parameter_groups,
                "input_layer_kan": head_cfg.input_layer_kan,
                "output_layer_kan": head_cfg.output_layer_kan,
                "trunk_shape": [hd[0], hd[1]],
                "fit_loss_first": loss_trace.first(),
                "fit_loss_last": loss_trace.last(),
                "loss_trace": loss_trace,
            }),
        );
        eprintln!(
            "  capacity fit: {:.6} -> {:.6}",
            loss_trace.first().copied().unwrap_or(f32::NAN),
            loss_trace.last().copied().unwrap_or(f32::NAN)
        );
    }

    let meta = serde_json::json!({
        "config": cli.config,
        "params": cli.params,
        "n_reaches": n_reaches,
        "n_features": n_feat,
        "subsample_index_stride": stride,
        "subsample_len": idx.len(),
        "fit_steps": cli.fit_steps,
        "lr": cli.lr,
        "target_corr": target_corr,
        "input_var_names": section.input_var_names,
        "arms": manifest,
    });
    let mut f = std::fs::File::create(cli.out_dir.join("manifest.json"))?;
    f.write_all(serde_json::to_string_pretty(&meta)?.as_bytes())?;
    eprintln!("\nwrote {}", cli.out_dir.display());
    Ok(())
}

/// `[len(rows), F]` tensor of the selected attribute rows.
fn to_tensor<B: Backend>(attrs: &Array2<f32>, rows: &[usize], device: &B::Device) -> Tensor<B, 2> {
    let f = attrs.shape()[1];
    let mut v = Vec::with_capacity(rows.len() * f);
    for &r in rows {
        for c in 0..f {
            v.push(attrs[(r, c)]);
        }
    }
    Tensor::from_data(TensorData::new(v, [rows.len(), f]), device)
}

/// Two targets in `(0, 1)` that are each an exact function of the attributes
/// and are mutually uncorrelated.
///
/// Built from `meanslope` and `log10_uparea` when present (the two attributes
/// §30 identified as carrying the timing and the scale channel respectively),
/// falling back to the first two columns. The second is Gram-Schmidt
/// orthogonalised against the first, so any residual correlation in the
/// *targets* is zero to numerical precision and every correlation measured in
/// the fitted outputs is the architecture's doing.
fn synthetic_targets(
    attrs: &Array2<f32>,
    rows: &[usize],
    names: &[String],
) -> (Vec<f32>, Vec<f32>, f64) {
    let pick = |want: &str, fallback: usize| -> usize {
        names.iter().position(|n| n == want).unwrap_or(fallback)
    };
    let (ca, cb) = (pick("meanslope", 0), pick("log10_uparea", 1));

    let a: Vec<f64> = rows.iter().map(|&r| attrs[(r, ca)] as f64).collect();
    let b: Vec<f64> = rows.iter().map(|&r| attrs[(r, cb)] as f64).collect();
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let saa: f64 = a.iter().map(|x| (x - ma).powi(2)).sum();
    let sab: f64 = a.iter().zip(&b).map(|(x, y)| (x - ma) * (y - mb)).sum();
    let beta = sab / saa;

    // b_perp is b with all of a projected out.
    let b_perp: Vec<f64> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (y - mb) - beta * (x - ma))
        .collect();

    let std = |v: &[f64]| -> f64 {
        let m = v.iter().sum::<f64>() / v.len() as f64;
        (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
    };
    let (sa, sb) = (std(&a), std(&b_perp));
    // Scale to unit variance then squash: sigmoid keeps the targets inside the
    // head's own output range so MSE is a fair ask.
    let sig = |z: f64| 1.0 / (1.0 + (-z).exp());
    let t1: Vec<f32> = a.iter().map(|x| sig((x - ma) / sa) as f32).collect();
    let t2: Vec<f32> = b_perp.iter().map(|x| sig(x / sb) as f32).collect();

    // Report the correlation of the squashed targets — sigmoid is monotone but
    // not linear, so this is small rather than exactly zero.
    let c = |u: &[f32], v: &[f32]| -> f64 {
        let (n, mu, mv) = (
            u.len() as f64,
            u.iter().map(|x| *x as f64).sum::<f64>() / u.len() as f64,
            v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64,
        );
        let num: f64 = u
            .iter()
            .zip(v)
            .map(|(x, y)| (*x as f64 - mu) * (*y as f64 - mv))
            .sum();
        let du: f64 = u.iter().map(|x| (*x as f64 - mu).powi(2)).sum::<f64>().sqrt();
        let dv: f64 = v.iter().map(|y| (*y as f64 - mv).powi(2)).sum::<f64>().sqrt();
        let _ = n;
        num / (du * dv)
    };
    let corr = c(&t1, &t2);
    (t1, t2, corr)
}

/// Fit the head to the two synthetic targets and return the final fields plus
/// the loss trace. Only the first two parameters are supervised; any others
/// float free, which is deliberate — the question is whether two supervised
/// outputs can be decorrelated, not whether all three can.
fn fit_capacity<B: AutodiffBackend>(
    mut head: KanHead<B>,
    x: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    cli: &Cli,
) -> (HashMap<String, Tensor<B::InnerBackend, 1>>, Vec<f32>)
where
    B::InnerBackend: Backend<Device = B::Device>,
{
    let mut optim = AdamConfig::new().init();
    let mut trace = Vec::with_capacity(cli.fit_steps / 50 + 2);
    let supervised: Vec<String> = cli.params.iter().take(2).cloned().collect();
    let n = x.dims()[0];

    for step in 0..cli.fit_steps {
        let out = head.forward(x.clone());
        let mut loss: Option<Tensor<B, 1>> = None;
        for (j, key) in supervised.iter().enumerate() {
            let pred = out[key].clone();
            let tgt: Tensor<B, 1> = targets
                .clone()
                .slice([0..n, j..j + 1])
                .reshape([n]);
            let term = (pred - tgt).powf_scalar(2.0).mean();
            loss = Some(match loss {
                Some(l) => l + term,
                None => term,
            });
        }
        let loss = loss.expect("at least one supervised parameter");
        if step % 50 == 0 || step == cli.fit_steps - 1 {
            trace.push(loss.clone().into_scalar().elem::<f32>());
        }
        let grads = GradientsParams::from_grads(loss.backward(), &head);
        head = optim.step(cli.lr, head, grads);
    }

    (head.valid().forward(x.inner()), trace)
}

/// Row-major `[N, P]` f32, parameters in the given order.
fn write_fields<B: Backend>(
    path: &Path,
    fields: &HashMap<String, Tensor<B, 1>>,
    params: &[String],
    n: usize,
) -> R<()> {
    let cols: Vec<Vec<f32>> = params
        .iter()
        .map(|k| {
            fields[k]
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("f32 field")
        })
        .collect();
    let mut out = Vec::with_capacity(n * params.len());
    for i in 0..n {
        for c in &cols {
            out.push(c[i]);
        }
    }
    write_f32(path, &out)
}

fn write_f32(path: &Path, v: &[f32]) -> R<()> {
    let mut f = std::fs::File::create(path)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    f.write_all(&bytes)?;
    Ok(())
}
