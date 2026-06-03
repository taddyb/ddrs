# DDR ↔ DDRS KAN parity testing plan

**Date:** 2026-06-02
**Branch:** `kan_improvements`
**Symptom:** Trained Manning's `n` distribution is centred ≈ 0.02–0.03 m⁻¹⋅⅓·s
across CONUS, hugging the lower bound of the `[0.015, 0.25]` log-space range.
That implies the KAN head's sigmoid output is centred ≲ 0.1 — either a real
training pathology or a divergence between DDR-Python's `kan` and the
DDRS port (`KanHead` + `rskan::KanLayer`).

This spec is the **testing plan** that will distinguish "the head is wired
correctly and the data/optimizer is to blame" from "the head itself is
diverging." It does NOT propose any code fix yet — the fix follows whichever
test layer first reports `MISMATCH`.

---

## 1. Why this matters / why now

The whole `rskan-head-swap` value proposition is that the routing parameters
that DDR learns and the routing parameters that DDRS learns come from the
**same function class with the same initial distribution**. Gradient parity in
the routing core (`compare_ddr_sandbox`) is already proven at f32 floor —
but if the parameter head diverges, every downstream metric (NSE, KGE, the
boxplots produced by `ddrs-eval-plots`) reflects a different model than DDR,
not a different solver. We lose the ability to attribute regressions.

The earliest place to catch a head divergence is **at the moment of
initialization**, before any training noise enters the picture. A
distributional divergence at init multiplies through 5 epochs of training
and is hard to disentangle from optimizer state, batch order, or numerical
drift in the routing pass.

---

## 2. Concerns

| # | Concern | Why it could go wrong |
|---|---------|----------------------|
| C1 | Test plan demands fixture exchange across Python and Rust runtimes (numpy `.npz` files written from DDR's `uv` venv, read in `ddrs-py`/Rust tests). | Schema drift between fixture writer and reader silently passes if shapes happen to match; mitigated by writing a small validator that prints fixture metadata and a `version` tag. |
| C2 | `rskan::KanLayer` exposes `init_from_parts` (per `layer.rs:55-108`) but `ddrs::nn::kan_head::KanHead` does not have a public `from_parts` constructor. | We will need to add a `KanHeadConfig::init_from_fixture(...)` that does the same for the surrounding `Linear` layers. Small but new public API — keep it `#[cfg(test)]`-gated if possible, or guard behind a `fixtures` feature flag, to avoid leaking a test-only path into the production surface. |
| C3 | We are about to discover that DDR's production `merit_training_config.yaml` uses `grid: 50, k: 2` while DDRS's mirror config uses `grid: 5, k: 3`. The DDRS comment at `config/merit_training.yaml:42` falsely claims "Match DDR's default". | This may be the root cause of the saturation symptom on its own. The test plan must verify under BOTH config settings, and the spec MUST be explicit about which we treat as ground truth before any fix lands. **Default position in this spec:** DDR's production YAML is ground truth → DDRS should move to `grid: 50, k: 2`. |
| C4 | rskan's own docs (`layer.rs:113-115`) state RNG bit-parity with pykan is impossible (`StdRng` vs Mersenne-Twister). | All parity at the *random sampling* layer is statistical, not bitwise. Any test that demands bitwise parity must source the parameters from a fixture, not from a seed. Plan reflects this — bitwise tests use fixtures; seed-based tests assert moments + KS statistics within tolerance. |
| C5 | The pykan code path has un-documented side effects from `MultKAN.__init__` (calls `torch.manual_seed`, `np.random.seed`, `random.seed` — `MultKAN.py:158-160`). | DDR's `kan.py:24-43` creates **one `KAN([H,H])` per outer hidden layer**, each call re-seeding global Torch/Numpy RNG to the *same* `seed`. So both inner blocks have **identical** coef / scale_base / scale_sp tensors. DDRS preserves this by reusing the same `seed` per inner `KanLayer`. Tests must explicitly check that `hidden[0].coef == hidden[1].coef` in both ports. |
| C6 | Forward parity tests assume `silu`, `extend_grid`, `coef2curve`, `curve2coef` match between rskan and pykan elementwise. rskan ships parity tests internally (`tests/parity_forward.rs`, `tests/parity_backward.rs`) but our test layer needs to confirm this is also true on the *backend we actually train on* (CUDA via `burn-cuda`), not just the `NdArray` backend rskan tests use. | Mitigated by running the fixture-forward test on both `NdArray` (CPU) and `LibTorch-CUDA` / `Cuda` backends — same fixture, same expected output. If the CUDA backend diverges and CPU doesn't, the bug is in `burn-cuda` kernels, not in the KAN code. |
| C7 | Burn 0.21's `Initializer::KaimingNormal` / `XavierNormal` may draw from `rand::thread_rng()` rather than a seedable per-init RNG. If so, DDRS's *Linear* weights are non-reproducible across runs at fixed `seed=42` — independent of any DDR parity question. | Discovered during the 0.5 audit (see "Are `input.weight` and `output.weight` reproducible…" row). If true, the fix is to bypass burn's initializer for `input` and `output` Linears and sample weights via a project-controlled `StdRng` (the same one rskan uses), then `Param::from_tensor(...)`. This would also make the embedding / head initialization use the **same RNG family** as the KAN blocks — eliminating one more source of cross-module drift. |

---

## 3. Assumptions

| # | Assumption | Justification |
|---|------------|---------------|
| A1 | DDR's `~/projects/ddr/config/merit_training_config.yaml` is the canonical "what DDR actually trains" config. | It's the file `ddr/scripts/train.py` reads at runtime (verified by inspection: `kan.grid=50, kan.k=2`). |
| A2 | The `seed: 42` / `np_seed: 42` in both configs is the right seed for parity runs. | Both DDRS's and DDR's configs already pin to 42; changing it for the parity test would muddy the comparison against existing training artefacts. |
| A3 | We can run DDR's reference notebook + a new "dump init params" script under DDR's existing `uv` venv at `~/projects/ddr/.venv/`. | Already standard practice; `~/projects/ddr/scripts/export_ddr_sandbox.py` is the established precedent. |
| A4 | A fixture covering input width 10, hidden 21, output 3, `num_hidden_layers=2` (i.e. the production shape from `merit_training.yaml`) is sufficient — we don't need to parameterize across many architectures. | The production shape is the only one anyone trains in. Smaller fixtures (in_dim=3, hidden=4) are useful for unit testing rskan itself but rskan already has those; DDR-vs-DDRS parity tests should target the actual model. |
| A5 | Statistical init parity within ≤ 5% relative tolerance on first four moments (mean, std, skew, kurtosis) and KS-statistic < 0.05 vs. 50,000-sample empirical CDF is "close enough" to call the distributions equivalent. | KS = 0.05 is the conventional bound for "indistinguishable at n ≈ 50k samples"; it bounds the L∞ between empirical CDFs. Tighter than that crosses into bit-parity territory which we already said is impossible across RNGs. |
| A6 | The `denormalize` path (`src/config.rs`) is correct and matches DDR's. | Validated in `tests/training_verification.rs` (the `loads_merit_training_yaml` block) and `ddrs-py/tests/smoke.py:test_denormalize_*`. Out of scope for this plan unless a test below points at it. |

---

## 4. Layered test plan

Five layers, ordered cheapest → most expensive. Each layer answers a
yes/no question; failures route to a specific module to investigate.

### Layer 0 — Exhaustive hyperparameter + initialization audit (no code; 1 hour)

**Question:** Does *every* declared and inherited hyperparameter — for the
embedding Linear, the KAN stack, and the head Linear — match between DDR
and DDRS, including all initializers, their modes / gains / nonlinearities,
all bias treatments, all rskan / pykan defaults that DDRS overrides, and
all rskan / pykan defaults that DDRS *doesn't* override (where the two
libraries pick different defaults)?

**Procedure:** produce four tables — one per sub-module — auditing every
field that affects the module's parameter tensors or its forward pass.
Where DDR's value is "torch default", quote the exact PyTorch source line.
Where DDRS's value is "burn default", quote the burn 0.21 source line.
Fill the **Match?** column with ✓ / ✗ / "STAT only" (statistically equivalent
but not bit-equivalent across the two RNGs — acceptable per A5 / C4).

#### 0.1 — Architectural layer counts and widths

| Field | DDR source | DDRS source | DDR value | DDRS value | Match? |
|-------|------------|-------------|-----------|------------|--------|
| `input_size` (F) | `kan.py:26` (from `len(input_var_names)`) | `kan_head.rs:48`, sized from `input_var_names.len()` | `len(merit_training_config.yaml::kan.input_var_names) = 10` | `len(merit_training.yaml::kan_head.input_var_names) = 10` | ✓ |
| `hidden_size` (H) | `kan.py:27` | `kan_head.rs:58` | `21` | `21` | ✓ |
| `num_hidden_layers` | `kan.py:33` (loop bound) | `kan_head.rs:62` | `2` | `2` | ✓ |
| `output_size` (P) | `kan.py:29` (from `len(learnable_parameters)`) | `kan_head.rs:50` | `len(["n","q_spatial","p_spatial"]) = 3` | `3` | ✓ |
| `input_var_names` (ordered) | YAML | YAML | (10 names) | (10 names) | ✓ — `diff <(yq '.kan.input_var_names[]' …) <(yq '.kan_head.input_var_names[]' …)` is empty; both lists are identical in content and order |
| `learnable_parameters` (ordered) | YAML | YAML | `["n","q_spatial","p_spatial"]` | `["n","q_spatial","p_spatial"]` | ✓ |

#### 0.2 — Embedding (`input: Linear(F, H)`)

| Field | DDR value (with src) | DDRS value (with src) | Match? |
|-------|----------------------|-----------------------|--------|
| Weight shape | `[H, F] = [21, 10]` | `[H, F] = [21, 10]` | ✓ |
| Weight initializer | `kaiming_normal_(weight, nonlinearity="relu")` (`kan.py:45`) | `Initializer::KaimingNormal { gain: sqrt(2), fan_out_only: false }` (`kan_head.rs:85-88`) | STAT only |
| Kaiming `mode` | `fan_in` (torch default at `nn/init.py:581`) | `fan_in` (burn `fan_out_only: false`) | ✓ |
| Kaiming `gain` (effective) | `sqrt(2)` from `nonlinearity="relu"` (`nn/init.py:calculate_gain`) | `KAIMING_GAIN_RELU = sqrt(2)` (`kan_head.rs:36`) | ✓ |
| Resulting weight std | `sqrt(2) / sqrt(F) = sqrt(2/10) ≈ 0.447` | same formula | ✓ analytically |
| Weight RNG source | PyTorch Mersenne-Twister (global), seeded by `torch.manual_seed(seed)` inside `MultKAN.__init__` (`MultKAN.py:158`) | burn's `Initializer::KaimingNormal` calls `rand::thread_rng()` internally — **not** seeded by `KanHeadConfig.seed`; confirmed non-reproducible by empirical probe (Task 2, 2026-06-03) | ✗ STAT only — formulas agree, but DDRS's Linear weights are non-deterministic across runs even at fixed seed. Tracked as C7; fix in Task 3. |
| Bias shape | `[H] = [21]` | `[H] = [21]` | ✓ |
| Bias initializer (pre-zero) | torch default `Linear` bias = `U(-1/sqrt(F), 1/sqrt(F))` (`nn/modules/linear.py`) | `LinearConfig::init` re-uses the weight initializer for bias unless overridden — so KaimingNormal(sqrt(2)) | irrelevant — overwritten |
| Bias post-init | `torch.nn.init.zeros_(self.input.bias)` (`kan.py:47`) | `zero_bias(input, device)` (`kan_head.rs:116`) | ✓ |

**Action items extracted from 0.2:**
- ~~Determine which RNG burn 0.21 uses when sampling `KaimingNormal`.~~ **RESOLVED (Task 2, 2026-06-03):** burn's `Initializer` uses `rand::thread_rng()` — not seeded by `KanHeadConfig.seed`. The Layer 1 statistical test is unaffected (we only assert moments), but Layer 2 fixture loading bypasses the initializer entirely via `Param::from_tensor(...)`. The reproducibility fix is Task 3 (project-controlled `StdRng`).

#### 0.3 — KAN stack (`hidden: Vec<KanLayer(H, H)> × num_hidden_layers`)

DDR side: each block is a `pykan.KAN([H, H], k=k, grid=grid, seed=seed)` —
i.e. `MultKAN(width=[H, H], ...)`, which internally constructs **one**
`KANLayer(in_dim=H, out_dim=H, ...)`. DDRS side: each block is one
`rskan::KanLayer(H, H, seed)`. Field comparison below is therefore between
pykan's `KANLayer` defaults (via `MultKAN`'s call path) and rskan's
`KanLayerConfig` defaults.

| Field | DDR / pykan source | DDRS / rskan source | DDR value | DDRS value | Match? |
|-------|--------------------|---------------------|-----------|------------|--------|
| `in_dim` | `MultKAN.py:214` (`width_in[l]`) | `kan_head.rs:102` | `H = 21` | `H = 21` | ✓ |
| `out_dim` | `MultKAN.py:214` (`width_out[l+1]`) | `kan_head.rs:102` | `H = 21` | `H = 21` | ✓ |
| `num` / `grid` | YAML `kan.grid` → `MultKAN.__init__(grid=…)` → KANLayer | YAML `kan_head.grid` → `KanLayerConfig.with_num` | **50** | **50** | ✓ — fixed in Task 1; `config/merit_training.yaml` updated to `grid: 50` |
| `k` | YAML `kan.k` → KANLayer | YAML `kan_head.k` → `with_k` | **2** | **2** | ✓ — fixed in Task 1; `config/merit_training.yaml` updated to `k: 2` |
| `noise_scale` | `MultKAN.__init__` default 0.3 (`MultKAN.py:96`) passed to `KANLayer(noise_scale=0.3)` (`MultKAN.py:214`) | `KAN_NOISE_SCALE = 0.3` (`kan_head.rs:40`) → `with_noise_scale(0.3)` | 0.3 | 0.3 | ✓ |
| `scale_base_mu` | `MultKAN.__init__` default `0.0` (`MultKAN.py:96`) → KANLayer | rskan default `0.0` (`layer.rs:43`) — **DDRS does not override** | 0.0 | 0.0 | ✓ |
| `scale_base_sigma` | `MultKAN.__init__` default `1.0` (`MultKAN.py:96`) → KANLayer | rskan default `1.0` (`layer.rs:44`) — **DDRS does not override** | 1.0 | 1.0 | ✓ |
| `scale_sp` | KANLayer default `1.0` (MultKAN passes `scale_sp=1.` explicitly, `MultKAN.py:214`) | rskan default `1.0` (`layer.rs:45`) — DDRS does not override | 1.0 | 1.0 | ✓ |
| `grid_range` | KANLayer default `[-1, 1]` (MultKAN passes `grid_range` from its own default `[-1, 1]`, `MultKAN.py:96`) | rskan default `[-1.0, 1.0]` (`layer.rs:46`) | `[-1, 1]` | `[-1, 1]` | ✓ |
| `sp_trainable` | KANLayer default `True` (MultKAN passes `True`, `MultKAN.py:96`) | rskan default `true` (`layer.rs:47`) | true | true | ✓ |
| `sb_trainable` | KANLayer default `True` (MultKAN passes `True`, `MultKAN.py:96`) | rskan default `true` (`layer.rs:48`) | true | true | ✓ |
| `grid_eps` | KANLayer default `0.02` (`KANLayer.py:44`) | **rskan does not expose** — `extend_grid` always uniform | 0.02 (irrelevant; only used by `update_grid_from_samples`) | N/A (we never call grid-update) | ✓ by exclusion |
| `base_fun` | KANLayer default `nn.SiLU()` (`KANLayer.py:44`) | rskan hard-codes `silu` (`layer.rs:192`) | SiLU | SiLU | ✓ |
| `sparse_init` | MultKAN default `False` (`MultKAN.py:96`) | rskan does not expose (mask = ones, `layer.rs:143`) | False → mask=ones | mask=ones | ✓ |
| `mask` | `torch.ones(in_dim, out_dim)` (`KANLayer.py:108`) | `NdArray2::<f32>::ones(...)` (`layer.rs:143`) | ones | ones | ✓ |
| `seed` (per-block) | `MultKAN.__init__(seed=seed)` re-seeds **all global RNGs** to `seed` (`MultKAN.py:158-160`) — DDR loops the KAN constructor in `kan.py:33-42` with the same `seed` for every block → both blocks initialised from identical RNG state | DDRS reuses `self.seed` for every inner `KanLayer` (`kan_head.rs:97-108`) — both blocks initialised from identical seed | both blocks identical | both blocks identical | ✓ (validate via `assert hidden[0].coef == hidden[1].coef` per port) |
| RNG used | torch / numpy / python global Mersenne-Twister | `rand::rngs::StdRng` (`init.rs:11`) | MT19937 | StdRng | STAT only (per C4) |

**Items that need follow-up before Layer 1 runs:**
- (none — every value is either ✓ or STAT-only).

#### 0.4 — Head (`output: Linear(H, P)`)

| Field | DDR value (with src) | DDRS value (with src) | Match? |
|-------|----------------------|-----------------------|--------|
| Weight shape | `[P, H] = [3, 21]` | `[P, H] = [3, 21]` | ✓ |
| Weight initializer | `xavier_normal_(weight, gain=0.1)` (`kan.py:46`) | `Initializer::XavierNormal { gain: 0.1 }` (`kan_head.rs:89-91`) | STAT only |
| Xavier formula | `std = gain * sqrt(2 / (fan_in + fan_out)) = 0.1 * sqrt(2/24) ≈ 0.0289` | same formula | ✓ analytically |
| Bias shape | `[P] = [3]` | `[P] = [3]` | ✓ |
| Bias post-init | `torch.nn.init.zeros_(self.output.bias)` (`kan.py:48`) | `zero_bias(output, device)` (`kan_head.rs:117`) | ✓ |
| Post-Linear nonlinearity | `F.sigmoid(_x)` (`kan.py:58`) | `sigmoid(logits)` (`kan_head.rs:157`) | ✓ |
| Output reshape | `_x.transpose(0, 1)` then index `x_transpose[idx]` per key (`kan.py:59-61`) | `probs.swap_dims(0, 1)` then `slice([idx..idx+1, 0..n]).reshape([n])` per key (`kan_head.rs:170-178`) | ✓ (verify by Layer 2) |

#### 0.5 — Top-level construction order and seed handling

| Field | DDR | DDRS | Match? |
|-------|-----|------|--------|
| Order of sub-module construction | `Linear input → loop(KAN blocks) → Linear output → init.kaiming_normal_ → init.xavier_normal_ → init.zeros_(input.bias) → init.zeros_(output.bias)` (`kan.py:31-48`) | `Linear input → loop(KanLayer blocks) → Linear output → zero_bias(input) → zero_bias(output)` (`kan_head.rs:93-117`) | ✓ for module order; **DDR's** seed-reset side effect from `MultKAN(seed=)` (`MultKAN.py:158-160`) re-seeds Torch / NumPy globals so the Linears initialised **after** the KAN loop draw from `seed`-derived state, while DDRS's `Initializer::KaimingNormal` and `XavierNormal` use burn-internal RNG independent of the KAN seed. **STAT only**, but flag it. |
| Are `input.weight` and `output.weight` reproducible given just `seed=42`? | Yes (after `MultKAN` re-seeds globals; the *order* of layer construction matters) | No — burn's initializers depend on whatever RNG burn picks; need to control that to make DDRS init repro across runs | ✗ — empirical probe shows different bytes across two consecutive `cfg.init()` calls at fixed seed=42 (Task 2 verification, 2026-06-03). `h1.input.weight[0..5] = [-0.5793, 0.0344, 0.2894, -0.5388, -0.6762]` vs `h2.input.weight[0..5] = [0.3699, 0.1127, -0.2515, -0.4630, -0.0366]`. burn 0.21 `Initializer::KaimingNormal` / `XavierNormal` call `rand::thread_rng()` — not a seedable per-init RNG. Task 3 must replace burn's Initializer with a project-controlled `StdRng` seeded from `KanHeadConfig.seed` for both Linear layers. |

**Pass criterion:** Every row in tables 0.1 – 0.5 is ✓ or STAT-only. The ❌
in 0.3 (`grid`, `k`) were resolved in Task 1. The ❌ in 0.5 (C7, Linear RNG
reproducibility) is tracked; it does not block Layer 1 (moments-only) but
must be fixed (Task 3) before Layer 2 fixture loading can assert bit parity.

**Failure routing:**
- Any row in 0.1, 0.2, 0.4, or 0.5 column "Match?" comes back ❌ that isn't
  already tracked → file a bug, fix before Layer 1.
- ~~Any "investigate" row~~ All "investigate" rows resolved in Task 2 (2026-06-03).

### Layer 1 — Statistical init parity (~1 day)

**Question:** With matching architecture and `seed=42`, do the *distributions*
of every initialized tensor match between DDR and DDRS, even though the
exact bytes differ (per C4)?

**Procedure:**
1. **Reference path (DDR):** new script
   `~/projects/ddrs/scripts/dump_kan_init_stats.py`, run under DDR's venv:
   - Build `ddr.nn.kan.kan(...)` with the parity config (Layer 0 §pass).
   - For each parameter tensor (`input.weight`, `input.bias`, `output.weight`,
     `output.bias`, and for each of the two `layers[i]`: `grid`, `coef`,
     `scale_base`, `scale_sp`, `mask`):
     - Write its shape, mean, std, min, max, abs-mean to one row of a CSV
       at `tests/fixtures/kan_init_stats_ddr.csv`.
2. **Port path (DDRS):** new integration test `tests/kan_head_init_parity.rs`
   - Build `KanHead<NdArrayBackend>` with the same parity config + seed=42.
   - Compute the same per-tensor statistics.
   - Read `kan_init_stats_ddr.csv`.
   - Assert per-tensor: `rel_err(mean) ≤ 5e-2`, `rel_err(std) ≤ 5e-2`,
     shape identical.
3. (Bonus) KS test on the flattened tensor vs. DDR-side using
   `ndarray-stats` or a small inline two-sample KS: assert `D < 0.05`.

**Pass criterion:** All tensors match in shape + first two moments + KS bound.

**Failure routing:**
- `input.weight` mismatch → burn's `KaimingNormal` vs torch's `kaiming_normal_`
  diverge → file rskan-independent bug.
- `output.weight` mismatch → `XavierNormal` divergence.
- `coef` / `scale_base` mismatch → rskan vs pykan formula drift (re-read
  `rskan/src/init.rs` against `KANLayer.py:98-112`).
- `hidden[0].coef != hidden[1].coef` in *either* port → the MultKAN
  re-seeding quirk is not being preserved.

### Layer 2 — Fixture-exchange forward parity (~2 days)

**Question:** Given **bit-identical initial parameters** loaded from a
fixture, does DDRS's forward pass produce bit-identical outputs to DDR's?

**Procedure:**
1. **Dumper (Python, DDR venv):** new script
   `~/projects/ddrs/scripts/dump_kan_fixture.py`:
   - Build `ddr.nn.kan.kan(...)` with parity config + `seed=42`.
   - Sample `inputs` with `torch.manual_seed(0); torch.randn(64, 10)` (64
     reaches, 10 attributes — the production shape).
   - Run `outputs = model(inputs=inputs)` (returns
     `{"n": [64], "q_spatial": [64], "p_spatial": [64]}`).
   - Save a single `.npz`:
     - `inputs` `[64, 10] float32`
     - `expected_n`, `expected_q_spatial`, `expected_p_spatial` `[64] float32`
     - `input_weight` `[H, F]`, `input_bias` `[H]`,
       `output_weight` `[P, H]`, `output_bias` `[P]`
     - For each block `b ∈ {0, 1}`:
       `block_{b}_grid` `[H, knots]`, `block_{b}_coef` `[H, H, n_basis]`,
       `block_{b}_scale_base` `[H, H]`, `block_{b}_scale_sp` `[H, H]`,
       `block_{b}_mask` `[H, H]`
     - `meta` json: `{"version": 1, "in": 10, "hidden": 21, "out": 3,
        "grid": 50, "k": 2, "num_hidden_layers": 2, "seed": 42}`
   - Output path: `tests/fixtures/kan_head_init_seed42.npz`.
2. **Loader (Rust, DDRS):** new helper in `src/nn/kan_head.rs`:
   ```rust
   #[cfg(feature = "fixtures")]
   impl<B: Backend> KanHead<B> {
       pub fn from_npz(path: &std::path::Path, device: &B::Device, cfg: &KanHeadConfig)
           -> Result<Self, std::io::Error> { … }
   }
   ```
   Uses `ndarray-npy` (already in deps for zarrs path) to read each array,
   then calls `KanLayerConfig::init_from_parts(…)` for each block, builds
   `Linear` layers by `Param::from_tensor(…)`. Bias tensors materialized
   from `input_bias`/`output_bias` directly (no `Initializer`).
3. **Test:** `tests/kan_head_fixture_forward.rs`
   ```rust
   #[test]
   fn forward_matches_ddr_fixture() {
       let device = Default::default();
       let head = KanHead::<NdArrayBackend>::from_npz(
           Path::new("tests/fixtures/kan_head_init_seed42.npz"),
           &device,
           &PARITY_CONFIG,
       ).unwrap();
       let inputs = … // from fixture
       let out = head.forward(inputs);
       for key in ["n", "q_spatial", "p_spatial"] {
           let got = out[key].to_data();
           let want = … // from fixture
           assert_max_abs_diff(&got, &want, 1e-6);  // f32 floor
       }
   }
   ```
4. Repeat the test on the CUDA backend (gated `#[cfg(feature = "cuda")]`) with
   `1e-4` tolerance (cuBLAS GEMM accumulates differently than CPU MKL).

**Pass criterion:** Max abs diff ≤ 1e-6 on `NdArray`, ≤ 1e-4 on CUDA, across
all three output keys, all 64 reaches.

**Failure routing:**
- Off by a constant scale → `silu` vs `silu` definition drift, or
  `coef2curve` normalisation drift.
- Off by a permutation → the `[P, N]` swap in `kan_head.rs:170` is wrong
  axis, or pykan's transpose convention differs.
- Off by a per-block factor → `mask` is being broadcast on the wrong axis.

### Layer 3 — Fixture-exchange gradient parity (~1 day)

**Question:** Given bit-identical params + inputs, are the gradients w.r.t.
each fixture parameter bit-identical to DDR's?

**Procedure:**
1. Extend the dumper script to also save `expected_grad_<param>` for every
   trainable parameter, computed by `loss = outputs["n"].sum() +
   outputs["q_spatial"].sum() + outputs["p_spatial"].sum(); loss.backward()`.
2. New test `tests/kan_head_fixture_backward.rs` wraps the head in
   `Autodiff<NdArrayBackend>`, runs the same loss, calls `loss.backward()`,
   reads `head.input.weight.grad(&grads)`, etc., asserts max abs diff.
3. Tolerance: `1e-5` on `NdArray-Autodiff`, `1e-3` on `Autodiff<Cuda>`.

**Pass criterion:** All gradients within tolerance.

**Failure routing:** This is where a custom `Backward` bug in rskan (if any)
would surface. rskan's own `tests/parity_backward.rs` already covers this
in isolation, but our test is the first to cover the **Linear→KAN×N→Linear→
sigmoid→swap_dims→slice** composition that DDRS uses for the head.

### Layer 4 — End-to-end CONUS init distribution (~half day)

**Question:** After init only (zero training), does DDRS's `dump_parameters`
produce the same CONUS-wide histogram of `n`, `q_spatial`, `p_spatial` as
DDR's `examples/merit/plot_parameter_map.ipynb` does for a freshly
constructed `kan` model?

**Procedure:**
1. Add a CLI flag `ddrs run --workflow dump-init` (or simpler: a one-shot
   `cargo run --release --example dump_init_params` that builds `KanHead`
   without loading a checkpoint, sweeps the full 346 321-reach attribute
   tensor through it, denormalises, writes `kan_init_params.nc`).
2. Add a parallel DDR script that does the same thing using `ddr.nn.kan.kan`.
3. Compare the histograms on each of the three parameters with the
   per-bin median annotation already implemented in
   `.claude/skills/ddrs-eval-plots/references/parameter_map.md`. Use the
   existing histogram cell from `parameter_map_n_conus.ipynb` as the
   plotting recipe.

**Pass criterion:** Visual + numerical: per-decile values within 5% relative.
KS test on the flattened distributions: `D < 0.05`.

**This is the test that answers the user's actual question** — "is the
saturation at 0.02–0.03 a real training pathology, or is the head different
at t=0?" If Layer 4 passes (init distributions agree) but trained
distributions diverge → it's a training-dynamics problem (optimizer,
batching, loss). If Layer 4 fails → the head itself is different and
Layers 1–3 already told you where.

---

## 5. Decision required before implementation: which parity config do we pin?

The DDR/DDRS `grid` and `k` divergence (C3) has to be resolved *before* we
generate any fixtures. Two paths:

| Option | Pro | Con |
|--------|-----|-----|
| **A — Move DDRS to `grid: 50, k: 2`** to match DDR's production config | DDRS becomes the actual port, not an architectural fork. Fixtures generated once stay valid for all future parity work. Fixes a latent comment lie at `config/merit_training.yaml:42`. | More KAN coefficients per layer → very small init-time and forward-time cost increase. Re-runs all training (no checkpoint compatibility). |
| **B — Keep DDRS at `grid: 5, k: 3`** and change DDR for the parity run only | No DDRS code/config change. | Fork — DDRS no longer matches the file it claims to mirror. Comment in `config/merit_training.yaml:42` stays a lie. Likely the cause of the saturation symptom — choosing this option leaves the symptom unaddressed even if all five parity layers pass. |

**Default recommendation in this spec: Option A.** It is the only choice
consistent with the project's stated invariant ("KAN head matches DDR-Python
exactly", `CLAUDE.md:8`).

---

## 6. Implementation order

1. (One-line) edit `config/merit_training.yaml` to `grid: 50, k: 2`,
   removing the false "Match DDR's default" comment. Run
   `cargo test --test kan_head` and the existing
   `loads_merit_training_yaml` smoke test (the latter is already failing
   per session notes — confirm the failure is independent of this change
   before continuing). **Re-train one checkpoint at the new config so the
   saturation hypothesis can be tested directly.**
2. Land Layer 0 audit as a table in this spec's PR description (no code).
3. Layer 1 — `dump_kan_init_stats.py` + `tests/kan_head_init_parity.rs`.
4. Layer 2 — extend dumper to fixtures, add `KanHead::from_npz`,
   `tests/kan_head_fixture_forward.rs` (NdArray then Cuda).
5. Layer 3 — gradient-fixture extension, `tests/kan_head_fixture_backward.rs`.
6. Layer 4 — `dump_init_params` example + comparison notebook.

Layers 1 and 2 are the load-bearing ones for the symptom; layers 3-4 are
for confidence in the long-term parity invariant.

---

## 7. What success looks like

After the plan ships:

- A single `cargo test --test kan_head_fixture_forward` run reproduces a
  bit-identical (within f32 floor) forward pass of the DDR KAN head.
- A single `cargo test --test kan_head_fixture_backward` run reproduces
  gradient parity.
- The CONUS init-distribution comparison shows DDR and DDRS produce the
  same `n` histogram before any training. Whichever direction the trained
  histogram drifts becomes diagnosable.
- The "0.02–0.03 centred" pathology either:
  (a) **disappears** after changing `grid/k` to match DDR (root cause was
       Option A divergence), in which case Layers 2–3 then defend the fix
       from regressing; or
  (b) **persists** even after Layer 2 forward parity is bit-exact at init,
       in which case the bug is in training (loss, optimizer, batch order,
       routing-gradient feedback) and a new spec — outside the scope of
       this one — needs to investigate.
