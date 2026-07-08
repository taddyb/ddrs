---
name: ddrs-docs-and-writing
description: "Use when writing, updating, or reviewing any ddrs project document: findings reports, session handoffs, experiment specs, implementation plans, or the paper at ddr_equifinality/paper.tex. Also use when asking what the correct doc type is for a given output, how to connect an experiment result to the paper narrative, or whether a hypothesis verdict belongs in a findings doc or a spec. Do NOT use for running experiments, editing Rust code, debugging builds, or plotting — use ddrs-run-and-operate, ddrs-debugging-playbook, or ddrs-eval-plots instead."
---

# ddrs docs and writing

This skill covers the documentation conventions, doc-type taxonomy, template structures, house style, and paper-to-experiment-log connection for the ddrs project.

## Glossary (defined once)

| Term | Meaning |
|---|---|
| **ddrs** | BURN-0.21 Rust port of the Python DDR (Differentiable Discharge Routing) solver using Muskingum-Cunge |
| **DDR** | Python/PyTorch reference at `~/projects/ddr/`. ddrs must stay gradient-exact against it |
| **KAN head** | Kolmogorov-Arnold Network head (`rskan::KanLayer`) that maps catchment attributes → routing parameters |
| **summed-Q′** | No-routing baseline: per-gauge sum of upstream divide Qr. Any trained model must beat this to prove routing earns its keep |
| **CONUS** | Contiguous United States; 346,321 MERIT reaches, 338,814 edges |
| **eval network** | The gauge-subgraph union used at evaluation time (64,892 reaches in the 2×2 experiments) |
| **zeta** | Per-reach GW–SW exchange flux (m³/s); positive = losing stream |
| **leakance** | Experimental GW–SW water-loss term; off by default; controlled by `use_leakance: true` in config |
| **disagg head** | Daily→hourly disaggregation sub-head inside KanHead; driven by AORC hourly precip |
| **spec** | Design doc written BEFORE an experiment. Lives in `docs/superpowers/specs/` |
| **plan** | Implementation task list derived from a spec. Lives in `docs/superpowers/plans/` |
| **findings doc** | Post-experiment narrative with hypothesis table, results, and verdict. Lives in `docs/` |
| **handoff** | Session-boundary doc with the action list for the next session. Lives in `docs/` |
| **journal** | Multi-experiment chronological record (used for cross-cutting investigations). Lives in `docs/` |

---

## Doc-type taxonomy

Choose the correct doc type before writing anything.

| Situation | Doc type | Location | Naming pattern |
|---|---|---|---|
| Planning a new experiment before any code runs | **spec** | `docs/superpowers/specs/` | `YYYY-MM-DD-<slug>-design.md` |
| Breaking a spec into implementation tasks | **plan** | `docs/superpowers/plans/` | `YYYY-MM-DD-<slug>.md` (same slug) |
| Summarizing what an experiment found after it ran | **findings doc** | `docs/` | `YYYY-MM-DD-<slug>-findings.md` |
| Handing state to the next session mid-experiment | **handoff** | `docs/` | `YYYY-MM-DD-<slug>-handoff.md` |
| Documenting a cross-cutting investigation that spans multiple runs | **journal** | `docs/` | `<date>_journal.md` or a named findings doc |
| Documenting a data source, contract, or external API | **reference** | `docs/reference/` or `docs/` | descriptive name, no date required |

### When NOT to write a new doc

- If a findings doc for that experiment already exists: update it in place (add a datestamped section) rather than creating a duplicate.
- Handoffs and journals are living docs — append, do not replace.
- The spec/plan pair is mandatory before running expensive GPU jobs. Do not skip the spec to save time: the pre-registered hypotheses and falsification criteria protect against HARKing (Hypothesizing After Results are Known).

---

## Findings doc template

Every experiment that runs on real data gets a findings doc. Use this structure exactly.

```markdown
# <Experiment name> — findings (<YYYY-MM-DD>)

Spec:  `docs/superpowers/specs/<spec-file>.md`
Plan:  `docs/superpowers/plans/<plan-file>.md`  (if applicable)
Script: `scripts/<script>.py`  (if applicable)
Prior finding: `docs/<prior-findings>.md`  (if applicable)

**One-line verdict:** <single declarative sentence. What happened, what it means. No hedging.>

---

## 1. Motivating observation and hypotheses

<What observation or question drove this experiment. 2-4 sentences.>

Pre-registered hypotheses table (must appear BEFORE the results section):

| # | Hypothesis | Mechanism proposed |
|---|---|---|
| H1 | <name> | <mechanism> |
...

## 2. Methods — how the experiment was done

<What was built or changed in Rust/Python. Which runs were used. Which scripts. Key config flags.>

<If a 2×2 or factorial: table of arms with run IDs and distinguishing config.>

## 3. Results — how it was resolved

Results table (verdicts SUPPORTED / REFUTED / INCONCLUSIVE only):

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| H1 | <name> | **SUPPORTED** / **REFUTED** / **INCONCLUSIVE** | <one metric or statistic> |
...

<Narrative paragraphs explaining each verdict and what the key numbers mean physically.>

## 4. Conclusions

<Numbered list. What changed (or didn't). What this supersedes. What the gate outcome was.>

## 5. Next steps

<Prioritized numbered list. Dropped items explicitly labeled "Dropped — <reason>".>

## 6. Raw script output (optional)

<Paste verbatim output of the analysis script so the doc is self-contained.>

## 7. Reproduce

<Copy-pasteable commands to reproduce the analysis from scratch.>
```

### Critical findings-doc rules

1. **Pre-register hypotheses BEFORE pasting results.** The hypothesis table in §1 must be derived from the spec, not reverse-engineered from what was found.
2. **Use SUPPORTED / REFUTED / INCONCLUSIVE.** Never "confirmed," "rejected," "partially supported," or "likely." Three states only.
3. **Every numeric claim needs a unit and a context.** Write "median |zeta| = 6.4e-4 m³/s (hourly-ON, 64,892 eval reaches, 1995/10–2010/09 window)" not "zeta was small."
4. **One-line verdicts are mandatory.** The verdict sentence goes immediately after the header block, before any sections. A reader who reads only that line should know whether to keep reading.
5. **When a gate fails, say so explicitly.** "The Phase-3 gate FAILED — the widened-K_D retrain was NOT run" is correct. "We decided not to run the retrain" is not.
6. **Superseding prior recommendations.** When a new finding overturns a prior doc's recommendation, name the prior doc and the exact item being superseded. Example from `docs/2026-07-02-leakance-diagnosis-findings.md` §4 item 2: "This supersedes the 'widen K_D past 1e-6 — top follow-up' recommendation in `docs/2026-07-01-leakance-hourly-findings.md` §5 item 2."

---

## Spec template

```markdown
# <Feature/experiment name> — design

Date: YYYY-MM-DD. Branch: `<branch>` (worktree off `<base>` @ `<sha>`).
Prior findings: `docs/<file>.md`.

## Problem

<What observation motivates this. What question it answers. 2-4 sentences.>

## Hypotheses and tests

| # | Hypothesis | Test | Falsified if |
|---|---|---|---|
| H1 | <name: mechanism> | <what data/computation to run> | <condition that refutes it> |
...

## Phase 1 — <instrumentation/setup>

<Smallest Rust/Python change needed to expose the test data. What tests stay green.>

## Phase 2 — <battery/analysis>

<Script, inputs consumed, outputs produced.>

## Phase 3 — <gated action> (if applicable)

**Gate: <condition>.** Then:
- <What to do if gate passes.>

If the gate fails: <what to do instead — do not spend GPU>.

## Deliverables

Numbered list of concrete artifacts.

## Concerns / assumptions (per planning rules)

- **Concern — <name>:** <what could go wrong and why>. *Mitigation:* <what guards it>.
- **Assumption — <name>:** <what is assumed and why it is defensible>.
- **Why this change:** <the benefit and alternative cost>.
```

---

## Handoff template

```markdown
# <Experiment name> — session handoff (<YYYY-MM-DD>)

Purpose: hand another session everything needed to continue. Read top-to-bottom;
**"What's left"** is the action list.

Spec:  `docs/superpowers/specs/<file>.md`
Plan:  `docs/superpowers/plans/<file>.md`
Branch: `<branch>`   HEAD at handoff: `<sha>`

---

## 0. Resolution / root cause (if applicable)

<If a prior session found and fixed a bug or false alarm, document it here first.>

## 1. Current state

<What is built, what runs exist, what was measured.>

## 2. What's left (action list)

Numbered, prioritized. Each item: what to do + how to verify it succeeded.

## 3. Known gotchas

<Anything that will bite the next session if not known.>
```

---

## House style

### Vocabulary

| Preferred | Avoid |
|---|---|
| SUPPORTED / REFUTED / INCONCLUSIVE | confirmed, rejected, showed, demonstrated |
| "beats baseline on NSE" | "improves performance" |
| "the gate FAILED" | "we decided not to proceed" |
| "as of 2026-07-05" | vague "recently" or "currently" |
| median ± IQR | mean ± std for skewed distributions |
| m³/s, m, 1/s with units always explicit | dimensionless numbers without units |

### Numbers

- Always include the gauge count and eval window when reporting median NSE / KGE. Example: "median NSE 0.715 / KGE 0.711 (2365 gauges, 1995/10–2010/09)".
- Volatile facts (benchmark numbers, run IDs) get a date-stamp: "as of 2026-07-05."
- Do not round to fewer significant figures than the effect size requires. If two runs differ by 0.0008 NSE, report four decimal places.

### Tables

- Hypothesis tables always come BEFORE results.
- Results tables always include verdicts in bold.
- 2×2 arms tables always include run ID, forcing, feature flag, and binary-validity status.

### Provenance

Every findings doc must identify: the run ID(s) used, the script that analyzed them, the eval window, and the gauge count. A reader should be able to reproduce the numbers from the doc alone.

---

## Current benchmark numbers (as of 2026-07-05)

These are the authoritative numbers to cite. Do not invent new numbers or cite from memory.

| Config | median NSE | median KGE | Gauges | Source |
|---|---|---|---|---|
| summed-Q′ baseline (no routing) | 0.689 | 0.723 | ~5,224 global matched | `docs/6_19_26_journal.md` |
| summed-Q′ baseline (CONUS eval) | 0.6781 | 0.7172 | 2365 | `docs/2026-06-23-precip-disaggregation-findings.md` |
| Best trained result: precip-disagg + L1 | **0.715** | **0.711** | 2365 | `docs/2026-06-23-precip-disaggregation-findings.md` |
| Δ vs CONUS baseline (best trained) | +0.037 NSE | −0.007 KGE | 2365 | same |

Key facts:
- **KGE does NOT beat the summed-Q′ baseline in any config as of 2026-07-05.** The best trained result loses −0.007 KGE. State this plainly when writing for any audience.
- **NSE beats the baseline** in the best config (+0.037).
- The KGE gap is structural, not loss-fixable: tested with nnse-kge loss, result was 0.7095 NSE / 0.7100 KGE — no dual win.

---

## Leakance experiment status (as of 2026-07-05)

The canonical reference for leakance status is `docs/2026-07-02-leakance-diagnosis-findings.md`. Summary for use in docs:

**2×2 verdict (2026-07-01):** GO — marginal. All three gate criteria met. Hourly forcing + leakance improves the losing-stream subset (ΔNSE +0.0005, ΔKGE +0.0018, 55.5% of gauges improve). |zeta| > 0.01 m³/s on 10.4% of 64,892 eval reaches.

**Diagnosis verdict (2026-07-02):** The small zeta is a training-signal problem, not a parameter-box problem.

| # | Hypothesis | Verdict |
|---|---|---|
| H2 | Driving-head starvation | **SUPPORTED** — median driving head 0.021 m; 47% of reaches gaining at eval-window mean |
| H4 | Gauge bias / gradient starvation | **SUPPORTED** — zeta–uparea ρ +0.76; gauged median |zeta| 11× ungauged; dry-tercile has LESS zeta than wet (inverted from physics) |
| H5 | Equifinality with routing params | **SUPPORTED** (daily only) — daily Δn = +0.012 (0.59 IQR); hourly Δn nil |
| H1 | Structural ceiling (K_D box) | **REFUTED** — 71.5% of reaches CAN exceed 0.01 m³/s inside current box; median utilization only 3.4% |
| H3 | KAN variance collapse | **REFUTED** — d_gw–meanP ρ +0.71, K_D–aridity +0.61 (strong learned structure) |
| H6 | Wrong yardstick (absolute bar) | **REFUTED** — fractional loss agrees: 8.4% lose >1% of local flow |
| H7 | Model-form error (d_gw pinning) | **REFUTED** — 0% of d_gw at bounds |

**K_D widening is NOT recommended.** The "widen K_D past 1e-6 — top follow-up" recommendation in `docs/2026-07-01-leakance-hourly-findings.md` §5 item 2 is superseded by the diagnosis. The binding constraint is the training signal, not the box.

**Identifiability positive control (2026-07-04, worktree):** FAILED. Recovery ratio 0.009 vs ≥0.5 bar. Root cause: windowed training objective has ~130× hotstart-transient noise floor vs synthetic zeta signal. Leakance identifiability is NOT proven. Phase B (state-cache hotstart, ≤0.25 mean L1 noise-floor target) is required before any identifiability claim can be made.

---

## Connecting experiments to the paper

The paper is at `/home/tbindas/projects/ddr_equifinality/paper.tex`. Title: "Beyond Equifinality in Differentiable River Routing" (Bindas, Shen).

**Core thesis: selective equifinality.** Geometry (p, q, top width) is identifiable — shaped by geomorphic physics. Manning's n is a bias-absorber — it shifts to compensate errors in lateral inflow rather than representing channel roughness truth.

**Experimental test:** train ddrs with four structurally different lateral inflow sources (two LSTM variants, two dHBV2.0 variants) on the same MERIT network. Compare convergence at: (1) raw learned parameters, (2) realized channel geometry at reference discharges, (3) routing performance. Parameters that converge across inflow sources are identifiable; parameters that diverge are compensatory.

**How ddrs findings feed the paper:**

| Experiment / finding | Paper connection |
|---|---|
| Manning's n shifts +20% when daily leakance is added (H5 equifinality, 2026-07-02) | Direct evidence of n as bias-absorber — quantitative support for the n-is-not-a-physical-property claim |
| K_D–aridity ρ +0.61, d_gw–meanP ρ +0.71 (H3 refuted, 2026-07-02) | Geometry/exchange params carry physical structure even when flux is gauge-shaped — supports selective identifiability |
| KGE structural ceiling vs summed-Q′ (2026-06-23) | Over-attenuation from L1 loss motivates the roughness-as-bias-absorber interpretation |
| Leakance identifiability control FAILED (2026-07-04) | Negative result: GW–SW exchange is NOT identifiable from gauged-only discharge under current windowed training — relevant to §Limitations |

**What is NOT yet in the paper:** the unit_catchments branch work (current branch). Do not cite findings from a branch that has not been merged to master without clearly marking them as preliminary.

**Writing guidance for paper sections:**
- Abstract: state results as hypotheses until experiments on the four-inflow design are complete. The current draft abstract claims results that don't exist yet.
- Methods §Riverbed Leakage (already drafted): the leakance equation is correct. Cite the diagnosis findings when explaining why zeta is small.
- Use "selective equifinality" not "subjective equifinality" — the claim is that some parameters are identifiable and others are not, which is about parameter type, not observer perspective.

---

## Common pitfalls

### 1. Citing the wrong baseline for the wrong gauge set

The summed-Q′ baseline has two versions: the global matched set (0.689 / 0.723, ~5,224 gauges) and the CONUS eval set (0.6781 / 0.7172, 2365 gauges). They come from different experiments and different gauge filters. Always specify which.

### 2. Treating K_D ceiling-pinning as a structural problem

K_D pins at 1e-6 on 100% of reaches. This is NOT evidence the box needs widening. The diagnosis showed median utilization is only 3.4% of the in-box capacity — the optimizer maxes the rate constant and then throttles the flux through the driving head. Widening the box is predicted to re-pin K_D at the new ceiling with no meaningful change to zeta. Do not recommend widening unless the diagnosis conclusion is explicitly overturned.

### 3. Claiming leakance is identifiable

As of 2026-07-05, leakance identifiability is NOT proven. The GO-marginal verdict from the 2×2 shows the term is active and skill-improving on the losing-stream subset under hourly forcing. It does not prove the learned parameters physically reflect true GW–SW exchange. The positive control experiment FAILED (recovery ratio 0.009). Do not write "leakance is identifiable" — write "leakance is active and non-collapsed under hourly forcing; identifiability is not yet established."

### 4. Missing the STALE-BINARY TRAP in methods sections

When describing how a run was produced, document the binary version used. The July 1 leakance 2×2 originally used a June 3 binary with no disaggregation or leakance — the runs were byte-identical despite different configs. Any findings doc for a run that used `ddrs run` or `cargo run` must note the binary state. Quick check: directory checkpoints (`epoch_E_mb_M/head.mpk`) = current binary; flat files (`epoch_E_mb_M.mpk`) = stale binary.

### 5. Using `use_cuda_graphs: true` silently masking NaN

If a run was trained with `use_cuda_graphs: true`, any NaN in the forward pass is masked — the printed training loss looks finite while weights are silently corrupted. Any findings doc for a run that showed unexpectedly NaN eval metrics with apparently healthy training loss should check this. Document it explicitly: "Diagnosis: CUDA graphs masked NaN precip in `AorcPrecipStore`."

---

## File paths for key documents (verified as of 2026-07-05)

| Doc | Path |
|---|---|
| Latest leakance findings | `/home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md` |
| 2×2 leakance findings | `/home/tbindas/projects/ddrs/docs/2026-07-01-leakance-hourly-findings.md` |
| Precip disagg findings | `/home/tbindas/projects/ddrs/docs/2026-06-23-precip-disaggregation-findings.md` |
| Cross-cutting gradient journal | `/home/tbindas/projects/ddrs/docs/6_19_26_journal.md` |
| Leakance diagnosis spec | `/home/tbindas/projects/ddrs/docs/superpowers/specs/2026-07-01-leakance-low-zeta-diagnosis-design.md` |
| Leakance feasibility spec | `/home/tbindas/projects/ddrs/docs/superpowers/specs/2026-06-29-leakance-hourly-feasibility-design.md` |
| Paper (LaTeX) | `/home/tbindas/projects/ddr_equifinality/paper.tex` |
| CLAUDE.md (source of truth for config, CLI, invariants) | `/home/tbindas/projects/ddrs/CLAUDE.md` |

---

## Provenance and maintenance

Re-verify benchmark numbers:
```bash
grep -n "median NSE\|median KGE\|0\.715\|0\.711\|0\.689\|0\.723" \
  /home/tbindas/projects/ddrs/docs/2026-06-23-precip-disaggregation-findings.md \
  /home/tbindas/projects/ddrs/docs/6_19_26_journal.md
```

Re-verify leakance verdict numbers:
```bash
grep -n "SUPPORTED\|REFUTED\|10\.4%\|6\.4e-4\|0\.034\|0\.76" \
  /home/tbindas/projects/ddrs/docs/2026-07-02-leakance-diagnosis-findings.md
```

Re-verify paper title and thesis:
```bash
grep -n "title\|equifinality\|selective\|n is\|roughness" \
  /home/tbindas/projects/ddr_equifinality/paper.tex | head -20
```

Check for new findings docs since last skill update:
```bash
ls -lt /home/tbindas/projects/ddrs/docs/*.md | head -10
```
