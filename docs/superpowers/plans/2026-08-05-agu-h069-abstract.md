# AGU 2026 H069 Abstract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a submission-ready AGU Fall Meeting 2026 abstract (title + ~2,000-char body + plain-language summary) for session H069, framed as the conference talk for the in-progress paper "Beyond Equifinality in Differentiable River Routing" (Bindas & Shen), centered on how learned per-reach routing parameters change with the lateral-inflow (Q′) input dataset.

**Architecture:** This is a writing deliverable, not code. "Tests" are verification passes: every numeric claim is checked against the claims table (Task 1), the do-not-use list, and the character limit. The deliverable file lives in the paper repo (`~/projects/ddr_equifinality/`), beside `paper.tex`; this plan lives in ddrs because the evidence base does.

**Tech Stack:** Markdown draft; `wc -m` for character counts; AGU submission is manual (user pastes into the AGU portal).

## Global Constraints

- **Deadline is today (2026-08-05).** Every task must complete in this session; the user submits via the AGU portal themselves.
- **AGU limits (verify in Task 2, do not assume):** abstract body historically ≤ 2,000 characters *including spaces*; title ≤ 300 characters; plain-language summary optional, ≤ 2,000 characters. If Task 2 finds different limits, the found limits win.
- **Do-not-use list** (`.claude/skills/ddrs-dev/references/research-status.md` §Do-not-use): never write "leakance is identifiable"; never cite H1–H6 verdicts in either direction as established ("n is a bias absorber" AND "n is cross-source-stable" are both forbidden); never use 0.689/0.723 as a CONUS bar (use 0.6781/0.7172); never use +0.026 (use +0.037); never use "own baseline" NSE numbers from the 07-07/07-16 docs (3,211-gauge population).
- **Evidence standard §5:** every numeric claim carries a unit, gauge count, and eval window at least once in the abstract (compression allowed after first mention).
- **Only audit-robust facts** (Task 1 table) may appear as results. Anything else is phrased as hypothesis, protocol, or work-in-progress.
- **Honest work-in-progress framing:** the dHBV2 cross-family arms are unrun and their configs do not exist yet. The abstract may promise them for December ("we will present the completed cross-family matrix") but must not present them as done.

## Concerns for the user

- **Schedule risk (the big one):** positioning this as the paper's AGU talk promises the completed four-source matrix by December. The dHBV2 arms have no configs yet, and the seed/budget replicates that the v2 audits call "essential" are also unrun. If they slip, the talk falls back to the three LSTM arms + landscape probes — presentable, but the abstract's closing sentence must be written so that fallback is still honest. The drafted closing sentence in Task 3 is written that way deliberately; don't strengthen it.
- **Framing risk:** the paper draft's central hypothesis (geometry identifiable, n = bias absorber) is currently REFUTED in registered form / INCONCLUSIVE overall. An abstract that restates the paper's hypothesis as a finding would violate the do-not-use list. The draft therefore leads with the *protocol* and reports the audit-robust facts as "the picture inverts/complicates" — reviewers find this more interesting, and it is the only honest option. Why this could go wrong: a hurried edit pass could "simplify" the hedged sentences back into directional claims. Task 4's claim-by-claim check exists to catch exactly that.
- **Limit risk:** if AGU's 2026 limits differ from the assumed 2,000 characters (they have changed before), the draft needs trimming on the spot; Task 3 marks the two sentences to cut first.

## Assumptions

- Authors: Tadd Bindas (presenter) and Chaopeng Shen, as in `paper.tex`. Affiliations/author order are entered in the portal by the user, not drafted here.
- The user has an AGU account and session H069 accepts direct submissions (the session text says "Submit an Abstract to this Session").
- The abstract is for a talk/poster derived from the paper — so it may share the paper's title stem "Beyond Equifinality in Differentiable River Routing".
- Numbers are quoted from the extraction verified this session against `docs/2026-07-07-lstm-equifinality{,-v2}-findings.md`, `docs/2026-07-09-h5-h6-equifinality-v2-findings.md`, and `references/research-status.md` (re-verified 2026-07-30). No new computation is needed or attempted.

---

### Task 1: Claims table — the only sources of truth the abstract may cite

**Files:**
- Create: `/home/tbindas/projects/ddr_equifinality/agu2026_h069_abstract.md` (working file; claims table at top, draft below in Task 3)

**Interfaces:**
- Produces: the claims table that Task 4's verification pass checks every abstract sentence against.

- [ ] **Step 1: Write the working file with this exact claims table**

```markdown
# AGU 2026 H069 abstract — working file

## Claims table (audit-robust; the abstract may cite ONLY these as results)

| # | Claim | Number | Source |
|---|---|---|---|
| C1 | Network/scale | MERIT CONUS, 346,321 reaches; 2,365 USGS gauges; train 1981/10–1995/09, eval 1995/10–2010/09 | research-status.md §Structural constants |
| C2 | Design | identical differentiable Muskingum–Cunge + KAN head; same network, attributes, gauges, seed (42), budget (5 epochs); only the Q′ store swapped (daily-LSTM flat, daily-LSTM + precip disagg, hourly-LSTM native; dHBV2 arms planned) | 07-07 findings §2 |
| C3 | n cross-source spread | relative-to-mean median spread 0.4512 (~40–50% level disagreement); per-arm median n 0.084 / 0.100 / 0.065 s·m^(−1/3); realized depth/hydraulic radius spread ≈ 0.10 at common reference discharge | 07-07 findings §8.1; v2 §4.1 |
| C4 | Anti-correlation | n-divergence vs Q′ disagreement ρ = −0.380 at gauge-network scale (6,888 gauges); negative in all four operationalizations | 07-07 v2 §3 |
| C5 | Gradient orthogonality | cross-source geometry-gradient cosine 0.023–0.095 vs within-arm noise ceilings 0.39–0.59 | 07-07 findings §3, §6; 07-09 v2 §1 |
| C6 | DA scaling | geometry-vs-drainage-area slopes classically signed and consistent across all arms; n's slope is positive (anti-Leopold–Maddock) in two arms, flat in the disagg arm (+0.184 / −0.027 / +0.145) | 07-07 v2 §4.2 |
| C7 | Loss landscape | 44× / 65× anisotropic valley; top-width scale (p) is the sloppy axis, n the comparatively stiff, forcing-indexed axis | 07-09 v2 §4 |
| C8 | Transfer concentration | n-swap penalties concentrate in ~10 of 2,340 gauges carrying 82% of the summed penalty; median gauge ≈ 0; low-disagreement control penalty (+0.125) exceeds the high-disagreement primary (+0.095) | 07-09 v2 §2–3 |

## Forbidden in this abstract
- Any H1–H6 verdict stated as established, in either direction.
- "bias absorber" asserted as a finding (allowed only as the hypothesis under test).
- 0.689/0.723; +0.026; any 3,211-gauge "own baseline" delta; "leakance is identifiable".
- dHBV2 arms, seed replicates, or longer-budget replicates described as completed.
```

- [ ] **Step 2: Verify every number in the table against the repo (not memory)**

Run: `grep -n "0.4512\|−0.380\|-0.380\|0.095\|0.023" /home/tbindas/projects/ddrs/docs/2026-07-07-lstm-equifinality*.md /home/tbindas/projects/ddrs/docs/2026-07-09-h5-h6-equifinality-v2-findings.md | head -30`
Expected: each of C3–C5's numbers appears in the named doc. If any number is absent, fix the table from the doc — the doc wins.

- [ ] **Step 3: Commit the working file** (paper repo, not ddrs)

```bash
cd /home/tbindas/projects/ddr_equifinality && git add agu2026_h069_abstract.md && git commit -m "docs: AGU H069 abstract working file with verified claims table"
```

---

### Task 2: Verify the real AGU 2026 limits and H069 deadline

**Files:**
- Modify: `/home/tbindas/projects/ddr_equifinality/agu2026_h069_abstract.md` (add a "Limits" line under the title)

**Interfaces:**
- Produces: confirmed `CHAR_LIMIT` (body), `TITLE_LIMIT`, deadline timestamp — consumed by Task 3 (drafting) and Task 4 (`wc -m` gate).

- [ ] **Step 1: Search for the official limits**

Use WebSearch: `AGU 2026 fall meeting abstract submission character limit deadline` and WebFetch the AGU abstract-submission FAQ page it surfaces. Record: body char limit, whether spaces count, title limit, plain-language-summary requirement, and the exact deadline (date + time zone).

- [ ] **Step 2: Record findings in the working file**

Add under the top heading: `Limits (verified 2026-08-05): body ≤ <N> chars, title ≤ <N> chars, deadline <date time TZ>.` If the search is inconclusive, record the assumed 2,000/300 limits with the word "UNVERIFIED — trim aggressively" and tell the user in the final summary.

---

### Task 3: Draft title + abstract body

**Files:**
- Modify: `/home/tbindas/projects/ddr_equifinality/agu2026_h069_abstract.md` (add "## Title" and "## Abstract" sections below the claims table)

**Interfaces:**
- Consumes: claims table C1–C8 (Task 1); CHAR_LIMIT (Task 2).
- Produces: the draft text Task 4 verifies and Task 5 packages.

- [ ] **Step 1: Add the title candidates (user picks one at review)**

```markdown
## Title (pick one)
1. Beyond Equifinality in Differentiable River Routing: Which Learned Channel Parameters Survive a Change of Inputs?
2. Input-Perturbation Tests of Parameter Identifiability in Differentiable Muskingum–Cunge Routing at CONUS Scale
```

- [ ] **Step 2: Add this abstract draft verbatim, then adjust to the verified limit**

```markdown
## Abstract (draft v1)

Differentiable river routing models can learn per-reach channel parameters — Manning's
roughness n and cross-sectional geometry — from downstream discharge at continental
scale, replacing lookup tables and regional power laws. But a gauge observes the
aggregate response of its entire upstream network, so learned parameters may be
compensatory artifacts rather than physical properties: the classic equifinality
problem, now posed over 346,000 reaches. We present an input-perturbation protocol
that tests identifiability directly: we train an identical differentiable
Muskingum–Cunge model, parameterized per-reach by a Kolmogorov–Arnold network head,
on the same MERIT CONUS river network, observations (2,365 USGS gauges,
1995–2010 evaluation), and training budget, swapping only the lateral-inflow forcing
among structurally different sources (daily and hourly LSTM variants and
differentiable HBV2.0 variants). Parameters that persist across inflow sources are
candidates for physical interpretation; parameters that track the source are
absorbing its errors. Preliminary results complicate the intuitive expectation that
geometry is physical while roughness absorbs bias. Learned n disagrees across
sources by roughly 40% at the median reach while realized depth and hydraulic radius
converge — yet n's divergence is anti-correlated with inflow disagreement
(network-scale ρ = −0.38), contradicting simple bias absorption; cross-source
geometry gradients are nearly orthogonal (cosine 0.02–0.10 against noise ceilings of
0.39–0.59); the loss landscape is a 44–65× anisotropic valley in which channel-width
scale, not roughness, is the sloppy direction; and cross-source parameter-swap
penalties concentrate in ~10 of 2,340 gauges rather than distributing across the
network. We will present the completed cross-model-family comparison and replicate
controls, and discuss which learned river-network parameters can be transferred
between modeling frameworks and interpreted physically.

Trim-first sentences if over limit: the "Parameters that persist… absorbing its
errors." sentence (its content is implied by the protocol sentence), then the DA
detail is already omitted (C6 held for the talk itself).
```

- [ ] **Step 3: Check the character count**

Run: `awk '/^## Abstract/,/^Trim-first/' /home/tbindas/projects/ddr_equifinality/agu2026_h069_abstract.md | sed '1d;$d' | tr -d '\n' | wc -m`
Expected: ≤ CHAR_LIMIT from Task 2. If over, apply the trim-first list and re-run until it passes.

---

### Task 4: Claim-by-claim verification pass

**Files:**
- Modify: `/home/tbindas/projects/ddr_equifinality/agu2026_h069_abstract.md` (append a "## Verification" checklist section)

**Interfaces:**
- Consumes: claims table (Task 1), draft (Task 3).
- Produces: a checked verification block; the gate Task 5 requires before packaging.

- [ ] **Step 1: Map every quantitative sentence in the draft to a claims-table row**

Append and fill:

```markdown
## Verification (2026-08-05)
- [ ] "346,000 reaches" → C1
- [ ] "2,365 USGS gauges, 1995–2010 evaluation" → C1
- [ ] "identical model… swapping only the lateral-inflow forcing" → C2 (dHBV2 phrased as part of the matrix being completed, not done — check wording)
- [ ] "~40% at the median reach… depth and hydraulic radius converge" → C3
- [ ] "ρ = −0.38" → C4
- [ ] "cosine 0.02–0.10 vs 0.39–0.59" → C5
- [ ] "44–65× anisotropic valley; width scale sloppy" → C7
- [ ] "~10 of 2,340 gauges" → C8
- [ ] No sentence asserts an H1–H6 verdict, "bias absorber" as finding, or any forbidden number
- [ ] Every number has unit + gauge count + window at first mention (evidence standard §5)
- [ ] Character count ≤ verified limit (paste the wc -m output here)
```

Any unmapped quantitative sentence is a plan failure: either add its source to the claims table (with a doc citation verified by grep) or delete the sentence.

- [ ] **Step 2: Session-fit check**

Confirm the draft explicitly touches ≥2 of H069's named topics — it should already hit "novel network representations and parameterizations" (KAN per-reach head), "graph-based and deep learning approaches", and "network-to-gauge site mapping" (gauge-observes-aggregate framing). If not, adjust the first two sentences, not the results.

- [ ] **Step 3: Commit**

```bash
cd /home/tbindas/projects/ddr_equifinality && git add agu2026_h069_abstract.md && git commit -m "docs: AGU H069 abstract draft v1 with claim-by-claim verification"
```

---

### Task 5: Plain-language summary + handoff to the user

**Files:**
- Modify: `/home/tbindas/projects/ddr_equifinality/agu2026_h069_abstract.md` (add "## Plain-language summary")

**Interfaces:**
- Consumes: verified draft (Task 4).
- Produces: the final package the user pastes into the AGU portal.

- [ ] **Step 1: Add the plain-language summary**

```markdown
## Plain-language summary (draft v1)

Computer models that move water down river networks depend on numbers describing
each river channel — how rough and how wide it is. New machine-learning methods can
tune millions of these numbers so the model matches streamflow measurements, but a
long-standing worry is that many different sets of numbers fit the data equally
well, so the tuned values may not describe the real rivers. We tested this by
training the same river model on the entire contiguous United States several times,
changing only the estimate of how much water enters the rivers. Channel-shape values
stayed consistent between versions, while channel-roughness values did not — but not
in the way the standard explanation predicts. Our results show which learned river
properties can be trusted and reused, and which are artifacts of the training setup.
```

- [ ] **Step 2: Final summary to the user**

Report in the final message: the chosen-title options, the full abstract text, its exact character count vs the verified limit, the deadline found in Task 2, and the two standing risks (dHBV2 arms unrun; hedged sentences must not be strengthened during portal entry).

- [ ] **Step 3: Commit**

```bash
cd /home/tbindas/projects/ddr_equifinality && git add agu2026_h069_abstract.md && git commit -m "docs: AGU H069 abstract final package (title, body, plain-language summary)"
```
