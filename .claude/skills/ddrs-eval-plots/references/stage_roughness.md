# Reference: animating `n(d)` over a water year

When a run sets `params.stage_roughness`, Manning's roughness stops being a
fixed per-reach number and starts breathing with the flow:

```
n(d) = n_0 · (d / d_ref)^(−gamma)
```

Rougher at low flow, smoother in flood, because relative roughness falls once
the bed material is drowned. This family shows that: a GIF of the network
coloured by `n(d)`, one frame per day, or a line plot of a few reaches through
the year.

**Script:** `experiments/stage_roughness/animate_n_of_d.py`. Self-contained
(`uv run --script` header), takes a run directory.

```bash
# line traces — cheaper, usually the more legible view
experiments/stage_roughness/animate_n_of_d.py <run-dir> --water-year 1996 --traces

# the GIF
experiments/stage_roughness/animate_n_of_d.py <run-dir> --water-year 1996
```

Both write into `<run-dir>/plots/`.

## Requirements

- The run must have been trained with **`--plot`**, so `plot/kan_parameters.nc`
  exists. Without it the script exits 1 and tells you to retrain.
- The run's `config.yaml` supplies `gamma` and `d_ref`. **A `gamma = 0` run
  animates to a still frame**, and the script says so up front rather than
  letting you wonder why nothing moves. That is the correct answer for such a
  run, not a bug.

## Where the depth comes from — read this before captioning a figure

Nothing exports per-reach daily depth. The script recomputes it from the same
closed form the solver inverts (`src/geometry.rs`):

```
d = ( Q · n_0 · (q+1) · d_ref^gamma / (p · √S) ) ^ ( 3 / (5 + 3q + 3·gamma) )
```

so the only free input is `Q`, the discharge each reach carries. Two modes:

| `--discharge` | what it uses | when it is honest |
|---|---|---|
| `accumulated` (default) | Q' summed over the upstream network, by topological solve against `conus_adjacency` | the realistic choice; this is roughly what the router carries |
| `local` | each reach's own Q' only | small networks where routing barely redistributes; **badly understates main-stem flow on CONUS** |

The mode is stamped on every frame and in the plot title, so a figure cannot get
separated from its caveat. Neither mode is the *routed* discharge the solver
actually produced, because that is not exported either; accumulated Q' is the
right order of magnitude and the right spatial pattern.

Falls back to `local` with a printed warning if the adjacency is missing or the
parameter COMIDs are not all in it, rather than failing.

## Dead reaches: the filter that has to be there

**54.5 % of CONUS reaches must be excluded or the figure lies.** Two problems,
both silent:

- ~42k MERIT reaches carry no Q' prediction and read as the `0.001` fill
  (CLAUDE.md, data sources). With upstream accumulation the count of reaches
  whose flow never moves reaches **147,040**, since a dead reach fed only by
  dead reaches stays dead.
- Others carry physically meaningless flows around `1e-9 m³/s`, which sit pinned
  at `attribute_minimums.depth`, so their `n(d)` is an artefact of the clamp
  rather than hydraulics.

Both classes plot as **flat lines**, and because the trace picker samples across
the discharge rank they get picked preferentially. They also drag the breathing
statistic toward 1. The script now requires a reach to both vary
(`max/min > 1.01`) and carry a median flow above `1e-3 m³/s`.

A relative-standard-deviation threshold is NOT enough: noise from the
accumulation solve pushes dead reaches past `std/mean > 1e-9`. Ask the question
directly.

## Verified numbers

Juniata, `2026-09-12T03-54-46Z-train-and-test` (213 reaches, `gamma = 0.35`,
water year 1996, accumulated discharge) — every reach live, no filtering needed:

```
n(d): min 0.0516  median 0.1487  max 0.4132
ratio of network-median n, wettest day vs driest: 2.312x
```

CONUS, `2026-09-12T06-06-19Z-train-and-test` (`gamma = 0.35`), after excluding
188,646 dead reaches:

```
n(d) over live reaches: min 0.0154  median 0.1567  max 0.7506
network-median n, wettest day vs driest:  1.286x
per-reach breathing: median 1.910x, p90 3.256x
```

**Report the per-reach number, not the network-median one.** The network median
barely moves (1.29x) because CONUS reaches peak on different days and the
spatial average smooths it out. What a hydrologist wants to know is what happens
at a reach: **the typical CONUS reach very nearly doubles its roughness between
its driest and wettest day.**

**What the traces should look like**, and the check that the pipeline is wired
correctly: the `n(d)` panel is the discharge panel upside down. Flashy reaches
drop sharply on every flood peak and climb back through recessions; the main
stem sits low and steady; reaches sort by size, headwaters riding high. If
`n(d)` does not track the inverse of the hydrograph, the discharge join is
wrong.

## Scaling to CONUS

The accumulation is one sparse triangular solve per day over 346,321 reaches, so
a water year is 365 solves — minutes, not seconds, and the dominant cost. The
GIF subsamples to at most 180 frames regardless of window length.

For a true map (reaches drawn on their MERIT geometry rather than the
`(log10 Q, n_0)` scatter the script defaults to), join `COMID` against the
fabric exactly as `references/parameter_map.md` does, then colour by the
`n_t[day]` column instead of a static field. The scatter default exists so the
script needs no shapefile and runs anywhere.

## Interpreting it

`n_0` is **roughness at `d_ref`**, not Manning's `n`. With `d_ref = 1 m` and
most CONUS baseflow depths well under 1 m, the realised `n` is larger than `n_0`
nearly everywhere. Published `n` numbers from `gamma = 0` runs are therefore not
comparable to `n_0` from a stage-roughness run — see the do-not-use list in
`ddrs-dev`'s `references/research-status.md`.

The swing is the point of the whole term: it is a flow-dependent travel time,
which findings §30 identified as the channel daily discharge can actually
constrain. But note the verdict in §35 — on CONUS `gamma = 0.35` cost about
0.01 median NSE while improving the downstream width exponent from 0.004 to
0.098, so a pretty animation is not evidence the term should be adopted. Design
and gates:
`docs/superpowers/specs/2026-09-12-stage-dependent-roughness-design.md`,
`tests/stage_roughness.rs`.
