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

## Verified on Juniata

Run `experiments/stage_roughness/juniata/.ddrs/runs/2026-09-12T03-54-46Z-train-and-test`
(213 reaches, `gamma = 0.35`, water year 1996, accumulated discharge):

```
n(d): min 0.0516  median 0.1487  max 0.4132
ratio of network-median n, wettest day vs driest: 2.312x
```

**What the traces should look like**, and the check that the pipeline is wired
correctly: the `n(d)` panel is the discharge panel upside down. Every reach's
roughness drops sharply on flood peaks and climbs back through recessions, and
the reaches sort by size — headwaters ride at `n ≈ 0.20–0.30`, the main stem at
`n ≈ 0.07–0.12`. If `n(d)` does not track the inverse of the hydrograph, the
discharge join is wrong.

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

The 2.3x swing is the point of the whole term: it is a flow-dependent travel
time, which is the thing findings §30 identified as the channel daily discharge
can actually constrain. Design and gates:
`docs/superpowers/specs/2026-09-12-stage-dependent-roughness-design.md`,
`tests/stage_roughness.rs`.
