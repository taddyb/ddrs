# NID dams snapped to MERIT and intersected with the eval gauges

**Date:** 2026-09-26 (NID fetched 2026-09-27 UTC)
**Asked (user):** the NWM reservoir table misses dams; download the National Inventory of Dams, snap it to MERIT,
and find the dams inside our gauges' networks.
**Scripts:** `~/projects/remote_sensing_extraction/nid/{fetch_nid.sh,snap_nid_to_merit.py}` (data and provenance at
`/mnt/ssd1/data/nid/`, catalogued in `/mnt/ssd1/data/README.md`); `experiments/reservoir/nid/intersect_nid_gauges.py`.
**Outputs:** `experiments/reservoir/nid/nid_dams_by_gauge.csv` (2,365 gauges), `nid_dams_in_eval_network.csv`
(5,935 dams, the candidate dam table for training), `nid_gauge_summary.json`.
**Follows:** `2026-09-26-option-c-dam-benchmark-findings.md` §9; the station-name flags in
`experiments/reservoir/benchmark/gauge_name_dam_flags.csv`.

## 1. Snapping

NID holds 92,766 US dams, 91,326 of them primary CONUS structures. Each is snapped to the MERIT-Basins reach within
3 km whose upstream area best matches the NID drainage area (details and classes in the script docstring and the
fetch repo README). 68,461 dams drain too little area for MERIT's network (farm ponds and small tributary dams);
11,796 are snapped.

| NID storage | Primary CONUS dams | Snapped |
|---|---:|---:|
| >= 1 MCM | 14,996 | 6,284 (42 %) |
| >= 10 MCM | 3,518 | 2,659 (76 %) |
| >= 100 MCM | 959 | 876 (91 %) |

- **Validation:** of 602 GRanD dams with coordinates and an independent COMID (the NWM crosswalk), 550 land on the
  same reach (91 %) and 593 on the same or an adjacent reach (98.5 %). Raystown, Abiquiu and Santa Rosa land on
  their known COMIDs.
- **NID drainage areas are sometimes incremental.** Mainstem USACE dams list only the area between them and the
  next dam up (Fort Randall 36,648 km2 against 675,585 km2 at its reach; Eufaula 21,769 against 122,831). A
  fallback (class D: >= 10 MCM, nearest reach <= 700 m draining more than the NID area) recovers 57 such dams.
- **What stays unsnapped among large structures:** lake dikes and off-channel works (Herbert Hoover Dike, Devils
  Lake jetties, Soo Locks), correctly.

## 2. Dams inside the eval gauges' networks

| | NID | NWM table |
|---|---:|---:|
| eval gauges with any dam upstream | 1,560 | 909 |
| with a dam >= 1 MCM | 1,265 | |
| with a dam >= 10 MCM | 917 | |
| with a dam >= 10 MCM and no NWM reservoir | 75 | |
| gauges with DOR > 0.5 (NID normal storage) | 308 | 347 |
| gauges with DOR > 0.5 (NID maximum storage) | 410 | |
| dams inside eval networks | 5,935 (2,442 >= 1 MCM, 1,099 >= 10 MCM, 304 >= 100 MCM) | |
| dams on an eval gauge's own reach | 721 (at 542 gauges) | |

- **Skill by NID degree of regulation** (trained run `2026-09-17T16-38-16Z`, median NSE WY1997-2010): no dam 0.735
  (805 gauges), DOR <= 0.1 0.775 (883), 0.1 to 0.5 0.725 (369), DOR > 0.5 0.487 (308). Same deficit as the NWM
  classification.
- **NID and the NWM table disagree on 139 gauges at DOR 0.5**: 50 cross it only under NID, 89 only under NWM.
  The 89 are a definition: NWM volume is HydroLAKES `Vol_total` (the whole lake), a median 3 times NID normal
  storage at those gauges (Ross Barnett, Broken Bow, Oologah, Clarence Cannon are all found by NID). With NID
  maximum storage, 410 gauges exceed DOR 0.5.
- **Cross-checks:** 92 of the 96 gauges whose station name puts them below or at a dam have an NID dam upstream,
  including 14 of the 18 the NWM table missed (the other 4 are a waterfall, an above-dam gauge, the natural
  Yellowstone Lake outlet, and Inspiration Dam, AZ). 111 of the 121 benchmark dams have an NID dam on the same
  COMID. The other 10 (among them Flaming Gorge, Trinity, Folsom and Cochiti) were not checked one by one; the GRanD
  validation above puts 98.5 % within one reach.

## 3. What this changes

- **The dam table for joint training** is `nid_dams_in_eval_network.csv`, not the 2,177-dam NWM table. 1,099 dams
  of >= 10 MCM sit inside eval networks, and 75 gauges with such a dam have no NWM reservoir at all.
- **Regulated-gauge counts built on the NWM table were lower bounds**, as `.claude/RESERVOIRS.md` warned; with
  NID, 1,560 rather than 909 eval gauges have a dam upstream, most of them small (DOR <= 0.1).
- **Each dam now carries release-relevant attributes** usable as learned-release features: maximum discharge,
  spillway type, outlet gate type, normal and maximum storage, purpose, year, height.

## 4. Caveats

- NID coordinates and drainage areas are self-reported by owners and agencies; class C dams (no drainage area) rest
  on location alone.
- A dam's reach is where its point snaps; a reservoir spanning several MERIT reaches is represented at one.
- Storage used for DOR: normal storage where reported, else NID storage (the larger of normal and maximum).
