# Literature scan: prescribing the width coefficient p from hydraulic-geometry scaling

`scholar` CLI is not installed (no binary at `~/.local/bin/scholar` or
`~/projects/google-scholar-cli/scholar`; `command -v scholar` fails). Per the skill's fallback,
metadata below is from Crossref and Semantic Scholar API lookups (title/author/journal/year/DOI
confirmed for every entry). Abstracts are quoted verbatim where the API returned one; where the
publisher elides the abstract from both indexes (noted per entry) and the publisher page 403'd on
direct fetch, the relationship description comes from a Google WebSearch snippet, marked
"source: web," never from memory.

## 1. Bindas et al. 2024, verified, abstract verbatim
```bibtex
@article{bindas2024improving, title={Improving River Routing Using a Differentiable Muskingum-Cunge Model and Physics-Informed Machine Learning}, author={Bindas, Tadd and Tsai, Wen-Ping and Liu, Jiangtao and Rahmani, Farshid and Feng, Dapeng and Bian, Yuchen and Lawson, Kathryn and Shen, Chaopeng}, journal={Water Resources Research}, year={2024}, doi={10.1029/2023WR035337}}
```
Retrieved abstract, verbatim: "Synthetic experiments show that while the channel geometry
parameter was unidentifiable, n can be identified with moderate precision." That is the full
extent of what the abstract says about channel geometry: the geometry parameter is
unidentifiable, nothing more. **The p = 21 claim is not stated in the retrieved text.** Confirming
it needs the paper body/config tables, not the abstract.

## 2. Hydraulic-geometry scaling relationships

**2a. Leopold & Maddock 1953.** Abstract elided by publisher in both indexes.
```bibtex
@techreport{leopold1953hydraulic, title={The Hydraulic Geometry of Stream Channels and Some Physiographic Implications}, author={Leopold, Luna Bergere and Maddock, Thomas}, institution={U.S. Geological Survey}, series={Professional Paper 252}, year={1953}, doi={10.3133/pp252}}
```
WebSearch snippet (source: web): depth, width, velocity, suspended load vary with discharge as
power functions, both at-a-station (single cross-section over time) and downstream (fixed
frequency, along the network). **Coefficients not in retrieved text.**

**2b. Moody & Troutman 2002.** Abstract elided by publisher.
```bibtex
@article{moody2002characterization, title={Characterization of the Spatial Variability of Channel Morphology}, author={Moody, John A. and Troutman, Brent M.}, journal={Earth Surface Processes and Landforms}, year={2002}, doi={10.1002/esp.403}}
```
WebSearch snippet (source: web): width/depth spatial variability across reaches spanning five
orders of magnitude of discharge, log-normal and scale-independent of discharge. A secondary web
source (a citing paper, not the abstract) attributes downstream-HG coefficients a=7.2, b=0.5
(width-discharge) to this paper. **Unverified against the original text, flag before using.**

**2c. Andreadis, Schumann & Pavelsky 2013.** Abstract elided by publisher.
```bibtex
@article{andreadis2013simple, title={A Simple Global River Bankfull Width and Depth Database}, author={Andreadis, Konstantinos M. and Schumann, Guy J.-P. and Pavelsky, Tamlin M.}, journal={Water Resources Research}, year={2013}, volume={49}, number={10}, pages={7164--7168}, doi={10.1002/wrcr.20440}}
```
WebSearch snippet (source: web): database built from "a regression relationship between bankfull
discharge, drainage area and hydraulic geometry characteristics," validated against Landsat/
in-situ data, 8-62% mean errors. **Regression coefficients not in retrieved text.**

**2d. Allen & Pavelsky 2018 (GRWL).** Abstract verbatim.
```bibtex
@article{allen2018global, title={Global Extent of Rivers and Streams}, author={Allen, George H. and Pavelsky, Tamlin M.}, journal={Science}, year={2018}, volume={361}, pages={585--588}, doi={10.1126/science.aat0636}}
```
Abstract, verbatim excerpt: "global river and stream surface area at mean annual discharge is
773,000 +/- 79,000 square kilometers... an area 44 +/- 15% larger than previous spatial
estimates." This is the source of the GRWL width database; the abstract itself is about total
surface area, not a W-Q or W-A coefficient. **No scaling coefficients in the retrieved abstract.**

**2e. Lin et al. 2020.** Abstract verbatim.
```bibtex
@article{lin2020global, title={Global Estimates of Reach-Level Bankfull River Width Leveraging Big Data Geospatial Analysis}, author={Lin, Peirong and Pan, Ming and Allen, George H. and Frasson, Renato Prata de Moraes and Zeng, Zhenzhong and Yamazaki, Dai and Wood, Eric F.}, journal={Geophysical Research Letters}, year={2020}, volume={47}, number={7}, doi={10.1029/2019GL086405}}
```
Abstract, verbatim excerpt: "state-of-the-art parameterization schemes only capture 30-40% of the
width variance globally, we developed a machine learning (ML) approach surveying 16 environmental
covariates, which considerably improved the predictive power (R2 = 0.81 and 0.77)." Explicitly
argues simple power-law W=a.Q^b / a.A^b parameterizations are weak (30-40% variance) versus a
16-covariate ML model. **No single a/b pair given, it's an ML model, not a power law.**

**2f. Frasson et al. 2019.** Abstract verbatim.
```bibtex
@article{frasson2019global, title={Global Relationships Between River Width, Slope, Catchment Area, Meander Wavelength, Sinuosity, and Discharge}, author={Frasson, Renato Prata de Moraes and Pavelsky, Tamlin M. and Fonstad, Mark A. and Durand, Michael T. and Allen, George H. and Schumann, Guy and Lion, Christine and Beighley, R. Edward and Yang, Xiao}, journal={Geophysical Research Letters}, year={2019}, volume={46}, number={6}, pages={3252--3262}, doi={10.1029/2019GL082027}}
```
Abstract, verbatim excerpt: "we found width to be directly associated with the magnitude of
meander wavelength and catchment area... power laws between mean annual discharge and width can
predict width typically to -35% to +81%, even when a single relationship is applied across all
rivers with discharge ranging from 100 to 50,000 m3/s." Directly supports a single global
W = a.Q^b relation with a stated accuracy envelope. **Specific a, b values not in the abstract,**
needs the paper's table.

**2g. Bjerklie et al., two candidates, both verified, abstracts elided.**
```bibtex
@article{bjerklie2003evaluating, title={Evaluating the Potential for Measuring River Discharge from Space}, author={Bjerklie, David M. and Dingman, S. Lawrence and Vorosmarty, Charles J. and Bolster, Carl H. and Congalton, Russell G.}, journal={Journal of Hydrology}, year={2003}, volume={278}, pages={17--38}, doi={10.1016/s0022-1694(03)00129-x}}
@article{bjerklie2005estimating, title={Estimating Discharge in Rivers Using Remotely Sensed Hydraulic Information}, author={Bjerklie, David M. and Moller, Delwyn and Smith, Laurence C. and Dingman, S. Lawrence}, journal={Journal of Hydrology}, year={2005}, volume={309}, pages={191--209}, doi={10.1016/j.jhydrol.2004.11.022}}
```
The 2005 title matches "estimating discharge from remotely sensed hydraulic data" most closely.
WebSearch snippet for the 2003 paper (source: web): couples water-surface width, channel slope,
max channel width from aerial/SAR imagery with general hydraulic equations to estimate discharge,
standard errors 50-100%. **Coefficients not in retrieved text for either.**

**2h. Dingman 2007.** Abstract elided by publisher.
```bibtex
@article{dingman2007analytical, title={Analytical Derivation of At-a-Station Hydraulic-Geometry Relations}, author={Dingman, S. Lawrence}, journal={Journal of Hydrology}, year={2007}, volume={334}, pages={17--27}, doi={10.1016/j.jhydrol.2006.09.021}}
```
WebSearch snippet (source: web): derives AHG exponents analytically from a power-law
velocity-depth relation and a power-law cross-section shape (bankfull width, max depth, shape
exponent); exponents depend only on the depth exponent and the shape exponent. At-a-station
(single reach over time), not downstream/network geometry. **Coefficients not in retrieved text.**

**2i. Wilkerson & Parker 2011.** Abstract elided by publisher.
```bibtex
@article{wilkerson2011physical, title={Physical Basis for Quasi-Universal Relationships Describing Bankfull Hydraulic Geometry of Sand-Bed Rivers}, author={Wilkerson, Gregory V. and Parker, Gary}, journal={Journal of Hydraulic Engineering}, year={2011}, volume={137}, number={7}, pages={739--753}, doi={10.1061/(asce)hy.1943-7900.0000352}}
```
WebSearch snippet (source: web): dimensionless relations among bankfull discharge, width, depth,
slope for single-thread sand-bed rivers (median grain size 0.062-0.50 mm); yields a predictive
relation for bankfull discharge as a function of width, depth, slope, grain size. **Coefficients
not in retrieved text.**

**2j. Gleason & Smith 2014 (AMHG).** Abstract elided; PNAS page 403'd on direct fetch.
```bibtex
@article{gleason2014toward, title={Toward Global Mapping of River Discharge Using Satellite Images and At-Many-Stations Hydraulic Geometry}, author={Gleason, Colin J. and Smith, Laurence C.}, journal={Proceedings of the National Academy of Sciences}, year={2014}, volume={111}, pages={4788--4791}, doi={10.1073/pnas.1317606111}}
```
WebSearch snippet (source: web): discharge retrievals within 20-30% of in-situ observations from
Landsat imagery alone via AMHG, which relates multiple at-a-station width observations across a
river's length. **This is a per-reach width-only inversion calibrated per reach, not a single
global W = a.Q^b prescription,** a weaker fit than Frasson 2019/Andreadis 2013 for a single
global p.

**2k. Neal et al. 2012.** Abstract elided; AGU page 403'd on direct fetch, no snippet text
retrievable at all.
```bibtex
@article{neal2012subgrid, title={A Subgrid Channel Model for Simulating River Hydraulics and Floodplain Inundation over Large and Data Sparse Areas}, author={Neal, Jeffrey and Schumann, Guy and Bates, Paul}, journal={Water Resources Research}, year={2012}, doi={10.1029/2012WR012514}}
```
**Cannot state what this paper says about width prescription from retrieved text.** Topically
plausible ("large-scale flood models prescribing width from area/discharge") but unconfirmed.

**2l. Yamazaki et al. 2014 (GWD-LR).** Abstract elided by publisher.
```bibtex
@article{yamazaki2014development, title={Development of the Global Width Database for Large Rivers}, author={Yamazaki, Dai and O'Loughlin, Fiachra and Trigg, Mark A. and Miller, Zachary F. and Pavelsky, Tamlin M. and Bates, Paul D.}, journal={Water Resources Research}, year={2014}, volume={50}, pages={3467--3480}, doi={10.1002/2013WR014664}}
```
WebSearch snippet (source: web): algorithm computes river width directly from satellite water
masks (SRTM Water Body Database) + HydroSHEDS flow direction, for channels between 60S-60N.
**Direct width extraction from imagery, not a discharge/area power law,** no a/b coefficients
exist for this method.

**2m. Brinkerhoff et al. 2019.** Abstract verbatim; note this differs from a literal
"width-discharge relations" paper.
```bibtex
@article{brinkerhoff2019reconciling, title={Reconciling At-a-Station and At-Many-Stations Hydraulic Geometry Through River-Wide Geomorphology}, author={Brinkerhoff, Colin B. and Gleason, Colin J. and Ostendorf, David W.}, journal={Geophysical Research Letters}, year={2019}, doi={10.1029/2019GL084529}}
```
Abstract, verbatim excerpt: "we present evidence... that AMHG can be hydraulically and
geomorphically reconciled with AHG. Our results indicate that AMHG is rightly understood as an
expression of a river-wide model of hydraulics driven by changes in slope imposed upon AHG
physics." A theoretical AMHG-AHG reconciliation using 155 US rivers, not a fitted global
width-discharge regression. **Weak fit** to "width-discharge relations": no other 2019
Brinkerhoff paper matching that description was found in Crossref under his name.

## 3. Summary table

| Relationship | Form | Predictor | Scale | Coefficients retrievable? |
|---|---|---|---|---|
| Leopold & Maddock 1953 | W,d,v power functions of Q | discharge | regional (US) | No |
| Moody & Troutman 2002 | log-normal W/d scaling vs Q | discharge | global (5 continents) | No (a=7.2,b=0.5 pair unverified) |
| Andreadis et al. 2013 | regression: bankfull W,d vs Q, area | discharge + area | global | No |
| Allen & Pavelsky 2018 (GRWL) | width observation database, not a formula | (none) | global | N/A |
| Lin et al. 2020 | ML (16 covariates), beats power law | discharge+area+others | global | No single a/b (ML) |
| Frasson et al. 2019 | power law W vs mean annual Q, -35%/+81% | discharge | global (rivers >90 m) | No |
| Bjerklie et al. 2003/2005 | discharge from width, slope, imagery | width + slope | regional | No |
| Dingman 2007 | analytical AHG exponents (theory) | depth (at-a-station) | theoretical | No |
| Wilkerson & Parker 2011 | dimensionless bankfull W,d,slope vs Q | discharge | sand-bed rivers | No |
| Gleason & Smith 2014 (AMHG) | per-reach W-Q inversion, not global | width (multi-station) | per-reach | No |
| Neal et al. 2012 | subgrid channel model (unconfirmed fit) | unknown | large-scale/global | No text retrieved |
| Yamazaki et al. 2014 (GWD-LR) | direct width extraction from imagery | (imagery) | global (60S-60N) | N/A |
| Brinkerhoff et al. 2019 | AMHG-AHG theory, not a fitted regression | (none) | per-reach (155 US) | N/A |

## Verification count

14 entries attempted (13 candidate relationship papers + Bindas et al. 2024). All 14 verified via
Crossref and/or Semantic Scholar: title/author/journal/year/DOI confirmed for every one; zero
dropped, zero fabricated. Verbatim abstracts retrieved for 5/14 (Bindas 2024, Allen & Pavelsky
2018, Lin et al. 2020, Frasson et al. 2019, Brinkerhoff et al. 2019); the other 9 have
publisher-elided abstracts in both indexes plus HTTP 403 on direct AGU/Wiley/PNAS fetch, so their
descriptions rely on WebSearch snippets (marked "source: web" per entry) rather than the paper's
own text.

Best-fit candidates for a single prescribable global p: **Frasson et al. 2019** (explicit global
power law, stated accuracy envelope) and **Andreadis et al. 2013** (explicit regression against
both discharge and drainage area, matching what the KAN head already conditions on). Both need
their coefficient tables pulled from the paper body, not just the abstract.

## 3. Coefficients from full texts

### 3.1 Dissertation, verbatim (`Bindas_PhD_Dissertation.docx`, converted with `pandoc -t plain`)

Chapter 2 (JRB-scale delta MC) states the p = 21 assumption directly:

> "For this, because at-a-site hydraulic geometries (Gleason, 2015; Leopold & Maddock, 1953;
> Orlandini & Rosso, 1998) lead to a power-law relation between top width (w [m]) and depth
> (d [m]), we can assume such a relationship: w = pd^q (2-6) ... where p [m] and q [-] are linear
> and exponential parameters ... To simplify the task (and because it is not sensitive based on
> our observations), we assumed p = 21 based on preliminary data fitting to USGS hydraulic
> geometries from field surveys of gages in the JRB."

So p = 21 is a site-specific fit to USGS field-survey hydraulic geometry at JRB gauges, not a
value taken from any published global power law. Confirmed later in the same chapter: "linear
channel coefficient p values were also never recoverable in single parameter tests and decreased
resulting NSE values when used as a tunable parameter... Thus, we did not include a learnable p
in the rest of the study."

Chapter 4 (delta MCv1.14 / KAN routing) restates the same power law, citing a fourth source, and
gives the Manning's-equation inversion used to get depth from discharge:

> "This relationship was derived in (Bindas et al. 2024) and is based on the power-law Leopold and
> Maddock river width and depth relationships (Leopold and Maddock 1953; Orlandini and Rosso 1998;
> Gleason 2015 Feb)." d = [Q_t n(q+1) / (p S_0^0.5)]^(3/(5+3q)) (4-2); w = pd^q (4-3)

There p becomes a learnable KAN output (unlike Chapter 2's fixed 21).

Dissertation reference list, verbatim, for the three hydraulic-geometry citations: Leopold, L.
B., & Maddock, T. Jr. (1953), "The hydraulic geometry of stream channels and some physiographic
implications," USGS Professional Paper 252, doi 10/ggj7hw. Orlandini, S., & Rosso, R. (1998),
"Parameterization of stream channel geometry in the distributed modeling of catchment dynamics,"
WRR 34(8), 1971-1985, doi 10.1029/98wr00257. Gleason, C. J. (2015), "Hydraulic geometry of
natural rivers: A review and future directions," Progress in Physical Geography, doi 10/f7dsqm.

Chapter 4 also sources measured (not power-law) top width from a random forest model, for
validation only, not to set p: trained by Chang et al. 2024 on HYDRoSWOT data. Full reference:
Chang SY, Ghahremani Z, Manuel L, Erfani SMH, Shen C, Cohen S, Van Meter KJ, Pierce JL, Meselhe
EA, Goharian E. 2024. "The geometry of flow..." WRR 60(10):e2023WR036733, doi:10.1029/2023WR036733.

### 3.2 `ddr` config: where the p_spatial default of 21 lives in code

`grep -rn "p_spatial" config/ src/ddr/` in `/home/tbindas/projects/ddr` confirms:
`src/ddr/validation/configs.py:100`, `"p_spatial": [1.0, 200.0],  # Leopold & Maddock width
coefficient, Log-space (m)` (the learnable bound range); line 112, `"p_spatial": 21,` inside a
`defaults` dict commented "Default parameter values for physical processes when not learned";
`src/ddr/geometry/predictor.py:314`, `default_p = self._defaults.get("p_spatial", 21.0)`, used
when the KAN does not learn p. `git log --all -S'"p_spatial": 21' --oneline` returns one commit,
`6b472d1 Feature: Abstraction of the dataset (#105)`, a refactor moving hardcoded defaults into a
typed Pydantic config; no commit message or comment ties 21 to a published power law, only the
generic "Leopold & Maddock" tag. Consistent with the dissertation: 21 is the JRB field-survey fit
carried forward as ddr's fallback, not a value from any paper below.

### 3.3 Open-access full texts, coefficients quoted verbatim

**Moody and Troutman (2002).** Direct fetch 403'd (USGS-hosted PDF and the AGU/Wiley page both).
Coefficients recovered from a paper quoting the equations verbatim: Modaresi Rad et al. 2024,
"Enhancing River Channel Dimension Estimation..." (JGR: ML and Computation, 10.1029/2024JH000173),
fetched from `https://repository.library.noaa.gov/view/noaa/68731/noaa_68731_DS1.pdf` (open
access), converted with `pdftotext`: "One of the earliest of these efforts was proposed by Moody
and Troutman (2002) for global channels (Equations 1 and 2). w = 7.2Q^(0.5+-0.02) (1); d =
0.27Q^(0.3+-0.01) (2); where w and d are bankfull top-width and depth, respectively, and Q is the
discharge." A citing source, not the original text, so flagged as such, but independently
corroborated below via Frasson et al. 2019's own validation table.

**Frasson et al. (2019).** Direct fetch succeeded via the University of Bristol repository mirror
(`https://research-information.bris.ac.uk/ws/files/196233179/Full_text_PDF_final_published_version_.pdf`
main text, `.../ws/files/196233182/Supplimentary_Information.pdf` SI), downloaded and converted
with `pdftotext -layout` (citation block confirms DOI 10.1029/2019GL082027). Main text states
their own width-area power law but never prints fitted a, b as numbers: "We assumed that the
relationship between river width and catchment area follows a power law of the form W = a*A^b
(2)... coefficients a and b are fitted using reduced major axis regression... applied in logarithm
space." Fit quality (Figure 3e caption, verbatim): "r2 = 0.25" -- a weak fit, a, b not given as
text anywhere fetched. The paper does not fit its own discharge-width law; it validates Moody and
Troutman (2002) against a larger sample using WBMsed mean annual discharge: "the equation by Moody
and Troutman (2002) agrees well with the medians of the discharge box plots between 100 and
5,000 m3/s... The normalized interquartile range varied from 85% to 165%... (Table S2)." Table
S2, verbatim (discharge m3/s; median width m; "M&T width" m; sample "rivers wider than 90 m,
60N-56S"): Q=100, median 79, M&T 72; Q=1000, median 241, M&T 228; Q=10000, median 780, M&T 720.
Every M&T value matches 7.2*Q^0.5 to the nearest meter, independently confirming the coefficient.

**Andreadis, Schumann and Pavelsky (2013).** Direct fetch 403'd on AGU/Wiley; the UNC Carolina
Digital Repository landing page returned only metadata; Semantic Scholar's API record for this
DOI reports `openAccessPdf.status: CLOSED`. Partial corroboration only, via a downstream
reimplementation that cites this paper: HydroMT's `rivdph_powlaw` in
`https://raw.githubusercontent.com/Deltares/hydromt/v0.9.0/hydromt/workflows/rivers.py` (fetched
directly): `def rivdph_powlaw(qbankfull, hc=0.27, hp=0.30, min_rivdph=1.0): return
np.maximum(hc * qbankfull**hp, min_rivdph)`, cited to "Andreadis et al. (2013)... WRR, 49(10),
7164-7168." This is d = 0.27*Qbf^0.30, identical to the Moody and Troutman depth equation, i.e.
Andreadis et al. 2013 appears (via this indirect citation) to reuse Moody and Troutman's depth
coefficients rather than refit new ones. No width coefficient recovered; the paper's own
drainage-area regression (per its abstract) remains unverified from primary text.

**Neal, Schumann and Bates (2012).** Every fetch attempted was blocked with HTTP 403: AGU/Wiley
full text, the Semantic-Scholar-listed `pdfdirect` OA link, and the Bristol research-information
page (redirect relists only the same two blocked links). No text retrieved; nothing filled in
from memory for this paper's width relation.

**Yamazaki, Kanae, Kim and Oki (2011)**, WRR (the CaMa-Flood model paper). Direct fetch succeeded
via a NASA GSFC mirror (`https://gmao.gsfc.nasa.gov/gmaoftp/sarith/ROUTING_MODEL/docs/yamazaki.pdf`),
converted with `pdftotext` (header confirms doi:10.1029/2010WR009726). Verbatim: "Channel width,
W, and bank height, B, are not resolved in GDBD and SRTM30. Hence, they were empirically
determined as a function of maximum 30 day upstream runoff, Rup (m3/s)... W = max[1.00*Rup^0.7,
10.0] (10); B = max[0.035*Rup^0.5, 1.00] (11). The parameters ... were carefully calibrated by
trial and error." Discharge here is maximum 30-day upstream runoff, not mean annual or bankfull
discharge, and the exponents were tuned to make routing perform well, not fit to observed
cross-sections -- a different kind of coefficient than the Leopold and Maddock regressions above.

**Lin et al. (2020).** Not re-fetched; unchanged from Section 2's verbatim abstract: single power
laws capture only 30-40% of global width variance, versus a 16-covariate ML model (R2 = 0.81,
0.77), i.e. no single a, b pair to prescribe.

### Summary

Frasson et al. 2019 does not report its own fitted width-discharge coefficients as text; it
validates Moody and Troutman (2002)'s w = 7.2*Q^0.5, 85-165% normalized IQR error over
100-10,000 m3/s. That equation, and its depth analog d = 0.27*Q^0.3 (also reused unchanged by
Andreadis et al. 2013 per the HydroMT code), is the only power law here with verified coefficients
traceable to primary or near-primary text. Andreadis et al. 2013's own drainage-area regression
and Neal et al. 2012's width relation remain unverified, both paywalled with no open copy found.
The dissertation's p = 21 is unrelated to any of these: an empirical fit to USGS field-survey
hydraulic geometry at JRB gauges, chosen because sensitivity tests found p had negligible effect
on routed discharge; that JRB fit, not a literature power law, is what ddr hardcodes as default.
