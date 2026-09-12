# Equifinality / identifiability — literature survey and proposed experiments (2026-07-08)

Produced by a Fable 5 subagent with live web verification (Crossref/arXiv/PMLR);
all references verified, none flagged [UNVERIFIED]. Commissioned from the paper
session (`~/projects/ddr_equifinality`) to design the next experiment wave after
the v2 findings (`docs/2026-07-07-lstm-equifinality-v2-findings.md`).

---

## 1. Hydrology / environmental identifiability lineage

- **Equifinality & GLUE:** Beven & Binley 1992 (Hydrol. Process. 6:279–298, 10.1002/hyp.3360060305); Beven 2006 "manifesto" (J. Hydrol. 320:18–36, 10.1016/j.jhydrol.2005.07.007) — explicitly warns calibrated parameters compensate forcing error; Beven & Freer 2001 (J. Hydrol. 249:11–29); Beven & Binley 2014 "GLUE: 20 years on" (Hydrol. Process. 28:5897–5918).
- **Dynamic identifiability:** Wagener et al. 2003 DYNIA (Hydrol. Process. 17:455–476, 10.1002/hyp.1135) — identifiability is time/regime-varying; Wagener & Wheater 2006 (J. Hydrol. 320:132–154) — weakly identifiable params carry noisy attribute relationships (relevant to our backwards n–DA slope).
- **Profile likelihood:** Raue et al. 2009 (Bioinformatics 25:1923–1929, 10.1093/bioinformatics/btp358) — flat profile = structural non-identifiability; template for n-vs-geometry profiles.
- **Sloppiness:** Gutenkunst et al. 2007 (PLoS Comput. Biol. 3:e189); Transtrum et al. 2015 (J. Chem. Phys. 143:010901) — stiff/sloppy eigendirections, the hydrology↔ML bridge; Chis et al. 2016 (Math. Biosci. 282:147–161) — sloppiness ≠ non-identifiability (the distinction between "n is sloppy" and "n is a compensator"); Vollert et al. 2023 (Environ. Model. Softw. 159:105578). **GAP: no verified FIM-sloppiness application to routing/rainfall-runoff — a novelty claim for us.**
- **Fisher-information OED:** Vrugt et al. 2002 PIMLI (WRR 38:1312); Hsu & Yeh 1989 (WRR 25:1025–1040); Pukelsheim 2006.
- **Response surfaces:** Sorooshian & Gupta 1983 (WRR 19:260–268) — the ancestor of loss-landscape analysis; Duan et al. 1992 SCE-UA (WRR 28:1015–1031); Gupta et al. 1998 multi-objective (WRR 34:751–763); Bárdossy & Singh 2008 (HESS 12:1273–1283) — compensating boundary optima are non-transferable.
- **Forcing-error compensation (closest precedent to our whole program):** Kavetski, Kuczera & Franks 2006a/b BATEA (WRR 42:W03407/W03408, 10.1029/2005WR004368, 10.1029/2005WR004376) — calibration yields storm-dependent parameter compensation for rainfall error; **Renard et al. 2010 (WRR 46:W05521, 10.1029/2009WR008328)** — input vs structural error non-identifiable from streamflow alone (the theoretical crux); Renard et al. 2011 (WRR 47:W11516); Vrugt et al. 2008 DREAM (WRR 44:W00B09); Del Giudice et al. 2013 (HESS 17:4209–4225) — explicit bias process protects parameters from absorbing input error; McMillan et al. 2011/2012 (realistic forcing-error magnitudes).
- **Differentiable/ML-parameterized geoscience:** Tsai et al. 2021 dPL (Nat. Commun. 12:5988, 10.1038/s41467-021-26107-z) — regionalization-regularizes-equifinality hypothesis (the claim our program tests); Feng et al. 2022 (WRR 58:e2022WR032404) — untrained-variable checks; Feng et al. 2023 (HESS 27:2357–2373) — ungauged extrapolation as identifiability hallmark; Shen et al. 2023 review (Nat. Rev. Earth Environ. 4:552–567); Höge et al. 2022 neural-ODE (HESS 26:5085–5102); Nearing et al. 2021 (WRR 57:e2020WR028091); Bindas et al. 2024 (WRR 60:e2023WR035337, 10.1029/2023WR035337).

## 2. ML loss-landscape methodology that transfers

- Li et al. 2018 (NeurIPS, arXiv:1712.09913) — filter-normalized 1D/2D loss slices comparable across independently trained models.
- Garipov et al. 2018 (NeurIPS, arXiv:1802.10026), Draxler et al. 2018 (ICML, PMLR 80:1309–1318) — mode connectivity / low-loss curves and barrier heights.
- Frankle et al. 2020 (ICML, PMLR 119:3259–3269, arXiv:1912.05671) — linear-interpolation loss barrier (cheapest basin test).
- Ghorbani et al. 2019 (ICML, PMLR 97:2232–2241); Yao et al. 2020 PyHessian; Sagun et al. 2016 (arXiv:1611.07476) — Hessian eigenspectrum: near-zero bulk (degenerate/compensating) vs outlier edge (identified).
- Hochreiter & Schmidhuber 1997 flat minima; Keskar et al. 2017; **Dinh et al. 2017 (ICML, PMLR 70:1019–1028) — reparameterization caveat: sharpness is scale-dependent, so use dimensionless/log parameters**; Foret et al. 2021 SAM (ICLR, arXiv:2010.01412).
- Ainsworth et al. 2023 Git Re-Basin (ICLR); Entezari et al. 2022 (ICLR) — permutation alignment. **Key note: run all barrier/connectivity/curvature analyses in PHYSICAL (n, p, q per reach) space, not KAN weight space — physical space is permutation-invariant by construction, sidestepping re-basin entirely.**

## 3. Proposed experiments (ranked: decisiveness × feasibility)

Notation: L_X(θ) = deterministic 96-window loss under Q'-source X at physical parameter field θ; θ_A = arm A's converged (n, p, q) field.

| # | Experiment | Tests | Cost |
|---|---|---|---|
| E1 | **Cross-Q' swap matrix** L_X(θ_Y), with n-only and geometry-only swaps | Is θ tuned to its own forcing (compensation) or forcing-agnostic? Renard-2010 operationalized | minutes–1 h, no retrain |
| E2 | **Linear loss barrier** between arm checkpoints, evaluated under EACH Q' (physical space; Garipov one-bend path if barrier high) | One shared basin vs forcing-specific basins — the "selective" signature is a barrier that depends on which Q' evaluates it | ~1 h |
| E3 | **Profile-likelihood slices**: global n-scaling α (re-fit geometry at each α) and geometry-scaling β (re-fit n), under each Q', per DA-stratum | Structural vs practical identifiability (Raue); distinguishes "n sloppy" from "n compensator" (Chis) | hours |
| E4 | **Hessian eigenspectrum by parameter class** (block n / p / q, log-space params, stochastic Lanczos via HVP or FD-of-analytic-gradient) | Is n in the near-zero bulk (flat) or the stiff edge? Eigenvalue-gap quantification of "selective"; novel FIM-sloppiness application to routing | hours |
| E5 | **Level-vs-slope decomposition**: impose physically-forward n–DA slope holding level (and vice versa), measure ΔL under each Q'; correlate n-divergence with ROUTED-flow divergence (not lateral-inflow disagreement) | Is the backwards DA slope doing compensation work? Reframes refuted H2 onto the routed-flow axis | hours |
| E6 | **2D loss surface** (n-scale α × geometry-scale β) gridded under each Q', minima overlaid | Flagship figure: does the valley floor MOVE with forcing (compensation) or stay pinned (identifiable)? | hours, embarrassingly parallel |
| E7 | **Held-out signature test**: timing signatures not in the loss (rising-limb celerity, peak lag, FDC mid-segment) per arm | Is n constrained by the physics that should constrain it? (Feng-2022 strategy) | post-hoc only |
| E8 | **BATEA-lite retrain**: add learnable per-basin inflow-bias multiplier, retrain arms | CAUSAL test: if n was compensating, its divergence collapses onto the bias term | 2 retrains, ~4–6 h |

**Minimal decisive set: E1 + E2 + E3** (near-zero cost). E4 adds mechanism; E6 is the paper figure; E8 is the causal confirmation.

Expected patterns: under "n is a compensator" — swap-loss inflation carried by the n-swap; low barrier under own-Q' but real barrier under cross-Q'; flat n-profile with wide compensating valley; n-block eigenvalues in the near-zero bulk; valley floor shifts with Q'. Under "n is identifiable" — n transfers across forcings; low barrier under both; sharp parabolic n-profile; stiff n-direction; pinned minimum.

**Enabling infra (one item):** a loss-eval mode that consumes an ARBITRARY per-reach (n, q_spatial, p_spatial) field from NetCDF instead of the KAN head output — reuse the probe machinery (deterministic windows, seed 42), skip head forward, inject loaded tensors. Everything E1–E6 reduces to calls of this binary.

## 4. BibTeX (verified load-bearing set)

```bibtex
@article{beven2006manifesto,
  author={Beven, Keith}, title={A manifesto for the equifinality thesis},
  journal={Journal of Hydrology}, year={2006}, volume={320}, number={1--2},
  pages={18--36}, doi={10.1016/j.jhydrol.2005.07.007}}
@article{wagener2003dynia,
  author={Wagener, Thorsten and McIntyre, Neil and Lees, M. J. and Wheater, H. S. and Gupta, H. V.},
  title={Towards reduced uncertainty in conceptual rainfall-runoff modelling: dynamic identifiability analysis},
  journal={Hydrological Processes}, year={2003}, volume={17}, number={2},
  pages={455--476}, doi={10.1002/hyp.1135}}
@article{raue2009profile,
  author={Raue, Andreas and Kreutz, Clemens and Maiwald, Thomas and Bachmann, Julie and Schilling, Marcel and Klingm{\"u}ller, Ursula and Timmer, Jens},
  title={Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood},
  journal={Bioinformatics}, year={2009}, volume={25}, number={15},
  pages={1923--1929}, doi={10.1093/bioinformatics/btp358}}
@article{gutenkunst2007sloppy,
  author={Gutenkunst, Ryan N. and Waterfall, Joshua J. and Casey, Fergal P. and Brown, Kevin S. and Myers, Christopher R. and Sethna, James P.},
  title={Universally sloppy parameter sensitivities in systems biology models},
  journal={PLoS Computational Biology}, year={2007}, volume={3}, number={10},
  pages={e189}, doi={10.1371/journal.pcbi.0030189}}
@article{transtrum2015sloppiness,
  author={Transtrum, Mark K. and Machta, Benjamin B. and Brown, Kevin S. and Daniels, Bryan C. and Myers, Christopher R. and Sethna, James P.},
  title={Perspective: Sloppiness and emergent theories in physics, biology, and beyond},
  journal={The Journal of Chemical Physics}, year={2015}, volume={143}, number={1},
  pages={010901}, doi={10.1063/1.4923066}}
@article{chis2016relationship,
  author={Chis, Oana-Teodora and Villaverde, Alejandro F. and Banga, Julio R. and Balsa-Canto, Eva},
  title={On the relationship between sloppiness and identifiability},
  journal={Mathematical Biosciences}, year={2016}, volume={282},
  pages={147--161}, doi={10.1016/j.mbs.2016.10.009}}
@article{kavetski2006bayesian2,
  author={Kavetski, Dmitri and Kuczera, George and Franks, Stewart W.},
  title={Bayesian analysis of input uncertainty in hydrological modeling: 2. Application},
  journal={Water Resources Research}, year={2006}, volume={42}, number={3},
  pages={W03408}, doi={10.1029/2005WR004376}}
@article{renard2010understanding,
  author={Renard, Benjamin and Kavetski, Dmitri and Kuczera, George and Thyer, Mark and Franks, Stewart W.},
  title={Understanding predictive uncertainty in hydrologic modeling: The challenge of identifying input and structural errors},
  journal={Water Resources Research}, year={2010}, volume={46}, number={5},
  pages={W05521}, doi={10.1029/2009WR008328}}
@article{tsai2021calibration,
  author={Tsai, Wen-Ping and Feng, Dapeng and Pan, Ming and Beck, Hylke and Lawson, Kathryn and Yang, Yuan and Liu, Jiangtao and Shen, Chaopeng},
  title={From calibration to parameter learning: Harnessing the scaling effects of big data in geoscientific modeling},
  journal={Nature Communications}, year={2021}, volume={12}, pages={5988},
  doi={10.1038/s41467-021-26107-z}}
@article{bindas2024improving,
  author={Bindas, Tadd and Tsai, Wen-Ping and Liu, Jiangtao and Rahmani, Farshid and Feng, Dapeng and Bian, Yuchen and Lawson, Kathryn and Shen, Chaopeng},
  title={Improving River Routing Using a Differentiable {M}uskingum--{C}unge Model and Physics-Informed Machine Learning},
  journal={Water Resources Research}, year={2024}, volume={60}, number={1},
  pages={e2023WR035337}, doi={10.1029/2023WR035337}}
@inproceedings{li2018visualizing,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Advances in Neural Information Processing Systems 31 (NeurIPS)},
  year={2018}, note={arXiv:1712.09913}}
@inproceedings{garipov2018loss,
  title={Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs},
  author={Garipov, Timur and Izmailov, Pavel and Podoprikhin, Dmitry and Vetrov, Dmitry P. and Wilson, Andrew Gordon},
  booktitle={Advances in Neural Information Processing Systems 31 (NeurIPS)},
  year={2018}, note={arXiv:1802.10026}}
@inproceedings{frankle2020linear,
  title={Linear Mode Connectivity and the Lottery Ticket Hypothesis},
  author={Frankle, Jonathan and Dziugaite, Gintare Karolina and Roy, Daniel M. and Carbin, Michael},
  booktitle={Proceedings of the 37th International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={119}, pages={3259--3269}, year={2020}, note={arXiv:1912.05671}}
@inproceedings{ghorbani2019investigation,
  title={An Investigation into Neural Net Optimization via Hessian Eigenvalue Density},
  author={Ghorbani, Behrooz and Krishnan, Shankar and Xiao, Ying},
  booktitle={Proceedings of the 36th International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={97}, pages={2232--2241}, year={2019}, note={arXiv:1901.10159}}
@inproceedings{dinh2017sharp,
  title={Sharp Minima Can Generalize For Deep Nets},
  author={Dinh, Laurent and Pascanu, Razvan and Bengio, Samy and Bengio, Yoshua},
  booktitle={Proceedings of the 34th International Conference on Machine Learning (ICML)},
  series={PMLR}, volume={70}, pages={1019--1028}, year={2017}, note={arXiv:1703.04933}}
```

## 5. Framing notes

1. Closest end-to-end precedent: Kavetski 2006b + Renard 2010. Our program is their
   differentiable-ML, spatially distributed, routing-parameter analog — a citable gap.
2. The sloppiness bridge (Transtrum/Gutenkunst ↔ Ghorbani/Sagun) lets one
   eigenspectrum figure speak to both hydrology and ML audiences; no verified prior
   FIM-sloppiness application to routing exists.
3. Chis 2016 provides the precise vocabulary for our two hypotheses: "n is sloppy"
   (poorly constrained, structurally identifiable) vs "n is a compensator"
   (structural non-identifiability with input error). E1+E3 separate these.
4. All landscape analyses in physical (n, p, q) space with log/dimensionless
   coordinates (Dinh 2017 caveat); never interpolate KAN weights directly.
