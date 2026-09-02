# Literature review: quantifying uncertainty in spatiotemporal graph neural networks

**Date:** 2026-08-11
**Compiled for:** ddrs (differentiable Muskingum-Cunge routing on the MERIT river network). Four parallel literature sweeps: (A) Bayesian/ensemble methods, (B) conformal prediction, (C) direct probabilistic output heads, (D) hydrology and environmental applications. ~70 papers surveyed; citations verified against arXiv/publisher pages during the sweep.

---

## 1. Taxonomy

UQ methods for spatiotemporal GNNs (ST-GNNs) split along two axes:

1. **What uncertainty is captured.** Aleatoric (irreducible data noise, via predictive distributions), epistemic (model/parameter uncertainty, via posteriors or ensembles), and, uniquely on graphs, **structural** uncertainty in the adjacency itself.
2. **What it costs.** Single-pass heads (quantile, parametric likelihood, evidential) vs multi-pass sampling (MC dropout, ensembles, SWAG, diffusion models) vs post-hoc wrappers (conformal) that touch neither training nor architecture.

The domain that developed nearly all of this is road-traffic forecasting (DCRNN/Graph WaveNet backbones on PEMS/METR-LA); hydrology has mature UQ only in the graph-free basin-lumped LSTM line.

---

## 2. Bayesian and ensemble methods

**The canonical benchmark** is Wu et al. (KDD 2021, arXiv:2105.11982), "Quantifying Uncertainty in Deep Spatiotemporal Forecasting": a decision-theoretic comparison of Bayesian (MC dropout, SG-MCMC) vs frequentist (quantile, interval-loss) UQ under identical graph and grid backbones, on traffic/COVID/air quality. Finding: Bayesian methods are more robust in the mean; frequentist direct heads are far cheaper and competitive on sharpness.

**MC dropout** is nearly free at training time but needs 20-50 forward passes and is the consistently weakest epistemic estimator in every comparative study (samples come from one mode of one model).

**Deep ensembles** are the consistent empirical winner for epistemic quality. Mallick et al. (arXiv:2204.01618; IEEE T-ITS 2024) show hyperparameter-diverse ensembles (Bayesian-optimization-sampled configurations, plus simultaneous quantile regression heads per member and a Gaussian copula over configurations) beat seed-only ensembles on DCRNN traffic forecasting. Cost: M full training runs.

**Weight-averaging posteriors** occupy the sweet spot. DeepSTUQ (Qian et al., ICDE 2023, arXiv:2208.05875; extended TKDE 2024 with PAC-Bayes analysis and multi-horizon conformal calibration) combines a heteroscedastic Gaussian NLL head (aleatoric) with MC dropout + Adaptive Weight Averaging (SWA-family, epistemic) + temperature scaling, beating both dropout and full ensembles at roughly one training run. Notably, **no published paper applies SWAG or a Laplace approximation directly to an ST-GNN forecaster**, a genuine gap.

**Bayesian GNNs with adjacency uncertainty** (Zhang et al., AAAI 2019 BGCN; Pal et al. node-copying variant; a GlobalSIP extension to time series on graphs) are the only family placing uncertainty on the graph structure itself, but have never scaled beyond small citation graphs and require sampling whole graphs.

**Surveys:** Wang et al., "Uncertainty in Graph Neural Networks: A Survey" (TMLR 2024, arXiv:2403.07185); "Uncertainty Quantification on Graph Learning" (arXiv:2404.14642).

---

## 3. Conformal prediction (distribution-free, post-hoc)

The organizing question is **what breaks exchangeability** and what restoring coverage costs.

**Static graphs, transductive:** nothing breaks. CF-GNN (Huang, Jin, Candès, Leskovec, NeurIPS 2023) proves split CP is valid under a permutation-invariance condition and trains a topology-aware correction GNN to shrink intervals up to 74%. DAPS (Zargarbashi et al., ICML 2023) and SNAPS (NeurIPS 2024) diffuse nonconformity scores along edges for efficiency. All give exact finite-sample marginal coverage.

**Inductive/dynamic graphs:** exactness survives only under structural constructions, NodeEx (ICLR 2024, recompute calibration scores against the current graph) or unfolded representations (Davis et al., ICLR 2025). Otherwise one accepts a quantified coverage gap: NAPS (Clarkson, ICML 2023) and NCPNET (KDD 2025, arXiv:2507.02151, temporal GNNs) via Barber et al.'s weighted-exchangeability bound.

**Time series:** ACI (Gibbs & Candès, NeurIPS 2021) and Conformal PID Control (Angelopoulos et al., NeurIPS 2023) give assumption-free long-run coverage under arbitrary shift; EnbPI and SPCI (Xu & Xie, ICML 2021/2023) give asymptotic marginal/conditional coverage under mixing; HopCPT (Auer, Gauch, Klotz, Hochreiter, NeurIPS 2023) uses a modern Hopfield network to reweight past errors by regime similarity, **evaluated on streamflow gauges among other domains**. The foundational theory is Barber, Candès, Ramdas, Tibshirani, "Conformal prediction beyond exchangeability" (Annals of Statistics 2023): fixed weights buy coverage minus an explicit total-variation gap.

**Spatiotemporal composites:** Mao, Martin, Reich (JASA 2024) prove local approximate exchangeability for spatial processes (calibrate within a neighborhood); CopulaCPTS (ICLR 2024) calibrates cross-horizon dependence for simultaneous multi-step bands; **STACI (arXiv:2503.04981) is a topology-aware conformal method built specifically for directed stream/river networks** with flow-respecting nonconformity scores plus ACI-style temporal adaptation, the closest match to gauge-network streamflow; CoRel (Cini et al., ICML 2025, arXiv:2502.09443) trains a graph quantile model on the residuals of any pretrained forecaster over correlated series.

Survey: Sun, "Conformal Methods for Quantifying Uncertainty in Spatiotemporal Data" (arXiv:2209.03580).

---

## 4. Direct probabilistic output heads

**Quantile heads** (MQ-RNN lineage, Wen et al. 2017): single-pass, assumption-free, but per-node marginals with no epistemic term and quantile-crossing issues (fixed architecturally in PE-GQNN, arXiv:2409.18865, or via joint coverage-width losses in QpiGNN).

**Parametric likelihood heads:** the travel-demand literature's clearest lesson is that **the distributional assumption matters more than the backbone**. Prob-GNN (Wang et al., IEEE T-ITS 2024, arXiv:2303.04040) compares Gaussian/truncated-Gaussian/Laplace/Poisson heads across ST-GNN backbones (truncated Gaussian and Laplace win; uncertainty stable under COVID-scale domain shift). STZINB-GNN (Zhuang et al., KDD 2022) uses zero-inflated negative binomial for sparse counts. For streamflow, non-negative, heavy-tailed, near-zero-inflated in dry regimes, this is the directly transferable finding, and it converges with hydrology's own conclusion (asymmetric Laplacian mixtures, §6).

**Evidential heads** (Deep Evidential Regression, Amini et al., NeurIPS 2020): both uncertainty types in one deterministic pass. Ported to ST-GNNs by Feng et al. 2023 (evidential DCRNN, Electronic Research Archive) and to spatiotemporal drought forecasting (Neural Computing & Applications 2025). Graph-native evidential work is mostly classification: Graph Posterior Network (Stadler et al., NeurIPS 2021), Natural Posterior Network (Charpentier et al., ICLR 2022, exponential-family likelihoods, attractive for discharge-like targets). Caveat: the epistemic term's theoretical status is contested (Meinert et al., AAAI 2023).

**Generative/diffusion:** the only family producing coherent **joint spatiotemporal sample paths** (needed when the downstream quantity is a trajectory functional: flood volume, peak timing). TimeGrad (ICML 2021) → DiffSTG (SIGSPATIAL 2023, non-autoregressive graph diffusion, 4-14% CRPS gains) → GCRDD (ADMA 2023) → SpecSTG (arXiv:2401.08119, spectral-domain diffusion, ~3x faster). Cost: 2-4 orders of magnitude more inference compute; aleatoric and epistemic conflated in sample spread.

**GP/neural-process hybrids:** Deep Graph GPs (Jiang et al., IEEE T-ITS 2022); STGNP (Hu et al., KDD 2023), graph neural processes for extrapolation to unobserved locations, the direct analogue of prediction in ungauged basins.

---

## 5. Hydrology: mature where the graph is absent, absent where the graph matters

**Basin-lumped LSTM UQ is settled.** Klotz et al. (HESS 2022) benchmarked GMM/CMAL/UMAL mixture-density heads plus MC dropout on CAMELS; CMAL (countable mixture of asymmetric Laplacians) wins because streamflow errors are skewed and heavy-tailed. This machinery ships in NeuralHydrology and runs operationally in Google Flood Hub (Nearing et al., Nature 2024: CMAL-headed global LSTM matching GloFAS at 5-day lead in ungauged basins). Baste et al. (EGUsphere 2026) show probabilistic heads specifically rescue extreme-event capture. Conformal wrappers exist (HopCPT on gauges; WCI-MDN, WRR 2026, weighted conformal over MDN heads on CAMELS-AUS).

**Every river-network-structured model surveyed is deterministic.** HydroNets (Moshe et al. 2020), CAMELS GNNs (Sun et al., WRR 2021; HESS 2022), physics-guided river-graph models (Jia et al., SDM 2021), topology-focused GNN work (Kirschstein & Sun, ICML 2024, river-tree over-squashing is why vanilla GNNs fail to exploit topology), **differentiable Muskingum-Cunge routing (Bindas et al., WRR 2024; Song et al. 2025)**, and the broader dPL program (Tsai et al., Nat. Comm. 2021; Feng et al., WRR 2022) all report point skill only. Shen et al. (Nat. Rev. Earth Environ. 2023) explicitly lists UQ as an open challenge for differentiable modeling. The lone partial exception is the USGS Delaware line (Zwart et al., JAWRA 2023; Chen et al. 2021), which gets uncertainty from data-assimilation ensembles, not the learned model.

**Two results frame the opportunity:**

- **Ruparell et al. (2026, arXiv:2607.03217)**, the single most relevant paper for ddrs. Probabilistic basin LSTMs routed downstream through Hayami routing: sampling upstream ensemble members independently **averages the uncertainty away** (downstream ensembles badly under-dispersed); the joint spatial distribution of upstream runoff must be preserved (quantile matching largely restores dispersion). Uncertainty is not a per-reach attachment; it is a spatially correlated object the routing operator propagates.
- **GenCast (Price et al., Nature 2025)**, proof that a deterministic graph-structured physical forecaster (GraphCast) can be upgraded to a calibrated generative ensemble beating ECMWF ENS on 97%+ of targets, scored with CRPS.

---

## 6. Synthesis: tradeoffs and the recipe the literature points to

| Family | Epistemic? | Passes at inference | Guarantee | Best exemplar |
|---|---|---|---|---|
| Quantile head | no | 1 | none | Mallick T-ITS 2024 (with ensemble) |
| Parametric NLL head | no | 1 | none | Prob-GNN |
| Evidential head | yes (contested) | 1 | none | GPN / NatPN / evidential DCRNN |
| MC dropout | weak | 20-50 | none | (baseline only) |
| Deep ensemble | yes (best) | M | none | Mallick et al. |
| SWA/SWAG | yes | K | none | DeepSTUQ (AWA); SWAG-on-ST-GNN unpublished |
| Diffusion/generative | conflated | S x N steps | none | DiffSTG, GenCast |
| Conformal wrapper | n/a | +0 | marginal coverage | CF-GNN, STACI, HopCPT, CoRel |

The composite recipe the traffic literature converged on (DeepSTUQ/TKDE 2024): **heteroscedastic or correctly-supported output head (aleatoric) + weight-averaged posterior or small hyperparameter-diverse ensemble (epistemic) + per-horizon conformal calibration (guarantee)**. Untested on river networks.

Evaluation: report CRPS (or MIS) plus coverage/width (PICP/MPIW) plus reliability (PIT); point metrics alone rank the families in a different, misleading order (Wu et al. 2021).

## 7. Gaps relevant to ddrs

1. **Parameter uncertainty.** KAN-emitted hydraulic parameters (Manning's n, geometry, leakance-type terms) are point estimates with no posterior. The leakance identifiability NO-GO (gauge = Σ over upstream network, sum not invertible per-reach) is precisely a statement about wide, correlated per-reach parameter posteriors, an argument only quantifiable with UQ. A Laplace/SWAG posterior over the KAN head would be the first such study on any ST-graph forecaster.
2. **Forcing uncertainty.** Probabilistic Q' (CMAL-headed runoff) propagated through the routing operator with joint spatial structure preserved (quantile matching per Ruparell, or copula/generative sampling). The differentiable solver routes ensembles cheaply; the sparse backward is untouched (post-hoc sampling, no autograd change).
3. **Predictive uncertainty at gauges.** Differentiability is an asset: CRPS-trainable probabilistic heads through the routing solve, which generic GNNs cannot support. Conformal calibration (STACI's flow-respecting scores, or HopCPT per-gauge regimes; CopulaCPTS for multi-day simultaneous hydrograph bands) layers on post-hoc at zero training cost.

No published work combines learned routing physics with any of the three. Nearest neighbors: Ruparell 2026 (classical routing of ML ensembles) and GenCast (weather precedent).

---

*Full per-paper detail (method, guarantee type, calibration metric, cost, links) is in the four agent survey outputs from the 2026-08-11 session; key arXiv ids inline above.*
