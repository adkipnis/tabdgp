# Related simulators

Other simulators of tabular datasets than scamd and TabICL's prior (those two are in
[`prior-art.md`](prior-art.md)), surveyed on 2026-10-11 from primary sources (repositories, papers,
docs) so the provenance pass need not be repeated. Each claim carries its URL; the figures were
collected by an agent and spot-checked only for "Mind the Gap?" (the paper and its headline hold,
the exact AUC and coverage figures were not re-read). Acronyms: prior-data fitted network (PFN), structural causal model (SCM), classifier two-sample test (C2ST), amortized Bayesian inference (ABI), item response theory (IRT).

**TL;DR.** No public generator covers TabDGP's niche: a randomized hyperprior over *grouped* tables, with grouping factors inside the SCM, certified against a registry of real grouped tables. Each piece exists somewhere else, though. The R study tools (fabricatr, simstudy, faux, simglm) have first-class nested and crossed grouping. Causal benchmarks (CSuite, gCastle, CDT) return the true graph. "Mind the Gap?" (May 2026) already runs a C2ST plus k-NN coverage of real against synthetic tables at the level of whole tables. Its warning matters most: the TabICL prior stayed at AUC > 0.95 after 86k hyperparameter configurations.

## Table

| Tool | Generates | Mechanism | Column types | White-box output | Config | Reproducibility | Licence, status | Realism check |
|---|---|---|---|---|---|---|---|---|
| TabPFN v1 prior ([paper](https://arxiv.org/abs/2207.01848), [standalone code](https://github.com/automl/tabpfn-v1-prior)) | flat X + y | MLP/BNN, SCM, GP; prior bag | numeric, thresholded categoricals | no (batch of x, y) | kwargs; hyperparameters drawn from meta-distributions ("differentiable hyperparameters") | torch iterator per batch | Apache-2.0; last push 2025-11 | downstream accuracy |
| TabPFN v2 prior ([Nature 2025](https://europepmc.org/article/PMC/PMC11711098)), 2.5 ([report](https://arxiv.org/abs/2511.08667)) | flat X + y (y is a node) | scale-free DAG (growing network with redirection), vector nodes; MLP, tree, discretization edges | numeric, categorical (nearest prototype), quantized, MCAR missing | not released | not released | not released | **generator code not released** (Nature "Code availability") | downstream accuracy |
| TabForestPFN ([paper](https://arxiv.org/abs/2405.13396), [code](https://github.com/FelixdenBreejen/TabForestPFN)) | flat X + y | random forest boundaries + TabPFN prior | numeric | no | YAML | per training run | MIT; stale since 2024-08 | downstream, fine-tuning |
| Mitra ([paper](https://arxiv.org/abs/2510.21204), [blog](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models)) | flat X + y | mixture: SCM, DT/ET/GB/RF fit to synthetic data, directly sampled RF | Gaussian numeric, geometric-cardinality categoricals | no | — | — | weights via AutoGluon; prior code not found | priors selected by real-data performance, diversity, distinctiveness |
| LimiX / LimiX-2 ([paper](https://arxiv.org/abs/2509.03505), [repo](https://github.com/limix-ldm/LimiX)) | flat joint x, y | hierarchical SCM; MLP, conv along rows, tree edges; graph-aware sampling | numeric, categorical | no | — | — | custom licence; generator not in repo | masking calibrated to real missingness; else downstream |
| TFM-Playground priors ([repo](https://github.com/automl/TFM-Playground), ex-[tabularpriors](https://github.com/automl/tabularpriors)); nanoTabPFN ([repo](https://github.com/automl/nanoTabPFN)) | flat X + y | wrappers over TabICL, TICL, TabPFN v1 priors | as wrapped | no | CLI flags | batches dumped to HDF5 | Apache-2.0; active | meta-feature similarity + downstream ([paper](https://arxiv.org/abs/2606.29241)) |
| TabDPT ([paper](https://arxiv.org/abs/2410.18164)) | real OpenML tables, no simulator | — | real | — | — | — | open weights, data | downstream |
| Do-PFN ([repo](https://github.com/jr2021/Do-PFN)); [Do-PFN-prior](https://github.com/oossen/Do-PFN-prior) | flat X, T, interventional y | random SCMs with interventions | numeric | SCM object; likelihood of y given x | Python | Do-PFN-prior: **one integer seed fixes all data** | no licence file | downstream; ATE shrinkage reported ([benchmark](https://hyper.ai/en/papers/2609.03003)) |
| CausalPFN ([repo](https://github.com/vdblm/CausalPFN)) | X, T, Y | DGPs satisfying ignorability | numeric | — | — | — | Apache-2.0; prior code not in repo | downstream |
| CausalFM ([paper](https://arxiv.org/abs/2506.10914), [toolkit](https://github.com/yccm/CausalFM-toolkit)) | X, T, Y; back-door, IV, front-door | SCM-based Bayesian priors | numeric | DataFrame | class per setting, `seed=` | seed per generator | Apache-2.0; active | downstream |
| gCastle ([repo](https://github.com/huawei-noah/trustworthyAI/tree/master/gcastle)) | flat, event sequences | ER, scale-free, bipartite, hierarchical, low-rank DAGs; linear/nonlinear SEM | numeric | **true adjacency `B`** | kwargs | **global `np.random.seed`** | Apache-2.0 | none |
| CDT ([repo](https://github.com/FenTechSolutions/CausalDiscoveryToolbox)) | flat | acyclic graph + mechanism (linear, polynomial, sigmoid, GP, NN) | numeric | true graph | kwargs | — | MIT; last push 2025-10 | none |
| causal-learn ([repo](https://github.com/py-why/causal-learn)) | algorithms; test data only | — | — | — | — | — | MIT | — |
| bnlearn (R [CRAN](https://cran.r-project.org/package=bnlearn), [Python](https://github.com/erdogant/bnlearn)) | flat, forward sampling from a (fitted) Bayesian network | discrete/Gaussian BN | categorical, Gaussian | network | network object | global RNG | GPL≥2 / MIT | none |
| CSuite ([repo](https://github.com/microsoft/csuite)) | 15 fixed datasets, 2–12 nodes | hand-crafted SEMs | continuous, mixed, discrete | **adjacency + interventional test sets** | — | **versioned files** | MIT; archived | none |
| Causal Chambers ([repo](https://github.com/juangamella/causal-chamber)) | real physical measurements | real devices | numeric, images | known physical graph | — | — | CC-BY-4.0 | real by construction |
| SDV ([repo](https://github.com/sdv-dev/SDV), [HMA](https://docs.sdv.dev/sdv/multi-table-data/modeling/synthesizers/hmasynthesizer)) | flat, multi-table, sequential, fit to one real dataset | Gaussian copula, CTGAN; HMA recursive copulas parent→child | all, via metadata | metadata, no graph | `Metadata` object | `reset_sampling()` ([docs](https://docs.sdv.dev/sdv/single-table-data/sampling/sample-realistic-data)) | **Business Source Licence** (MIT after 4 years); v1.38.5 2026-09 | SDMetrics quality report, detection metric ([docs](https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/detection-single-table)) |
| synthcity ([repo](https://github.com/vanderschaarlab/synthcity)) | flat, time series, survival; fit to data | GANs, VAEs, flows, diffusion, BN, ARF | all; **no missing values** | no | plugin kwargs | per plugin | Apache-2.0; last release 2025-05 | fidelity, detection, utility, privacy metrics |
| synthpop ([CRAN](https://cran.r-project.org/package=synthpop)) | flat, fit to data | sequential CART / parametric | all | visit sequence | R args | `set.seed` | GPL-2/3; 2026-09 | utility measures |
| simstudy ([CRAN](https://cran.r-project.org/package=simstudy), [clustered](https://kgoldfeld.github.io/simstudy/articles/clustered.html)) | flat, nested clusters, longitudinal, cluster RCTs, survival | user formulas per level | numeric, binary, categorical, counts | the definition tables | `defData` formula tables | `set.seed` | GPL-3; 2026-02 | none |
| faux ([site](https://www.scienceverse.org/faux/reference/index.html)) | factorial, mixed, **crossed** (`add_random`, `sim_mixed_cc`); `sim_df` from data | multivariate normal + converters | numeric, binary, factors | `design` attribute | function args | seed arg | MIT; 2026-03 | none |
| fabricatr ([site](https://declaredesign.org/r/fabricatr/), [crossed](https://declaredesign.org/r/fabricatr/articles/cross_classified.html)) | nested levels, panel, **cross-classified with correlated joins** | user expressions, `draw_*` | binary, ordered, Likert, counts | levels | `fabricate()` | `set.seed` | MIT; 2024-01 | none |
| simglm ([CRAN](https://cran.r-project.org/package=simglm)) | GLMM up to 3 nesting levels, unbalanced, missing | user GLMM | numeric, binary, counts | the spec | nested list | `set.seed` | MIT; 2026-09 | power |
| simr, lme4 `simulate` ([simr](https://cran.r-project.org/package=simr), [lme4](https://cran.r-project.org/package=lme4)) | outcome given user covariates and grouping | fitted or specified GLMM | outcome only | the model | model object | `set.seed` | GPL≥2 | power |
| simsurv, coxed, survsim ([simsurv](https://cran.r-project.org/web/packages/simsurv/vignettes/simsurv_usage.html), [coxed](https://rdrr.io/cran/coxed/man/sim.survdata.html), [survsim](https://cran.r-project.org/package=survsim)) | survival times; coxed also draws normal X | Weibull, splines, user hazards; flexible baseline | time, status | coxed returns true betas, baseline, marginal effect | args | `set.seed` | GPL; coxed last 2020 | none |
| mirt `simdata`, catR ([simdata](https://rdrr.io/cran/mirt/man/simdata.html), [catR](https://cran.r-project.org/package=catR)) | item responses | (M)IRT item models | dichotomous, polytomous | Theta + item objects | args | `set.seed` | GPL≥3; catR last 2022 | none |
| mice `ampute` ([help](https://rdrr.io/cran/mice/man/ampute.html)) | missingness on given data | MCAR/MAR/MNAR by weighted sum scores | any | patterns, weights (`mads`) | patterns, freq, weights, type | `set.seed` | GPL≥2; 2025-12 | none |
| BayesFlow ([repo](https://github.com/bayesflow-org/bayesflow)) | user simulators; benchmarks (two moons, SIR, Bernoulli GLM) | user code; `HierarchicalSimulator` stacks levels | user | dicts of arrays | Python functions | `rng` Generator per simulator | MIT; v2.0.14 2026-09 | simulation-based calibration |
| sbi ([repo](https://github.com/sbi-dev/sbi)) | user simulators; linear Gaussian, Gaussian mixture | user code | user | — | — | torch seed | Apache-2.0; v0.27 2026-08 | calibration diagnostics |

## Short notes

- **TabPFN v2's prior is the reference design, and it is closed.** It has a scale-free DAG with disjoint subgraphs that make uninformative features, vector-valued nodes, categoricals by nearest prototype, Kumaraswamy warps, quantization, MCAR missingness and **non-i.i.d. rows by mixing prototypes**, a weak form of clustering without group labels. "The code to generate synthetic pre-training data has not been released" ([Nature, Code availability](https://europepmc.org/article/PMC/PMC11711098)). TabPFN-2.5 only says the generator was "substantially improved" ([report](https://arxiv.org/abs/2511.08667)).
- **Realism studies of priors exist now.** "Mind the Gap?" ([arXiv 2605.06343](https://arxiv.org/html/2605.06343)) embeds each table as 70 aggregate statistics and runs an XGBoost discriminator. It reports AUC 0.997–0.9997 for TabICL against real corpora, real-table coverage (k-NN, k = 5) of 6–16%, and no reduction after Tree-structured Parzen Estimator (TPE) search, a genetic algorithm and an 86k-point grid. It also found no correlation between proximity and downstream performance. "Towards Evaluating Data Priors" ([arXiv 2606.29241](https://arxiv.org/html/2606.29241v1)) trains an identical nanoTabPFN on each of TabICL, TabForestPFN, TabPFN v1, TICL and real data, and compares them on meta-features.
- **Causal PFN priors** (Do-PFN, CausalPFN, CausalFM, SurvivalPFN at [rgklab/SurvivalPFN](https://github.com/rgklab/SurvivalPFN), SIC at [arXiv 2603.29475](https://arxiv.org/abs/2603.29475), code pending) all generate the outcome with the covariates. None models grouping.
- **Fit-to-data synthesizers** (SDV, synthcity, synthpop) clone one dataset; none is a prior over datasets. SDV's HMA models parent→child tables with one copula per table and is limited to about 5 tables of depth 1 ([docs](https://docs.sdv.dev/sdv/multi-table-data/modeling/synthesizers/hmasynthesizer)). A grouping factor with group-level covariates is the same structure as a parent table.
- **The R study tools have the grouping vocabulary.** Most of it is hand-specified and fixed rather than drawn from a distribution. simstudy's cluster-size dispersion parameter: "a dispersion of 0 (the default) implies exact balance" ([vignette](https://kgoldfeld.github.io/simstudy/articles/clustered.html)). fabricatr crosses levels with a Gaussian-copula join at Spearman ρ, so cells come out partially filled ([article](https://declaredesign.org/r/fabricatr/articles/cross_classified.html)). faux and simglm also have crossed factors and unbalanced designs.
- **ABI frameworks ship no covariate generators.** BayesFlow's benchmarks draw design matrices as i.i.d. normals (`bernoulli_glm.py`). `HierarchicalSimulator` multiplies batch shapes per level, so it only builds balanced groups ([source](https://github.com/bayesflow-org/bayesflow/blob/main/bayesflow/simulators/hierarchical_simulator.py)). sbi ships two toy simulators. Habermann et al.'s amortized multilevel models use model-specific simulators ([arXiv 2408.13230](https://arxiv.org/abs/2408.13230)).

## (a) Practices worth adopting

1. **Use a standard support metric for the containment test.** The options are k-NN precision/recall coverage as in [Mind the Gap?](https://arxiv.org/html/2605.06343), or PRDC (precision, recall, density, coverage) and α-precision as implemented in [synthcity](https://github.com/vanderschaarlab/synthcity). Report recall (real covered by generated) and precision (generated covered by real) together.
2. **Expect the C2ST bar to be hard, and say why up front.** TabICL's prior could not be tuned below AUC 0.95 ([Mind the Gap?](https://arxiv.org/html/2605.06343)). Widening the matched hyperprior also places mass where no real table lies, which a symmetric C2ST detects. Ablate the summary-feature groups the way the paper does, and report AUC with an interval over splits.
3. **Add a downstream check beside the distributional ones.** Train the same small model on TabDGP and on a baseline prior, then evaluate on mixedbench, following the protocol of [Towards Evaluating Data Priors](https://arxiv.org/html/2606.29241v1). That paper found data similarity "an informative but incomplete proxy".
4. **Put the version on the data.** CSuite versions its datasets so that additions stay backwards compatible ([repo](https://github.com/microsoft/csuite)). Stamp each `Dataset` with the TabDGP version and a hash of the hyperprior defaults.
5. **One integer key, never global state.** [Do-PFN-prior](https://github.com/oossen/Do-PFN-prior) is "fully seeded: on-the-fly data generation is completely determined by one integer seed". gCastle seeds `np.random` globally ([source](https://github.com/huawei-noah/trustworthyAI/blob/master/gcastle/castle/datasets/simulator.py)), which is the anti-pattern ADR 0001 already avoids.
6. **Return the estimands for the outcome node.** coxed returns the true coefficients, baseline hazard and marginal effect ([help](https://rdrr.io/cran/coxed/man/sim.survdata.html)). Do-PFN-prior exposes the likelihood of y given x for soft labels.
7. **Borrow parameterizations for quirk axes.** If missingness earns an axis: `ampute`'s patterns, frequencies, weights and LEFT/MID/TAIL/RIGHT types ([help](https://rdrr.io/cran/mice/man/ampute.html)), and LimiX's masks calibrated to real missingness by variable type ([paper](https://arxiv.org/abs/2509.03505)). For crossed factors: fabricatr's copula join. For imbalance: simstudy's dispersion. For non-i.i.d. rows: TabPFN v2's prototype mixing.
8. **Plug into the prior ecosystem.** An adapter to [TFM-Playground's prior interface](https://github.com/automl/TFM-Playground/tree/main/tfmplayground/priors) would make TabDGP comparable with every public prior at no cost to the core. A wrapper exposing TabDGP as a BayesFlow `Simulator` would do the same for ABI users.
9. **Pick mechanism families by measured redundancy.** Mitra chooses priors through a generalizability matrix of diversity and distinctiveness ([paper](https://arxiv.org/abs/2510.21204)), which suits choosing TabDGP's mechanism families.

## (b) What is unique, and what is not

**Genuinely distinctive (no counterexample found):**
- **A prior over grouped tables.** No PFN, causal or ABI generator draws grouping factors as SCM nodes with group-level and within-group covariates. The nearest is TabPFN v2's prototype mixing, which has no labels. The R tools have grouping but no randomized hyperprior: every study is hand-written.
- **Certification of the defaults against a registry of real grouped tables, as a release gate.** Published realism work measures priors after the fact (Mind the Gap?, Towards Evaluating Data Priors) or tunes them on downstream accuracy (Mitra). None ships defaults required to pass a C2ST and a containment test.
- **Measure before granting a quirk.** O'PRIOR ([arXiv 2605.18971](https://arxiv.org/abs/2605.18971)) and TabPFN v2 add realism perturbations by design choice, not from a measurement on real tables.

**Not unique:**
- **C2ST and coverage as the method** are already applied to priors at whole-table level (Mind the Gap?). Row-level detection metrics are standard in SDMetrics and synthcity. TabDGP's novelty is the reference corpus and the gating, not the test.
- **"Calibrated to real data"** in a broad sense: Mitra selects priors by real-data performance, LimiX calibrates missingness to real patterns, and Real-TabPFN continues pre-training on real tables ([arXiv 2507.03971](https://arxiv.org/abs/2507.03971)). SDV, synthcity and synthpop are fit to data outright.
- **Returning the graph.** gCastle (`dataset.B`), CDT, CSuite (`adj_matrix.csv`) and bnlearn all return the true graph. Returning the resolved hyperparameters, latent node values and column roles together is rarer but still incremental.
- **Grouping structure itself.** Nested, crossed, imbalanced and correlated-crossed designs exist in fabricatr, simstudy, faux and simglm.
- **Many quirks.** TabPFN v2 has quantization (ties), warps (bounds and skew), MCAR missingness and uninformative features. simstudy has unequal cluster sizes and MCAR/MAR/NMAR. fabricatr has empty crossed cells. Group confounding and leverage points are the ones not found as axes elsewhere.
- **Distributions over hyperparameters.** TabPFN v1 ("differentiable hyperparameters", [README](https://github.com/automl/tabpfn-v1-prior)) and TabICL already use meta-distributions. The typed Pydantic schema is an implementation nicety, not a differentiator.
- **Per-dataset seeding.** Do-PFN-prior derives everything from one seed, and BayesFlow simulators take a `Generator`. The (seed, index) purity contract is cleaner than most but not new.

## Data types beyond GLMMs, and who covers them

| Data type | Simulators |
|---|---|
| Survival, censoring, competing risks | simsurv, coxed, survsim; synthcity survival plugins; SurvivalPFN, SIC, SurvPFN ([arXiv 2606.04564](https://arxiv.org/abs/2606.04564)) |
| IRT, item responses | mirt `simdata`, catR |
| Time series, forecasting | ForecastPFN ([arXiv 2311.01933](https://arxiv.org/abs/2311.01933)); synthcity TimeGAN; SDV sequential |
| Temporal drift | Drift-Resilient TabPFN ([repo](https://github.com/automl/Drift-Resilient_TabPFN)) |
| Longitudinal, panel | simstudy, fabricatr, simglm |
| Missing data (MCAR/MAR/MNAR) | mice `ampute`, simstudy; TabPFN v2 (MCAR only); LimiX masks |
| Causal effects, interventions | Do-PFN, CausalPFN, CausalFM (back-door, IV, front-door), CSuite |
| Causal discovery | gCastle, CDT, bnlearn, CSuite; Causal Chambers (real) |
| Structural equation / latent variable models | simsem ([CRAN](https://cran.r-project.org/package=simsem)) |
| Correlated discrete marginals | GenOrd, bindata, SimCorMultRes ([CRAN](https://cran.r-project.org/package=GenOrd)) |
| Multi-table relational | SDV HMA |
| Survey populations | simPop, synthpop |

## Not verified

The internal seeding and batching of LimiX, Mitra and CausalPFN, whose generator code is not public. The exact prior of SurvivalPFN and SIC (abstracts only). Mitra's final mixture weights. CDT's seeding.
