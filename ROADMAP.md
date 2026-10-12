# TabDGP — Roadmap

Decided in metabeta-v3's Phase 1 grill (2026-10-11; its root ADR 0005 and
`packages/mixed` ADR 0005), moved here from metabeta-v3's `ROADMAP.md` when the repository
was created. Order reflects dependencies. Specs and tickets are GitHub Issues on this
repository.

- [ ] **1 The flat SCM.** The repository on mixedbench's template; the flat structural causal
  model with a marginal catalogue and typed post-hoc columns, written from scratch with scamd
  and TabICL's prior read for ideas; the hyperprior config with overrides; tests; the tensor
  library decided in this spec
- [ ] **2 Grouping nodes.** Group-level and within-group covariates, the imbalance draw of
  metabeta's envelope rule 7, within-group time, binary treatment; metabeta.mixed's slice A1
  starts after this. Before its spec, the prior-art search (`docs/prior-art.md`) extended to
  the newest reports of TabPFN, Mitra and LimiX (technical reports, papers and posts on how the
  cutting edge simulates tabular training data, not its source code), amending the ADRs or the
  remaining roadmap items where they warrant it; then whether TabPFN v2's DAG can be built from
  what is published and, if so, whether it replaces the Cauchy DAG or joins it as a second
  graph family
- [ ] **3 Measurement pass and proofs.** The measurement pass over mixedbench's cached tables
  ([mixedbench#71](https://github.com/adkipnis/mixedbench/issues/71): the quirk statistics per
  column and per factor, released on PyPI before this item's spec) and the two proof
  tests, the two-sample test and the containment test
- [ ] **4 Quirk axes.** The quirk axes the pass justifies and the matched, widened hyperprior;
  its flat case matched to a comparable sample of flat tables, not to grouped tables alone
  ([mixedbench#72](https://github.com/adkipnis/mixedbench/issues/72)); the marginal catalogue
  extended by the families the pass finds missing (candidates: gamma, exponential, Pareto,
  counts, censoring at a floor or ceiling, the node's own shape without rank matching); the fit
  also decides whether MLP width and depth are drawn per node, like the activation, rather than
  per dataset ([#15](https://github.com/adkipnis/tabdgp/issues/15) kept them per dataset for
  want of a reason either way)
- [ ] **5 Outcome node.** The optional outcome node for white-box misspecified datasets
- [ ] **6 Extensions for other amortization targets.** Data that amortized inference and
  prior-data fitted networks (PFNs) train on beyond GLMMs, each a new node type or an entry of
  the outcome catalogue rather than a redesign: survival data (event times, censoring
  mechanisms, frailty grouping) for metasurv; item response theory (IRT) data (persons and
  items as crossed factors with features on each, binary or ordered responses); missingness
  at random or not at random as indicator nodes; dyadic and network data (two factors sharing one level
  set); spatial dependence (a kernel on the node noise). Each extension is specced when a
  consumer needs it

## Release and paper

Decided on 2026-10-11 (ADR 0002): a package release and a JOSS paper first; the matched
hyperprior's validation as a section of the metabeta paper; a standalone paper only with a
finding of its own.

- [ ] **7 Release.** After item 4 or 5: PyPI, the Zenodo DOI, the demo notebook, and the
  JOSS paper as a citable handle; the README states the proofs as consistency checks on the
  summary vector, never as realism
- [ ] **8 Quirk census ownership.** Before the mixedbench paper is drafted: decide whether the
  quirk statistics of the measurement pass (mb#71) go into mixedbench's registry census or
  stay for TabDGP's paper
- [ ] **9 Downstream ablations.** Once metabeta.mixed's slice A1 trains: the posterior trained
  on TabDGP against an i.i.d. prior and a TabICL-style prior, scored on mixedbench; then each
  quirk axis switched off by an override in turn. The first goes into the metabeta paper
- [ ] **10 Standalone paper.** Only if item 9's per-quirk ablation shows an effect and a second
  consumer (item 6, metasurv) exists: TMLR, or NeurIPS Datasets & Benchmarks in a cycle other
  than mixedbench's; the Table Representation Learning workshop as a first test
