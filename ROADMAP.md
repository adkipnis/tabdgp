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
  starts after this
- [ ] **3 Measurement pass and proofs.** The measurement pass over mixedbench's cached tables
  (a mixedbench issue: the quirk statistics per column and per factor) and the two proof
  tests, the two-sample test and the containment test
- [ ] **4 Quirk axes.** The quirk axes the pass justifies and the matched, widened hyperprior
- [ ] **5 Outcome node.** The optional outcome node for white-box misspecified datasets
