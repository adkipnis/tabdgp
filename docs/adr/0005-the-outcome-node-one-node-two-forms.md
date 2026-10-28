---
status: accepted
date: 2026-10-22
amended: 2026-10-23, #41: the form is a point, never drawn, and its field is outcome_form; 2026-10-23, ADR 0007: a form is a model, `'glm'` and `'glmm'` the built-in models' names, no `Hyperprior.glmm()`; 2026-10-27, #65: `family_shares` is `y_family_shares`; 2026-10-28, ADR 0007: the forms are `'none'` and `'scm'` again, a model is called on the covariates after the draw; 2026-10-28, #67: the form is the outcome's source and its field `outcome_source`, the scm form y from the SCM, the glmm form the GLMM; the module `hyperprior` is `scm`, `draw` is `drawDataset`
---

# 0005 - The outcome node: one node, from the SCM or a model, the SCM by default

The outcome joins the grouped SCM of #1 and #22. Grilled on
2026-10-22 after a prior-art sweep of what the tabular priors give their target that a
covariate node does not, the decision is: **the outcome is one node of the generating graph,
role `outcome`, name `y`, at the row unit and last in order, so a sink; a field,
`outcome_source`, sets its source: `scm`, a node like the others, or a model, `glm` and `glmm` a
likelihood on the design matrix (ADRs 0006 and 0007); `grouped()` and `flat()` draw y from the SCM
by default, and the proofs certify it.** The words are `CONTEXT.md`'s: outcome node, outcome source, deviation, count, marginal, post-hoc
type, group share.

The sweep (TabPFN v1-3, TabICL v1-2, O'Prior, SurvivalPFN, SurvPFN, Do-PFN, scamd): the tabular
priors give y a class-count draw with permuted labels, a learnability filter on (X, y) (classes
present in both halves, ExtraTrees beats the mean) and, in TabICL v2 and TabPFN v2, a place on
any node of the SCM, with x downstream of y; an own noise level is rare (O'Prior, metabeta-v2's
R² cap), label flips and target-only subgroup effects appear once; none gives y a minimum parent
count, none ablates a y mechanism alone, none matches y to real data. scamd has no outcome node.

## Decisions

- **One node, drawn by default.** The outcome is a node of the graph with the observed columns'
  machinery: a parent set from the Cauchy DAG among every earlier node, latent ones included
  (a latent group-level parent is the endogeneity ADR 0003 asked a switch for; the switch is
  `latent_nodes`, and in the GLMM the deviation of ADR 0006), a mechanism and, in a
  grouped table, a group share and a group factor like any row-unit node. Every registry row
  has a response, so a matched table has one: `grouped()` and `flat()` draw y from the SCM,
  `outcome_source = 'none'` is a study's override, `'glm'` and `'glmm'` the model-based sources
  (ADR 0007). The field is a point, `'none'`, `'scm'`, `'glm'`, `'glmm'` or a model, and an
  override replaces it by another point, never by shares, since a dataset of each source returns
  different things and no corpus mixes them.
- **A sink at the row.** y is last in the node order and no node reads it: TabICL's y-as-node,
  where x may descend from y, is not taken over, since a GLMM's covariates are causes or
  correlates of its outcome, never its effects. The outcome is drawn once per row; a
  group-level outcome is a later preset's.
- **What the SCM's y gets beyond a covariate node**, two fields: `outcome_noise_scale`, since
  the shared `noise_scale` bounds a node's noise share at one half and the outcome's share of
  unexplained variance reaches far higher in real tables, default `LogUniform(0.1, 3)`, a guess,
  to be fitted; and `outcome_type_shares`, since the mix of response types (the practice
  survey's families: Normal 53 %, Bernoulli 26 %, counts and ordered scales the rest) is not
  the covariate mix, default a round guess at those shares, over the five types below. The
  marginal, a categorical y's k and its permuted labels, and the rank matching are the column
  rule's: a multiclass or ordered outcome needs nothing new.
- **A `count` type and a `poisson` marginal.** A Poisson outcome is integer-valued and neither
  continuous nor ordered, so `count` joins the column types (int64, outside the design-column
  budget until the measurement pass admits count covariates), with a Poisson marginal whose
  log rate is a field, one more entry of the catalog. scamd-audit T11 had delayed counts to the
  measurement pass; the outcome brings the type early, the covariate case still waits for the
  pass.
- **No learnability filter.** A rejection changes the distribution the proofs certify, #15's
  "no forced child" already lets a node be unpredictable, and the registry has null results:
  an outcome with no parents is a valid table.
- **What is matched**: the defaults of the SCM's y, to the response statistics MixedBench's
  measurement pass adds once its exclusion of the response is lifted. The GLMM's priors
  are coverage-first (ADR 0006) and never fitted; only its covariates are matched, since they
  are `grouped()`'s.
- **One outcome type per corpus** is a point override, `outcome_type_shares = {'continuous': 1}`
  or `y_family_shares = {'gaussian': 1}`, a study of that type, not a preset.

## Considered options

- **One model, a GLMM whose linear predictor always carries an SCM part** (λ ∈ [0, ∞)): a
  marginal rank-matched onto η has no likelihood, so the two are not one family; and the
  matched outcome and the trained-on outcome answer different questions (what real responses look
  like; what a GLMM's posterior is), so they stay two with one knob between them, the
  deviation (ADR 0006).
- **"Never drawn unless asked for"**, `CONTEXT.md`'s wording until this ADR: it would certify
  the outcome under a proof configuration of its own and leave `grouped()`'s tables without
  the column every registry table has.
- **Not taken over, from the sweep**: y as any node (TabICL v2, TabPFN v2 and 3), label flips
  (O'Prior, TabFM), O'Prior's target-only subgroup effects (the group share is that, on every
  node), TabPFN v2's exemption of y from post-processing (the registry's responses have real
  marginals, to be matched), the ExtraTrees and shared-class filters, a dedicated
  mechanism family for y (TabPFN v1's multinomial head).
- **Time and treatment** (ADR 0003 deferred them to the item that reads them) and **the GP and
  tree mechanism families** (#1's closing comment): neither is bound to the node, the random
  slope's rule "time first when present" works without time existing, and the registry has no
  time role to fit against; each is its own later item. scamd-audit G11's objection to a GP
  family (no summary statistic tells function classes apart) is for that item to answer.

## Consequences

- `graph` places y last at the row unit; `column` gains the role `outcome`, the type `count`
  and the `poisson` marginal; `scm` gains `outcome_source`, `outcome_noise_scale` and
  `outcome_type_shares`; `drawDataset` returns y among the columns. Every default draw changes: a
  version change.
- The proofs' summary vector gains the response's statistics; MixedBench's measurement pass
  measures the response for it.
- `CONTEXT.md` gains outcome source, deviation, prior context and count; the outcome node's entry is
  rewritten; random effects are α.
- ADR 0003's open points close: the endogeneity switch is `latent_nodes` and the deviation;
  time and treatment go to their own item.
