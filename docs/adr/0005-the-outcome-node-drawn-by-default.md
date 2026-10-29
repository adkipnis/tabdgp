---
status: accepted
date: 2026-10-22
---

# 0005 - The outcome is a node of the SCM, drawn by default. It is a sink at the row unit.

1. The outcome node joins the SCM (structural causal model) of #1 and #22 as the column `y` with role `outcome` (#35).
2. A dataset's outcome source (`outcome_source`) is a point, `'scm'` by default or `'none'`:
   - under `'scm'`, y is a node like the others; under `'none'`, the dataset has no y.
   - every registry table has a response, so `grouped()` and `flat()` draw y, and the proofs certify it.
   - an override sets the other point, never shares, since the two return different datasets and no corpus mixes them.
   - a model (a GLM or GLMM, ADR 0007) is no source of the node: it draws a y of its own on the covariates after the draw.
3. y's column is planned last, outside the design-column budget, its type chosen with the shares `outcome_type_shares`:
   - the default is a round guess: `continuous` 0.6, `binary` 0.2, `count` 0.1, `categorical` and `ordered` 0.05 each.
   - precedent: in the practice survey, 53 % of responses are Normal, 26 % Bernoulli, counts and ordered scales the rest.
   - a categorical or ordered y has at most `outcome_max_levels` levels (10).
   - y's marginal, its k and its permuted labels follow the column rule, so a multiclass or ordered y needs nothing new.
4. A Poisson outcome is integer-valued and neither continuous nor ordered, so the column types gain `count`:
   - its values are int64, drawn from a `poisson` marginal whose log rate (`poisson_log_rate`) is a field.
   - the SCM draws no count covariates until the measurement pass admits them; a table's count covariate is one design column.
5. y is the last node, at the row unit, and no node reads it (a sink):
   - its parents are drawn by the Cauchy DAG (directed acyclic graph) among every earlier node, latent ones included.
   - a latent group-level parent makes y endogenous; `latent_nodes = 0` switches this off.
   - in a grouped table, y draws a group share and a group factor like any row-unit node (ADR 0003).
   - a group-level outcome is a later preset's.
6. y's noise scale is a field of its own (`outcome_noise_scale`), `LogUniform(0.1, 3)`, a guess to be fitted:
   - the shared `noise_scale` bounds a node's noise share at one half, but an outcome's unexplained share reaches far higher in real tables.
7. y's node value (the score) is rank-matched onto y's marginal or post-hoc type, like a covariate's.
8. No learnability filter rejects a dataset, so an outcome without parents is a valid table.
9. The SCM's y is what the hyperprior matches to real responses; a model's priors are coverage-first (ADR 0006) and never fitted.
10. Time, treatment and the GP (Gaussian process) and tree mechanism families are not bound to the outcome node, so each is its own later item.

## Considered options

| question | chosen | rejected |
|---|---|---|
| outcome by default | drawn | drawn only when asked for |
| SCM and GLMM | two models, joined by the deviation | one GLMM with an SCM part |
| y's place | a sink at the row unit | any node |
| y's marginal | the column rule | exempt from post-processing |
| y's mechanism | the nodes' families | a head of its own |
| label noise | none | label flips, target-only subgroup effects |
| learnability | no filter | ExtraTrees and shared-class filters |

1. **Outcome by default:**
   - y drawn only when asked for would be certified under a proof configuration of its own, and `grouped()`'s tables would lack the column every registry table has.
2. **SCM and GLMM:**
   - A marginal rank-matched onto the GLMM's (generalized linear mixed model's) linear predictor η has no likelihood, so the two are not one family.
   - They answer different questions (what real responses look like, what a GLMM's posterior is), so one knob joins them, the deviation (ADR 0006).
3. **y's place:**
   - TabICL v2 and TabPFN v2 and v3 put y on any node, so a covariate may descend from y.
   - As a sink, y's own noise reaches no covariate, so a deviation is endogenous only through a latent parent, which `latent_nodes = 0` switches off.
4. **y's marginal:**
   - TabPFN v2 exempts y from post-processing, but the registry's responses have real marginals, to be matched.
5. **y's mechanism:**
   - No tabular prior ablates a mechanism for y alone (e.g. TabPFN v1's multinomial head).
6. **Label noise:**
   - Label flips are rare among the tabular priors (only O'Prior and TabFM).
   - O'Prior's target-only subgroup effects are the group share, which every node has.
7. **Learnability:**
   - A rejection changes the distribution the proofs certify.
   - #15's "no forced child" already lets a node be unpredictable, and the registry has null results.

## Consequences

- **The proofs:** certify the SCM's y under `grouped()`.
  - Their summary vector gains the response's statistics, which MixedBench's measurement pass measures.
- **One type per corpus:** a study's override (`outcome_type_shares = {'continuous': 1}`), not a preset.
  - A model states its response family instead (`Glm(family='gaussian')`).
- **Later items, none built:**
  - time and treatment (ADR 0003), which have no role in the registry to fit against.
  - the GP family, which has to answer scamd-audit G11: no summary statistic tells function classes apart.
  - a group-level outcome.
