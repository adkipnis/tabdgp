---
status: accepted
date: 2026-10-22
---

# 0005 - The outcome node, drawn by default

Every real table has a response, which the SCM draws by default as the outcome node `y` (#35).
y is the last node, at the row unit, and no node reads it.
Its type is drawn from shares, which include the new column type `count`.
Its noise scale is a field of its own.
An override draws a dataset without y.
A model (ADR 0007) draws a y of its own on the covariates, after the draw.
The SCM's y is what the hyperprior matches to real responses.

## Decisions

1. A dataset's outcome source (`outcome_source`) is a point, `'scm'` by default or `'none'`:
   - under `'scm'`, y is a node like the others; under `'none'`, the dataset has no y.
   - every table of the registry (MixedBench's list of real grouped tables and their documented models) has a response, so `grouped()` and `flat()` draw y, and the proofs (the tests that compare generated and real tables) certify it.
   - an override sets the other point, never shares, since the two return different datasets and no corpus mixes them.
2. y's column is planned last, outside the design-column budget, its type chosen with the shares `outcome_type_shares`:
   - the default is a round guess: `continuous` 0.6, `binary` 0.2, `count` 0.1, `categorical` and `ordered` 0.05 each.
   - precedent: in the practice survey (4,325 mixed models stated in 2,244 papers), 53 % of responses are Normal, 26 % Bernoulli, counts and ordered scales the rest.
   - a categorical or ordered y has at most `outcome_max_levels` levels (10).
   - y's marginal, its k and its permuted labels follow the column rule (the draws every covariate column makes), so a multiclass or ordered y needs nothing new.
3. A Poisson outcome is integer-valued and neither continuous nor ordered, so the column types gain `count`:
   - its values are int64, drawn from a `poisson` marginal whose log rate (`poisson_log_rate`) is a field.
   - the SCM draws no count covariates until the measurement pass (MixedBench's statistics of each quirk on the real tables) admits them; a table's count covariate is one design column.
4. y is the last node, at the row unit, and no node reads it (a sink):
   - its parents are drawn by the Cauchy DAG (directed acyclic graph, whose edge logits add a Cauchy-distributed term of each of their two nodes) among every earlier node, latent ones included.
   - a latent group-level parent makes y endogenous; `latent_nodes = 0` switches this off.
   - in a grouped table, y draws a group share and a group factor like any row-unit node (ADR 0003).
   - a group-level outcome is a later preset's.
5. y's noise scale is a field of its own (`outcome_noise_scale`), `LogUniform(0.1, 3)`, a guess to be fitted:
   - the shared `noise_scale` bounds a node's noise share at one half, but an outcome's unexplained share reaches far higher in real tables.
6. y's node value (the score) is rank-matched onto y's marginal or post-hoc type, like a covariate's.
7. No learnability filter rejects a dataset, so an outcome without parents is a valid table.
8. Time, treatment and the GP (Gaussian process) and tree mechanism families are not bound to the outcome node, so each is left for a later decision.

## Considered options

| question | chosen | rejected |
|---|---|---|
| outcome by default | drawn | drawn only when asked for |
| SCM and GLMM | two models, joined by the deviation | one GLMM with an SCM part |
| y's place | a sink at the row unit | any node |
| y's marginal | the column rule | exempt from post-processing |
| y's mechanism | the nodes' families | a head of its own (a mechanism only y uses) |
| label noise | none | label flips, target-only subgroup effects |
| learnability | no filter | ExtraTrees and shared-class filters |

1. **Outcome by default:**
   - y drawn only when asked for would be certified under a proof configuration of its own (a hyperprior only the proofs use), and `grouped()`'s tables would lack the column every registry table has.
2. **SCM and GLMM:**
   - A marginal rank-matched onto the GLMM's (generalized linear mixed model's) linear predictor η has no likelihood, so the two are not one family.
   - They answer different questions (what real responses look like, what a GLMM's posterior is), so one knob joins them, the deviation (ADR 0006).
3. **y's place:**
   - TabICL v2 and TabPFN v2 and v3 put y on any node, so a covariate may descend from y.
   - As a sink, y's own noise reaches no covariate, so a deviation is endogenous only through a latent parent, which `latent_nodes = 0` switches off.
4. **y's marginal:**
   - TabPFN v2 exempts y from post-processing (the marginal or post-hoc type applied after the SCM), but the registry's responses have real marginals, to be matched.
5. **y's mechanism:**
   - No tabular prior ablates a mechanism for y alone (e.g. TabPFN v1's multinomial head).
6. **Label noise:**
   - Label flips are rare among the tabular priors (only O'Prior and TabFM).
   - O'Prior's target-only subgroup effects are the group share, which every node has.
7. **Learnability:**
   - A rejection changes the distribution the proofs certify.
   - #15's "no forced child" (a latent node may have no children) already lets a node be unpredictable, and the registry has null results.

## Consequences

- **The proofs:** certify the SCM's y under `grouped()`.
  - Their summary vector gains the response's statistics, which MixedBench's measurement pass measures.
- **One type per corpus:** a study's override (`outcome_type_shares = {'continuous': 1}`), not a preset.
  - A model states its response family instead (`Glm(family='gaussian')`).
- **Extension points, none built:**
  - time and treatment (ADR 0003), which have no role in the registry to fit against.
  - the GP family, though no summary statistic tells function classes apart yet.
  - a group-level outcome.
