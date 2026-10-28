---
status: accepted
date: 2026-10-23
amended: 2026-10-26, #56: the prior context is the `Prior`, `GlmOutcome`'s field `prior`; 2026-10-27, #65: y's family is `y_family`, as β's is `beta_family`, its shares `y_family_shares`; 2026-10-27, #66: `GlmOutcome` holds the `Prior` and the `Parameters` as drawn, and `GlmmOutcome` adds its random terms; 2026-10-27, #56: ground truth, not truth
---

# 0007 - The form is a model: a callable over the covariates, the GLM and the GLMM its first cases, the outcome a container per model

ADR 0005 made the outcome one node with two forms; ADR 0006 made the glmm form read
columns and a grouping, never the graph, so that a real table's covariates can carry a
resampled outcome with a known ground truth. #41 then decided that the outcome may be any explicit
function of the covariates a study specifies, the glmm form its first case. Grilled on
2026-10-23, the decision is: **a form is a model, a callable that reads the covariates and a
generator and returns the columns it writes with their ground truth; the generalized linear model
(GLM) and the generalized linear mixed model (GLMM) are TabDGP's own models, configs of
their own in `prior` with a shared skeleton, named by `outcome_form = 'glm'` and `'glmm'`;
every form gets the outcome node's score from the SCM; `Dataset.outcome` holds the form's
container, `None` only when no outcome is drawn; and any table can come in through
`Covariates.fromPandas`.** The words are `CONTEXT.md`'s: form, model, score, deviation,
prior, dataset.

## Decisions

- **The form is a model.** `outcome_form` takes `'none'`, `'scm'`, `'glm'`, `'glmm'` or a
  `Model`, a callable `(Covariates, numpy Generator) -> Outcome`: a function, a
  `functools.partial` with a study's settings bound, or an object with `__call__`. A string
  names a built-in model's default config, and a validator expands it (`'glm'` to `Glm()`,
  `'glmm'` to `Glmm()`), so the resolved hyperprior shows the object; an override of a model's
  field is the object, `outcome_form=Glmm(y_family_shares={'gaussian': 1})`. There is no
  `Hyperprior.glmm()`: `grouped()` already decides what a GLMM needs, and metabeta draws
  `Hyperprior.grouped(outcome_form='glmm')`, passing nothing else. The scm form stays the
  SCM's own: its three fields stay on `Hyperprior`, which is the SCM's config.
- **Two models, one skeleton.** `Glm` holds the fixed block, the families and links, the
  joint scale draw over its blocks, the design matrix coder and its container; `Glmm(Glm)`
  adds the random blocks and `random_effects: PositiveCount`, so a GLMM without random
  effects cannot be written. The difference is the list of blocks `prior` draws over,
  `[fixed]` or `[fixed, one per factor]`, so nothing is written twice. `'glmm'` on a
  hyperprior whose `factors` may be 0 fails at construction; `'glm'` on a grouped table is
  allowed, the pooled ground truth over grouped covariates, a dataset where a fitter's random-effect
  scale should go to zero. A user's model is checked at draw time only, on what it returns.
- **A model resolves its own fields.** `Glm`'s fields are points or distributions of the
  kinds `Hyperprior` uses, resolved inside the call with the generator it is handed, as a
  column resolves its own fields; the per-dataset values are its prior. The field
  kinds and the resolution machinery (`resolveField`, the share ordering, `categories`) move
  from `hyperprior` into a module of their own, `field`, as a base class both configs inherit.
  ADR 0006's tokens, the standard deviations' family and the residual law, are shares fields
  of `Glm` with point defaults, chosen once per dataset by `Shares.choose` where `prior` reads
  them, the chosen names recorded in the prior. The marginal alias `Family` becomes
  `Marginal`, the glossary's word, so that "family" means the response family throughout.
- **One score for every form.** `draw` always propagates y's node through the SCM, as the scm
  form does today, and hands the z-scored value, the score, to the model in `Covariates`
  with the observed columns, their metadata and the grouping. The scm form rank-matches the
  score onto its marginal; a GLM or GLMM adds λ · sd(η) · z(score) to its linear predictor,
  ADR 0006's deviation; a user's model may ignore it. y stays one node of the graph under
  every form. A table without an SCM has no score and so no deviation: the returned deviation
  is zeros and λ is recorded as given.
- **The container.** `Outcome` holds the columns a form writes and their `Column` metadata,
  `y` today, a survival form's `time` and `event` later; `draw` merges them into the dataset's
  columns without copying. Subclasses add the ground truth: `ScmOutcome` the score; `GlmOutcome`
  the prior and ADR 0006's parameters as drawn, with their raw-scale views;
  `GlmmOutcome(GlmOutcome)` each factor's random terms. `Dataset.outcome` is `None` only
  under `'none'`. A user's model may
  return the bare `Outcome`; TabDGP checks the shapes and dtypes of what comes back.
- **Any table in.** `Covariates` is public. `Covariates.fromPandas(table, factors)` builds it
  from a DataFrame with categorical dtypes: types from dtypes, levels and codes from the
  categoricals, factor codes and nesting from the data. A resampled outcome is then the model
  call with a generator from the key's rule, `default_rng([seed, index])`; the oracle test set
  itself is built downstream.
- **The dataset's views.** `Dataset.toPandas()` (coded columns as pandas categoricals, the
  latent nodes left out), `toDesign()` (the design matrix with its names and coding
  statistics, from the coder the GLM uses, so the user and the model see one coding; y and the
  grouping factors are not in it, by the glossary's definition) and `toDict()` (a JSON-ready
  dict of the whole dataset, dtypes recorded, a user's model by its qualified name). pandas is
  a dependency, not an extra: `fromPandas` is on the main path and both consumers carry it.

## Considered options

- **A general `'model'` value beside a field naming the model**: the second field means
  nothing unless the first is `'model'`, a cross-field check the interface can rule out.
- **A prior DSL**, nodes whose parameters are other nodes, sampled in order, the GLMM one
  preset of it: a small probabilistic programming language to maintain, for a generality the
  callable already gives a user, whose prior pipeline is ten lines of numpy inside their model.
- **Nested configs for every form**, the scm form's fields moving into `Scm(...)`: symmetric,
  but it changes the 0.3.0 override spelling and refactors the path the proofs certify, for one
  built-in model with priors.
- **`Hyperprior.glmm(model=Glmm(), **overrides)`**: clunky, and `grouped()` already decides.
- **One class with `random_effects = 0` for the GLM**: no duplication, but a GLMM without
  random effects contradicts the glossary, and the type would let a user write the zero by
  accident. **`'glm'` as a synonym of `'glmm'`** would draw random effects on a grouped table
  under a name that denies them; **a staging spelling** renamed later is two spellings.
- **A generic container** (`model`, `truth`, `context` as dicts): no checks at construction,
  string keys metabeta would have to know.
- **A `Dataset` in, a `Dataset` out** for ready covariates: a real table has no graph, no
  latent nodes, no resolved hyperprior, so most of it would be fake.
- **A token field kind**, drawn uniformly: a new kind whose one distribution shares already
  express, without weights.
- **The score only when λ > 0**: saves a few draws, adds a `None` branch to every model, and
  under the glmm form at λ = 0 y would have no parents in the graph.
- **`toNumpy()` with codes as floats**: hides a coding decision; the design matrix is the
  coding. **pandas as an extra**: right while `toPandas` was its only reader.

## Consequences

- ADR 0005: the form's values; `Hyperprior.glmm()` is gone. ADR 0006: the preset bullet, the
  fields' home, the tokens, the container and the module table are amended; the envelope as
  defaults, the joint scale draw, no caps and the estimands on both scales stand, and the lab
  note on the R2D2M2 prior stays the spec's first ticket.
- `CONTEXT.md`: Form, Preset, Deviation, Prior context and Dataset rewritten; Model and Score
  added.
- `hyperprior` loses the field kinds to `field`; `prior` and `outcome` are new modules; the
  package exports `Covariates`, `Outcome`, `Glm` and `Glmm`.
- No default draw changes: the scm form already computes y's node, and `'none'` draws nothing
  more, so `grouped()` and `flat()` draw the same bits.
- metabeta's `envelope.md` and its ADRs name `Hyperprior.glmm()`; they read
  `grouped(outcome_form='glmm')` now.
- A survival form later writes several columns, keeps its latent times and censoring in its
  container, and redraws its censoring inside its own call; a rejection that changes what a
  preset certifies stays refused for `grouped()` and `flat()`.
