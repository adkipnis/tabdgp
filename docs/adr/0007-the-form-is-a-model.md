---
status: accepted
date: 2026-10-23
amended: 2026-10-26, #56: the prior context is the `Prior`, `GlmOutcome`'s field `prior`; 2026-10-27, #65: y's family is `y_family`, as β's is `beta_family`, its shares `y_family_shares`; 2026-10-27, #66: `GlmOutcome` holds the `Prior` and the `Parameters` as drawn, and `GlmmOutcome` adds its random terms; 2026-10-27, #56: ground truth, not truth; 2026-10-28, #68: a model's outcome is drawn after the dataset by `modelOutcome` (was `resampleOutcome`) on the key's child stream, `outcome_form` takes `'none'` or `'scm'`, `Dataset.covariates` holds the score, `ScmOutcome` folds into `Outcome`, and a deviation without a score fails; 2026-10-28, #68: a fitter reads the covariates and a model's outcome as a pair, `toDesign` moves to `Covariates` and `toDict` goes; 2026-10-28, #67: the form is the outcome's source and its field `outcome_source`, the glmm form the GLMM, and the title follows; `Hyperprior` is `Scm`, the base of every config `Hyperprior`, `draw` is `drawDataset`
---

# 0007 - A model is a callable over the covariates, the GLM and the GLMM its first cases, the outcome a container per model

ADR 0005 made the outcome one node with two sources; ADR 0006 made the GLMM read
columns and a grouping, never the graph, so that a real table's covariates can carry a
resampled outcome with a known ground truth. #41 then decided that the outcome may be any explicit
function of the covariates a study specifies, the GLMM its first case. Grilled on
2026-10-23, the decision is: **the outcome's source may be a model, a callable that reads the
covariates and a generator and returns the columns it writes with their ground truth; the
generalized linear model (GLM) and the generalized linear mixed model (GLMM) are TabDGP's own
models, configs of their own in `prior` with a shared skeleton, named by `outcome_source = 'glm'`
and `'glmm'`; every source gets the outcome node's score from the SCM; `Dataset.outcome` holds the
source's container, `None` only when no outcome is drawn; and any table can come in through
`Covariates.fromPandas`.** The words are `CONTEXT.md`'s: outcome source, model, score, deviation,
prior, dataset.

## Decisions

- **The source may be a model.** `outcome_source` takes `'none'`, `'scm'`, `'glm'`, `'glmm'` or a
  `Model` (until 2026-10-28: a model is called after the draw), a callable
  `(Covariates, numpy Generator) -> Outcome`: a function, a
  `functools.partial` with a study's settings bound, or an object with `__call__`. A string
  names a built-in model's default config, and a validator expands it (`'glm'` to `Glm()`,
  `'glmm'` to `Glmm()`), so the resolved hyperprior shows the object; an override of a model's
  field is the object, `outcome_source=Glmm(y_family_shares={'gaussian': 1})`. There is no
  `Scm.glmm()`: `grouped()` already decides what a GLMM needs, and metabeta draws
  `Scm.grouped(outcome_source='glmm')`, passing nothing else. The SCM's y stays the
  SCM's own: its three fields stay on `Scm`, which is the SCM's config.
- **Two models, one skeleton.** `Glm` holds the fixed block, the families and links, the
  joint scale draw over its blocks, the design matrix coder and its container; `Glmm(Glm)`
  adds the random blocks and `random_effects: PositiveCount`, so a GLMM without random
  effects cannot be written. The difference is the list of blocks `prior` draws over,
  `[fixed]` or `[fixed, one per factor]`, so nothing is written twice. `'glmm'` on a
  hyperprior whose `factors` may be 0 fails at construction; `'glm'` on a grouped table is
  allowed, the pooled ground truth over grouped covariates, a dataset where a fitter's random-effect
  scale should go to zero. A user's model is checked at draw time only, on what it returns.
- **A model resolves its own fields.** `Glm`'s fields are points or distributions of the
  kinds `Scm` uses, resolved inside the call with the generator it is handed, as a
  column resolves its own fields; the per-dataset values are its prior. The field
  kinds and the resolution machinery (`resolveField`, the share ordering, `categories`) move
  from `scm` into a module of their own, `field`, as a base class both configs inherit,
  `Hyperprior`.
  ADR 0006's tokens, the standard deviations' family and the residual law, are shares fields
  of `Glm` with point defaults, chosen once per dataset by `Shares.choose` where `prior` reads
  them, the chosen names recorded in the prior. The marginal alias `Family` becomes
  `Marginal`, the glossary's word, so that "family" means the response family throughout.
- **One score for every source.** `drawDataset` always propagates y's node through the SCM, as
  `'scm'` does today, and hands the z-scored value, the score, to the model in `Covariates`
  with the observed columns, their metadata and the grouping (until 2026-10-28: the dataset
  keeps them). Under `'scm'`, the SCM rank-matches the
  score onto y's marginal; a GLM or GLMM adds λ · sd(η) · z(score) to its linear predictor,
  ADR 0006's deviation; a user's model may ignore it. y stays one node of the graph under
  every source. A table without an SCM has no score and so no deviation: the returned deviation
  is zeros and λ is recorded as given (until 2026-10-28: a nonzero λ fails).
- **The container.** `Outcome` holds the columns a source writes and their `Column` metadata,
  `y` today, a survival model's `time` and `event` later; `drawDataset` merges them into the
  dataset's columns without copying. Subclasses add the ground truth: `ScmOutcome` the score (until
  2026-10-28: the covariates hold it); `GlmOutcome`
  the prior and ADR 0006's parameters as drawn, with their raw-scale views;
  `GlmmOutcome(GlmOutcome)` each factor's random terms. `Dataset.outcome` is `None` only
  under `'none'`. A user's model may
  return the bare `Outcome`; TabDGP checks the shapes and dtypes of what comes back.
- **Any table in.** `Covariates` is public. `Covariates.fromPandas(table, factors)` builds it
  from a DataFrame with categorical dtypes: types from dtypes, levels and codes from the
  categoricals, factor codes and nesting from the data. A resampled outcome is then the model
  call with a generator from the key's rule, `default_rng([seed, index])` (until 2026-10-28:
  the key's child stream); the oracle test set
  itself is built downstream.
- **The dataset's views.** `Dataset.toPandas()` (coded columns as pandas categoricals, the
  latent nodes left out), `toDesign()` (the design matrix with its names and coding
  statistics, from the coder the GLM uses, so the user and the model see one coding; y and the
  grouping factors are not in it, by the glossary's definition) and `toDict()` (a JSON-ready
  dict of the whole dataset, dtypes recorded, a user's model by its qualified name) (until
  2026-10-28: `toDesign` is the covariates', and `toDict` goes). pandas is
  a dependency, not an extra: `fromPandas` is on the main path and both consumers carry it.

## Considered options

- **A general `'model'` value beside a field naming the model**: the second field means
  nothing unless the first is `'model'`, a cross-field check the interface can rule out.
- **A prior DSL**, nodes whose parameters are other nodes, sampled in order, the GLMM one
  preset of it: a small probabilistic programming language to maintain, for a generality the
  callable already gives a user, whose prior pipeline is ten lines of numpy inside their model.
- **Nested configs for every source**, the SCM's outcome fields moving into a config of their
  own: symmetric,
  but it changes the 0.3.0 override spelling and refactors the path the proofs certify, for one
  built-in model with priors.
- **`Scm.glmm(model=Glmm(), **overrides)`**: clunky, and `grouped()` already decides.
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
  under the GLMM at λ = 0 y would have no parents in the graph.
- **`toNumpy()` with codes as floats**: hides a coding decision; the design matrix is the
  coding. **pandas as an extra**: right while `toPandas` was its only reader.

## Consequences

- ADR 0005: the source's values; `Scm.glmm()` is gone. ADR 0006: the preset bullet, the
  fields' home, the tokens, the container and the module table are amended; the envelope as
  defaults, the joint scale draw, no caps and the estimands on both scales stand, and the lab
  note on the R2D2M2 prior stays the spec's first ticket.
- `CONTEXT.md`: Outcome source, Preset, Deviation, Prior context and Dataset rewritten; Model and
  Score added.
- `scm` loses the field kinds to `field`; `prior` and `outcome` are new modules; the
  package exports `Covariates`, `Outcome`, `Glm` and `Glmm`.
- No default draw changes: the SCM already computes y's node, and `'none'` draws nothing
  more, so `grouped()` and `flat()` draw the same bits.
- metabeta's `envelope.md` and its ADRs name `Hyperprior.glmm()`; they read
  `grouped(outcome_form='glmm')` now.
- A survival model later writes several columns, keeps its latent times and censoring in its
  container, and redraws its censoring inside its own call; a rejection that changes what a
  preset certifies stays refused for `grouped()` and `flat()`.

## Amendment 2026-10-28: a model's outcome is drawn after the dataset

Once y's family is stated (ADR 0006's amendment of the same day), `'glmm'` has no default to
expand to. A model on a drawn dataset's covariates could also not be called apart from its
draw, since the dataset did not keep them. Grilled on 2026-10-28, the decision is: **`drawDataset`
runs no model. `modelOutcome(seed, index, covariates, model)` draws a model's outcome after the
dataset, on drawn and real covariates alike, from the key's child stream. The dataset keeps its
covariates, the score among them.**

- **Two values.** `outcome_source` takes `'none'` or `'scm'`, as 0.3.0 shipped it. `Model` moves
  from `scm` to `outcome`, beside `runModel`, so `scm` no longer imports `prior`.
  `runModel` refuses a GLMM on covariates without a grouping factor, as before.
- **`modelOutcome`.** It replaces `resampleOutcome`, named for what it does, since it is the
  only route to a model's outcome: a second y under `'scm'`, a first under `'none'`.
- **The key's child stream.** `modelOutcome` draws from the first child of the key's seed
  sequence (numpy's `SeedSequence.spawn`), so a dataset's own key gives an independent outcome.
  At the key's root a model reused the draw's numbers: over 200 keys, Spearman's ρ between
  `rows` and the intercept's location was 1.00, and 0.05 under another seed.
- **The covariates in the dataset.** `Dataset.covariates` holds the `Covariates` the draw
  built, the score among them (`None` under `'none'`, which never draws y's node). `columns`
  and `metadata` become properties that merge them with the outcome's, the grouping factors
  first and y last. `ScmOutcome` folds into `Outcome`, since the score was its only addition.
- **A fitter reads the pair.** A fitter reads an outcome on covariates, the same pair whether
  the covariates were drawn or read from a table, so the dataset never holds a model's outcome:
  `Dataset.outcome` is the SCM's y, or `None`. The design matrix is the covariates' view,
  `Covariates.toDesign()`, since the glossary derives it from the covariates alone and a table's
  covariates need it too. `toDict` goes: no consumer calls it, a dataset is never written to a
  file, and a fitter in R reads `toPandas()` from parquet more easily than JSON's lists.
- **The score, required.** The deviation needs only the score, λ · sd(Xβ + Σ_k Z_k α_k) ·
  score. A resolved λ > 0 on covariates without one fails in `runModel`, where it returned
  zeros, so a deviation needs covariates drawn under `'scm'`.

### Considered options

- **The model kept in `drawDataset`** (`outcome_source=Glmm(family=...)`): one call per dataset with
  the score handed over inside, but two routes to a model's outcome where one serves both.
- **The strings kept, failing with a message of their own:** a value that always fails, a case
  the interface can rule out.
- **`Dataset.toCovariates()`,** a view rebuilding them: a conversion where a field holds them
  once.
- **`Dataset.y_score`,** with `modelOutcome` taking a `Dataset` too: a union input that hides
  the conversion inside.
- **Another seed by convention:** every caller has to remember it, and forgetting is silent.
- **The dataset's outcome swapped for a model's** (`dataset.withOutcome(outcome)`): one object
  for drawn tables only, since a table still gives the pair; its record would say
  `outcome_source='scm'` above a GLMM's y, and a study's plain `Outcome` would look like the SCM's.
- **The outcome holding its covariates:** every study's model would carry them, and `'none'`
  would still need `Dataset.covariates`.
- **`toDict` kept** for a later `Outcome.toDict`: speculative, as no consumer calls either.

### Consequences

- ADR 0005: the sources are `'none'` and `'scm'` again, and a model is no source of the node but a
  call on its covariates. ADR 0001: a model's outcome takes the key's child stream.
- `CONTEXT.md`: Outcome source, Model, Score, Deviation, Covariates and Dataset rewritten, and
  "resampled outcome" goes.
- The GLM and GLMM draws through `drawDataset` go, and `modelOutcome`'s are pinned instead; the
  SCM's draws under `grouped()` and `flat()`, and under `'none'`, stand bitwise.
- metabeta draws a dataset and calls `modelOutcome` on its covariates with the family it
  states; its `envelope.md` and ADRs follow.
- The field now only switches y's node; #67 renamed it `outcome_source`, and the SCM's config
  `Scm`.
- `Design` and its coder move from `outcome` to `covariates`; `Dataset.toDesign()` and
  `Dataset.toDict()` go, neither released.
