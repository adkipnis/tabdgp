---
status: accepted
date: 2026-10-23
---

# 0007 - Models as functions of the covariates

A model is a function of the covariates.
The generalized linear model (GLM) and the generalized linear mixed model (GLMM) are its first cases.
It draws a y of its own after the dataset, on a dataset's or a table's covariates, with a known ground truth (#46).
The ground truth is what a fitter (a method that estimates a model's parameters from a table) should recover.
`modelOutcome` draws a model's outcome on a stream of its own, keyed like a dataset.
A study's own model is checked only on what it returns.
The GLM and the GLMM read the covariates as their design matrix.
A dataset never holds a model's outcome.

## Decisions

1. A model reads covariates (`Covariates`), drawn or a table's:
   - a dataset keeps the ones its draw built (`Dataset.covariates`), the score among them under `'scm'`.
   - `Covariates.fromPandas(table, factors)` reads a table's, infers each column's metadata from its dtype (or takes it declared) and checks it against the values; a table has no score.
2. `modelOutcome(seed, index, covariates, model)` draws a model's outcome from the key's first child stream (the first of the independent streams the key's seed sequence spawns, NumPy's `SeedSequence.spawn`), so at a dataset's own key it reads other numbers than the draw (#68):
   - it is the only route to a model's outcome: a second y on covariates drawn under `'scm'`, a first one otherwise.
3. A model (`Model`) is a function of the covariates and a generator that returns an `Outcome` (a function, a `functools.partial` with a study's settings bound, or an object with `__call__`):
   - a study's model is checked only on what it returns: an `Outcome` of the covariates' length that writes none of their names.
4. The GLM and the GLMM are TabDGP's own models, two configs in `prior` with one skeleton (the fields and the draw they share, ADR 0006):
   - `Glmm(Glm)` adds the random blocks and `random_effects`, a positive count, so a GLMM without random effects cannot be written.
   - the prior draws over the fixed block (the fixed effects) alone, or one more block (its random effects) per grouping factor, so nothing is written twice.
   - a GLMM on covariates without a grouping factor fails; a GLM on grouped ones draws the pooled ground truth (one set of fixed effects for every group), where a fitter's random-effect scale should go to zero.
5. `Scm`, `Glm` and `Glmm` share one base, `Hyperprior` (`field`), whose fields are each a point or a distribution:
   - a model resolves its fields inside its call, with the generator it is handed.
   - a field per factor (`random_effects`, `lkj_eta`) resolves once per grouping factor, as a column resolves its own.
6. The GLM and the GLMM read the covariates as their design matrix (`Covariates.toDesign()`):
   - the intercept comes first, and a z-score divides by the population standard deviation (sd) over the n rows.
   - a coded column drops its first level in the codes' order (`track` of three levels gives `track=1` and `track=2`).
   - the grouping factors and y are left out.
7. The outcome (`Outcome`) holds the columns a model writes (`y` today, a survival model's `time` and `event` later) and their metadata, each with the role `outcome` and its type's dtype:
   - `GlmOutcome` adds y's family, link and shape, the prior, the parameters and the deviation (ADR 0006), and `GlmmOutcome` each factor's random terms.
   - y's family and shape are the model's specification, which a fitter states, so they sit beside the link, not on the `Prior`.
8. A fitter reads a model's outcome on its covariates, so a dataset never holds one:
   - `Dataset.outcome` holds only the outcome node's y (ADR 0005).
   - `Dataset.columns` and `Dataset.metadata` merge the covariates' with it, the grouping factors first and y last.
9. `Dataset.toPandas()` returns the table, with coded columns as pandas categoricals and without the latent nodes.

## Considered options

| question | chosen | rejected |
|---|---|---|
| the covariates | a field of the dataset | a view, the score alone, on the outcome |
| where a model runs | after the draw | in `drawDataset` |
| a model's stream | the key's first child | the key's root, another seed |
| a model | a function | a prior DSL |
| the GLM | a config of its own | a GLMM without random effects |
| the design matrix | the covariates' view | the dataset's, codes as floats |
| the outcome | a class per model | dicts |
| y's family and shape | on the outcome | on the `Prior`, the resolved config |
| a model's outcome | beside the dataset | swapped into it |
| the table | pandas, a dependency | pandas as an extra, JSON |

1. **The covariates:**
   - A view (`Dataset.toCovariates()`) rebuilds what a field holds once.
   - The score alone (`Dataset.y_score`) makes `modelOutcome` take a `Dataset` too, a union input that hides the conversion, and a `Dataset` read from a table would fake most of its fields (no graph, latent nodes or resolved hyperprior).
   - On the outcome, they would be carried by every study's model, and `'none'` would still need them on the dataset.
2. **Where a model runs:**
   - In `drawDataset` (`outcome_source=Glmm(...)`), a model's outcome has two routes, where one serves drawn and real covariates alike.
3. **A model's stream:**
   - At the key's root, a model reuses the draw's numbers: over 200 keys, Spearman's ρ between `rows` and the intercept's location was 1.00, and 0.05 under another seed.
   - Another seed by convention is one every caller has to remember, and forgetting it is silent.
4. **A model:**
   - A prior DSL (domain-specific language: nodes whose parameters are other nodes, the GLMM one preset) is a small probabilistic programming language to maintain, for a generality a function already gives (a study's prior is ten lines of NumPy inside its model).
5. **The GLM:**
   - A GLMM with `random_effects = 0` contradicts the glossary, and its type lets a user write the zero by accident.
6. **The design matrix:**
   - On the dataset, a table's covariates would lack it, though the glossary derives it from the covariates alone.
   - Codes as floats (`toNumpy()`) hide the coding the design matrix states.
7. **The outcome:**
   - Dicts (`model`, `truth`, `context`) have no checks at construction, and string keys a consumer (a package that trains on the datasets) has to know.
8. **y's family and shape:**
   - On the `Prior`, they would mix the model's specification into a prior over its parameters.
   - The resolved config on the outcome carries every field a fitter does not need.
9. **A model's outcome:**
   - Swapped into the dataset (`dataset.withOutcome(outcome)`), its record would say `outcome_source='scm'` above a GLMM's y.
10. **The table:**
   - pandas as an extra cannot serve `fromPandas`, which is on the main path.
   - JSON (JavaScript Object Notation, `toDict`) has no caller, since a dataset is never written to a file and a fitter in R reads `toPandas()` from parquet (a columnar file format) more easily than JSON's lists.

## Consequences

- **Oracle sets:** a study builds an oracle set (many outcomes on one table's covariates, each with its ground truth) downstream, one `modelOutcome` per key, reproducible per key.
- **One coding:** a user's fitter reads the design matrix the model drew on, and the outcome's means and sds map its parameters onto the raw columns (ADR 0006).
- **Later models, none built:** a survival model writes several columns (`time`, `event`), keeps its latent times and censoring in its outcome, and redraws its censoring inside its own call.
