---
status: accepted
date: 2026-10-23
---

# 0007 - A model is a function of the covariates. The GLM and the GLMM are its first cases. It draws after the dataset.

1. A model (e.g. a generalized linear model (GLM) or a generalized linear mixed model (GLMM)) draws a y of its own on a dataset's or a table's covariates after the draw (ADR 0005), with a known ground truth (#46).
2. A model reads covariates (`Covariates`), drawn or a table's:
   - a dataset keeps the ones its draw built (`Dataset.covariates`), the score among them under `'scm'`.
   - `Covariates.fromPandas(table, factors)` reads a table's, infers each column's metadata from its dtype (or takes it declared) and checks it against the values; a table has no score.
3. `modelOutcome(seed, index, covariates, model)` draws a model's outcome from the key's first child stream (NumPy's `SeedSequence.spawn`), so at a dataset's own key it reads other numbers than the draw (#68):
   - it is the only route to a model's outcome: a second y on covariates drawn under `'scm'`, a first one otherwise.
4. A model (`Model`) is a function of the covariates and a generator that returns an `Outcome` (a function, a `functools.partial` with a study's settings bound, or an object with `__call__`):
   - a study's model is checked only on what it returns: an `Outcome` of the covariates' length that writes none of their names.
5. The GLM and the GLMM are TabDGP's own models, two configs in `prior` with one skeleton (ADR 0006):
   - `Glmm(Glm)` adds the random blocks and `random_effects`, a positive count, so a GLMM without random effects cannot be written.
   - the prior draws over the fixed block alone, or one more per grouping factor, so nothing is written twice.
   - a GLMM on covariates without a grouping factor fails; a GLM on grouped ones draws the pooled ground truth, where a fitter's random-effect scale should go to zero.
6. `Scm`, `Glm` and `Glmm` share one base, `Hyperprior` (`field`), whose fields are each a point or a distribution:
   - a model resolves its fields inside its call, with the generator it is handed.
   - a field per factor (`random_effects`, `lkj_eta`) resolves once per grouping factor, as a column resolves its own.
7. A model reads the covariates as their design matrix (`Covariates.toDesign()`):
   - the intercept comes first, continuous and count columns are z-scored by the population sd over the n rows, and binary ones stay 0/1.
   - a categorical or ordered column gives k − 1 dummies, the first level in the codes' order dropped (`track` of three levels gives `track=1` and `track=2`).
   - the grouping factors and y are left out.
8. The outcome (`Outcome`) holds the columns a model writes (`y` today, a survival model's `time` and `event` later) and their metadata, each with the role `outcome` and its type's dtype:
   - `GlmOutcome` adds y's family, link and shape, the prior, the parameters and the deviation (ADR 0006), and `GlmmOutcome` each factor's random terms.
   - y's family and shape are the model's specification, which a fitter states, so they sit beside the link, not on the `Prior`.
9. A fitter reads a model's outcome on its covariates, so a dataset never holds one:
   - `Dataset.outcome` is the SCM's (structural causal model's) y, or `None` under `'none'`.
   - `Dataset.columns` and `Dataset.metadata` merge the covariates' with it, the grouping factors first and y last.
10. `Dataset.toPandas()` returns the table, with coded columns as pandas categoricals and without the latent nodes.

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
   - Dicts (`model`, `truth`, `context`) have no checks at construction, and string keys a consumer has to know.
8. **y's family and shape:**
   - On the `Prior`, they would mix the model's specification into a prior over its parameters.
   - The resolved config on the outcome carries every field a fitter does not need.
9. **A model's outcome:**
   - Swapped into the dataset (`dataset.withOutcome(outcome)`), its record would say `outcome_source='scm'` above a GLMM's y.
10. **The table:**
   - pandas as an extra cannot serve `fromPandas`, which is on the main path.
   - JSON (`toDict`) has no caller, since a dataset is never written to a file and a fitter in R reads `toPandas()` from parquet more easily than JSON's lists.

## Consequences

- **Oracle sets:** a study builds them downstream, one `modelOutcome` per key on a table's covariates, reproducible per key.
- **One coding:** a user's fitter reads the design matrix the model drew on, and the outcome's means and sds map its parameters onto the raw columns (ADR 0006).
- **Later models, none built:** a survival model writes several columns (`time`, `event`), keeps its latent times and censoring in its container, and redraws its censoring inside its own call.
