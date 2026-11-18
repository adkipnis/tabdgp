---
status: accepted
date: 2026-11-17
---

# 0009 - A schema proof and a content proof

The proofs (the tests that compare generated and real tables) certify a preset's defaults against a set of real tables.
One module measures any table into its summary vector, a real table like a generated one.
The schema proof compares the real tables' sizes, type mix and levels with the hyperprior's fields.
The content proof draws 39 datasets at each real table's schema and ranks the real table among them on every axis.
A preset passes on three gates: recall, a floor on precision and a negative control.
The two-sample test is reported.
The flat preset is certified first, then the grouped one.

## Decisions

### The summary vector

1. One module in `src/` measures a table into its summary vector, one row per dataset:
   - it measures `Covariates` and an `Outcome`, so a real table enters through `Covariates.fromPandas`.
   - it measures both samples of a proof, the real tables and the generated ones.
   - `src/` imports nothing downstream. A script in `experiments/` loads the real tables, measures them once per pinned version of their source and commits their summary vectors.
2. The axes (one summary statistic each) come in six blocks: the sizes, each column's shape, the relations between columns, y, the grouping, and y's relation to the groups:
   - a statistic per column enters as its median and its maximum over the dataset's columns of one type.
   - an axis joins if every table of its kind can show it. A table that cannot show it (e.g. a second factor's statistic) leaves it missing.
   - an axis stays if real tables show values there that the generator cannot produce. It is then a named gap.
   - the statistics of missing values wait until the generator draws missing values.
3. y is measured like every other column, on axes of its own:
   - it gets the column statistics, its largest rank correlation with a covariate and its between-group share per factor.
   - no model is fitted to measure a table.

### The two proofs

1. A table's schema is its sizes (rows, design columns, factors, groups per factor, nested or crossed), each column's type and its number of levels, y's included.
2. The schema proof compares the real tables' schemas with the hyperprior's fields, and draws no table:
   - the sizes are drawn coverage-first (ADR 0004), so the proof asks only whether each real table's sizes can be drawn.
   - the type mix and the levels are compared by their distance per axis (ADR 0008).
   - the hyperprior's side is sampled from its fields, since a count bounded by another field is truncated.
3. The content proof draws 39 datasets per real table, each at that table's schema:
   - an override of `Scm` states each column's type and its levels.
   - on every axis, the real table is ranked among its 39 draws (see Talts et al., 2020).
   - a real table with several documented models enters once, with its first model.

### The gates

1. Recall is the share of real tables outside the range of their draws, per axis:
   - a calibrated generator leaves 5 % outside, and the gate adds a tolerance for the sampling noise.
   - the mean rank is reported. It is 0.5 when the draws are centered on the real tables.
   - an axis above the gate is a named gap, which the test lists by name. The list may only shrink.
2. Precision is the share of generated tables that lie at least as near to the real tables as the real table they were drawn for, on every axis at once:
   - a table's reach is its mean distance to the 5 nearest real tables from other sources.
   - the distance between two tables is their mean absolute difference over the axes, each axis coded on the pooled ranks (ADR 0008).
   - an axis that only one of the two tables shows costs 1, and an axis that neither shows costs 0.
   - a generated table counts if its reach is at most the reach of its real table. A tie counts half.
   - a calibrated generator scores 0.5, because a real table and its 39 draws are then exchangeable.
   - the floor is 0.25. A generator scores it if half of its tables are realistic and the other half lie far from every real table.
   - it is gated by a floor and never maximized.
3. The negative control runs the same proof on tables that lack a structure, and the proof has to fail there:
   - `dependence='none'` has to fail on the relations between columns.
   - a flat table with group labels assigned at random has to fail on the grouping.
4. The two-sample test (C2ST) and the distances are reported, beside the same statistics on two halves of the real tables.
5. The real tables are split by their source into a part the hyperprior is fitted on and a part that certifies it. The split has a fixed seed.
6. A gate, a seed, the split and the list of named gaps change like a default, with an issue first.

### The order

1. The flat preset is certified first, on a sample of real flat tables, and the grouped preset after it on the registry (MixedBench's list of real grouped tables and their documented models).
2. A later kind of table (survival, item response) brings its own set of real tables and runs the same two proofs.
3. TabICL v2's prior runs through the flat proof as a reported reference, from `experiments/`.

## Considered options

| question | chosen | rejected |
|---|---|---|
| the claim | recall, a precision floor, a control | recall alone, a C2ST gate, a bar from the first run |
| the measuring code | TabDGP's own, for both samples | MixedBench's measurer, a copy of it |
| the unit | one vector per dataset | rows per column beside it |
| y's statistics | a column's | a linear R², a refit by lme4 |
| the pairing | the schema | the sizes alone |
| the size defaults | read off the fields | read off drawn tables |
| recall | paired ranks | a pooled band |
| a model's y | outside the proofs | a proof of its own |
| precision | the reach, compared within a pair | a real table among the neighbors, improved precision, density, α-precision, one threshold |
| the floor | 0.25 | the noise below 0.5 |

1. **The claim:**
   - Recall alone is loose on a column's shape, since a wide mix of marginals contains every real table there.
   - A classifier separates every prior from real tables. Three searches over TabICL's prior kept its area under the curve above 0.95 (see Mind the Gap?, arXiv 2605.06343).
   - A prior rebalanced toward the feature correlations of real tables lost accuracy downstream (see Mitra-v2, arXiv 2609.04540, appendix C). Precision therefore gets a floor and no target.
   - A bar set from the first run only certifies that nothing got worse.
2. **The measuring code:**
   - MixedBench's measurer needs a registry row and a grouped table, so it cannot measure a set of flat, survival or item-response tables.
   - A copy has to reproduce the registry's measured files, which float32 columns and a second coding make fragile.
   - With one measure for both samples, the two implementations do not need to agree.
3. **The unit:**
   - The columns of one dataset are not independent, so a null over rows per column would not hold.
   - The maximum keeps a quirk that sits in one column of several.
4. **y's statistics:**
   - A linear R² is a model's statistic, and the SCM's y is drawn without a model.
   - A refit by lme4 (R's package for mixed models by maximum likelihood) repeats the between-group share as an estimate. It adds only lme4's behavior, which matters for a GLMM's y.
5. **The pairing:**
   - Most statistics depend on the sizes, so unpaired draws would test the sizes.
   - A type mix drawn anew per draw leaves the axes of a type missing at random, which adds noise to every comparison.
   - The type mix is compared directly, as shares.
6. **The size defaults:**
   - A real table's sizes are read off the table, and the hyperprior's off its fields, so no table has to be drawn.
7. **Recall:**
   - A pooled band (ADR 0008's containment) shows only a generator that is too narrow.
   - A paired rank shows both: too narrow leaves more than 5 % outside, and too wide piles the ranks in the middle.
   - The scale-draw experiment ranked each registry fit among 39 refits in the same way (#77).
8. **A model's y:**
   - A model's priors are coverage-first and never fitted (ADR 0006), and the scale-draw experiment already compared them with the registry's fits (#77).
9. **Precision:**
   - The share of generated tables with a real table among their nearest neighbors rises when the generator is too wide. In many dimensions the real tables lie in the middle of the generated ones (see Khayatkhoei and AbdAlmageed, 2023).
   - Improved precision has no known value for a calibrated generator, and its value changes with the number of axes (see Naeem et al., 2020). Its floor could only be read off a run.
   - Density has an expected value of 1, but it is unbounded and noisy at a few hundred tables (see Naeem et al., 2020).
   - α-precision needs an embedding that is trained on the real tables (see Alaa et al., 2022).
   - One threshold for all tables is set by the oddest real tables, so a wide generator passes it at a common schema (see Mind the Gap?, arXiv 2605.06343).
   - The few hundred real tables are a sparse sample. The reach of a real table grows with the sparsity as much as the reach of its draws, so their comparison is unaffected.
   - Tables from one source resemble each other, and a generated table has no such sibling. The own source is therefore left out.
   - The calibrated value of 0.5 holds for any distance and any number of neighbors. The mean absolute difference is the simplest distance on ranks, and 5 neighbors are the choice of Naeem et al. (2020) and of Mind the Gap?.
   - A Mahalanobis distance reacts more sharply where the axes are correlated, but it has no rule for a table that lacks axes. The two are compared on the measured tables (#97).
10. **The floor:**
    - The defaults are matched to real tables and then widened, so they are wider than the real tables on purpose. A floor at the noise below 0.5 would reject them.
    - Below 0.25 most of a training corpus could be unlike real tables. The half is a judgment.
    - The floor has no named gaps. If a preset's defaults score below it, the defaults change or the floor does (decision 6 of the gates).

## Consequences

- **Named gaps before the first run:**
  - count covariates, which `type_shares` does not draw.
  - how well the covariates predict y together, which no axis measures.
- **The stated columns:** the override is public, since a study with a fixed design (e.g. a power analysis) states its columns too. It gets an issue of its own.
- **`fidelity`:** gains the paired ranks and the precision. Its containment stays for two unpaired samples.
- **The real flat tables:** MixedBench reads its flat sample in a survey script, so a loader in its package is owed.
- **A reference for the grouped proof:** no public prior draws grouped tables. PluRel (arXiv 2602.04029) draws relational databases, whose child table joined to its parent is a grouped table.
- **Consistency checks only:** the gates are not a claim of realism (ADR 0008), and a downstream ablation remains the final test.
- **Left to the spec:** the list of axes, the tolerance on recall and the schema proof's bar on the type mix and the levels.
