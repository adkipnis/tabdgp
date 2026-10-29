---
status: accepted
date: 2026-10-14
---

# 0004 - Presets choose the kind of table. A bounded count is truncated. Sizes are drawn coverage-first.

1. `drawDataset(seed, index, scm)` takes a preset of the SCM's (structural causal model's) hyperprior and has no default:
   - `Scm.grouped()` keeps the defaults, which are what the proofs certify; `Scm.flat()` sets `factors = 0`.
   - a later kind of table (item response, survival, causal) is a preset that fixes fields, never a branch in `drawDataset` nor a new mechanism.
   - an override serves a study, an ablation or a size outside a preset.
2. A count bounded by another field draws from its distribution truncated at that field's resolved value:
   - the table's length (`rows`) resolves first.
   - the design-column budget (`design_columns`) is truncated at `rows // 2`, so a table has at least two rows per design column.
   - a factor's number of groups (`groups`) is truncated at `rows`, or at one below its subordinate factor's.
3. Sizes are drawn coverage-first: a size is log-uniform, so every decade gets equal mass, and a count is uniform.
4. The number of grouping factors (`factors`) is `UniformInt(1, 3)`:
   - precedent: in MixedBench 1.0.1, the p90 is 2 and the maximum 5.
5. Each factor after the first nests in its predecessor with the share `nested_share` (ADR 0003), a guess of `Uniform(0.2, 0.6)`:
   - precedent: 134 two-factor tables are nested and 184 crossed.
6. The factors draw their `groups` bottom-up, `LogUniformInt(2, 400)` each (precedent: p50 32, p90 361):
   - a superior factor has fewer groups than its subordinate one, so a nested pair is never one partition relabeled.
   - a factor's floor is the lowest value of `groups` (2 by default) plus its depth (the number of factors superior to it), so every superior factor still gets that lowest value.
   - e.g. a chain of 3 factors draws at least 4 groups at the bottom, 3 in the middle and 2 at the top, so it needs 3 values of `groups` and 4 rows; fewer fail at construction (`groups`) or at resolve (`rows`).
   - `groups` also takes a tuple, one point per factor in the factors' order, so the sizes can be paired to a real table's fit (#59).
7. The group sizes follow the symmetric Dirichlet the registry fitted per factor (the Gamma(κ) weights of the prior-art search normalize to it):
   - with the share `balanced_share` (1/3; precedent: 38.8 % of factors) the groups are equally large, the remainder one more for the first groups (10 rows into 3 groups: 4, 3 and 3).
   - otherwise each group gets one row, and the remaining n − m rows are multinomial over Dirichlet(`size_alpha`) proportions, with `size_alpha` drawn per factor from `LogUniform(0.5, 500)` (precedent: p5 0.97, median 50).
   - every group is non-empty, so no draw is retried.
8. The two shares of ADR 0003 are guesses: `group_level_share` is `Beta(2, 4)`, at the registry's 31 %, and `group_share` is `Beta(0.5, 2)`, with mass near zero.

## Considered options

| question | chosen | rejected |
|---|---|---|
| kind of table | a preset | a kwarg on `drawDataset` |
| default kind | `factors` from 1 | `factors` from 0 |
| bounded count | truncated | clipped after the draw |
| a superior factor's groups | fewer than the subordinate's | at most as many |

1. **Kind of table:**
   - A bool `flat` is not future-proof.
   - A `Literal` of kinds makes every kind a branch inside `drawDataset`, which can contradict an override.
2. **Default kind:**
   - The defaults are what a consumer trains on and the proofs certify, so grouped tables are the default and the flat table the other preset.
3. **Bounded count:**
   - A draw clipped afterwards piles mass on the bound: at n = 20, 57 % of the draws from `LogUniformInt(2, 400)` would become all-singleton tables.
4. **A superior factor's groups:**
   - Equal counts put one subordinate group in each superior group, the same partition relabeled, which gives two random effects no GLMM (generalized linear mixed model) tells apart.
   - At most as many groups forces equal counts when the subordinate factor has 2 groups, and equal counts hit 15.9 % of nested pairs under `grouped()`.
   - Turning such a pair crossed honors `nested_share` only sometimes.

## Consequences

- **No override by default:** a consumer calls `drawDataset(seed, index, Scm.grouped())` and passes nothing.
- **The proofs:** run on `grouped()`; MixedBench's flat sample (mb#72) tests `flat()`.
- **α's floor:** the registry's smallest α, 0.025, lies below the floor of 0.5.
  - the containment test decides whether to widen it.
