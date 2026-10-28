---
status: accepted
date: 2026-10-28
amended: 2026-10-28, #72: the gate waits for the fitted defaults, and #47's arms run in the proofs' script
---

# 0009 - The summary and the proofs: how a dataset is measured and how the defaults are certified

mixedbench 1.0.11 ships six measured files over its registry (`columns.csv`,
`column_factors.csv`, `factor_pairs.csv`, `dimensions.csv`, `fits.csv`, `responses.csv`, 52
axes), and `fidelity` (ADR 0008) compares two samples of summary rows but says nothing about
how a dataset becomes one, which support metric joins the band, or what the bar is. Grilled on
2026-10-28, the decision is: **a dataset is measured into summary tables of the registry's six
shapes by a copy of mixedbench's own survey functions, and into one summary vector per dataset
aggregated from them; the proofs draw the grouped preset at each registry row's sizes, compare
per key by containment and the distances and per dataset by the C2ST and a neighborhood test,
and gate on containment alone: a bar's share of registry rows inside the generated band on every
axis, the rest reported.** The words are `CONTEXT.md`'s: summary tables, summary vector,
containment test, neighborhood test, pairing.

## Decisions

- **Two shapes.** The summary tables are a dataset's statistics in the registry's six shapes,
  with the registry's column names, so a proof reads a generated file against its real
  counterpart row for row (a generated column against registry columns). The summary vector is
  one row per dataset, built from its tables and not from the dataset again: per column axis the
  median and the 90th percentile over the table's columns (the p90 because a quirk sits in one
  column of several), the type mix as shares of columns, the per-factor axes of the first and
  the second factor (missing without one), the factor pairs' relation as shares, and the per-row
  axes (`d_all`, `condition_all`, the response's statistics, a GLMM's fit statistics) as they
  are. Containment and the distances run per key, where the quirks live; the C2ST and the
  neighborhood test run on the vector, whose rows are independent, so their permutation null
  holds.
- **The code is copied.** mixedbench's statistic functions live in its `survey/` scripts, outside
  its wheel, so `summary` copies them with a citation (the borrowed-code rule) over `Covariates`
  and an `Outcome`, the fit statistics (R², ICC per factor, sd(η); Nakagawa, Johnson and
  Schielzeth 2017) from a GLMM's ground truth. A slow cross-check test loads a handful of
  registry tables through mixedbench's `Tables` and `Covariates.fromPandas` and reproduces their
  rows to four decimals; it reads the checkout's cache (3.8 GB, not shipped) from an environment
  variable and skips without it. mixedbench is a dev dependency, pinned.
- **Strata.** `columns.csv` and `responses.csv` are compared per type (continuous, binary,
  categorical, ordered; the response types), each on the axes its type defines, and the type
  mix as shares, since a binary column "inside the skewness band" is vacuous and the type mix is
  a quantity of its own. A statistic a stratum leaves blank in a few rows (a constant column)
  keeps ADR 0008's −1.
- **The pairing.** One registry row gives `rows`, `factors`, `groups` as a tuple (superior
  first), `design_columns` = `d_all` and `nested_share` 0 or 1 by its relation (`mixed`, 21 rows,
  as crossed); ten draws per row at one seed. The 14 rows with `d_all` above `rows // 2` cannot
  be paired and are reported as outside the size envelope, never drawn at a clipped d: lowering
  the floor is a default change of the fit. No nested pair has as many groups as its superior,
  so the nesting rule needs no exception.
- **The neighborhood test.** `prdc` in `fidelity`: precision, recall, density and coverage
  (Naeem et al. 2020), k = 5, Euclidean on the pooled scaled ranks, which puts every axis on
  (0, 1), over rows complete on every axis. Reported, never gated: coverage is the measure that
  tracked downstream rank (Structural Coverage), which is why it is read at all.
- **The bar.** A proof asserts that at least a bar's share of the registry rows lies inside the
  generated 1st to 99th band on every axis of a file and stratum. The bar is one round number
  per proof, set from the first run of the fitted defaults just below what they reach; a change
  to it is a default change. Until the fit, the defaults are guesses it replaces, so the gate
  waits for it and the script reports the recall it will gate on. The every-axis recall, the precision, the
  C2ST, the distances and the neighborhood test are reported with their nulls, never gated. The
  C2ST gate at 0.6 is retired: a widened prior puts mass off the registry by design, which a
  symmetric test punishes, and a discriminator's accuracy did not predict performance (Mind the
  Gap).
- **Where the numbers live.** An `experiments/proofs.py` script draws the paired sample, writes
  the summary tables, a report (`SUMMARY.md` and one comparison table per file and stratum as
  CSV) and the keys; the slow tests, written with the gate, call the same function and assert
  the bar. The generated
  summaries are not committed, since the keys regenerate them bit for bit. `Scm.flat()` runs
  against the flat sample by the same machinery, paired by `n_obs` and `d_all`, last.
- **The proofs test the scm outcome.** The preset's y against `responses.csv`, per response
  type, is a proof. A GLMM's y against `fits.csv` and `responses.csv` is an experiment
  in the same script, the rerun of the lab note that chose ADR 0006's joint scale draw and the
  question the script answers first: three arms as subclasses of `Glmm` overriding `drawPrior`
  in it (the current
  prior, R2D2M2 at Aguilar and Bürkner's defaults, and unit scales with no budget, the prior the
  budget replaced), the package's draw with sizes paired per fit, the ground-truth statistics
  and y refitted by lme4 through `mixedbench.frequentist` on the cluster, so estimates meet
  estimates. The arms are tabled side by side; there is no rule, and ADR 0006's amendment stands
  unless the margins say otherwise.

## Considered options

- **One shape**: per key alone leaves the C2ST on rows that are not independent within a
  dataset; per dataset alone loses a rare level in one column of ten to a median.
- **mixedbench shipping its survey functions**: one definition, but the generated side's
  summary would exist only in tests, and the fit needs it in `src/`.
- **The neighborhood test on the per-key rows** with missing at −1: a missing axis sets a
  distance, so the strata make the rows complete first.
- **A gate on the every-axis recall**: falls with the number of axes, so it is reported.
- **Replicating the lab note's numbers**: dropped; its k-NN and its numpy copy of the prior are
  what the rerun replaces.

## Consequences

- `summary` joins the layers, imported from `covariates` and `outcome`; `fidelity` stays
  imported by nothing in `src/`.
- What the first run will show is a reading, not a failure of the test: the SCM draws no count
  or integer covariates, while the registry's continuous stratum holds ages, Likert scales and
  counts, where `unique_share`, `zero_share` and `min_share` get their mass. The fit's count
  families answer it; a count response's dispersion (variance over mean) is not measured by
  mixedbench and goes to a ticket there.
- metabeta's mixed ADR 0005 and its envelope's rule 9 name the 0.6 bar; they get a note.
