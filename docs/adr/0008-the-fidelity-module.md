---
status: accepted
date: 2026-10-24
amended: 2026-10-24, #58: the p-value counts the statistic among its null, (k + 1)/(B + 1); the ranks are scaled as (r − ½)/N; the band ends at observed values; sbi's c2st runs plain folds and accuracy on z-scores, so the recipe follows it, not copies it
---

# 0008 — The fidelity module: how generated and registry summaries are compared

The proofs (metabeta-v3's mixed ADR 0005) compare summary vectors of generated and registry
tables by a two-sample test and a containment test, and the hyperprior's fit (roadmap item 4)
will read the same comparisons. Two lab notes (2026-10-12, 2026-10-24) each carried a copy of a
k-NN two-sample test and a percentile band, with a guessed k, no null and the generated sizes
drawn rather than paired. Grilled on 2026-10-24 ([#58](https://github.com/adkipnis/tabdgp/issues/58)),
the decision is: **one module, `fidelity`, that takes two samples of summary vectors as float
arrays and returns scored comparisons with a null, draws nothing and reads no registry: a
random-forest classifier two-sample test with 5-fold cross-validation and a permutation null,
a Wasserstein-1 distance per axis on ranks with the same null, and the band containment per
axis; the summaries ranked over both samples with average ranks on ties and a missing value at
−1; balanced accuracy at any sample sizes.** The words are `CONTEXT.md`'s: summary vector,
two-sample test, containment test, null, pairing.

## Decisions

- **Arrays in, comparisons out.** A sample is a float array `(s, p)`, s datasets by p summaries,
  a missing summary NaN. How a dataset becomes a summary row (the fit's axes from the truth, the
  measurement pass's statistics mirrored on generated tables) is the caller's, decided in item
  3; the registry's files are read by the caller, so `src/` imports nothing downstream.
- **The classifier.** scikit-learn's random forest at its defaults, 5 stratified folds, the
  accuracy averaged over folds, after `sbi.utils.metrics.c2st`, which runs plain folds and
  accuracy on z-scores; the forest and the folds are seeded from the module's generator. One
  recipe, no classifier argument. The k-NN of the notes is dropped: with the forest present its
  only merit was the missing dependency. scikit-learn is an extra, `tabdgp[fidelity]`, imported
  inside `c2st`.
- **The null.** Every accuracy and distance reads against a permutation null: the labels of
  the pooled sample are shuffled and the test rerun, 200 times by default, returning the
  statistic, the null's values and the p-value, the share of the null at or above the statistic
  with the statistic counted among it, (k + 1)/(B + 1), which is never 0 (Phipson and Smyth
  2010). The bar is the caller's; the module sets none.
- **The coding.** Each axis is ranked over both samples, tied values at their mean rank, scaled
  to (r − ½)/N over the N observed values, a missing value at −1 below every rank. A forest's
  trees are the same under any monotone recoding, so the choice over sbi's z-scores only fixes
  the missing value's place and keeps a heavy-tailed axis from setting a distance.
- **The distance per axis.** Wasserstein-1 between the two samples' ranks: the area between
  their empirical distribution functions, bounded like Kolmogorov–Smirnov's largest gap and
  less noisy, since it integrates the whole gap. It says where a gap is; the band says how large.
- **Containment.** The share of one sample's rows inside the other's band per axis, the band
  the 1st to 99th percentile by default, its ends at the nearest observed values so that it
  reads the same on ranks as on values, and the share inside on every axis. Called twice it
  gives recall (registry rows inside the generated band) and precision (generated rows inside
  the registry's band), the cost of widening.
- **Sizes.** Any sizes; the score is balanced accuracy, so chance stays 0.5 while the generated
  side grows as large as the caller wants. The pairing of generated sizes to registry fits is
  an override of the grouped preset, `rows`, `factors` and `groups` as points, which needs
  `groups` to take one point per factor ([#59](https://github.com/adkipnis/tabdgp/issues/59)).

## Considered options

- **Maximum mean discrepancy** with the median heuristic: bandwidth-sensitive on multimodal
  samples (sbibm), and a third global score beside the C2ST says nothing it does not.
- **The binomial null** around 0.5 (Lopez-Paz and Oquab 2017): wrong for cross-validated
  accuracies, whose folds are correlated; the permutation null is exact for any classifier.
- **The SBI benchmark's MLP** (two hidden layers of 10 p units): needs scaling and many
  iterations for the forest's accuracy at these sizes.
- **A support metric** (k-NN coverage, PRDC) for containment: needs a metric on the summary
  space, which ranks only fake; deferred to item 3's grill, where the summary vector is decided.
- **Reproducing the lab notes' k-NN numbers**: dropped, since both notes are rerun anyway.

## Consequences

- Among the three tests, containment comes first: of the distributional measures, coverage is
  the one that tracked downstream performance (Structural Coverage), while a discriminator
  separates every prior from real tables (Mind the Gap) and its accuracy did not predict
  performance. The C2ST diagnoses how far, the per-axis distance where; item 3's grill decides
  the bar's form in that light, and the downstream ablation stays the arbiter.
- The lab notes' functions become calls into the module; the #47 rerun is an `experiments/`
  script on the package's own draw.
