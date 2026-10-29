---
status: accepted
date: 2026-10-24
---

# 0008 - Summary vectors are compared on pooled ranks. Every statistic reads against a permutation null. The bar is the caller's.

1. One module, `fidelity`, compares two samples of summary vectors for the proofs and the hyperprior's fit alike, draws nothing and reads no registry (#58):
   - a sample is a float array (s, p), s datasets by p summaries, a missing summary NaN.
   - how a dataset becomes a summary row is the caller's, and so are the registry's files, so `src/` imports nothing downstream.
   - the caller pairs the generated sizes to the real tables' fits by an override (`rows`, `factors` and `groups` as points, ADR 0004).
2. Each axis is ranked over both samples, tied values at their mean rank, scaled to (r − ½)/N over the N observed values, and a missing value at −1, below every rank:
   - e.g. 2, 5, 5 and a missing value give 1/6, 2/3, 2/3 and −1.
3. The classifier two-sample test (C2ST, `c2st`) is scikit-learn's random forest at its defaults, its balanced accuracy averaged over 5 stratified folds (after sbi's `c2st`):
   - one recipe, no classifier argument; the forest and the folds are seeded from the caller's generator.
   - balanced accuracy keeps chance at 0.5 at any sizes, so the generated side grows as large as the caller wants.
   - scikit-learn is an extra (`tabdgp[fidelity]`), imported inside `c2st`.
4. The distance per axis (`distances`) is the Wasserstein-1 distance between the two samples' ranks, the area between their empirical distribution functions.
5. Every accuracy and distance reads against a permutation null: the pooled sample's labels are shuffled and the test rerun, 200 times by default:
   - the p-value counts the statistic among its null, (k + 1)/(B + 1), so it is never 0 (Phipson and Smyth 2010).
   - the module sets no bar.
6. Containment (`containment`) is the share of one sample's rows inside the other's band, per axis and on every axis:
   - the band is the 1st to 99th percentile by default, its ends at the nearest observed values, so it reads the same on ranks as on values.
   - called twice, it gives the recall (real rows inside the generated band) and the precision (generated rows inside the real band), the cost of widening.

## Considered options

| question | chosen | rejected |
|---|---|---|
| coding | pooled ranks | z-scores |
| classifier | a random forest | k-NN, the SBI benchmark's MLP |
| global score | the C2ST alone | maximum mean discrepancy |
| distance per axis | Wasserstein-1 | Kolmogorov-Smirnov |
| null | permutations | binomial around 0.5 |
| containment | a percentile band per axis | a support metric |

1. **Coding:**
   - A forest's trees are the same under any monotone recoding, so ranks over sbi's z-scores only fix the missing value's place.
   - On z-scores, a heavy-tailed axis would set the distance.
2. **Classifier:**
   - k-NN (k nearest neighbors) needs a guessed k, and its only merit over the forest was a dependency less.
   - The SBI (simulation-based inference) benchmark's MLP (multilayer perceptron, two hidden layers of 10 p units) needs scaling and many iterations to reach the forest's accuracy at these sizes.
3. **Global score:**
   - Maximum mean discrepancy with the median heuristic is bandwidth-sensitive on multimodal samples (sbibm), and says nothing the C2ST does not.
4. **Distance per axis:**
   - Kolmogorov-Smirnov's largest gap reads one point, so it is noisier than Wasserstein-1, which integrates the whole gap.
5. **Null:**
   - A binomial null around 0.5 (Lopez-Paz and Oquab 2017) is wrong for cross-validated accuracies, whose folds are correlated; a permutation null is exact for any classifier.
6. **Containment:**
   - A support metric (k-NN coverage, or precision, recall, density and coverage) needs a metric on the summary space, which ranks only fake; it waits for the summary vector.

## Consequences

- **Containment first:** of the distributional measures, coverage is the one that tracked downstream performance (Structural Coverage).
  - a discriminator separates every prior from real tables, and its accuracy did not predict performance (Mind the Gap).
  - the C2ST says how far, the distance per axis where, and a downstream ablation stays the arbiter.
- **Consistency checks only:** the proofs' summary vector and bar are documented as consistency checks, never claimed as realism in the README or a paper.
