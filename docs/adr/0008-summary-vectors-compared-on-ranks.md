---
status: accepted
date: 2026-10-24
---

# 0008 - One module compares two samples of summary vectors. Each axis is ranked over both samples. Each statistic is tested against a permutation null.

1. The module `fidelity` compares two samples of summary vectors (#58). The proofs use it, and so will the hyperprior's fit.
2. A sample is a float array of shape (s, p), with s datasets and p summary statistics. A missing statistic is NaN.
3. The module draws nothing and reads no registry:
   - the caller turns each dataset into its summary vector.
   - the caller reads the registry's files, so `src/` imports nothing downstream.
   - the caller pairs the generated sizes with the real tables' fits by an override (ADR 0004).
4. Each axis is ranked over both samples:
   - tied values get their mean rank.
   - the ranks are scaled to (r − 1/2)/N, where N is the number of observed values.
   - a missing value gets −1, below every rank.
   - e.g. 2, 5, 5 and a missing value get 1/6, 2/3, 2/3 and −1.
5. The classifier two-sample test (C2ST, `c2st`) uses scikit-learn's random forest with its default settings. Its statistic is the balanced accuracy, averaged over 5 stratified folds (see sbi's `c2st`):
   - there is one recipe and no classifier argument.
   - the caller's generator seeds the forest and the folds.
   - chance stays at 0.5 for any sample sizes, so the generated sample can be as large as the caller wants.
   - scikit-learn is an optional dependency (`tabdgp[fidelity]`), which `c2st` imports when it runs.
6. `distances` computes the Wasserstein-1 distance per axis between the two samples' ranks, which is the area between their empirical distribution functions.
7. Each statistic is tested against a permutation null:
   - the null shuffles the sample labels of the pooled rows and reruns the test, 200 times by default.
   - the p-value counts the statistic as part of its null, (k + 1)/(B + 1), so it is never 0 (see Phipson and Smyth, 2010).
   - the module sets no bar; the caller does.
8. `containment` returns the share of one sample's rows inside the other sample's band, per axis and on every axis:
   - the band spans the 1st to 99th percentile by default.
   - its ends are observed values, so ranks and values agree on which rows are inside.
   - with the registry as the reference, the share is the precision (generated rows inside the real band); otherwise it is the recall.

## Considered options

| question | chosen | rejected |
|---|---|---|
| coding | ranks | z-scores |
| classifier | a random forest | k-NN, the SBI benchmark's MLP |
| global score | the C2ST | maximum mean discrepancy |
| distance per axis | Wasserstein-1 | Kolmogorov-Smirnov |
| null | permutations | a binomial around 0.5 |
| containment | a percentile band per axis | a support metric |

1. **Coding:**
   - A random forest grows the same trees under any monotone transformation, so ranks and z-scores (as in sbi) differ only in where the missing values go.
   - On z-scores, a heavy-tailed axis would dominate the distances.
2. **Classifier:**
   - k-NN (k nearest neighbors) needs a guessed k, and its only advantage was one dependency less.
   - The MLP (multilayer perceptron) of the SBI (simulation-based inference) benchmark has two hidden layers of 10 p units. It needs scaling and many iterations to reach the forest's accuracy at these sizes.
3. **Global score:**
   - Maximum mean discrepancy with the median heuristic is sensitive to its bandwidth on multimodal samples (see sbibm).
   - It adds nothing the C2ST does not already show.
4. **Distance per axis:**
   - Kolmogorov-Smirnov uses only the largest gap between the distribution functions, so it is noisier than Wasserstein-1, which integrates the whole gap.
5. **Null:**
   - A binomial null around 0.5 (see Lopez-Paz and Oquab, 2017) is wrong for cross-validated accuracies, because the folds are correlated.
   - A permutation null is exact for any classifier.
6. **Containment:**
   - A support metric (e.g. k-NN coverage, or precision, recall, density and coverage) needs a distance between summary vectors, which ranks do not provide.
   - It is deferred until the summary vector is decided.

## Consequences

- **Containment first:** of the distributional measures, coverage tracked downstream performance (see Structural Coverage).
  - A discriminator separates every prior from real tables, and its accuracy did not predict performance (see Mind the Gap).
  - The C2ST shows how far apart the samples are, and the distances show on which axes.
  - A downstream ablation remains the final test.
- **Consistency checks only:** the proofs' summary vector and bar are documented as consistency checks, never claimed as realism in the README or a paper.
