"""fidelity.py defines how two samples of summary vectors are compared.

A sample is an array (s datasets with p summary statistics).
Each test scores each axis on its pooled ranks.
The classifier two-sample test (C2ST) and distances compare their statistic against a null reference.

pipeline:
---------
two samples, (s, p) each
-> each axis coded on the pooled ranks (a missing value at −1)        _rankPooled
-> the statistic on the samples' labels                               c2st, distances
-> the same statistic on shuffled labels (its null)                   _drawNull
-> the share of one sample inside the other's band                    containment
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

# =====================================================================================
# the comparisons
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Comparison:
    """A test's statistic on two samples, and its permuted null."""

    statistic: float
    null: np.ndarray  # (permutations,)

    @property
    def p_value(self) -> float:
        """The share of the null at or above the statistic
        (Phipson and Smyth 2010, https://doi.org/10.2202/1544-6115.1585).
        """
        above = int(np.sum(self.null >= self.statistic))
        return (above + 1) / (len(self.null) + 1)


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Containment:
    """The share of one sample's rows inside the other's band, per axis and on every axis."""

    per_axis: np.ndarray  # (p,)
    every_axis: float


# =====================================================================================
# the tests
# =====================================================================================
# -------------------------------------------------------------------------------------
# the two-sample test
# -------------------------------------------------------------------------------------


def c2st(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, permutations: int = 200
) -> Comparison:
    """A random forest's balanced classification accuracy of two sample sources.
    It follows sbi's c2st (https://github.com/sbi-dev/sbi/blob/main/sbi/utils/metrics.py),
    but stratifies the folds and balances the accuracy (so chance stays 0.5 at unequal sizes).
    It also uses ranks instead of z-scores.
    """
    # sklearn is the extra tabdgp[fidelity]
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    def scoreForest(ranks: np.ndarray, labels: np.ndarray) -> float:
        forest = RandomForestClassifier(random_state=int(rng.integers(2**32)))
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(rng.integers(2**32)))
        scores = cross_val_score(
            forest, ranks, labels, cv=folds, scoring='balanced_accuracy', error_score='raise'
        )
        return float(scores.mean())

    statistic, null = _drawNull(scoreForest, a, b, rng, permutations)
    return Comparison(statistic=statistic, null=null)


def _drawNull(
    score: Callable[[np.ndarray, np.ndarray], float | np.ndarray],
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
    permutations: int,
) -> tuple[float | np.ndarray, np.ndarray]:
    """The score on the samples' labels, and its null
    (the score on shuffled labels, the ranks unchanged).
    """
    ranks, labels = _rankPooled(a, b)
    statistic = score(ranks, labels)
    null = np.empty((permutations, *np.shape(statistic)))  # (permutations,) or (permutations, p)
    for permutation in range(permutations):
        shuffled = rng.permutation(labels)
        null[permutation] = score(ranks, shuffled)
    return statistic, null


def _rankPooled(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ranks all values per axis by (r − 1/2)/N,
    where r is a value's rank among the N observed values of both samples.
    Tied values get their average rank. Missingness is denoted with −1.
    """
    pooled = np.concatenate([a, b])  # (s, p)
    labels = np.repeat([0, 1], [len(a), len(b)])  # (s,)
    ranks = np.full(pooled.shape, -1.0)  # (s, p)
    for axis in range(pooled.shape[1]):
        observed = ~np.isnan(pooled[:, axis])
        values = pooled[observed, axis]
        ordered = np.sort(values)
        below = np.searchsorted(ordered, values, side='left')
        at_or_below = np.searchsorted(ordered, values, side='right')
        ranks[observed, axis] = (below + at_or_below) / 2 / len(values)
    return ranks, labels


# -------------------------------------------------------------------------------------
# the distance per axis
# -------------------------------------------------------------------------------------


def distances(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, permutations: int = 200
) -> tuple[Comparison, ...]:
    """The Wasserstein-1 distance per axis between two samples' ranks."""
    # the distances (p,) and their null (permutations, p)
    statistics, null = _drawNull(_measureWasserstein, a, b, rng, permutations)
    comparisons = []
    for statistic, axis_null in zip(statistics, null.T):
        comparison = Comparison(statistic=float(statistic), null=axis_null)
        comparisons.append(comparison)
    return tuple(comparisons)


def _measureWasserstein(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The area between two samples' empirical distribution functions (separately per axis)."""
    order = np.argsort(ranks, axis=0)  # (s, p)
    ordered_ranks = np.take_along_axis(ranks, order, axis=0)  # (s, p)
    ordered_labels = labels[order]  # (s, p)
    cdf_a = np.cumsum(ordered_labels == 0, axis=0) / np.sum(labels == 0)  # (s, p)
    cdf_b = np.cumsum(ordered_labels == 1, axis=0) / np.sum(labels == 1)  # (s, p)
    delta = np.abs(cdf_a - cdf_b)[:-1]  # (s − 1, p)
    gaps = np.diff(ordered_ranks, axis=0)  # (s − 1, p)
    return np.sum(delta * gaps, axis=0)  # (p,)


# -------------------------------------------------------------------------------------
# the containment
# -------------------------------------------------------------------------------------


def containment(
    sample: np.ndarray, reference: np.ndarray, band: tuple[float, float] = (1, 99)
) -> Containment:
    """The share of `sample`'s rows inside `reference`'s percentile `band` per axis.
    It reads as the precision if the registry is the `reference`, and as the recall otherwise.
    """
    ranks, labels = _rankPooled(sample, reference)
    sample_ranks = ranks[labels == 0]  # (s, p)
    reference_ranks = ranks[labels == 1]  # (s, p)
    # the band ends at observed values ('nearest') -> ranks and raw values agree on which rows are inside
    low, high = np.percentile(reference_ranks, band, axis=0, method='nearest')  # (p,)
    inside = (low <= sample_ranks) & (sample_ranks <= high)  # (s, p)
    per_axis = inside.mean(axis=0)  # (p,)
    every_axis = float(inside.all(axis=1).mean())
    return Containment(per_axis=per_axis, every_axis=every_axis)
