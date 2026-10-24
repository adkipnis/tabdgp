"""The fidelity module's seams: c2st, distances and containment,
each on two samples of summary vectors, the first two against their null.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tabdgp.fidelity import Comparison, c2st, containment, distances

# worked by hand: an axis is coded as (r − ½)/N on the pooled ranks of its N observed values,
# ties at their average rank, a missing value at −1;
# the distance is the area between the two samples' distribution functions on the codes
WORKED_DISTANCES = [
    pytest.param([0.0, 1.0], [2.0, 3.0], 0.5, id='no overlap: codes 1/8, 3/8 vs 5/8, 7/8'),
    pytest.param(
        [1.0, 2.0], [2.0, np.nan], 7 / 12, id='a tie and a missing value: 1/6, 2/3 vs 2/3, −1'
    ),
    pytest.param([0.0], [1.0, 2.0], 0.5, id='unequal sizes: 1/6 vs 1/2, 5/6'),
]


@pytest.mark.parametrize(('a', 'b', 'expected'), WORKED_DISTANCES)
def test_a_distance_is_wasserstein_1_between_the_pooled_ranks(
    a: list[float], b: list[float], expected: float
) -> None:
    (comparison,) = distances(np.array([a]).T, np.array([b]).T, np.random.default_rng(0))
    assert comparison.statistic == pytest.approx(expected)


def _normal(rng: np.random.Generator, shift: float = 0.0) -> np.ndarray:
    """100 datasets by 3 standard normal summaries, the second shifted by `shift`."""
    sample = rng.standard_normal((100, 3))  # (s, p)
    return sample + np.array([0.0, shift, 0.0])


def test_a_shift_on_one_axis_is_named_by_its_distance() -> None:
    """The per-axis distance says where two samples differ.
    Only the shifted axis leaves its null.
    """
    rng = np.random.default_rng(1)
    a, b = _normal(rng), _normal(rng, shift=1.0)
    comparisons = distances(a, b, rng, permutations=199)
    p_values = [comparison.p_value for comparison in comparisons]
    assert p_values[1] == 1 / 200
    assert min(p_values[0], p_values[2]) > 0.01


def _insideNull(comparison: Comparison) -> bool:
    """The statistic lies between the smallest and the largest value of its null."""
    return comparison.null.min() <= comparison.statistic <= comparison.null.max()


def test_two_samples_of_one_distribution_stay_inside_their_distances_null() -> None:
    rng = np.random.default_rng(2)
    a, b = _normal(rng), _normal(rng)
    comparisons = distances(a, b, rng, permutations=199)
    assert all(_insideNull(comparison) for comparison in comparisons)


# any sample of 1 to 20 datasets by 1 to 4 summaries, missing and infinite values included
SAMPLES = arrays(np.float64, st.tuples(st.integers(1, 20), st.integers(1, 4)))


@given(sample=SAMPLES)
def test_a_sample_lies_wholly_inside_its_own_band_from_0_to_100(sample: np.ndarray) -> None:
    """The band's ends are the sample's own smallest and largest codes,
    a missing value's −1 included.
    """
    contained = containment(sample, sample, band=(0, 100))
    assert np.all(contained.per_axis == 1) and contained.every_axis == 1


def test_containment_is_the_share_inside_the_other_band_on_each_axis_and_on_all() -> None:
    """Worked by hand at band (0, 100).
    The wide sample's missing value sits at −1, below the narrow one's band, but inside its own.
    The narrow sample lies wholly in the wide one's band, so swapping the two raises the shares.
    """
    wide = np.array([[0.0, 6.0], [1.0, 9.5], [2.0, 7.0], [np.nan, 8.0]])
    narrow = np.array([[1.0, 6.0], [2.0, 9.0]])
    recall = containment(wide, narrow, band=(0, 100))
    precision = containment(narrow, wide, band=(0, 100))
    assert recall.per_axis.tolist() == [0.5, 0.75] and recall.every_axis == 0.25
    assert precision.per_axis.tolist() == [1.0, 1.0] and precision.every_axis == 1.0


def test_two_samples_of_one_distribution_score_at_chance_inside_the_c2st_null() -> None:
    rng = np.random.default_rng(3)
    a, b = _normal(rng), _normal(rng)
    comparison = c2st(a, b, rng, permutations=9)
    assert comparison.statistic == pytest.approx(0.5, abs=0.1)
    assert _insideNull(comparison)


def test_a_shifted_sample_is_told_apart_above_its_whole_c2st_null() -> None:
    rng = np.random.default_rng(4)
    a, b = _normal(rng), _normal(rng, shift=1.0)
    comparison = c2st(a, b, rng, permutations=4)
    assert comparison.p_value == 1 / 5


def _comparisonsAt(a: np.ndarray, b: np.ndarray, seed: int) -> list[tuple[float, list[float]]]:
    """c2st's and the distances' statistics and nulls, drawn from a generator of `seed`."""
    rng = np.random.default_rng(seed)
    comparisons = [c2st(a, b, rng, permutations=1), *distances(a, b, rng, permutations=1)]
    return [(comparison.statistic, comparison.null.tolist()) for comparison in comparisons]


def test_one_generator_seed_gives_one_comparison() -> None:
    """Every draw comes from the generator: the shuffles, and the forest's and the folds' seeds."""
    rng = np.random.default_rng(5)
    a, b = _normal(rng), _normal(rng)
    assert _comparisonsAt(a, b, seed=6) == _comparisonsAt(a, b, seed=6)


def test_distances_and_containment_run_without_scikit_learn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """scikit-learn is the extra tabdgp[fidelity], which the tests always install;
    only c2st may import it.
    """
    # a None entry makes its import fail, whether or not an earlier test imported it
    for name in [*sys.modules, 'sklearn']:
        if name.split('.')[0] == 'sklearn':
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.delitem(sys.modules, 'tabdgp.fidelity')
    fidelity = importlib.import_module('tabdgp.fidelity')
    a, b = np.zeros((2, 1)), np.ones((2, 1))
    fidelity.distances(a, b, np.random.default_rng(0), permutations=1)
    fidelity.containment(a, b)
    with pytest.raises(ImportError):
        fidelity.c2st(a, b, np.random.default_rng(0))
