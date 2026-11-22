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

from tabdgp.fidelity import Comparison, c2st, containment, distances, precision, recall

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
    wide_in_narrow = containment(wide, narrow, band=(0, 100))
    narrow_in_wide = containment(narrow, wide, band=(0, 100))
    assert wide_in_narrow.per_axis.tolist() == [0.5, 0.75] and wide_in_narrow.every_axis == 0.25
    assert narrow_in_wide.per_axis.tolist() == [1.0, 1.0] and narrow_in_wide.every_axis == 1.0


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


# worked by hand: a real value's rank among its draws is the share of the draws below it,
# a tied draw counting half, over the draws that show the axis
WORKED_RANKS = [
    pytest.param(2.5, [1.0, 2.0, 3.0, 4.0], 0.5, id='two of four draws below'),
    pytest.param(2.0, [1.0, 2.0, 3.0, 4.0], 0.375, id='a tied draw counts half: (1 + 1/2)/4'),
    pytest.param(1.0, [1.0, 2.0, 3.0, 4.0], 0.125, id="a tie with the range's end: (0 + 1/2)/4"),
    pytest.param(0.0, [1.0, 2.0, 3.0, 4.0], 0.0, id='below every draw'),
    pytest.param(5.0, [1.0, 2.0, 3.0, 4.0], 1.0, id='above every draw'),
    pytest.param(2.5, [1.0, 2.0, 3.0, np.nan], 2 / 3, id='a draw that lacks the axis: 2/3'),
    pytest.param(np.nan, [1.0, 2.0, 3.0, 4.0], np.nan, id='the real table lacks the axis'),
    pytest.param(2.5, [np.nan, np.nan, np.nan, np.nan], np.nan, id='every draw lacks the axis'),
]


@pytest.mark.parametrize(('real', 'draws', 'expected'), WORKED_RANKS)
def test_a_real_value_is_ranked_among_its_paired_draws(
    real: float, draws: list[float], expected: float
) -> None:
    paired = recall(np.array([[real]]), np.array([draws]).T[None])
    assert paired.ranks.item() == pytest.approx(expected, nan_ok=True)


def test_recall_counts_per_axis_the_tables_it_ranks_and_those_outside_their_draws() -> None:
    """Worked by hand, every table's draws at 1, 2, 3, 4 on both axes.
    Axis 0 ranks 0 and 1/2, and the third table lacks it.
    Axis 1 ranks 1/8 (a tie with the range's end, so inside), 1 and 3/8.
    """
    real = np.array([[0.0, 1.0], [2.5, 5.0], [np.nan, 2.0]])
    draws = np.tile(np.array([1.0, 2.0, 3.0, 4.0])[None, :, None], (3, 1, 2))
    paired = recall(real, draws)
    assert paired.tables.tolist() == [2, 3]
    assert paired.outside.tolist() == [1, 1]
    assert paired.mean_rank.tolist() == pytest.approx([0.25, 0.5])


def _pairedNormal(rng: np.random.Generator, spread: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """2,000 real tables by 3 standard normal summaries, and 39 draws each, `spread` times as wide."""
    real = rng.standard_normal((2000, 3))  # (t, p)
    draws = spread * rng.standard_normal((2000, 39, 3))  # (t, s, p)
    return real, draws


def test_draws_from_the_real_tables_distribution_leave_5_percent_outside_at_a_mean_rank_of_half() -> (
    None
):
    """A real table is the smallest or the largest of itself and its s draws with chance 2/(s + 1),
    5 % at 39 draws.
    """
    paired = recall(*_pairedNormal(np.random.default_rng(7)))
    assert paired.outside / paired.tables == pytest.approx(0.05, abs=0.02)
    assert paired.mean_rank == pytest.approx(0.5, abs=0.03)


def _distanceFromMiddle(ranks: np.ndarray) -> float:
    """The ranks' mean distance from 1/2, 1/4 when they are uniform."""
    return float(np.mean(np.abs(ranks - 0.5)))


def test_narrow_draws_leave_more_tables_outside_and_wide_draws_pile_the_ranks_in_the_middle() -> (
    None
):
    """Recall shows both failures: draws half as wide as the real tables leave more of them outside,
    and draws twice as wide leave fewer and rank them near 1/2.
    """
    rng = np.random.default_rng(8)
    calibrated = recall(*_pairedNormal(rng))
    narrow = recall(*_pairedNormal(rng, spread=0.5))
    wide = recall(*_pairedNormal(rng, spread=2.0))
    assert np.all(narrow.outside > 2 * calibrated.outside)
    assert np.all(2 * wide.outside < calibrated.outside)
    assert _distanceFromMiddle(calibrated.ranks) == pytest.approx(0.25, abs=0.02)
    assert _distanceFromMiddle(wide.ranks) < 0.2


# worked by hand on one axis: a table is coded by its share of the real tables of other sources
# below it (a tie counting half), and those tables by their share among themselves.
# with 3 others coded 1/6, 1/2 and 5/6, a value between two of them codes at their midpoint
# and reaches 1/6 on 2 neighbors, and a value beyond them all codes at 0 or 1 and reaches 1/3.
# the real tables lie at 2, 5, 8 and 11, with the draws 1 and 6, 3 and 12, 5 and 9, 10 and 7
WORKED_REACHES = [
    pytest.param(
        [0, 1, 2, 3],
        [3 / 4, 1 / 4, 1 / 2, 3 / 4],
        id='own sources: the real tables reach 1/3, 1/6, 1/6, 1/3; '
        'the draws 1/3, 1/6; 1/6, 1/3; 1/6, 1/6; 1/3, 1/6',
    ),
    pytest.param(
        [0, 0, 1, 2],
        [1 / 2, 1 / 2, 1 / 2, 3 / 4],
        id='a sibling is left out: the first two tables have 2 others, coded 1/4 and 3/4, '
        'so they and their draws reach 1/2',
    ),
]


@pytest.mark.parametrize(('sources', 'expected'), WORKED_REACHES)
def test_precision_is_the_share_of_draws_within_their_real_tables_reach(
    sources: list[int], expected: list[float]
) -> None:
    """A table's reach is its mean distance to the 2 nearest real tables of other sources.
    A draw is within reach if its own is at most its real table's.
    """
    real = np.array([[2.0], [5.0], [8.0], [11.0]])
    draws = np.array([[[1.0], [6.0]], [[3.0], [12.0]], [[5.0], [9.0]], [[10.0], [7.0]]])
    measured = precision(real, draws, np.array(sources), neighbors=2)
    assert measured.per_table.tolist() == pytest.approx(expected)
    assert measured.share == pytest.approx(np.mean(expected))


def test_a_tables_precision_does_not_depend_on_the_other_tables_draws() -> None:
    """The real tables of other sources are the ruler, the same for every generator.
    Draws ten times as wide at the last table leave every other table's precision unchanged.
    """
    rng = np.random.default_rng(10)
    real = rng.standard_normal((30, 3))  # (t, p)
    draws = rng.standard_normal((30, 39, 3))  # (t, s, p)
    wider = np.concatenate([draws[:-1], 10 * draws[-1:]])
    sources = np.arange(30)
    before = precision(real, draws, sources).per_table
    after = precision(real, wider, sources).per_table
    assert np.array_equal(before[:-1], after[:-1])


def test_an_axis_only_one_of_two_tables_shows_costs_1_and_one_neither_shows_costs_0() -> None:
    """Worked by hand on one axis, which only the first real table shows, so both real tables
    reach 1. The first one's draw that shows the axis reaches 1 at any value (a tie, which counts
    half), and its draw that lacks the axis reaches 0. The second one's draws reach 1.
    """
    real = np.array([[3.0], [np.nan]])
    draws = np.array([[[5.0], [np.nan]], [[np.nan], [np.nan]]])
    measured = precision(real, draws, np.array([0, 1]), neighbors=1)
    assert measured.per_table.tolist() == [0.75, 0.5]


def _shareAt(rng: np.random.Generator, spread: float) -> float:
    """The precision of 39 draws per real table, `spread` times as wide as the real tables are
    around their centers: 200 tables by 4 summaries, each table its own source.
    """
    centers = rng.standard_normal((200, 4))  # (t, p)
    real = centers + rng.standard_normal((200, 4))
    draws = centers[:, None] + spread * rng.standard_normal((200, 39, 4))  # (t, s, p)
    return precision(real, draws, sources=np.arange(200)).share


def test_precision_is_half_for_draws_from_the_real_tables_distribution_and_falls_as_they_widen() -> (
    None
):
    """A real table and its draws are then exchangeable, so each draw is within reach with chance 1/2."""
    rng = np.random.default_rng(9)
    calibrated, wide, wider = _shareAt(rng, 1.0), _shareAt(rng, 2.0), _shareAt(rng, 4.0)
    assert calibrated == pytest.approx(0.5, abs=0.06)
    assert calibrated - 0.1 > wide > wider


# any sample of 2 to 10 real tables by 1 to 4 summaries, missing and infinite values included
REAL_TABLES = arrays(np.float64, st.tuples(st.integers(2, 10), st.integers(1, 4)))


@given(real=REAL_TABLES)
def test_draws_that_copy_their_real_table_rank_it_at_half_and_lie_half_within_its_reach(
    real: np.ndarray,
) -> None:
    """Every draw ties with its real table, on each axis and on the reach, and a tie counts half."""
    draws = np.repeat(real[:, None], 3, axis=1)
    ranks = recall(real, draws).ranks
    assert np.all(ranks[~np.isnan(real)] == 0.5) and np.all(np.isnan(ranks[np.isnan(real)]))
    measured = precision(real, draws, sources=np.arange(len(real)), neighbors=1)
    assert np.all(measured.per_table == 0.5)


def test_precision_refuses_a_source_that_leaves_fewer_real_tables_than_neighbors() -> None:
    """The first source holds two of the three tables, so its tables have one neighbor."""
    real = np.array([[2.0], [5.0], [8.0]])
    draws = np.repeat(real[:, None], 2, axis=1)
    with pytest.raises(ValueError, match='1 real tables of other sources'):
        precision(real, draws, np.array([0, 0, 1]), neighbors=2)


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


def test_only_c2st_needs_scikit_learn(monkeypatch: pytest.MonkeyPatch) -> None:
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
    fidelity.recall(a, b[:, None])
    fidelity.precision(a, b[:, None], np.arange(2), neighbors=1)
    with pytest.raises(ImportError):
        fidelity.c2st(a, b, np.random.default_rng(0))
