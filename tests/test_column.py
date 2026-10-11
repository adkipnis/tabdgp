"""The column module's seams: the column plan under the budget, and the post-hoc types."""

from __future__ import annotations

import numpy as np
import pytest

from tabdgp.column import apply, plan
from tabdgp.dataset import Column
from tabdgp.hyperprior import Hyperprior

TYPES = ['continuous', 'binary', 'categorical', 'ordered']


def _corner(type: str) -> dict[str, float]:
    """Type shares that put all mass on one type."""
    shares = {}
    for name in TYPES:
        shares[name] = 1.0 if name == type else 0.0
    return shares


def _cost(column: Column) -> int:
    """A column's width in the design matrix: 1 if continuous, K − 1 otherwise."""
    return 1 if column.levels is None else column.levels - 1


def _plan(seed: int, **overrides: object) -> list[Column]:
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(**overrides).resolve(rng)
    return plan(resolved, rng)


@pytest.mark.parametrize('type', TYPES)
def test_the_budget_is_met_exactly_for_budgets_0_to_19_under_every_type_share_corner(
    type: str,
) -> None:
    """metabeta's design matrix has the width it drew.

    A corner such as all-categorical must still fill a remainder of 1.
    """
    for budget in range(20):
        for seed in range(10):
            columns = _plan(seed, design_columns=budget, type_shares=_corner(type))
            assert sum(_cost(column) for column in columns) == budget


def test_k_is_clipped_when_it_would_overflow_the_budget() -> None:
    """A tiny levels_p draws K in the millions; with a budget of 3, one column of K = 4 fills it."""
    columns = _plan(0, design_columns=3, type_shares=_corner('categorical'), levels_p=1e-6)
    assert columns == [Column(type='categorical', levels=4)]


def test_max_levels_caps_k() -> None:
    """metabeta passes 10, so no categorical or ordered column may exceed it however large K is drawn."""
    columns = _plan(
        0, design_columns=19, type_shares=_corner('ordered'), levels_p=1e-6, max_levels=10
    )
    assert columns == [
        Column(type='ordered', levels=10),
        Column(type='ordered', levels=10),
        Column(type='binary', levels=2),
    ]


FAMILIES = {
    'normal',
    'lognormal',
    'student_t',
    'beta',
    'uniform',
    'zero_inflated_lognormal',
    'normal_mixture',
}


def test_every_continuous_column_names_a_marginal_family() -> None:
    """draw hands Column.marginal to match, which knows only the catalogue's families."""
    columns = _plan(0, design_columns=19, type_shares=_corner('continuous'))
    assert {column.marginal for column in columns} <= FAMILIES


DISCRETE = [
    Column(type='binary', levels=2),
    Column(type='categorical', levels=3),
    Column(type='ordered', levels=7),
    Column(type='categorical', levels=20),
]


def _apply(n: int, column: Column, seed: int) -> np.ndarray:
    """The codes of one column cut from n standard normal values."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior().resolve(rng)
    values = rng.standard_normal(n)  # (N,)
    return apply(values, column, resolved, rng)


@pytest.mark.parametrize('column', DISCRETE, ids=str)
def test_apply_gives_codes_0_to_k_minus_1_with_every_level_present_when_n_allows(
    column: Column,
) -> None:
    """A level no row has would not exist in a real table; n = K is the tightest case that allows all."""
    for n in [column.levels, 100]:
        for seed in range(10):
            codes = _apply(n, column, seed)
            assert codes.dtype == np.int64
            assert np.array_equal(np.unique(codes), np.arange(column.levels))


def test_apply_gives_valid_codes_when_n_is_below_k() -> None:
    """A small n_obs override can leave fewer rows than levels; each row still gets a code below K."""
    column = Column(type='categorical', levels=6)
    for seed in range(10):
        codes = _apply(3, column, seed)
        assert len(codes) == 3
        assert set(codes.tolist()) <= set(range(column.levels))


def _monotone(n: int, column: Column, seed: int) -> bool:
    """Whether the codes never decrease as the values increase."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior().resolve(rng)
    values = rng.standard_normal(n)  # (N,)
    codes = apply(values, column, resolved, rng)  # (N,)
    return bool(np.all(np.diff(codes[np.argsort(values)]) >= 0))


def test_categorical_labels_are_permuted() -> None:
    """Unpermuted codes would order the categories along the node, a structure nominal levels lack."""
    column = Column(type='categorical', levels=5)
    assert not all(_monotone(200, column, seed) for seed in range(20))


def test_ordered_codes_are_monotone_in_the_values() -> None:
    """An ordered column's codes carry the node's order, else it would be a categorical."""
    column = Column(type='ordered', levels=5)
    assert all(_monotone(200, column, seed) for seed in range(20))


def test_plan_and_apply_are_deterministic_under_a_fixed_generator() -> None:
    """A key reproduces its dataset bitwise only if every step draws from the generator alone."""
    assert _plan(7) == _plan(7)
    column = Column(type='categorical', levels=6)
    assert np.array_equal(_apply(100, column, 7), _apply(100, column, 7))
