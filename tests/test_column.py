"""The column module's seams: the column plan under the budget, and a column's values from its node's."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from tabdgp.column import apply, plan
from tabdgp.dataset import Column
from tabdgp.hyperprior import Hyperprior
from tabdgp.marginal import Family
from tests.fixtures.build import point

TYPES = ['continuous', 'binary', 'categorical', 'ordered']


def _plan(seed: int, **overrides: object) -> dict[str, Column]:
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
            columns = _plan(seed, design_columns=budget, type_shares=point('type_shares', type))
            assert sum(column.width for column in columns.values()) == budget


def test_k_is_clipped_when_it_would_overflow_the_budget() -> None:
    """A tiny levels_p draws K in the millions; with a budget of 3, one column of K = 4 fills it."""
    columns = _plan(
        0, design_columns=3, type_shares=point('type_shares', 'categorical'), levels_p=1e-6
    )
    assert list(columns.values()) == [Column(type='categorical', levels=4)]


def test_max_levels_caps_k() -> None:
    """metabeta passes 10, so no categorical or ordered column may exceed it however large K is drawn."""
    columns = _plan(
        0,
        design_columns=19,
        type_shares=point('type_shares', 'ordered'),
        levels_p=1e-6,
        max_levels=10,
    )
    assert list(columns.values()) == [
        Column(type='ordered', levels=10),
        Column(type='ordered', levels=10),
        Column(type='binary', levels=2),
    ]


def test_plan_names_its_columns_x0_to_x_p_minus_1() -> None:
    """The one home of the rule that plan column k is node x{k}; every later step reads the names."""
    columns = _plan(0, design_columns=4, type_shares=point('type_shares', 'continuous'))
    assert list(columns) == ['x0', 'x1', 'x2', 'x3']


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


CONTINUOUS = [Column(type='continuous', marginal=family) for family in get_args(Family)]


def _node(seed: int) -> np.ndarray:
    """500 z-scored node values, as propagate leaves them."""
    values = np.random.default_rng(seed).standard_normal(500, dtype=np.float32)  # (N,)
    return (values - values.mean()) / values.std()


def _apply_to(values: np.ndarray, column: Column, seed: int) -> np.ndarray:
    """The column cut or matched from given node values, under a hyperprior resolved with one generator."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior().resolve(rng)
    return apply(values, column, resolved, rng)


@pytest.mark.parametrize('column', CONTINUOUS, ids=str)
def test_a_continuous_column_has_exactly_its_marginal_samples_values(column: Column) -> None:
    """Two different nodes under one generator get the same values, only in another order.

    So the values come from the sample alone and the node supplies only the order: a quantile
    lookup that resamples (scamd's copula, 37% ties) would fail this.
    """
    values = _apply_to(_node(1), column, seed=0)
    other = _apply_to(_node(2), column, seed=0)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(np.sort(values), np.sort(other))
    assert not np.array_equal(values, other)


@pytest.mark.parametrize('column', CONTINUOUS, ids=str)
def test_a_continuous_column_has_the_nodes_ranks(column: Column) -> None:
    """Rank matching keeps the dependence the SCM built: read in the node's order, the column rises.

    Non-decreasing rather than increasing, since the zero-inflated sample has ties at 0.
    """
    node = _node(1)
    values = _apply_to(node, column, seed=0)
    in_node_order = values[np.argsort(node)]  # (N,)
    assert (np.diff(in_node_order) >= 0).all()
    assert np.ptp(values) > 0


@pytest.mark.parametrize('column', [*CONTINUOUS, Column(type='categorical', levels=6)], ids=str)
def test_plan_and_apply_are_deterministic_under_a_fixed_generator(column: Column) -> None:
    """A key reproduces its dataset bitwise only if every step draws from the generator alone."""
    assert _plan(7) == _plan(7)
    node = _node(1)
    assert np.array_equal(_apply_to(node, column, 7), _apply_to(node, column, 7))
