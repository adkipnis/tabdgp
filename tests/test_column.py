"""The column module's seams: the column plan under the budget, and a column's values from its node's."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from tabdgp.column import apply, plan
from tabdgp.dataset import Column
from tabdgp.hyperprior import Hyperprior
from tabdgp.marginal import PARAMETERS, Family
from tests.fixtures.build import point

TYPES = ['continuous', 'binary', 'categorical', 'ordered']
ALPHA = {'level_alpha': 1.0}
# one value per marginal parameter, for columns built by hand
EXAMPLE = {
    'student_df': 5.0,
    'lognormal_sigma': 0.5,
    'zero_share': 0.2,
    'beta_a': 2.0,
    'beta_b': 3.0,
    'mixture_gap': 2.0,
    'mixture_weight': 0.3,
}


def _plan(seed: int, **overrides: object) -> dict[str, Column]:
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(**overrides).resolveDataset(rng)
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


def _shapes(columns: dict[str, Column]) -> list[tuple[str, int | None]]:
    """Each column's type and K, without its drawn parameters."""
    return [(column.type, column.levels) for column in columns.values()]


def test_k_is_clipped_when_it_would_overflow_the_budget() -> None:
    """A tiny levels_p draws K in the millions; with a budget of 3, one column of K = 4 fills it."""
    columns = _plan(
        0, design_columns=3, type_shares=point('type_shares', 'categorical'), levels_p=1e-6
    )
    assert _shapes(columns) == [('categorical', 4)]


def test_max_levels_caps_k() -> None:
    """metabeta passes 10, so no categorical or ordered column may exceed it however large K is drawn."""
    columns = _plan(
        0,
        design_columns=19,
        type_shares=point('type_shares', 'ordered'),
        levels_p=1e-6,
        max_levels=10,
    )
    assert _shapes(columns) == [('ordered', 10), ('ordered', 10), ('binary', 2)]


def test_each_column_draws_its_own_parameters() -> None:
    """Real tables mix skews and level imbalances within one table,
    so two lognormal columns of one dataset differ in sigma and two coded ones in concentration.
    """
    lognormal = _plan(
        0,
        design_columns=5,
        type_shares=point('type_shares', 'continuous'),
        marginal_shares=point('marginal_shares', 'lognormal'),
    )
    sigmas = {column.parameters['lognormal_sigma'] for column in lognormal.values()}
    assert len(sigmas) == 5
    # levels_p = 1 draws K = 3 every time, so 8 design columns are 4 ordered columns
    ordered = _plan(0, design_columns=8, type_shares=point('type_shares', 'ordered'), levels_p=1.0)
    alphas = {column.parameters['level_alpha'] for column in ordered.values()}
    assert len(alphas) == 4


def test_a_point_override_gives_every_column_its_value() -> None:
    """A caller who wants one sigma for the whole dataset passes it as a point."""
    columns = _plan(
        0,
        design_columns=5,
        type_shares=point('type_shares', 'continuous'),
        marginal_shares=point('marginal_shares', 'lognormal'),
        lognormal_sigma=0.7,
    )
    for column in columns.values():
        assert column.parameters == {'lognormal_sigma': 0.7}


def test_the_fields_drawn_per_column_are_the_families_and_the_level_concentration() -> None:
    """A field drawn per column that no column reads would silently stay a distribution."""
    read = {'level_alpha'}
    for fields in PARAMETERS.values():
        read.update(fields)
    assert read == set(Hyperprior.PER_COLUMN)


def test_plan_names_its_columns_x0_to_x_p_minus_1() -> None:
    """The one home of the rule that plan column k is node x{k}; every later step reads the names."""
    columns = _plan(0, design_columns=4, type_shares=point('type_shares', 'continuous'))
    assert list(columns) == ['x0', 'x1', 'x2', 'x3']


DISCRETE = [
    Column(type='binary', levels=2, parameters=ALPHA),
    Column(type='categorical', levels=3, parameters=ALPHA),
    Column(type='ordered', levels=7, parameters=ALPHA),
    Column(type='categorical', levels=20, parameters=ALPHA),
]


def _apply(n: int, column: Column, seed: int) -> np.ndarray:
    """The codes of one column cut from n standard normal values."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n)  # (N,)
    return apply(values, column, rng)


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
    """A small rows override can leave fewer rows than levels; each row still gets a code below K."""
    column = Column(type='categorical', levels=6, parameters=ALPHA)
    for seed in range(10):
        codes = _apply(3, column, seed)
        assert len(codes) == 3
        assert set(codes.tolist()) <= set(range(column.levels))


def _monotone(n: int, column: Column, seed: int) -> bool:
    """Whether the codes never decrease as the values increase."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n)  # (N,)
    codes = apply(values, column, rng)  # (N,)
    return bool(np.all(np.diff(codes[np.argsort(values)]) >= 0))


def test_categorical_labels_are_permuted() -> None:
    """Unpermuted codes would order the categories along the node, a structure nominal levels lack."""
    column = Column(type='categorical', levels=5, parameters=ALPHA)
    assert not all(_monotone(200, column, seed) for seed in range(20))


def test_ordered_codes_are_monotone_in_the_values() -> None:
    """An ordered column's codes carry the node's order, else it would be a categorical."""
    column = Column(type='ordered', levels=5, parameters=ALPHA)
    assert all(_monotone(200, column, seed) for seed in range(20))


def _continuous(family: str) -> Column:
    """A continuous column of the family with its example parameters."""
    parameters = {field: EXAMPLE[field] for field in PARAMETERS[family]}
    return Column(type='continuous', marginal=family, parameters=parameters)


CONTINUOUS = [_continuous(family) for family in get_args(Family)]


def _node(seed: int) -> np.ndarray:
    """500 z-scored node values, as propagate leaves them."""
    values = np.random.default_rng(seed).standard_normal(500, dtype=np.float32)  # (N,)
    return (values - values.mean()) / values.std()


def _apply_to(values: np.ndarray, column: Column, seed: int) -> np.ndarray:
    """The column cut or matched from given node values."""
    return apply(values, column, np.random.default_rng(seed))


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


@pytest.mark.parametrize(
    'column', [*CONTINUOUS, Column(type='categorical', levels=6, parameters=ALPHA)], ids=str
)
def test_plan_and_apply_are_deterministic_under_a_fixed_generator(column: Column) -> None:
    """A key reproduces its dataset bitwise only if every step draws from the generator alone."""
    assert _plan(7) == _plan(7)
    node = _node(1)
    assert np.array_equal(_apply_to(node, column, 7), _apply_to(node, column, 7))


def test_a_large_level_alpha_gives_even_level_counts() -> None:
    """level_alpha is the Dirichlet concentration of the level shares,
    so a large one leaves each share near 1/K.
    """
    column = Column(type='ordered', levels=5, parameters={'level_alpha': 1e4})
    rng = np.random.default_rng(0)
    codes = apply(rng.standard_normal(1000), column, rng)
    np.testing.assert_allclose(np.bincount(codes), 200, atol=10)
