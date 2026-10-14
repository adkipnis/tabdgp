"""The column module's seams: what a Column guarantees on its own,
the grouping factors and their units, the column plan under the budget,
and a column's values from its node's.
"""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.column import Column, Grouping, apply, assign, plan
from tabdgp.hyperprior import MARGINAL_FIELDS, Family, Hyperprior
from tests.fixtures.build import groupingOf, point

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


def test_a_column_type_outside_the_four_fails() -> None:
    """metabeta's encoder branches on the type, so a fifth would pass through unencoded."""
    with pytest.raises(ValidationError):
        Column(type='count')


CONTRADICTING_LEVELS = [
    pytest.param('continuous', 3, id='a continuous column with levels'),
    pytest.param('binary', 3, id='a binary column of three levels'),
    pytest.param('binary', None, id='a binary column without levels'),
    pytest.param('categorical', 2, id='a categorical column of two levels'),
    pytest.param('ordered', 2, id='an ordered column of two levels'),
    pytest.param('categorical', None, id='a categorical column without levels'),
]


@pytest.mark.parametrize(('type', 'levels'), CONTRADICTING_LEVELS)
def test_levels_that_contradict_the_type_fail(type: str, levels: int | None) -> None:
    """Binary is exactly K = 2 and the other discrete types K ≥ 3, so the types never overlap."""
    with pytest.raises(ValidationError):
        Column(type=type, levels=levels)


def test_a_marginal_on_a_discrete_column_fails() -> None:
    """Only continuous columns are rank matched; a discrete column's values are its codes."""
    with pytest.raises(ValidationError):
        Column(type='binary', levels=2, marginal='lognormal')


def test_a_continuous_column_without_a_marginal_fails() -> None:
    """Its parameters are its family's,
    so a continuous column names the family it was drawn from.
    """
    with pytest.raises(ValidationError):
        Column(type='continuous')


FOREIGN_PARAMETERS = [
    pytest.param(
        {'type': 'continuous', 'marginal': 'lognormal'}, {}, id='a lognormal without its sigma'
    ),
    pytest.param(
        {'type': 'continuous', 'marginal': 'normal'},
        {'lognormal_sigma': 1.0},
        id='a normal with a lognormal sigma',
    ),
    pytest.param({'type': 'binary', 'levels': 2}, {}, id='a binary without its concentration'),
    pytest.param(
        {'type': 'categorical', 'levels': 3},
        {'level_alpha': 1.0, 'beta_a': 2.0},
        id='a categorical with a beta shape',
    ),
]


@pytest.mark.parametrize(('fields', 'parameters'), FOREIGN_PARAMETERS)
def test_parameters_other_than_the_columns_own_fail(fields: dict, parameters: dict) -> None:
    """A missing value hides a draw; an extra one claims a draw that played no part."""
    with pytest.raises(ValidationError):
        Column(**fields, parameters=parameters)


def test_an_unknown_marginal_family_fails_at_construction() -> None:
    """A family outside the catalogue would otherwise fail only when the column is matched."""
    with pytest.raises(ValidationError):
        Column(type='continuous', marginal='gamma')


DESIGN_COLUMNS = [
    pytest.param(Column(type='continuous', marginal='normal'), 1, id='continuous'),
    pytest.param(Column(type='binary', levels=2, parameters=ALPHA), 1, id='binary'),
    pytest.param(Column(type='categorical', levels=5, parameters=ALPHA), 4, id='categorical'),
    pytest.param(Column(type='ordered', levels=3, parameters=ALPHA), 2, id='ordered'),
]


@pytest.mark.parametrize(('column', 'design_columns'), DESIGN_COLUMNS)
def test_a_column_is_one_design_column_wide_if_continuous_and_k_minus_1_if_not(
    column: Column, design_columns: int
) -> None:
    """The design-column budget counts metabeta's dummy coding: K levels take K − 1 columns."""
    assert column.design_columns == design_columns


def test_a_grouping_factor_is_0_design_columns_wide() -> None:
    """Its codes index random effects and are never dummy coded,
    so the budget counts the covariates alone.
    """
    column = Column(type='categorical', levels=2, role='grouping factor')
    assert column.design_columns == 0


def test_a_grouping_factor_with_fewer_than_2_levels_fails() -> None:
    """One group is no grouping; two are, so a factor need not be K ≥ 3 like a categorical."""
    with pytest.raises(ValidationError, match='1 levels'):
        Column(type='categorical', levels=1, role='grouping factor')


@pytest.mark.parametrize(('type', 'levels'), [('binary', 2), ('ordered', 3)])
def test_a_grouping_factor_of_another_type_fails(type: str, levels: int) -> None:
    """Its codes are labels of groups, unordered at any G, so a factor is categorical even at 2."""
    with pytest.raises(ValidationError):
        Column(type=type, levels=levels, role='grouping factor')


def test_a_marginal_on_a_grouping_factor_fails() -> None:
    """A factor's values are its codes, drawn by the size draw, never matched to a marginal."""
    with pytest.raises(ValidationError, match='marginal'):
        Column(type='categorical', levels=3, role='grouping factor', marginal='normal')


@pytest.mark.parametrize('parameters', [ALPHA, {'size_alpha': 1.0, 'beta_a': 2.0}], ids=str)
def test_a_parameter_other_than_size_alpha_on_a_grouping_factor_fails(parameters: dict) -> None:
    """A factor draws only its sizes, so its level concentration or a shape would be a claim
    of a draw that played no part.
    """
    with pytest.raises(ValidationError, match='draws'):
        Column(type='categorical', levels=3, role='grouping factor', parameters=parameters)


def test_nests_in_on_a_covariate_fails() -> None:
    """Nesting relates grouping factors; a covariate is at most constant within groups."""
    with pytest.raises(ValidationError):
        Column(type='categorical', levels=3, parameters=ALPHA, nests_in='g0')


def test_chain_follows_the_nesting_on_a_hand_built_grouping() -> None:
    """A node may read the units of its chain: g2 nests in g1, g1 in g0, g3 is crossed with all.

    The row is finer than every factor, so its chain holds them all.
    """
    grouping = groupingOf(
        [[0, 0, 1, 1], [0, 1, 2, 2], [0, 1, 2, 3], [0, 1, 0, 1]],
        nests_in=[None, 'g0', 'g1', None],
    )
    assert grouping.chain('g0') == ['g0']
    assert grouping.chain('g2') == ['g0', 'g1', 'g2']
    assert grouping.chain('g3') == ['g3']
    assert grouping.chain('row') == ['g0', 'g1', 'g2', 'g3', 'row']


# g1 nests in g0, which holds g1's groups 1 and 2 in group 0 and its groups 0 and 3 in group 1;
# g2 is crossed with both
NESTED = groupingOf(
    [[0, 1, 1, 0, 0, 1, 1, 0], [2, 0, 3, 1, 2, 0, 3, 1], [0, 1, 0, 1, 0, 1, 0, 1]],
    nests_in=[None, 'g0', None],
)


def test_broadcast_repeats_a_units_values_over_its_rows_on_a_hand_built_grouping() -> None:
    """A group's value is shared by its rows and by the finer groups within it,
    so a node is read at the row or at a finer unit of its chain.
    """
    at_g0 = np.array([10.0, 20.0])
    np.testing.assert_array_equal(NESTED.broadcast(at_g0, 'g0', 'g0'), [10, 20])
    np.testing.assert_array_equal(NESTED.broadcast(at_g0, 'g0', 'g1'), [20, 10, 10, 20])
    np.testing.assert_array_equal(
        NESTED.broadcast(at_g0, 'g0', 'row'), [10, 20, 20, 10, 10, 20, 20, 10]
    )
    at_g1 = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(NESTED.broadcast(at_g1, 'g1', 'row'), [3, 1, 4, 2, 3, 1, 4, 2])


OFF_CHAIN = [
    pytest.param('g2', 'g1', id='between crossed factors'),
    pytest.param('g1', 'g0', id='from a finer factor to a coarser one'),
    pytest.param('row', 'g0', id='from the row to a factor'),
]


@pytest.mark.parametrize(('from_unit', 'to_unit'), OFF_CHAIN)
def test_broadcasting_out_of_a_units_chain_fails(from_unit: str, to_unit: str) -> None:
    """Off the chain, one target group spans several source groups and has no single value;
    the first row's would be returned without complaint and mean nothing.
    """
    with pytest.raises(ValueError, match=f'{from_unit}.*{to_unit}'):
        NESTED.broadcast(np.zeros(8), from_unit, to_unit)


def _assign(seed: int, **overrides: object) -> Grouping:
    rng = np.random.default_rng(seed)
    resolved = Hyperprior.grouped(**overrides).resolveDataset(rng)
    return assign(resolved, rng)


def test_factors_0_gives_an_empty_grouping_whose_broadcast_is_the_identity() -> None:
    """A flat table is a grouping of no factors, so every later step runs on it unchanged."""
    grouping = _assign(0, factors=0)
    values = np.arange(5.0)
    assert grouping.metadata == {}
    assert grouping.codes == {}
    assert grouping.chain('row') == ['row']
    np.testing.assert_array_equal(grouping.broadcast(values, 'row', 'row'), values)


def test_a_balanced_factors_sizes_differ_by_at_most_one_and_record_no_size_alpha() -> None:
    """Rule 7's balanced case gives each group n // G rows and the first n mod G one more;
    a size_alpha that played no part is not recorded.
    """
    for seed in range(20):
        grouping = _assign(seed, factors=3, nested_share=0, balanced_share=1)
        assert list(grouping.codes) == ['g0', 'g1', 'g2']
        for name, codes in grouping.codes.items():
            sizes = np.bincount(codes)
            assert sizes.max() - sizes.min() <= 1
            assert grouping.metadata[name].parameters == {}


def test_an_unbalanced_factor_records_its_size_alpha() -> None:
    """The concentration shapes the sizes, so the proofs read it off the column;
    each factor draws its own, and a point override gives all one.
    """
    drawn = _assign(0, factors=3, nested_share=0, balanced_share=0)
    alphas = {column.parameters['size_alpha'] for column in drawn.metadata.values()}
    assert len(alphas) == 3
    given = _assign(0, factors=3, nested_share=0, balanced_share=0, size_alpha=7.0)
    for column in given.metadata.values():
        assert column.parameters == {'size_alpha': 7.0}


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
    100 rows admit every budget up to 19 (n ≥ 2d).
    """
    for budget in range(20):
        for seed in range(10):
            columns = _plan(
                seed, rows=100, design_columns=budget, type_shares=point('type_shares', type)
            )
            assert sum(column.design_columns for column in columns.values()) == budget


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
    """metabeta passes 10,
    so no categorical or ordered column may exceed it however large K is drawn.
    """
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


def test_plan_names_its_columns_x0_to_x_p_minus_1() -> None:
    """The one home of the rule that plan column k is node x{k};
    every later step reads the names.
    """
    columns = _plan(0, design_columns=4, type_shares=point('type_shares', 'continuous'))
    assert list(columns) == ['x0', 'x1', 'x2', 'x3']


DISCRETE = [
    Column(type='binary', levels=2, parameters=ALPHA),
    Column(type='categorical', levels=3, parameters=ALPHA),
    Column(type='ordered', levels=7, parameters=ALPHA),
    Column(type='categorical', levels=20, parameters=ALPHA),
]


def _apply(n: int, column: Column, seed: int) -> np.ndarray:
    """One column cut or matched from n standard normal values."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n)  # (n,)
    return apply(values, column, rng)


@pytest.mark.parametrize('column', DISCRETE, ids=str)
def test_apply_gives_codes_0_to_k_minus_1_with_every_level_present_when_n_allows(
    column: Column,
) -> None:
    """A level no row has would not exist in a real table;
    n = K is the tightest case that allows all.
    """
    for n in [column.levels, 100]:
        for seed in range(10):
            codes = _apply(n, column, seed)
            assert codes.dtype == np.int64
            assert np.array_equal(np.unique(codes), np.arange(column.levels))


def test_apply_gives_valid_codes_when_n_is_below_k() -> None:
    """A small rows override can leave fewer rows than levels;
    each row still gets a code below K.
    """
    column = Column(type='categorical', levels=6, parameters=ALPHA)
    for seed in range(10):
        codes = _apply(3, column, seed)
        assert len(codes) == 3
        assert set(codes.tolist()) <= set(range(column.levels))


def _monotone(n: int, column: Column, seed: int) -> bool:
    """Whether the codes never decrease as the values increase."""
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n)  # (n,)
    codes = apply(values, column, rng)  # (n,)
    return bool(np.all(np.diff(codes[np.argsort(values)]) >= 0))


def test_categorical_labels_are_permuted() -> None:
    """Unpermuted codes would order the categories along the node,
    a structure nominal levels lack.
    """
    column = Column(type='categorical', levels=5, parameters=ALPHA)
    assert not all(_monotone(200, column, seed) for seed in range(20))


def test_ordered_codes_are_monotone_in_the_values() -> None:
    """An ordered column's codes carry the node's order, else it would be a categorical."""
    column = Column(type='ordered', levels=5, parameters=ALPHA)
    assert all(_monotone(200, column, seed) for seed in range(20))


def _continuous(family: str) -> Column:
    """A continuous column of the family with its example parameters."""
    parameters = {field: EXAMPLE[field] for field in MARGINAL_FIELDS[family]}
    return Column(type='continuous', marginal=family, parameters=parameters)


CONTINUOUS = [_continuous(family) for family in get_args(Family)]


def _node(seed: int) -> np.ndarray:
    """500 z-scored node values, as propagate leaves them."""
    values = np.random.default_rng(seed).standard_normal(500, dtype=np.float32)  # (n,)
    return (values - values.mean()) / values.std()


def _applyTo(values: np.ndarray, column: Column, seed: int) -> np.ndarray:
    """The column cut or matched from given node values."""
    return apply(values, column, np.random.default_rng(seed))


@pytest.mark.parametrize('column', CONTINUOUS, ids=str)
def test_a_continuous_column_has_exactly_its_marginal_samples_values(column: Column) -> None:
    """Two different nodes under one generator get the same values, only in another order.

    So the values come from the sample alone and the node supplies only the order: a quantile
    lookup that resamples (scamd's copula, 37% ties) would fail this.
    """
    values = _applyTo(_node(1), column, seed=0)
    other = _applyTo(_node(2), column, seed=0)
    assert values.dtype == np.float32
    np.testing.assert_array_equal(np.sort(values), np.sort(other))
    assert not np.array_equal(values, other)


@pytest.mark.parametrize('column', CONTINUOUS, ids=str)
def test_a_continuous_column_has_the_nodes_ranks(column: Column) -> None:
    """Rank matching keeps the dependence the SCM built: read in the node's order, the column rises.

    Non-decreasing rather than increasing, since the zero-inflated sample has ties at 0.
    """
    node = _node(1)
    values = _applyTo(node, column, seed=0)
    in_node_order = values[np.argsort(node)]  # (n,)
    assert (np.diff(in_node_order) >= 0).all()
    assert np.ptp(values) > 0


def test_plan_is_deterministic_under_a_fixed_generator() -> None:
    """A key reproduces its dataset bitwise only if every step draws from the generator alone."""
    assert _plan(7) == _plan(7)


@pytest.mark.parametrize(
    'column', [*CONTINUOUS, Column(type='categorical', levels=6, parameters=ALPHA)], ids=str
)
def test_apply_is_deterministic_under_a_fixed_generator(column: Column) -> None:
    """Every marginal family and the level codes draw from the generator alone."""
    node = _node(1)
    assert np.array_equal(_applyTo(node, column, 7), _applyTo(node, column, 7))


def test_a_large_level_alpha_gives_even_level_counts() -> None:
    """level_alpha is the Dirichlet concentration of the level shares,
    so a large one leaves each share near 1/K.
    """
    column = Column(type='ordered', levels=5, parameters={'level_alpha': 1e4})
    codes = _apply(1000, column, seed=0)
    np.testing.assert_allclose(np.bincount(codes), 200, atol=10)


def _fromDefaults(family: str, n: int, seed: int) -> np.ndarray:
    """A column of the family under parameters drawn from the defaults, as plan draws them."""
    rng = np.random.default_rng(seed)
    hyperprior = Hyperprior()
    parameters = {field: hyperprior.resolveField(field, rng) for field in MARGINAL_FIELDS[family]}
    column = Column(type='continuous', marginal=family, parameters=parameters)
    return apply(rng.standard_normal(n), column, rng)


SUPPORTS = {
    'normal': (-np.inf, np.inf),
    'lognormal': (0, np.inf),
    'student_t': (-np.inf, np.inf),
    'beta': (0, 1),
    'uniform': (0, 1),
    'zero_inflated_lognormal': (0, np.inf),
    'normal_mixture': (-np.inf, np.inf),
}


@pytest.mark.parametrize('family', get_args(Family))
def test_each_familys_values_lie_in_its_support(family: str) -> None:
    """A bounded column stays bounded and every column is finite, over many drawn parameters.

    Closed bounds: a zero-inflated or uniform draw can be 0.
    """
    low, high = SUPPORTS[family]
    for seed in range(50):
        values = _fromDefaults(family, 500, seed)
        assert np.isfinite(values).all()
        assert (low <= values).all() and (values <= high).all()


def _large(family: str, **parameters: float) -> np.ndarray:
    """A column of 100,000 rows, so that its statistics sit within 0.01 of their law's."""
    column = Column(type='continuous', marginal=family, parameters=parameters)
    return _apply(100_000, column, seed=0)


def test_each_familys_parameters_are_honoured() -> None:
    """Each family reads its own parameters; expected values are closed forms, not the sampler's.

    The mixture's gap is set wide, so the share above half of it is the minor weight.
    The t's two-tailed P(|T| > 3) at df = 3 is 0.0577, from its closed-form CDF.
    """
    lognormal = _large('lognormal', lognormal_sigma=0.5)
    assert abs(np.log(lognormal).std() - 0.5) < 0.01
    inflated = _large('zero_inflated_lognormal', lognormal_sigma=1.0, zero_share=0.3)
    assert abs((inflated == 0).mean() - 0.3) < 0.01
    beta = _large('beta', beta_a=2.0, beta_b=5.0)
    assert abs(beta.mean() - 2 / 7) < 0.01
    mixture = _large('normal_mixture', mixture_gap=20.0, mixture_weight=0.2)
    assert abs((mixture > 10).mean() - 0.2) < 0.01
    assert abs(mixture[mixture > 10].mean() - 20) < 0.05
    student = _large('student_t', student_df=3.0)
    assert abs((np.abs(student) > 3).mean() - 0.0577) < 0.005
