"""The column module's seams: what a Column guarantees on its own,
the grouping factors and their units, the column plan under the budget,
and a column's values from its node's.
"""

from __future__ import annotations

import math
from typing import get_args

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tabdgp.column import Column, Grouping, assign, plan, rankMatch
from tabdgp.field import UniformInt
from tabdgp.scm import MARGINAL_FIELDS, Marginal, Scm
from tests.fixtures.build import groupingOf, point

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
    'poisson_log_rate': 1.0,
}


def test_a_column_type_outside_the_five_fails() -> None:
    """metabeta's encoder branches on the type, so a sixth would pass through unencoded."""
    with pytest.raises(ValidationError):
        Column(type='integer')


CONTRADICTING_LEVELS = [
    pytest.param('continuous', 3, id='a continuous column with levels'),
    pytest.param('binary', 3, id='a binary column of three levels'),
    pytest.param('binary', None, id='a binary column without levels'),
    pytest.param('categorical', 2, id='a categorical column of two levels'),
    pytest.param('ordered', 2, id='an ordered column of two levels'),
    pytest.param('categorical', None, id='a categorical column without levels'),
    pytest.param('count', 3, id='a count column with levels'),
]


@pytest.mark.parametrize(('type', 'levels'), CONTRADICTING_LEVELS)
def test_levels_that_contradict_the_type_fail(type: str, levels: int | None) -> None:
    """Binary is exactly k = 2 and the other discrete types k >= 3, so the types never overlap."""
    with pytest.raises(ValidationError):
        Column(type=type, levels=levels)


CONTRADICTING_MARGINALS = [
    pytest.param(
        {'type': 'binary', 'levels': 2, 'marginal': 'lognormal'},
        id='a binary column with a marginal',
    ),
    pytest.param(
        {'type': 'count', 'marginal': 'lognormal', 'parameters': {'lognormal_sigma': 0.5}},
        id='a count with a continuous family',
    ),
    pytest.param(
        {'type': 'continuous', 'marginal': 'poisson', 'parameters': {'poisson_log_rate': 1.0}},
        id='a continuous column with the poisson marginal',
    ),
]


@pytest.mark.parametrize('fields', CONTRADICTING_MARGINALS)
def test_a_marginal_that_contradicts_the_type_fails(fields: dict) -> None:
    """A discrete column's values are its codes, so it has no marginal.
    A count's family is poisson, a continuous column's one of marginal_shares, which has no poisson:
    a Poisson sample cast to float32 would be a count typed as continuous.
    """
    with pytest.raises(ValidationError, match='marginal'):
        Column(**fields)


FOREIGN_PARAMETERS = [
    pytest.param(
        {'type': 'continuous', 'marginal': 'lognormal'}, {}, id='a lognormal without its sigma'
    ),
    pytest.param(
        {'type': 'continuous', 'marginal': 'normal'},
        {'lognormal_sigma': 1.0},
        id='a normal with a lognormal sigma',
    ),
    pytest.param({'type': 'binary', 'levels': 2}, {'beta_a': 2.0}, id='a binary with a beta shape'),
    pytest.param(
        {'type': 'continuous'}, {'lognormal_sigma': 1.0}, id='no marginal with a lognormal sigma'
    ),
    pytest.param(
        {'type': 'continuous'}, {'level_alpha': 1.0}, id='a continuous column with a concentration'
    ),
    pytest.param(
        {'type': 'categorical', 'levels': 3},
        {'level_alpha': 1.0, 'beta_a': 2.0},
        id='a categorical with a beta shape',
    ),
    pytest.param({'type': 'count', 'marginal': 'poisson'}, {}, id='a count without its log rate'),
    pytest.param(
        {'type': 'count', 'marginal': 'poisson'},
        {'poisson_log_rate': 1.0, 'level_alpha': 1.0},
        id='a count with a level concentration',
    ),
]


@pytest.mark.parametrize(('fields', 'parameters'), FOREIGN_PARAMETERS)
def test_parameters_other_than_the_columns_own_fail(fields: dict, parameters: dict) -> None:
    """A marginal's parameters are exactly its fields, a coded column's at most its concentration:
    a missing value hides a draw; an extra one claims a draw that played no part.
    """
    with pytest.raises(ValidationError):
        Column(**fields, parameters=parameters)


BARE = [
    pytest.param({'type': 'continuous'}, id='continuous'),
    pytest.param({'type': 'count'}, id='count'),
    pytest.param({'type': 'binary', 'levels': 2}, id='binary'),
    pytest.param({'type': 'categorical', 'levels': 3}, id='categorical'),
]


@pytest.mark.parametrize('fields', BARE)
def test_a_column_without_a_marginal_or_parameters_validates(fields: dict) -> None:
    """A model writes y without a marginal, and a real table's columns come with none;
    only a column a plan draws carries a marginal or a level concentration.
    """
    Column(**fields)


def test_an_unknown_marginal_family_fails_at_construction() -> None:
    """A family outside the catalog would otherwise fail only when the column is matched."""
    with pytest.raises(ValidationError):
        Column(type='continuous', marginal='gamma')


DESIGN_COLUMNS = [
    pytest.param(Column(type='continuous', marginal='normal'), 1, id='continuous'),
    pytest.param(Column(type='binary', levels=2, parameters=ALPHA), 1, id='binary'),
    pytest.param(Column(type='categorical', levels=5, parameters=ALPHA), 4, id='categorical'),
    pytest.param(Column(type='ordered', levels=3, parameters=ALPHA), 2, id='ordered'),
    pytest.param(
        Column(type='count', marginal='poisson', parameters={'poisson_log_rate': 1.0}),
        1,
        id='count',
    ),
]


@pytest.mark.parametrize(('column', 'design_columns'), DESIGN_COLUMNS)
def test_a_column_is_one_design_column_wide_without_levels_and_k_minus_1_with_them(
    column: Column, design_columns: int
) -> None:
    """The design-column budget counts metabeta's dummy coding: k levels take k − 1 columns.
    A count enters the design matrix as a continuous column.
    """
    assert column.design_columns == design_columns


OFF_BUDGET = [
    pytest.param(
        Column(type='categorical', levels=2, role='grouping_factor'),
        id='a grouping factor, whose codes index random effects',
    ),
    pytest.param(
        Column(type='categorical', levels=4, role='outcome', parameters=ALPHA),
        id='a categorical y, which the covariates predict',
    ),
]


@pytest.mark.parametrize('column', OFF_BUDGET)
def test_a_grouping_factor_or_the_outcome_is_0_design_columns_wide(column: Column) -> None:
    """The design matrix dummy codes the covariates alone, so the budget counts them alone."""
    assert column.design_columns == 0


def test_a_grouping_factor_with_fewer_than_2_levels_fails() -> None:
    """One group is no grouping; two are, so a factor need not be k >= 3 like a categorical."""
    with pytest.raises(ValidationError, match='1 levels'):
        Column(type='categorical', levels=1, role='grouping_factor')


@pytest.mark.parametrize(('type', 'levels'), [('binary', 2), ('ordered', 3)])
def test_a_grouping_factor_of_another_type_fails(type: str, levels: int) -> None:
    """Its codes are labels of groups, unordered at any m, so a factor is categorical even at 2."""
    with pytest.raises(ValidationError):
        Column(type=type, levels=levels, role='grouping_factor')


def test_a_marginal_on_a_grouping_factor_fails() -> None:
    """A factor's values are its codes, drawn by the size draw, never matched to a marginal."""
    with pytest.raises(ValidationError, match='marginal'):
        Column(type='categorical', levels=3, role='grouping_factor', marginal='normal')


@pytest.mark.parametrize('parameters', [ALPHA, {'size_alpha': 1.0, 'beta_a': 2.0}], ids=str)
def test_a_parameter_other_than_size_alpha_on_a_grouping_factor_fails(parameters: dict) -> None:
    """A factor draws only its sizes, so its level concentration or a shape would be a claim
    of a draw that played no part.
    """
    with pytest.raises(ValidationError, match='draws'):
        Column(type='categorical', levels=3, role='grouping_factor', parameters=parameters)


@pytest.mark.parametrize('role', ['covariate', 'outcome'])
def test_nests_in_on_a_covariate_or_the_outcome_fails(role: str) -> None:
    """Nesting relates grouping factors; any other column is at most constant within groups."""
    with pytest.raises(ValidationError, match='nest'):
        Column(type='categorical', levels=3, role=role, parameters=ALPHA, nests_in='g0')


def test_chain_follows_the_nesting_on_a_hand_built_grouping() -> None:
    """A node may read the units of its chain: g2 nests in g1, g1 in g0, g3 is crossed with all.

    The row is subordinate to every factor, so its chain holds them all.
    """
    grouping = groupingOf(
        [[0, 0, 1, 1], [0, 1, 2, 2], [0, 1, 2, 3], [0, 1, 0, 1]],
        nests_in=[None, 'g0', 'g1', None],
    )
    assert grouping.traceChain('g0') == ['g0']
    assert grouping.traceChain('g2') == ['g0', 'g1', 'g2']
    assert grouping.traceChain('g3') == ['g3']
    assert grouping.traceChain('row') == ['g0', 'g1', 'g2', 'g3', 'row']


# g1 nests in g0, which holds g1's groups 1 and 2 in group 0 and its groups 0 and 3 in group 1;
# g2 is crossed with both
NESTED = groupingOf(
    [[0, 1, 1, 0, 0, 1, 1, 0], [2, 0, 3, 1, 2, 0, 3, 1], [0, 1, 0, 1, 0, 1, 0, 1]],
    nests_in=[None, 'g0', None],
)


def test_broadcast_repeats_a_units_values_over_its_rows_on_a_hand_built_grouping() -> None:
    """A group's value is shared by its rows and by the subordinate groups within it,
    so a node is read at the row or at a subordinate unit of its chain.
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
    pytest.param('g1', 'g0', id='from a subordinate factor to a superior one'),
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
    resolved = Scm.grouped(**overrides).resolveDataset(rng)
    return assign(resolved, rng)


def test_factors_0_gives_an_empty_grouping_whose_broadcast_is_the_identity() -> None:
    """A flat table is a grouping of no factors, so every later step runs on it unchanged."""
    grouping = _assign(0, factors=0)
    values = np.arange(5.0)
    assert grouping.metadata == {}
    assert grouping.codes == {}
    assert grouping.traceChain('row') == ['row']
    np.testing.assert_array_equal(grouping.broadcast(values, 'row', 'row'), values)


@given(
    seed=st.integers(0, 2**32 - 1),
    rows=st.integers(5, 60),
    factors=st.integers(0, 4),
    nested_share=st.sampled_from([0.0, 0.5, 1.0]),
    balanced_share=st.sampled_from([0.0, 0.5, 1.0]),
)
def test_assign_gives_g_non_empty_groups_with_g_at_most_n_for_every_factor(
    seed: int, rows: int, factors: int, nested_share: float, balanced_share: float
) -> None:
    """metabeta indexes one random effect per group,
    so an empty group or more groups than rows would need a retry or a repair.
    The truncation and metabeta's rule 7 rule both out on every draw.
    Five rows fit a chain of four factors.
    """
    grouping = _assign(
        seed,
        rows=rows,
        factors=factors,
        nested_share=nested_share,
        balanced_share=balanced_share,
    )
    assert len(grouping.codes) == factors
    for name, codes in grouping.codes.items():
        levels = grouping.metadata[name].levels
        assert codes.dtype == np.int64
        assert len(codes) == rows
        assert levels <= rows
        np.testing.assert_array_equal(np.unique(codes), np.arange(levels))


def test_a_balanced_factors_sizes_differ_by_at_most_one_and_record_no_size_alpha() -> None:
    """Rule 7's balanced case gives each group n // m rows and the first n mod m one more;
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


def test_a_tiny_size_alpha_gives_singleton_groups() -> None:
    """A tiny concentration puts nearly all the share on one group,
    so every other group keeps only the one row that metabeta's rule 7 guarantees it.
    """
    for seed in range(10):
        grouping = _assign(seed, rows=1000, factors=1, groups=10, balanced_share=0, size_alpha=1e-6)
        sizes = np.sort(np.bincount(grouping.codes['g0']))
        np.testing.assert_array_equal(sizes, [1] * 9 + [991])


def test_two_crossed_factors_codes_are_independent() -> None:
    """Crossed assignments are drawn independently: a dependence between them would be a quirk
    the registry has not measured, so each cell's count is near the product of the margins.
    """
    grouping = _assign(0, rows=100_000, factors=2, nested_share=0, groups=4, balanced_share=1)
    cells = np.bincount(4 * grouping.codes['g0'] + grouping.codes['g1']).reshape(4, 4)
    expected = np.outer(cells.sum(axis=1), cells.sum(axis=0)) / 100_000  # (4, 4)
    np.testing.assert_allclose(cells, expected, rtol=0.05)


def _nestsIn(grouping: Grouping) -> list[str | None]:
    return [column.nests_in for column in grouping.metadata.values()]


def test_nested_share_1_gives_a_chain_and_0_crossed_factors_only() -> None:
    """nested_share is the probability that a factor nests in the previous one,
    so its ends give one chain and no nesting at all.
    """
    assert _nestsIn(_assign(0, factors=3, nested_share=1)) == [None, 'g0', 'g1']
    assert _nestsIn(_assign(0, factors=3, nested_share=0)) == [None, None, None]


def test_each_group_of_a_nested_factor_lies_in_one_superior_group() -> None:
    """Nesting holds in the codes, not only in the column's claim:
    a subordinate group meets exactly one superior group,
    so the pairs of codes are as many as its groups.
    """
    for seed in range(20):
        grouping = _assign(seed, rows=500, factors=3, nested_share=1)
        for subordinate, superior in [('g1', 'g0'), ('g2', 'g1')]:
            pairs = set(
                zip(grouping.codes[subordinate].tolist(), grouping.codes[superior].tolist())
            )
            assert len(pairs) == grouping.metadata[subordinate].levels


@pytest.mark.parametrize('rows', [4, 6, 100])
def test_a_superior_factor_has_fewer_groups_than_its_subordinate_one(rows: int) -> None:
    """Equal counts would put one subordinate group in each superior one:
    one partition relabeled, two random effects no model tells apart (ADR 0004, amended).
    The top factor keeps 2 groups and each subordinate one more,
    so 4 rows are the fewest a chain of three fits in.
    """
    for seed in range(50):
        grouping = _assign(seed, rows=rows, factors=3, nested_share=1)
        levels = [column.levels for column in grouping.metadata.values()]
        assert levels[0] < levels[1] < levels[2]


def test_a_chain_fits_under_a_groups_range_that_starts_above_2() -> None:
    """Every superior factor needs the range's lowest count, not 2,
    so each subordinate one keeps room for the ones above it:
    a floor of 2 plus the depth left the top factor no count on some keys.
    """
    for seed in range(200):
        grouping = _assign(seed, rows=2000, factors=3, nested_share=1, groups=UniformInt(10, 400))
        levels = [column.levels for column in grouping.metadata.values()]
        assert 10 <= levels[0] < levels[1] < levels[2]


OWN_POINTS = [
    pytest.param((5, 30), 1, [None, 'g0'], id='a chain, the superior factor first'),
    pytest.param((30, 5), 0, [None, None], id='crossed factors in any order'),
]


@pytest.mark.parametrize(('groups', 'nested_share', 'nests_in'), OWN_POINTS)
def test_a_tuple_of_groups_gives_each_factor_its_own_point(
    groups: tuple[int, ...], nested_share: float, nests_in: list[str | None]
) -> None:
    """A registry fit with two factors has two group counts,
    and a dataset paired to it draws each factor at its own.
    """
    grouping = _assign(0, rows=500, factors=2, groups=groups, nested_share=nested_share)
    levels = [column.levels for column in grouping.metadata.values()]
    assert levels == list(groups)
    assert _nestsIn(grouping) == nests_in


def test_a_tuples_point_above_the_rows_fails_naming_groups_and_the_bound() -> None:
    """A tuple's point is honored exactly or not at all, like a point every factor shares:
    30 balanced groups over 20 rows would leave ten of them empty.
    """
    with pytest.raises(ValueError, match=r'groups.*\b20\b'):
        _assign(0, rows=20, factors=2, groups=(5, 30), nested_share=1, balanced_share=1)


def test_assign_is_deterministic_under_a_fixed_generator() -> None:
    """A key reproduces its dataset bitwise only if the grouping draws from the generator alone."""
    first = _assign(7, factors=3)
    second = _assign(7, factors=3)
    assert first.metadata == second.metadata
    for name, codes in first.codes.items():
        np.testing.assert_array_equal(codes, second.codes[name])


def _plan(seed: int, **overrides: object) -> dict[str, Column]:
    rng = np.random.default_rng(seed)
    resolved = Scm(**overrides).resolveDataset(rng)
    return plan(resolved, rng)


@pytest.mark.parametrize('type', Scm.categories('type_shares'))
def test_the_budget_is_met_exactly_for_budgets_0_to_19_under_every_type_share_corner(
    type: str,
) -> None:
    """metabeta's design matrix has the width it drew.

    A corner such as all-categorical must still fill a remainder of 1.
    100 rows admit every budget up to 19 (n >= 2d).
    """
    for budget in range(20):
        for seed in range(10):
            columns = _plan(
                seed, rows=100, design_columns=budget, type_shares=point('type_shares', type)
            )
            assert sum(column.design_columns for column in columns.values()) == budget


def _shapes(columns: dict[str, Column]) -> list[tuple[str, int | None]]:
    """Each column's type and k, without its drawn parameters."""
    return [(column.type, column.levels) for column in columns.values()]


def test_k_is_clipped_when_it_would_overflow_the_budget() -> None:
    """A tiny levels_p draws k in the millions; with a budget of 3, one column of k = 4 fills it."""
    columns = _plan(
        0,
        design_columns=3,
        type_shares=point('type_shares', 'categorical'),
        levels_p=1e-6,
        outcome_form='none',
    )
    assert _shapes(columns) == [('categorical', 4)]


def test_max_levels_caps_k() -> None:
    """metabeta passes 10,
    so no categorical or ordered column may exceed it however large k is drawn.
    """
    columns = _plan(
        0,
        design_columns=19,
        type_shares=point('type_shares', 'ordered'),
        levels_p=1e-6,
        max_levels=10,
        outcome_form='none',
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
        outcome_form='none',
    )
    sigmas = {column.parameters['lognormal_sigma'] for column in lognormal.values()}
    assert len(sigmas) == 5
    # levels_p = 1 draws k = 3 every time, so 8 design columns are 4 ordered columns
    ordered = _plan(
        0,
        design_columns=8,
        type_shares=point('type_shares', 'ordered'),
        levels_p=1.0,
        outcome_form='none',
    )
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
        outcome_form='none',
    )
    for column in columns.values():
        assert column.parameters == {'lognormal_sigma': 0.7}


def test_plan_gives_every_column_its_marginal_or_level_alpha() -> None:
    """rankMatch reads them, and Column admits a column without either,
    so the guarantee is the plan's: every column it draws, y included, carries its draws.
    """
    for seed in range(20):
        for column in _plan(seed).values():
            drawn = column.marginal is not None or 'level_alpha' in column.parameters
            assert drawn


def test_plan_names_its_columns_x0_to_x_d_minus_1() -> None:
    """The one home of the rule that plan column i is node x{i};
    every later step reads the names.
    """
    columns = _plan(
        0, design_columns=4, type_shares=point('type_shares', 'continuous'), outcome_form='none'
    )
    assert list(columns) == ['x0', 'x1', 'x2', 'x3']


def test_plan_ends_with_y_under_scm_and_has_no_y_under_none() -> None:
    """plan owns the names, so y is minted once there, after the covariates,
    and a covariate-only study draws none.
    y is drawn after the budget loop, so the covariates are the same either way.
    """
    with_y = _plan(0, design_columns=5)
    without_y = _plan(0, design_columns=5, outcome_form='none')
    assert list(with_y)[-1] == 'y'
    assert with_y['y'].role == 'outcome'
    assert 'y' not in without_y
    assert {name: with_y[name] for name in without_y} == without_y


@pytest.mark.parametrize('type', Scm.categories('outcome_type_shares'))
def test_ys_type_follows_outcome_type_shares_under_every_point_mass(type: str) -> None:
    """A corpus of one outcome type is one override,
    count included, which the covariates' type shares never draw.
    """
    for seed in range(10):
        columns = _plan(seed, outcome_type_shares=point('outcome_type_shares', type))
        assert columns['y'].type == type


def test_ys_k_ignores_the_budget() -> None:
    """y takes no design column, so under a budget of 1 a categorical y keeps its k,
    where a categorical covariate is clipped to binary.
    levels_p = 1 draws k = 3 every time.
    """
    columns = _plan(
        0,
        design_columns=1,
        type_shares=point('type_shares', 'categorical'),
        outcome_type_shares=point('outcome_type_shares', 'categorical'),
        levels_p=1.0,
    )
    assert _shapes(columns) == [('binary', 2), ('categorical', 3)]


def test_ys_k_stops_at_outcome_max_levels_and_at_max_levels() -> None:
    """A tiny levels_p draws k in the millions, and no budget clips y's.
    outcome_max_levels caps y alone; max_levels caps every column, y included.
    """
    ordered = point('outcome_type_shares', 'ordered')
    capped = _plan(0, outcome_type_shares=ordered, levels_p=1e-6, outcome_max_levels=4)
    assert capped['y'].levels == 4
    lower = _plan(0, outcome_type_shares=ordered, levels_p=1e-6, outcome_max_levels=8, max_levels=5)
    assert lower['y'].levels == 5


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
    return rankMatch(values, column, rng)


@pytest.mark.parametrize('column', DISCRETE, ids=str)
def test_rank_match_gives_codes_0_to_k_minus_1_with_every_level_present_when_n_allows(
    column: Column,
) -> None:
    """A level no row has would not exist in a real table;
    n = k is the tightest case that allows all.
    """
    for n in [column.levels, 100]:
        for seed in range(10):
            codes = _apply(n, column, seed)
            assert np.array_equal(np.unique(codes), np.arange(column.levels))


def test_rank_match_gives_valid_codes_when_n_is_below_k() -> None:
    """A small rows override can leave fewer rows than levels;
    each row still gets a code below k.
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
    codes = rankMatch(values, column, rng)  # (n,)
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


def _matched(family: str, parameters: dict[str, float]) -> Column:
    """A column rank matched to the family: a count if poisson, else continuous."""
    type = 'count' if family == 'poisson' else 'continuous'
    return Column(type=type, marginal=family, parameters=parameters)


def _example(family: str) -> Column:
    """A column of the family with its example parameters."""
    parameters = {field: EXAMPLE[field] for field in MARGINAL_FIELDS[family]}
    return _matched(family, parameters)


CONTINUOUS = [_example(family) for family in Scm.categories('marginal_shares')]
COUNT = _example('poisson')


def _node(seed: int) -> np.ndarray:
    """500 z-scored node values, as propagate leaves them."""
    values = np.random.default_rng(seed).standard_normal(500, dtype=np.float32)  # (n,)
    return (values - values.mean()) / values.std()


def _applyTo(values: np.ndarray, column: Column, seed: int) -> np.ndarray:
    """The column cut or matched from given node values."""
    return rankMatch(values, column, np.random.default_rng(seed))


@pytest.mark.parametrize('column', [*CONTINUOUS, COUNT], ids=str)
def test_a_rank_matched_column_has_exactly_its_marginal_samples_values(column: Column) -> None:
    """Two different nodes under one generator get the same values, only in another order.

    So the values come from the sample alone and the node supplies only the order: a quantile
    lookup that resamples (scamd's copula, 37% ties) would fail this.
    """
    values = _applyTo(_node(1), column, seed=0)
    other = _applyTo(_node(2), column, seed=0)
    np.testing.assert_array_equal(np.sort(values), np.sort(other))
    assert not np.array_equal(values, other)


@pytest.mark.parametrize('column', [*CONTINUOUS, COUNT], ids=str)
def test_a_rank_matched_column_has_the_nodes_ranks(column: Column) -> None:
    """Rank matching keeps the dependence the SCM built: read in the node's order, the column rises.

    Non-decreasing rather than increasing, since the zero-inflated sample has ties at 0.
    """
    node = _node(1)
    values = _applyTo(node, column, seed=0)
    in_node_order = values[np.argsort(node)]  # (n,)
    assert (np.diff(in_node_order) >= 0).all()
    assert np.ptp(values) > 0


@pytest.mark.parametrize('column', [*CONTINUOUS, COUNT, *DISCRETE], ids=str)
def test_tied_node_values_share_one_column_value(column: Column) -> None:
    """A row-unit node at group share 1 is constant within its group factor's groups,
    so it ties over each group's rows.
    The column keeps the ties, so it is constant within the groups too.
    """
    node = np.repeat(_node(1)[:50], 10)  # (500,) 50 distinct values, each on 10 rows
    values = _applyTo(node, column, seed=0)
    assert len(np.unique(values)) <= 50
    for value in np.unique(node):
        assert len(np.unique(values[node == value])) == 1


@pytest.mark.parametrize('column', [*CONTINUOUS, COUNT, *DISCRETE], ids=str)
def test_rank_match_gives_each_column_its_dtype(column: Column) -> None:
    """An outcome checks its arrays against Column.dtype, so the plan's columns must meet it."""
    assert _apply(50, column, 0).dtype == column.dtype


FLOORED = [
    # at zero_share = 0.99, two values are both zero in 98 % of seeds without the floor
    pytest.param(
        Column(
            type='continuous',
            marginal='zero_inflated_lognormal',
            parameters={'lognormal_sigma': 0.5, 'zero_share': 0.99},
        ),
        id='a zero-inflated column, whose zeros could fill the sample',
    ),
    # at poisson_log_rate = -2, two values tie in about 78 % of seeds without the floor
    pytest.param(
        Column(type='count', marginal='poisson', parameters={'poisson_log_rate': -2.0}),
        id='a count, whose low rate could tie the sample',
    ),
]


@pytest.mark.parametrize('column', FLOORED)
def test_a_marginal_sample_at_two_groups_is_not_constant(column: Column) -> None:
    """A group-level column, or a row-unit one at group share 1, has one value per group,
    so at two groups its sample could be constant, a column a consumer cannot z-score.
    """
    node = np.array([-1.0, 1.0], dtype=np.float32)  # (u,) two distinct values, as at two groups
    for seed in range(50):
        values = _applyTo(node, column, seed)
        assert np.ptp(values) > 0


def test_plan_is_deterministic_under_a_fixed_generator() -> None:
    """A key reproduces its dataset bitwise only if every step draws from the generator alone."""
    assert _plan(7) == _plan(7)


@pytest.mark.parametrize(
    'column', [*CONTINUOUS, COUNT, Column(type='categorical', levels=6, parameters=ALPHA)], ids=str
)
def test_rank_match_is_deterministic_under_a_fixed_generator(column: Column) -> None:
    """Every marginal family and the level codes draw from the generator alone."""
    node = _node(1)
    assert np.array_equal(_applyTo(node, column, 7), _applyTo(node, column, 7))


def test_a_large_level_alpha_gives_even_level_counts() -> None:
    """level_alpha is the Dirichlet concentration of the level shares,
    so a large one leaves each share near 1/k.
    """
    column = Column(type='ordered', levels=5, parameters={'level_alpha': 1e4})
    codes = _apply(1000, column, seed=0)
    np.testing.assert_allclose(np.bincount(codes), 200, atol=10)


def _fromDefaults(family: str, n: int, seed: int) -> np.ndarray:
    """A column of the family under parameters drawn from the defaults, as plan draws them."""
    rng = np.random.default_rng(seed)
    scm = Scm()
    parameters = {field: scm.resolveField(field, rng) for field in MARGINAL_FIELDS[family]}
    column = _matched(family, parameters)
    return rankMatch(rng.standard_normal(n), column, rng)


SUPPORTS = {
    'normal': (-np.inf, np.inf),
    'lognormal': (0, np.inf),
    'student_t': (-np.inf, np.inf),
    'beta': (0, 1),
    'uniform': (0, 1),
    'zero_inflated_lognormal': (0, np.inf),
    'normal_mixture': (-np.inf, np.inf),
    'poisson': (0, np.inf),
}


@pytest.mark.parametrize('family', get_args(Marginal))
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
    column = _matched(family, parameters)
    return _apply(100_000, column, seed=0)


def test_each_familys_parameters_are_honored() -> None:
    """Each family reads its own parameters; expected values are closed forms, not the sampler's.

    The mixture's gap is set wide, so the share above half of it is the minor weight.
    The t's two-tailed P(|T| > 3) at df = 3 is 0.0577, from its closed-form CDF.
    A Poisson's mean is the exponential of its log rate.
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
    poisson = _large('poisson', poisson_log_rate=5.0)
    assert abs(poisson.mean() / math.exp(5) - 1) < 0.01
