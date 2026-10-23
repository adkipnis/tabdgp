"""The hyperprior's seams: construction with overrides, and their resolution to points."""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import ValidationError

from tabdgp.column import ColumnType, assign
from tabdgp.graph import Activation, Graph, Mechanism
from tabdgp.hyperprior import (
    MARGINAL_FIELDS,
    Beta,
    Dirichlet,
    Family,
    Hyperprior,
    LogUniform,
    LogUniformInt,
    Uniform,
    UniformInt,
)
from tabdgp.mechanism import ACTIVATIONS
from tests.fixtures.build import point


def test_the_defaults_validate() -> None:
    """A standalone user overrides nothing, so the defaults alone must pass every check."""
    Hyperprior()


GROUPING_DEFAULTS = {
    'factors': UniformInt(1, 3),
    'groups': LogUniformInt(2, 400),
    'balanced_share': 1 / 3,
    'size_alpha': LogUniform(0.5, 500),
    'nested_share': Uniform(0.2, 0.6),
    'group_level_share': Beta(2, 4),
    'group_share': Beta(0.5, 2),
}


def test_the_grouping_fields_take_the_grouped_presets_defaults() -> None:
    """ADR 0004's table: what the proofs certify and metabeta trains on."""
    hyperprior = Hyperprior()
    defaults = {field: getattr(hyperprior, field) for field in GROUPING_DEFAULTS}
    assert defaults == GROUPING_DEFAULTS


OUTCOME_DEFAULTS = {
    'outcome': 'scm',
    'outcome_type_shares': {
        'continuous': 0.6,
        'binary': 0.2,
        'categorical': 0.05,
        'ordered': 0.05,
        'count': 0.1,
    },
    'outcome_max_levels': 10,
    'outcome_noise_scale': LogUniform(0.1, 3),
    'poisson_log_rate': Uniform(-1, 6),
}


@pytest.mark.parametrize('preset', [Hyperprior.grouped, Hyperprior.flat], ids=['grouped', 'flat'])
def test_the_outcome_fields_take_the_presets_defaults(preset: Callable[[], Hyperprior]) -> None:
    """ADR 0005's table: both presets draw the outcome node in its scm form."""
    hyperprior = preset()
    defaults = {field: getattr(hyperprior, field) for field in OUTCOME_DEFAULTS}
    assert defaults == OUTCOME_DEFAULTS


NOT_A_FORM = [
    pytest.param('glmm', id='glmm'),
    pytest.param('SCM', id='a misspelled form'),
    pytest.param({'none': 0.3, 'scm': 0.7}, id='shares'),
    pytest.param(Dirichlet({'none': 1, 'scm': 1}), id='a dirichlet'),
]


@pytest.mark.parametrize('outcome', NOT_A_FORM)
def test_an_outcome_other_than_none_or_scm_fails_at_construction(
    outcome: str | dict | Dirichlet,
) -> None:
    """glmm is no form until its preset exists, and a dataset of each form returns different things,
    so no corpus mixes them: each fails where it is written, like a typo, not at a draw.
    """
    with pytest.raises(ValidationError):
        Hyperprior(outcome=outcome)


def test_grouped_equals_the_defaults_and_flat_sets_factors_to_0() -> None:
    """Every draw starts from a preset; Hyperprior() stays valid as the grouped one,
    and the flat preset differs from it in factors alone.
    """
    assert Hyperprior.grouped() == Hyperprior()
    assert Hyperprior.flat() == Hyperprior(factors=0)


def test_an_override_on_a_preset_replaces_its_default() -> None:
    """A study narrows one field of a preset and keeps the rest."""
    assert Hyperprior.grouped(rows=500) == Hyperprior(rows=500)
    assert Hyperprior.flat(rows=500) == Hyperprior(rows=500, factors=0)


def test_flat_rejects_an_override_of_factors() -> None:
    """The flat preset is defined by its factors;
    flat(factors=2) would be a grouped table under the wrong name.
    """
    with pytest.raises(TypeError, match='factors'):
        Hyperprior.flat(factors=2)


def test_a_point_override_resolves_to_itself() -> None:
    """metabeta passes its sizes as points; resolveDataset must hand them back untouched."""
    shares = {'continuous': 1.0, 'binary': 0.0, 'categorical': 0.0, 'ordered': 0.0}
    hyperprior = Hyperprior(rows=500, type_shares=shares, edge_offset=-1.5)
    resolved = hyperprior.resolveDataset(np.random.default_rng(0))
    assert (resolved.rows, resolved.type_shares, resolved.edge_offset) == (500, shares, -1.5)


def test_resolving_twice_changes_nothing() -> None:
    """Dataset.hyperparameters holds the resolved hyperprior, which a caller reads as final.
    A per-dataset distribution left over would be drawn again, so resolving twice would change it.
    """
    resolved = Hyperprior().resolveDataset(np.random.default_rng(0))
    assert resolved.resolveDataset(np.random.default_rng(1)) == resolved


def test_resolve_leaves_the_fields_drawn_per_column_node_or_factor_as_given() -> None:
    """Each column, node or grouping factor draws its own value of these fields,
    so no dataset-wide value is drawn.
    """
    per_node = {'noise_scale', 'init_scale', 'group_share', 'outcome_noise_scale'}
    assert set(Hyperprior.PER_NODE) == per_node
    assert set(Hyperprior.PER_FACTOR) == {'groups', 'size_alpha'}
    hyperprior = Hyperprior(noise_scale=0.5, lognormal_sigma=LogUniform(2, 3))
    resolved = hyperprior.resolveDataset(np.random.default_rng(0))
    assert (resolved.noise_scale, resolved.lognormal_sigma) == (0.5, LogUniform(2, 3))
    for field in Hyperprior.PER_COLUMN + Hyperprior.PER_NODE + Hyperprior.PER_FACTOR:
        assert getattr(resolved, field) == getattr(hyperprior, field)


@given(seed=st.integers(0, 2**32 - 1))
def test_resolving_one_field_gives_its_point_or_one_draw_inside_its_support(seed: int) -> None:
    """A column or node takes its own value of a field through resolveField, whatever it holds."""
    hyperprior = Hyperprior(noise_scale=0.5, lognormal_sigma=LogUniform(2, 3))
    rng = np.random.default_rng(seed)
    assert hyperprior.resolveField('noise_scale', rng) == 0.5
    assert 2 <= hyperprior.resolveField('lognormal_sigma', rng) <= 3


@pytest.mark.parametrize('distribution', [UniformInt, LogUniformInt])
@given(
    low=st.integers(2, 100),
    width=st.integers(1, 1000),
    margin=st.integers(0, 2000),
    seed=st.integers(0, 2**32 - 1),
)
def test_a_bounded_distribution_resolves_at_or_below_its_bound(
    distribution: type, low: int, width: int, margin: int, seed: int
) -> None:
    """A factor has at most as many groups as the table has rows.
    A bound above the distribution's high leaves the high in force.
    """
    hyperprior = Hyperprior(groups=distribution(low, low + width), nested_share=0)
    bound = low + margin
    groups = hyperprior.resolveField('groups', np.random.default_rng(seed), high=bound)
    assert low <= groups <= min(bound, low + width)


@pytest.mark.parametrize('distribution', [UniformInt, LogUniformInt])
@given(
    low=st.integers(2, 100),
    width=st.integers(1, 1000),
    floor=st.integers(0, 1100),
    seed=st.integers(0, 2**32 - 1),
)
def test_a_distribution_resolves_at_or_above_its_floor(
    distribution: type, low: int, width: int, floor: int, seed: int
) -> None:
    """A factor keeps one group more for each factor superior to it.
    A floor below the distribution's low leaves the low in force.
    """
    assume(floor <= low + width)
    hyperprior = Hyperprior(groups=distribution(low, low + width), nested_share=0)
    groups = hyperprior.resolveField('groups', np.random.default_rng(seed), low=floor)
    assert max(low, floor) <= groups <= low + width


TRUNCATED = [
    # the ten values 2 to 11 keep a tenth each; a clip would put 91 % on 11
    pytest.param(UniformInt(2, 101), 11, 0.1, id='uniform'),
    # 20 keeps log(21/20) / log(21/2) of the mass; a clip would put 57 % on it
    pytest.param(LogUniformInt(2, 400), 20, 0.021, id='log-uniform'),
]


@pytest.mark.parametrize(('distribution', 'bound', 'share'), TRUNCATED)
def test_a_truncated_draw_piles_no_mass_on_its_bound(
    distribution: UniformInt | LogUniformInt, bound: int, share: float
) -> None:
    """A draw clipped at rows would give most small tables one row per group (ADR 0004);
    truncation keeps the distribution's shape below its bound.
    """
    hyperprior = Hyperprior(groups=distribution)
    rng = np.random.default_rng(0)
    draws = [hyperprior.resolveField('groups', rng, high=bound) for _ in range(10_000)]
    assert draws.count(bound) / len(draws) == pytest.approx(share, abs=0.01)


@pytest.mark.parametrize(
    'distribution', [UniformInt(2, 400), LogUniformInt(2, 400)], ids=['uniform', 'log-uniform']
)
def test_a_truncated_support_of_one_value_draws_it(
    distribution: UniformInt | LogUniformInt,
) -> None:
    """A table of two rows admits factors of two groups only.
    Construction rejects a support of one value; a bound may narrow a support to it.
    """
    hyperprior = Hyperprior(groups=distribution)
    rng = np.random.default_rng(0)
    draws = {hyperprior.resolveField('groups', rng, high=2) for _ in range(100)}
    assert draws == {2}


@pytest.mark.parametrize(
    'distribution', [UniformInt(10, 400), LogUniformInt(10, 400)], ids=['uniform', 'log-uniform']
)
def test_a_bound_below_the_distributions_low_fails_naming_the_field_and_its_bound(
    distribution: UniformInt | LogUniformInt,
) -> None:
    """No value of the distribution lies at or below the bound,
    and a draw above it would break m ≤ n without a word.
    """
    hyperprior = Hyperprior(groups=distribution)
    with pytest.raises(ValueError, match=r'groups.*\b7\b'):
        hyperprior.resolveField('groups', np.random.default_rng(0), high=7)


OUT_OF_REACH = [
    pytest.param(LogUniformInt(2, 400), 3, id='a floor above the bound'),
    pytest.param(UniformInt(2, 3), 10, id='a floor above the distribution'),
]


@pytest.mark.parametrize(('distribution', 'bound'), OUT_OF_REACH)
def test_a_floor_out_of_reach_fails_naming_the_field_and_its_floor(
    distribution: UniformInt | LogUniformInt, bound: int
) -> None:
    """The bottom factor of a chain of three needs the range's lowest count plus 2, here 4.
    A bound of 3 or a range that stops at 3 cannot reach it,
    and a draw below it would leave no count for a superior factor.
    """
    hyperprior = Hyperprior(groups=distribution, nested_share=0)
    with pytest.raises(ValueError, match=r'groups.*\b4\b'):
        hyperprior.resolveField('groups', np.random.default_rng(0), low=4, high=bound)


def test_a_point_above_its_bound_fails_naming_the_field_and_its_bound() -> None:
    """A point override is honored exactly or not at all:
    50 groups cannot be assigned to 30 rows, and 19 design columns under 10 rows break n ≥ 2d.
    """
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=r'groups.*\b30\b'):
        Hyperprior(groups=50, nested_share=0).resolveField('groups', rng, high=30)
    with pytest.raises(ValueError, match=r'design_columns.*\b5\b'):
        Hyperprior(rows=10, design_columns=19).resolveDataset(rng)


def test_a_point_groups_fails_at_construction_when_factors_can_nest() -> None:
    """A point gives every factor one count,
    and a superior factor needs fewer than the subordinate one;
    failing at construction beats failing in the datasets that happen to draw a nesting.
    """
    with pytest.raises(ValidationError, match='nested_share'):
        Hyperprior(groups=5)
    Hyperprior(groups=5, nested_share=0)
    Hyperprior(groups=5, factors=1)
    Hyperprior(groups=5, factors=UniformInt(0, 1))
    Hyperprior.flat(groups=5)


def test_a_groups_range_too_narrow_for_the_longest_chain_fails_at_construction() -> None:
    """A chain of f factors needs f counts, each superior one below the subordinate one,
    so a range of fewer values fails as a point does, not in the datasets that draw a chain.
    """
    with pytest.raises(ValidationError, match='nested_share'):
        Hyperprior(groups=UniformInt(2, 3))
    with pytest.raises(ValidationError, match='nested_share'):
        Hyperprior(groups=LogUniformInt(10, 11))
    Hyperprior(groups=UniformInt(2, 4))
    Hyperprior(groups=UniformInt(2, 3), factors=UniformInt(1, 2))
    Hyperprior(groups=UniformInt(2, 3), nested_share=0)


@given(rows=st.integers(2, 60), seed=st.integers(0, 2**32 - 1))
def test_design_columns_resolves_at_or_below_half_the_rows(rows: int, seed: int) -> None:
    """A table has at least two rows per design column (metabeta's rule 8, n ≥ 2d)."""
    resolved = Hyperprior(rows=rows).resolveDataset(np.random.default_rng(seed))
    assert resolved.design_columns <= rows // 2


def test_rows_precedes_design_columns_in_field_order() -> None:
    """resolveDataset resolves in field order and bounds design_columns by the resolved rows."""
    fields = list(Hyperprior.model_fields)
    assert fields.index('rows') < fields.index('design_columns')


def test_resolve_is_deterministic_under_a_fixed_generator() -> None:
    """A dataset is a pure function of its key, so the same generator gives the same points."""
    hyperprior = Hyperprior()
    first = hyperprior.resolveDataset(np.random.default_rng(7))
    second = hyperprior.resolveDataset(np.random.default_rng(7))
    assert first == second


ALL_CONTINUOUS = {'continuous': 1.0, 'binary': 0.0, 'categorical': 0.0, 'ordered': 0.0}
OUT_OF_RANGE = [
    pytest.param({'rows': 0}, id='a point below the bound'),
    pytest.param({'rows': UniformInt(0, 100)}, id='a support below the bound'),
    pytest.param({'rows': LogUniform(20, 5000)}, id='a continuous distribution for a count'),
    pytest.param({'levels_p': 1.5}, id='a probability above 1'),
    pytest.param({'zero_share': 1.0}, id='a point on an open bound'),
    pytest.param({'student_df': Uniform(-1, 5)}, id='a support across zero for a positive field'),
    pytest.param({'max_levels': 2}, id='a cap that leaves no categorical'),
    pytest.param({'outcome_max_levels': 2}, id="a cap on y's k that leaves no categorical"),
    pytest.param({'factors': -1}, id='a negative number of factors'),
    pytest.param({'groups': 1}, id='a factor of one group'),
    pytest.param({'size_alpha': 0.0}, id='a zero concentration of the group sizes'),
    pytest.param({'balanced_share': 1.5}, id='a share above 1'),
    pytest.param({'group_share': Uniform(0, 2)}, id='a support above 1 for a share'),
    pytest.param({'outcome_noise_scale': -1.0}, id='a negative outcome noise scale'),
    pytest.param({'poisson_log_rate': Uniform(0, 45)}, id="a log rate past numpy's poisson limit"),
    pytest.param({'type_shares': ALL_CONTINUOUS | {'binary': 0.5}}, id='shares above 1'),
    pytest.param(
        {'type_shares': ALL_CONTINUOUS | {'continuous': 1.5, 'binary': -0.5}}, id='a negative share'
    ),
    pytest.param({'type_shares': {'continuous': 1.0}}, id='shares missing a category'),
    pytest.param(
        {'outcome_type_shares': OUTCOME_DEFAULTS['outcome_type_shares'] | {'continuous': 1.6}},
        id='outcome type shares summing to 2',
    ),
    pytest.param(
        {'mechanism_shares': Dirichlet({'linear': 1, 'tree': 1})}, id='an unknown category'
    ),
]


@pytest.mark.parametrize('overrides', OUT_OF_RANGE)
def test_an_out_of_range_value_fails_at_construction(overrides: dict) -> None:
    """An invalid override fails where it is written, not deep inside a later draw."""
    with pytest.raises(ValidationError):
        Hyperprior(**overrides)


def test_a_misspelled_override_fails_at_construction() -> None:
    """A typo in a field name would otherwise leave the default in place without a word."""
    with pytest.raises(ValidationError):
        Hyperprior(n_ob=500)


SUPPORTED = [
    ('rows', UniformInt),
    ('rows', LogUniformInt),
    ('hub_scale', Uniform),
    ('hub_scale', LogUniform),
]


@pytest.mark.parametrize(('field', 'distribution'), SUPPORTED)
@given(low=st.integers(1, 1000), width=st.integers(1, 1000), seed=st.integers(0, 2**32 - 1))
def test_a_distribution_override_resolves_inside_its_support(
    field: str, distribution: type, low: int, width: int, seed: int
) -> None:
    """Narrow integer ranges probe the log-uniform's floor at its upper end.
    A budget of 0 admits every row count, a single row included.
    """
    hyperprior = Hyperprior(**{field: distribution(low, low + width)}, design_columns=0)
    resolved = hyperprior.resolveDataset(np.random.default_rng(seed))
    assert low <= getattr(resolved, field) <= low + width


def test_resolved_shares_choose_a_category_by_its_share() -> None:
    """column and graph choose a type, family or activation from resolved shares."""
    rng = np.random.default_rng(0)
    types = {'continuous': 0.0, 'binary': 1.0, 'categorical': 0.0, 'ordered': 0.0}
    resolved = Hyperprior(type_shares=types).resolveDataset(rng)
    assert resolved.type_shares.choose(rng) == 'binary'


def test_resolving_one_shares_field_gives_shares_that_choose() -> None:
    """resolveField hands back shares that choose,
    whether the field holds a Dirichlet or a point.
    """
    rng = np.random.default_rng(0)
    assert Hyperprior().resolveField('mechanism_shares', rng).choose(rng) in {'linear', 'mlp'}


def test_a_shares_field_names_its_default_categories() -> None:
    """The catalogs and the point masses read a field's categories from its default."""
    assert Hyperprior.categories('mechanism_shares') == ['linear', 'mlp']


def test_categories_reads_a_point_default() -> None:
    """A point default has no Dirichlet to read,
    yet the shares check and the point masses need its categories.
    """
    shares = point('outcome_type_shares', 'count')
    assert Hyperprior(outcome_type_shares=shares).outcome_type_shares == shares


def test_a_point_mass_puts_every_share_on_its_category() -> None:
    """Tests force one type, family or activation
    through a point mass over the field's categories.
    """
    shares = point('type_shares', 'ordered')
    assert shares == {'continuous': 0.0, 'binary': 0.0, 'categorical': 0.0, 'ordered': 1.0}
    assert Hyperprior(type_shares=shares).type_shares == shares


@pytest.mark.parametrize(
    'shares',
    [
        pytest.param(
            {'continuous': 0.4, 'binary': 0.3, 'categorical': 0.2, 'ordered': 0.1}, id='point'
        ),
        pytest.param(
            Dirichlet({'continuous': 1, 'binary': 2, 'categorical': 3, 'ordered': 4}),
            id='dirichlet',
        ),
    ],
)
def test_the_order_an_override_lists_its_categories_in_changes_no_draw(
    shares: dict[str, float] | Dirichlet,
) -> None:
    """A category is chosen by its place in the shares,
    so an override listed in another order must not reorder them.
    """
    alpha = shares.alpha if isinstance(shares, Dirichlet) else shares
    reversed_alpha = dict(reversed(alpha.items()))
    reversed_shares = Dirichlet(reversed_alpha) if isinstance(shares, Dirichlet) else reversed_alpha
    listed = Hyperprior(type_shares=shares).resolveDataset(np.random.default_rng(0))
    reversed_listed = Hyperprior(type_shares=reversed_shares).resolveDataset(
        np.random.default_rng(0)
    )
    assert list(reversed_listed.type_shares.items()) == list(listed.type_shares.items())


def test_the_fields_drawn_per_column_are_the_families_and_the_level_concentration() -> None:
    """A field drawn per column that no column reads would silently stay a distribution."""
    read = {'level_alpha'}
    for fields in MARGINAL_FIELDS.values():
        read.update(fields)
    assert read == set(Hyperprior.PER_COLUMN)


def test_the_fields_drawn_per_factor_are_exactly_those_a_factor_column_records() -> None:
    """A field drawn per factor that no factor's column records would silently stay a distribution.
    groups is recorded as the column's levels, size_alpha among the parameters of an unbalanced one.
    """
    rng = np.random.default_rng(0)
    resolved = Hyperprior.grouped(factors=2, balanced_share=0).resolveDataset(rng)
    recorded = {'groups'}
    for column in assign(resolved, rng).metadata.values():
        recorded.update(column.parameters)
    assert recorded == set(Hyperprior.PER_FACTOR)


def test_the_fields_drawn_per_node_are_exactly_those_the_graph_records() -> None:
    """A field drawn per node that the graph does not record would silently stay a distribution."""
    recorded = set(Graph.__dataclass_fields__) & set(Hyperprior.model_fields)
    recorded.add('outcome_noise_scale')  # recorded as y's noise_scale
    assert recorded == set(Hyperprior.PER_NODE)


CATALOGS = [
    # a count's poisson marginal is its type's by construction, so it takes no share
    pytest.param(set(get_args(Family)) - {'poisson'}, 'marginal_shares', id='continuous families'),
    pytest.param(set(ACTIVATIONS), 'activation_shares', id='activations'),
    pytest.param(set(get_args(Activation)), 'activation_shares', id='activation names'),
    pytest.param(set(get_args(Mechanism)), 'mechanism_shares', id='mechanism families'),
    pytest.param(set(get_args(ColumnType)) - {'count'}, 'type_shares', id='covariate types'),
    pytest.param(set(get_args(ColumnType)), 'outcome_type_shares', id='outcome types'),
]


@pytest.mark.parametrize(('catalog', 'field'), CATALOGS)
def test_each_catalog_names_exactly_the_hyperprior_defaults_categories(
    catalog: set[str], field: str
) -> None:
    """A category with a share but no entry in its catalog would fail only when drawn."""
    assert catalog == set(Hyperprior.categories(field))


INVALID = [
    pytest.param(Uniform, (2, 1), id='a uniform with low above high'),
    pytest.param(LogUniform, (0, 5), id='a log-uniform from zero'),
    pytest.param(UniformInt, (3, 3), id='an integer uniform of one value'),
    pytest.param(LogUniformInt, (0, 10), id='an integer log-uniform from zero'),
    pytest.param(Beta, (0, 1), id='a beta with a zero shape'),
    pytest.param(Dirichlet, ({'linear': 0, 'mlp': 1},), id='a dirichlet with a zero concentration'),
]


@pytest.mark.parametrize(('distribution', 'parameters'), INVALID)
def test_an_invalid_distribution_fails_at_construction(
    distribution: type, parameters: tuple
) -> None:
    """A field's bound check reads the support, which is meaningless for invalid parameters."""
    with pytest.raises(ValueError):
        distribution(*parameters)
