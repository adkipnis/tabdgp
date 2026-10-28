"""The hyperprior's seams: construction with overrides, and their resolution to points."""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tabdgp.column import ColumnType, assign
from tabdgp.field import Beta, Dirichlet, LogUniform, LogUniformInt, Uniform, UniformInt
from tabdgp.graph import Activation, Graph, Mechanism
from tabdgp.hyperprior import MARGINAL_FIELDS, Hyperprior, Marginal
from tabdgp.mechanism import ACTIVATIONS
from tabdgp.prior import Glmm
from tests.fixtures.build import drawNothing


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
    'outcome_form': 'scm',
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
    pytest.param('SCM', id='a misspelled form'),
    pytest.param('glm', id="a model's name, which states no family"),
    pytest.param('glmm', id="a glmm's name"),
    pytest.param(Glmm(family='gaussian'), id='a model, drawn after the dataset'),
    pytest.param(drawNothing, id="a study's model"),
    pytest.param({'none': 0.3, 'scm': 0.7}, id='shares'),
    pytest.param(Dirichlet({'none': 1, 'scm': 1}), id='a dirichlet'),
    pytest.param({'signal_share': 0.5}, id="a dict of a glm's fields"),
    pytest.param({}, id='an empty dict'),
]


@pytest.mark.parametrize('form', NOT_A_FORM)
def test_a_form_other_than_none_or_scm_fails_at_construction(
    form: str | dict | Dirichlet | Glmm | Callable[..., None],
) -> None:
    """A dataset of each form returns different things, so no corpus mixes them:
    each fails where it is written, like a typo, not at a draw.
    A model is no form: modelOutcome draws its outcome after the dataset.
    """
    with pytest.raises(ValidationError):
        Hyperprior(outcome_form=form)


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
    # each of these fields defaults to a distribution, so a draw would show as a point
    hyperprior = Hyperprior()
    resolved = hyperprior.resolveDataset(np.random.default_rng(0))
    for field in Hyperprior.PER_COLUMN + Hyperprior.PER_NODE + Hyperprior.PER_FACTOR:
        assert getattr(resolved, field) == getattr(hyperprior, field)


def test_a_design_columns_point_above_half_the_rows_fails_naming_the_field_and_its_bound() -> None:
    """A point override is honored exactly or not at all:
    19 design columns under 10 rows break n ≥ 2d.
    """
    with pytest.raises(ValueError, match=r'design_columns.*\b5\b'):
        Hyperprior(rows=10, design_columns=19).resolveDataset(np.random.default_rng(0))


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


def test_a_tuple_of_groups_fails_at_construction_unless_it_has_one_point_per_factor() -> None:
    """A tuple pairs each factor with one count of a registry fit,
    so a factor left without a point, or a point without a factor, would pair the wrong sizes.
    A distribution of factors cannot match a tuple's length.
    """
    with pytest.raises(ValidationError, match='factors'):
        Hyperprior(groups=(5, 30), factors=3, nested_share=0)
    with pytest.raises(ValidationError, match='factors'):
        Hyperprior(groups=(5, 30), nested_share=0)
    Hyperprior(groups=(5, 30), factors=2, nested_share=0)
    Hyperprior.flat(groups=())


def test_a_tuple_of_groups_that_cannot_nest_fails_at_construction() -> None:
    """Wherever a factor may nest in the one before it, it needs more groups than that one,
    pair by pair along the tuple; crossed factors take any counts.
    """
    with pytest.raises(ValidationError, match='nested_share'):
        Hyperprior(groups=(30, 5), factors=2)
    with pytest.raises(ValidationError, match='nested_share'):
        Hyperprior(groups=(5, 5), factors=2, nested_share=1)
    with pytest.raises(ValidationError, match='nested_share'):
        Hyperprior(groups=(5, 30, 10), factors=3)
    Hyperprior(groups=(5, 30), factors=2)
    Hyperprior(groups=(30, 5), factors=2, nested_share=0)


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
    pytest.param(
        {'groups': (1, 5), 'factors': 2, 'nested_share': 0}, id='a factor of one group in a tuple'
    ),
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
    pytest.param(
        set(get_args(Marginal)) - {'poisson'}, 'marginal_shares', id='continuous families'
    ),
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
