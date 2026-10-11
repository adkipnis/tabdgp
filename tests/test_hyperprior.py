"""The hyperprior's seams: construction with overrides, and resolve to points."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tabdgp.dataset import Mechanism
from tabdgp.hyperprior import (
    Beta,
    Dirichlet,
    Distribution,
    Hyperprior,
    LogUniform,
    LogUniformInt,
    Shares,
    Uniform,
    UniformInt,
)
from tabdgp.marginal import FAMILIES
from tabdgp.mechanism import ACTIVATIONS


def test_the_defaults_validate() -> None:
    """A standalone user overrides nothing, so the defaults alone must pass every check."""
    Hyperprior()


def test_a_point_override_resolves_to_itself() -> None:
    """metabeta passes its sizes as points; resolve must hand them back untouched."""
    shares = {'continuous': 1.0, 'binary': 0.0, 'categorical': 0.0, 'ordered': 0.0}
    hyperprior = Hyperprior(n_obs=500, type_shares=shares, edge_offset=-1.5)
    resolved = hyperprior.resolve(np.random.default_rng(0))
    assert (resolved.n_obs, resolved.type_shares, resolved.edge_offset) == (500, shares, -1.5)


def test_the_resolved_hyperprior_has_no_distribution_left() -> None:
    """Dataset.hyperparameters holds the resolved hyperprior, which a caller reads as points."""
    resolved = Hyperprior().resolve(np.random.default_rng(0))
    values = [getattr(resolved, name) for name in Hyperprior.model_fields]
    assert not any(isinstance(value, Distribution) for value in values)


def test_resolve_is_deterministic_under_a_fixed_generator() -> None:
    """A dataset is a pure function of its key, so the same generator gives the same points."""
    hyperprior = Hyperprior()
    first = hyperprior.resolve(np.random.default_rng(7))
    second = hyperprior.resolve(np.random.default_rng(7))
    assert first == second


ALL_CONTINUOUS = {'continuous': 1.0, 'binary': 0.0, 'categorical': 0.0, 'ordered': 0.0}
OUT_OF_RANGE = [
    pytest.param({'n_obs': 0}, id='a point below the bound'),
    pytest.param({'n_obs': UniformInt(0, 100)}, id='a support below the bound'),
    pytest.param({'n_obs': LogUniform(20, 5000)}, id='a continuous distribution for a count'),
    pytest.param({'levels_p': 1.5}, id='a probability above 1'),
    pytest.param({'zero_share': 1.0}, id='a point on an open bound'),
    pytest.param({'student_df': Uniform(-1, 5)}, id='a support across zero for a positive field'),
    pytest.param({'max_levels': 2}, id='a cap that leaves no categorical'),
    pytest.param({'type_shares': ALL_CONTINUOUS | {'binary': 0.5}}, id='shares above 1'),
    pytest.param(
        {'type_shares': ALL_CONTINUOUS | {'continuous': 1.5, 'binary': -0.5}}, id='a negative share'
    ),
    pytest.param({'type_shares': {'continuous': 1.0}}, id='shares missing a category'),
    pytest.param(
        {'mechanism_shares': Dirichlet({'linear': 1, 'tree': 1})}, id='an unknown category'
    ),
]


@pytest.mark.parametrize('overrides', OUT_OF_RANGE)
def test_an_out_of_range_value_fails_at_construction(overrides: dict) -> None:
    """An invalid override fails where it is written, not deep inside a later draw."""
    with pytest.raises(ValidationError):
        Hyperprior(**overrides)


def test_a_misspelt_override_fails_at_construction() -> None:
    """A typo in a field name would otherwise leave the default in place without a word."""
    with pytest.raises(ValidationError):
        Hyperprior(n_ob=500)


SUPPORTED = [
    ('n_obs', UniformInt),
    ('n_obs', LogUniformInt),
    ('student_df', Uniform),
    ('student_df', LogUniform),
]


@pytest.mark.parametrize(('field', 'distribution'), SUPPORTED)
@given(low=st.integers(1, 1000), width=st.integers(1, 1000), seed=st.integers(0, 2**32 - 1))
def test_a_distribution_override_resolves_inside_its_support(
    field: str, distribution: type, low: int, width: int, seed: int
) -> None:
    """Narrow integer ranges probe the log-uniform's floor at its upper end."""
    hyperprior = Hyperprior(**{field: distribution(low, low + width)})
    resolved = hyperprior.resolve(np.random.default_rng(seed))
    assert low <= getattr(resolved, field) <= low + width


def test_resolved_shares_choose_a_category_by_its_share() -> None:
    """column, graph and mechanism draw categories from resolved shares, one or an array at a time."""
    rng = np.random.default_rng(0)
    types = {'continuous': 0.0, 'binary': 1.0, 'categorical': 0.0, 'ordered': 0.0}
    resolved = Hyperprior(type_shares=types, mechanism_shares={'linear': 0.5, 'mlp': 0.5}).resolve(
        rng
    )
    assert resolved.type_shares.choose(rng) == 'binary'
    chosen = resolved.mechanism_shares.choose(rng, size=200)  # (200,)
    assert chosen.shape == (200,)
    assert set(chosen) == {'linear', 'mlp'}


def test_a_shares_field_names_its_default_categories() -> None:
    """The catalogues and the point masses read a field's categories from its default."""
    assert Hyperprior.categories('mechanism_shares') == ['linear', 'mlp']


def test_a_point_mass_puts_every_share_on_its_category() -> None:
    """Tests force one type, family or activation through a point mass over the field's categories."""
    shares = Shares.point(Hyperprior.categories('type_shares'), 'ordered')
    assert shares == {'continuous': 0.0, 'binary': 0.0, 'categorical': 0.0, 'ordered': 1.0}
    assert Hyperprior(type_shares=shares).type_shares == shares


CATALOGUES = [
    pytest.param(set(FAMILIES), 'marginal_shares', id='marginal families'),
    pytest.param(set(ACTIVATIONS), 'activation_shares', id='activations'),
    pytest.param(set(get_args(Mechanism)) - {'noise'}, 'mechanism_shares', id='mechanism families'),
]


@pytest.mark.parametrize(('catalogue', 'field'), CATALOGUES)
def test_each_catalogue_names_exactly_the_hyperprior_defaults_categories(
    catalogue: set[str], field: str
) -> None:
    """A category with a share but no entry in its catalogue would fail only when drawn."""
    assert catalogue == set(Hyperprior.categories(field))


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
