"""The field seams: a config of point-or-distribution fields built on Fields,
and the distributions its fields take.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from pydantic import ValidationError

from tabdgp.field import (
    Beta,
    Count,
    Dirichlet,
    Fields,
    LogUniform,
    LogUniformInt,
    Positive,
    Simplex,
    Uniform,
    UniformInt,
)


class Toy(Fields):
    """A config of its own, so that these tests read no hyperprior."""

    SHARES: ClassVar[tuple[str, ...]] = ('colors', 'sizes')

    count: Count = UniformInt(0, 10)
    scale: Positive = LogUniform(0.1, 3)
    colors: Simplex = Dirichlet({'red': 1, 'green': 2, 'blue': 3})
    sizes: Simplex = {'small': 0.5, 'large': 0.5}  # noqa: RUF012, a pydantic field


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


def test_a_distribution_may_touch_an_open_bound_and_a_point_may_not() -> None:
    """A draw lands on a support's endpoint with probability 0,
    so Beta(1, 1) is a positive scale while the point 0 is not.
    """
    assert Toy(scale=Beta(1, 1)).scale == Beta(1, 1)
    with pytest.raises(ValidationError, match='gt=0'):
        Toy(scale=0.0)


class OnlyColors(Toy):
    """The toy with sizes left out of its shares fields."""

    SHARES: ClassVar[tuple[str, ...]] = ('colors',)


def test_the_share_ordering_acts_on_a_subclasss_own_shares_fields_and_on_no_other() -> None:
    """A subclass names its shares fields in SHARES and nothing else,
    so a dict field of its own is left as given, unchecked and unordered.
    """
    with pytest.raises(ValidationError, match='purple'):
        Toy(sizes={'small': 0.5, 'purple': 0.5})
    assert OnlyColors(sizes={'purple': 2.0}).sizes == {'purple': 2.0}


CATEGORIES = [
    pytest.param('colors', ['red', 'green', 'blue'], id='a dirichlet default'),
    pytest.param('sizes', ['small', 'large'], id='a point default, which has no dirichlet to read'),
]


@pytest.mark.parametrize(('field', 'categories'), CATEGORIES)
def test_categories_reads_a_shares_fields_default_in_order(
    field: str, categories: list[str]
) -> None:
    """The shares check, the catalogs and the point masses
    read a field's categories from its default.
    """
    assert Toy.categories(field) == categories


def test_a_point_mass_puts_every_share_on_its_category() -> None:
    """Tests force one type, family or activation
    through a point mass over the field's categories.
    """
    shares = {'red': 0.0, 'green': 0.0, 'blue': 1.0}
    assert Toy(colors=shares).colors == shares


@pytest.mark.parametrize(
    ('shares', 'reversed_shares'),
    [
        pytest.param(
            {'red': 0.5, 'green': 0.3, 'blue': 0.2},
            {'blue': 0.2, 'green': 0.3, 'red': 0.5},
            id='point',
        ),
        pytest.param(
            Dirichlet({'red': 1, 'green': 2, 'blue': 3}),
            Dirichlet({'blue': 3, 'green': 2, 'red': 1}),
            id='dirichlet',
        ),
    ],
)
def test_the_order_an_override_lists_its_categories_in_changes_no_draw(
    shares: dict[str, float] | Dirichlet, reversed_shares: dict[str, float] | Dirichlet
) -> None:
    """A category is chosen by its place in the shares,
    so an override listed in another order must not reorder them.
    """
    listed = Toy(colors=shares).resolveField('colors', np.random.default_rng(0))
    reversed_listed = Toy(colors=reversed_shares).resolveField('colors', np.random.default_rng(0))
    assert list(reversed_listed.items()) == list(listed.items())


@given(seed=st.integers(0, 2**32 - 1))
def test_resolving_one_field_gives_its_point_or_one_draw_inside_its_support(seed: int) -> None:
    """A column or node takes its own value of a field through resolveField, whatever it holds."""
    rng = np.random.default_rng(seed)
    assert Toy(scale=0.5).resolveField('scale', rng) == 0.5
    assert 2 <= Toy(scale=LogUniform(2, 3)).resolveField('scale', rng) <= 3


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
    toy = Toy(count=distribution(low, low + width))
    bound = low + margin
    count = toy.resolveField('count', np.random.default_rng(seed), high=bound)
    assert low <= count <= min(bound, low + width)


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
    toy = Toy(count=distribution(low, low + width))
    count = toy.resolveField('count', np.random.default_rng(seed), low=floor)
    assert max(low, floor) <= count <= low + width


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
    toy = Toy(count=distribution)
    rng = np.random.default_rng(0)
    draws = [toy.resolveField('count', rng, high=bound) for _ in range(10_000)]
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
    toy = Toy(count=distribution)
    rng = np.random.default_rng(0)
    draws = {toy.resolveField('count', rng, high=2) for _ in range(100)}
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
    toy = Toy(count=distribution)
    with pytest.raises(ValueError, match=r'count.*\b7\b'):
        toy.resolveField('count', np.random.default_rng(0), high=7)


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
    toy = Toy(count=distribution)
    with pytest.raises(ValueError, match=r'count.*\b4\b'):
        toy.resolveField('count', np.random.default_rng(0), low=4, high=bound)


def test_a_point_above_its_bound_fails_naming_the_field_and_its_bound() -> None:
    """A point override is honored exactly or not at all:
    50 groups cannot be assigned to 30 rows.
    """
    with pytest.raises(ValueError, match=r'count.*\b30\b'):
        Toy(count=50).resolveField('count', np.random.default_rng(0), high=30)


def test_resolved_shares_choose_a_category_by_its_share() -> None:
    """column and graph choose a type, family or activation from resolved shares."""
    rng = np.random.default_rng(0)
    toy = Toy(colors={'red': 0.0, 'green': 1.0, 'blue': 0.0})
    assert toy.resolveField('colors', rng).choose(rng) == 'green'


def test_resolving_one_shares_field_gives_shares_that_choose() -> None:
    """resolveField hands back shares that choose,
    whether the field holds a Dirichlet or a point.
    """
    rng = np.random.default_rng(0)
    assert Toy().resolveField('colors', rng).choose(rng) in {'red', 'green', 'blue'}
    assert Toy().resolveField('sizes', rng).choose(rng) in {'small', 'large'}


def test_resolve_field_is_deterministic_under_a_fixed_generator() -> None:
    """A dataset is a pure function of its key, so the same generator gives the same point."""
    toy = Toy()
    for field in Toy.model_fields:
        first = toy.resolveField(field, np.random.default_rng(7))
        second = toy.resolveField(field, np.random.default_rng(7))
        assert first == second
