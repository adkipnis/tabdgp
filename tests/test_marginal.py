"""The marginal module's seam: match, a sample of the family placed by the node's ranks."""

from __future__ import annotations

import numpy as np
import pytest

from tabdgp.hyperprior import Hyperprior
from tabdgp.marginal import match

FAMILIES = [
    'normal',
    'lognormal',
    'student_t',
    'beta',
    'uniform',
    'zero_inflated_lognormal',
    'normal_mixture',
]


def _match(values: np.ndarray, family: str, seed: int, **overrides: object) -> np.ndarray:
    """The matched column under a hyperprior resolved and matched with one generator."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(**overrides).resolve(rng)
    return match(values, family, resolved, rng)


def _node(seed: int, n: int = 500) -> np.ndarray:
    """z-scored node values, as propagate leaves them."""
    values = np.random.default_rng(seed).standard_normal(n, dtype=np.float32)  # (N,)
    return (values - values.mean()) / values.std()


@pytest.mark.parametrize('family', FAMILIES)
def test_the_matched_column_has_exactly_the_marginal_samples_values(family: str) -> None:
    """Two different nodes under one generator get the same values, only in another order.

    So the values come from the sample alone and the node supplies only the order: a quantile
    lookup that resamples (scamd's copula, 37% ties) would fail this.
    """
    column = _match(_node(1), family, seed=0)
    other = _match(_node(2), family, seed=0)
    assert column.dtype == np.float32
    np.testing.assert_array_equal(np.sort(column), np.sort(other))
    assert not np.array_equal(column, other)


@pytest.mark.parametrize('family', FAMILIES)
def test_the_matched_column_has_the_nodes_ranks(family: str) -> None:
    """Rank matching keeps the dependence the SCM built: read in the node's order, the column rises.

    Non-decreasing rather than increasing, since the zero-inflated sample has ties at 0.
    """
    values = _node(1)
    column = _match(values, family, seed=0)
    in_node_order = column[np.argsort(values)]  # (N,)
    assert (np.diff(in_node_order) >= 0).all()
    assert np.ptp(column) > 0


SUPPORTS = {
    'normal': (-np.inf, np.inf),
    'lognormal': (0, np.inf),
    'student_t': (-np.inf, np.inf),
    'beta': (0, 1),
    'uniform': (0, 1),
    'zero_inflated_lognormal': (0, np.inf),
    'normal_mixture': (-np.inf, np.inf),
}


@pytest.mark.parametrize('family', FAMILIES)
def test_each_familys_sample_lies_in_its_support(family: str) -> None:
    """A bounded column stays bounded and every column is finite, over many resolved hyperpriors.

    Closed bounds: float32 rounds a beta draw near an endpoint onto it.
    """
    low, high = SUPPORTS[family]
    for seed in range(50):
        column = _match(_node(seed), family, seed)
        assert np.isfinite(column).all()
        assert (low <= column).all() and (column <= high).all()


@pytest.mark.parametrize('family', FAMILIES)
def test_match_is_deterministic_under_a_fixed_generator(family: str) -> None:
    """A key reproduces its dataset bitwise, so match may read no state but its generator."""
    values = _node(1)
    np.testing.assert_array_equal(_match(values, family, seed=0), _match(values, family, seed=0))


def _large(family: str, **overrides: object) -> np.ndarray:
    """A matched column of 100,000 rows, so that its statistics sit within 0.01 of their law's."""
    return _match(_node(1, n=100_000), family, 0, **overrides)


def test_each_familys_parameters_are_honoured() -> None:
    """Each family reads its own fields; expected values are closed forms, not the sampler's.

    The mixture's gap is set wide, so the share above half of it is the minor weight.
    The t's two-tailed P(|T| > 3) at df = 3 is 0.0577, from its closed-form CDF.
    """
    lognormal = _large('lognormal', lognormal_sigma=0.5)
    assert abs(np.log(lognormal).std() - 0.5) < 0.01
    inflated = _large('zero_inflated_lognormal', zero_share=0.3)
    assert abs((inflated == 0).mean() - 0.3) < 0.01
    beta = _large('beta', beta_a=2.0, beta_b=5.0)
    assert abs(beta.mean() - 2 / 7) < 0.01
    mixture = _large('normal_mixture', mixture_gap=20.0, mixture_weight=0.2)
    assert abs((mixture > 10).mean() - 0.2) < 0.01
    assert abs(mixture[mixture > 10].mean() - 20) < 0.05
    student = _large('student_t', student_df=3.0)
    assert abs((np.abs(student) > 3).mean() - 0.0577) < 0.005
