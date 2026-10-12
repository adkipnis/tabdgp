"""The marginal module's seam: the catalogue, a sample of each family under a column's parameters."""

from __future__ import annotations

from typing import get_args

import numpy as np
import pytest

from tabdgp.hyperprior import Hyperprior
from tabdgp.marginal import FAMILIES, PARAMETERS, Family


def _sample(family: str, n: int, seed: int) -> np.ndarray:
    """A sample of the family under parameters drawn from the defaults, as a column draws them."""
    rng = np.random.default_rng(seed)
    hyperprior = Hyperprior()
    parameters = {field: hyperprior.point(field, rng) for field in PARAMETERS[family]}
    return FAMILIES[family](n, parameters, rng)


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
def test_each_familys_sample_lies_in_its_support(family: str) -> None:
    """A bounded column stays bounded and every column is finite, over many drawn parameters.

    Closed bounds: a zero-inflated or uniform draw can be 0.
    """
    low, high = SUPPORTS[family]
    for seed in range(50):
        sample = _sample(family, 500, seed)
        assert np.isfinite(sample).all()
        assert (low <= sample).all() and (sample <= high).all()


def _large(family: str, **parameters: float) -> np.ndarray:
    """A sample of 100,000, so that its statistics sit within 0.01 of their law's."""
    return FAMILIES[family](100_000, parameters, np.random.default_rng(0))


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
