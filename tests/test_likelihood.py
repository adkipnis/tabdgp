"""The likelihood module's seams: each family's draw in range and in its column's dtype at any eta,
with the mean and variance of its law.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from tabdgp.likelihood import COLUMNS, LINKS, draw
from tabdgp.prior import Glm


def test_each_family_of_glms_shares_has_a_link_a_column_a_range_and_a_law() -> None:
    """A family with a share but no link, column or case in draw would fail only when a dataset
    chose it, which the tables below catch only if they list it.
    """
    families = Glm.categories('y_family_shares')
    assert list(LINKS) == list(COLUMNS) == families
    assert _families(RANGES) == _families(LAWS) == families


def _families(cases: list) -> list[str]:
    return [case.values[0] for case in cases]


FLOAT32 = np.finfo(np.float32)
BELOW_ONE = 1 - 2.0**-24  # the largest float32 below 1
INT64_MAX = np.iinfo(np.int64).max
# a family, y's dtype and its range, and a shape that suits it (None for the base three)
RANGES = [
    pytest.param('gaussian', np.float32, -np.inf, np.inf, None, id='gaussian'),
    pytest.param('bernoulli', np.int64, 0, 1, None, id='bernoulli'),
    pytest.param('poisson', np.int64, 0, np.inf, None, id='poisson'),
    pytest.param('student_t', np.float32, -np.inf, np.inf, 3.0, id='student_t'),
    pytest.param('negative_binomial', np.int64, 0, INT64_MAX, 1.0, id='negative_binomial'),
    pytest.param('latent_bernoulli', np.int64, 0, 1, 2.0, id='latent_bernoulli'),
    pytest.param('gamma', np.float32, FLOAT32.tiny, FLOAT32.max, 1.0, id='gamma'),
    pytest.param('beta', np.float32, FLOAT32.tiny, BELOW_ONE, 2.0, id='beta'),
]
# η over what a wide prior reaches, and far beyond it, where float64 and float32 run out
ETAS = [
    pytest.param(np.linspace(-40, 40, 1001), id='wide prior'),
    pytest.param(np.array([-800.0, -100.0, -10.0, 0.0, 10.0, 100.0, 800.0]), id='beyond'),
]


@pytest.mark.parametrize(('family', 'dtype', 'low', 'high', 'shape'), RANGES)
@pytest.mark.parametrize('eta', ETAS)
def test_each_familys_draw_is_in_range_and_of_its_dtype_without_a_warning(
    family: str, dtype: type, low: float, high: float, shape: float | None, eta: np.ndarray
) -> None:
    """y's column holds the family's draw in its type's dtype, at any η:
    a Gamma y is positive and a beta y inside (0, 1) as float32 holds them, so neither is
    a bound the fitter's likelihood would reject; a count saturates at int64's largest.
    """
    scale = np.full(len(eta), 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = draw(family, eta, scale, shape, np.random.default_rng(0))
    assert y.dtype == COLUMNS[family].dtype == dtype
    assert y.shape == eta.shape
    assert np.isfinite(y).all()
    assert low <= y.min() and y.max() <= high


def _latentMean(eta: float, scale: float) -> float:
    """E[expit(η + σ ε)] for ε standard normal, by Gauss-Hermite quadrature."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    return float((weights / math.sqrt(2 * math.pi) * _expit(eta + scale * nodes)).sum())


def _expit(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


LATENT_MEAN = _latentMean(math.log(3), 1.5)  # the latent Bernoulli's mean in LAWS

# η, the shape and y's mean and variance in closed form: the identity, the logit and the log of η
# at which the means are round (1.5, logit⁻¹(log 3) = 0.75, e^(log 4) = 4), the variance of each
# family's own noise at scale 1
LAWS = [
    pytest.param('gaussian', 1.5, None, 1.5, 1.0, id='gaussian: the identity'),
    pytest.param('bernoulli', math.log(3), None, 0.75, 0.75 * 0.25, id='bernoulli: the logit'),
    pytest.param('poisson', math.log(4), None, 4.0, 4.0, id='poisson: the log'),
    pytest.param('student_t', 1.5, 10.0, 1.5, 10 / 8, id='student_t: df / (df − 2) at σ_ε = 1'),
    pytest.param(
        'negative_binomial',
        math.log(4),
        2.0,
        4.0,
        4 * (1 + 4 / 2),
        id='negative_binomial: μ (1 + μ / φ)',
    ),
    pytest.param(
        'latent_bernoulli',
        math.log(3),
        1.5,
        LATENT_MEAN,
        LATENT_MEAN * (1 - LATENT_MEAN),
        id='latent_bernoulli: the logit of η plus noise, attenuated toward ½',
    ),
    pytest.param('gamma', math.log(2), 4.0, 2.0, 2.0**2 / 4, id='gamma: the log, μ² / shape'),
    pytest.param(
        'beta',
        math.log(3),
        4.0,
        0.75,
        0.75 * 0.25 / (4 + 1),
        id='beta: the logit, μ (1 − μ) / (φ + 1)',
    ),
]


@pytest.mark.parametrize(('family', 'eta', 'shape', 'mean', 'variance'), LAWS)
def test_each_familys_draw_has_the_mean_and_variance_of_its_law_at_its_shape(
    family: str, eta: float, shape: float | None, mean: float, variance: float
) -> None:
    """Each family has its canonical link (ADR 0006), which a fitter of that family assumes,
    and its own noise at the shape: the draw's moments are the law's, not numpy's by construction.
    """
    rows = 400_000
    y = draw(family, np.full(rows, eta), np.ones(rows), shape, np.random.default_rng(0))
    assert y.mean() == pytest.approx(mean, rel=0.02)
    assert y.var() == pytest.approx(variance, rel=0.05)


def test_a_latent_bernoullis_mean_is_attenuated_by_its_noise() -> None:
    """The latent noise makes y misspecified for a Bernoulli fitter: its mean is pulled to ½,
    away from the Bernoulli's 0.75 at the same η, so LAWS tells the two apart.
    """
    assert 0.5 < LATENT_MEAN < 0.7


def test_a_negative_binomial_count_above_numpys_limit_saturates_as_the_poisson_does() -> None:
    """numpy's own negative_binomial raises when its gamma rate passes the limit,
    which happens at a mean far below it when φ is small, so y saturates instead.
    """
    eta = np.array([0.0, 800.0])  # (n,)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = draw('negative_binomial', eta, np.ones(2), 1.0, np.random.default_rng(0))
    assert y[0] < INT64_MAX
    assert y[1] == INT64_MAX


def test_a_poisson_rate_above_numpys_limit_draws_without_failing() -> None:
    """A log link at a wide scale reaches rates numpy refuses (above about 9.2e18, η ≈ 43.7),
    and int64 holds no count above that, so y saturates at int64's largest:
    the draw never fails or warns, and the rate below the limit is drawn as is.
    """
    eta = np.array([0.0, 43.0, 44.0, 800.0])  # (n,)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = draw('poisson', eta, np.ones(4), None, np.random.default_rng(0))
    assert y[1] == pytest.approx(math.exp(43), rel=1e-6)
    assert y[2:].tolist() == [INT64_MAX, INT64_MAX]
