"""likelihood.py defines y's likelihood, which draws y from eta through its family's canonical link.

A draw never fails or warns at any eta, and y always fits its column's dtype.

pipeline:
---------
y's family, eta, the residual scale and y's shape
-> y in its column's dtype                           drawY
"""

from __future__ import annotations

import numpy as np

from tabdgp.column import Column

# =====================================================================================
# the families' links and columns
# =====================================================================================

Y_LINKS = {
    'normal': 'identity',
    'bernoulli': 'logit',
    'poisson': 'log',
    'student_t': 'identity',
    'negative_binomial': 'log',
    'latent_bernoulli': 'logit',
    'gamma': 'log',
    'beta': 'logit',
}
Y_COLUMNS = {
    'normal': Column(type='continuous', role='outcome'),
    'bernoulli': Column(type='binary', levels=2, role='outcome'),
    'poisson': Column(type='count', role='outcome'),
    'student_t': Column(type='continuous', role='outcome'),
    'negative_binomial': Column(type='count', role='outcome'),
    'latent_bernoulli': Column(type='binary', levels=2, role='outcome'),
    'gamma': Column(type='continuous', role='outcome'),
    'beta': Column(type='continuous', role='outcome'),
}
RATE_LIMIT = 9.2e18  # numpy refuses a Poisson rate above about 9.22e18, int64's largest count
MEAN_EDGE = 2.0**-53  # keeps both of the beta's parameters positive, which numpy needs, at any eta

# =====================================================================================
# the draw
# =====================================================================================


def drawY(
    family: str,
    eta: np.ndarray,
    scale: np.ndarray | None,
    shape: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws y from eta through the family's canonical link, in its column's dtype."""
    match family:
        case 'normal':
            # y ~ N(eta, σ_ε)
            noise = rng.standard_normal(len(eta))  # (n,)
            y = eta + scale * noise  # (n,)
        case 'bernoulli':
            # y ~ Bernoulli(p), eta the logit of p
            y = _drawBernoulli(eta, rng)
        case 'poisson':
            # y ~ Poisson(λ), eta the log of λ
            rate = _exponentiate(eta)  # (n,)
            y = _drawCounts(rate, rng)
        case 'student_t':
            # y = eta + σ_ε * t_df, the shape df
            noise = rng.standard_t(shape, len(eta))  # (n,)
            y = eta + scale * noise  # (n,)
        case 'negative_binomial':
            # y ~ Poisson(λ) with λ ~ Gamma(φ, mean exp(eta)), the shape φ:
            # the mean is μ = exp(eta) and the variance μ * (1 + μ / φ), the Poisson's at φ → ∞
            mean = _exponentiate(eta)  # (n,)
            # two steps, which saturate where numpy's negative_binomial raises (above the limit)
            rate = rng.gamma(shape, mean / shape)  # (n,)
            y = _drawCounts(rate, rng)
        case 'latent_bernoulli':
            # y ~ Bernoulli(expit(eta + σ_c * ε)) with ε standard normal, the shape σ_c
            latent = eta + shape * rng.standard_normal(len(eta))  # (n,)
            y = _drawBernoulli(latent, rng)
        case 'gamma':
            # y ~ Gamma(k, mean exp(eta)), the shape k: the variance is μ² / k
            mean = _exponentiate(eta)  # (n,)
            draws = rng.gamma(shape, mean / shape)  # (n,)
            # inside float32's positive range, which a fitter's Gamma likelihood needs
            limits = np.finfo(np.float32)
            y = np.clip(draws, limits.tiny, limits.max)
        case 'beta':
            # y ~ Beta(μ * φ, (1 − μ) * φ), the shape φ, the precision:
            # the mean is μ = expit(eta) and the variance μ * (1 − μ) / (φ + 1)
            mean = np.clip(_invertLogit(eta), MEAN_EDGE, 1 - MEAN_EDGE)  # (n,)
            draws = rng.beta(mean * shape, (1 - mean) * shape)  # (n,)
            # inside float32's open interval (0, 1), which a fitter's beta likelihood needs
            below_one = 1 - 2.0**-24  # the largest float32 below 1
            y = np.clip(draws, np.finfo(np.float32).tiny, below_one)
    return y.astype(Y_COLUMNS[family].dtype)


def _invertLogit(eta: np.ndarray) -> np.ndarray:
    """The inverse logit, without exp's overflow."""
    return 0.5 * (1 + np.tanh(eta / 2))


def _drawBernoulli(eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = _invertLogit(eta)  # (n,)
    uniforms = rng.random(len(eta))  # (n,)
    return uniforms < p


def _exponentiate(eta: np.ndarray) -> np.ndarray:
    """e^eta, cut at 700 below float64's overflow."""
    return np.exp(np.minimum(eta, 700.0))


def _drawCounts(rate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Poisson counts per row, saturated at int64's largest above the limit (beyond int64)."""
    drawn = rate < RATE_LIMIT  # (n,)
    counts = rng.poisson(np.where(drawn, rate, 1.0))  # (n,) int64
    return np.where(drawn, counts, np.iinfo(np.int64).max)
