"""family.py defines y's families: how each draws y from the linear predictor eta.

Each family draws through its canonical link,
at a residual scale per row that only the Normal and Student-t families have (None otherwise)
and a shape, y's second parameter, that only the contaminated families have.
The draw never fails or warns at any eta, and y stays inside what its column's dtype holds.

pipeline:
---------
y's family, eta, the residual scale per row and y's shape
-> y in its column's dtype                                   drawY
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
CONTINUOUS = Column(type='continuous', role='outcome')
BINARY = Column(type='binary', levels=2, role='outcome')
COUNT = Column(type='count', role='outcome')
Y_COLUMNS = {
    'normal': CONTINUOUS,
    'bernoulli': BINARY,
    'poisson': COUNT,
    'student_t': CONTINUOUS,
    'negative_binomial': COUNT,
    'latent_bernoulli': BINARY,
    'gamma': CONTINUOUS,
    'beta': CONTINUOUS,
}
RATE_LIMIT = 9.2e18  # numpy refuses a Poisson rate above about 9.22e18, int64's largest count
MEAN_EDGE = 2.0**-53  # keeps both of the beta's parameters positive, which numpy needs, at any eta

# =====================================================================================
# the draw
# =====================================================================================


def drawY(
    y_family: str,
    eta: np.ndarray,
    scale: np.ndarray | None,
    y_shape: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draws y from eta through the family's canonical link, in its column's dtype."""
    match y_family:
        case 'normal':
            # y ~ N(eta, σ_ε)
            noise = rng.standard_normal(len(eta))  # (n,)
            y = eta + scale * noise  # (n,)
        case 'bernoulli':
            # y ~ Bernoulli(p), eta the logit of p
            y = _drawBernoulli(eta, rng)
        case 'poisson':
            # y ~ Poisson(λ), eta the log of λ
            rate = np.exp(np.minimum(eta, 700.0))  # (n,) float64, exp's overflow cut at 700
            y = _drawCounts(rate, rng)
        case 'student_t':
            # y = eta + σ_ε * t_df, the shape df
            noise = rng.standard_t(y_shape, len(eta))  # (n,)
            y = eta + scale * noise  # (n,)
        case 'negative_binomial':
            # y ~ Poisson(λ) with λ ~ Gamma(φ, mean e^eta), the shape φ:
            # the mean is μ = e^eta and the variance μ * (1 + μ / φ), the Poisson's at φ → ∞
            mean = np.exp(np.minimum(eta, 700.0))  # (n,)
            # numpy's negative_binomial raises on a gamma rate above the Poisson's limit, which
            # this draws in two steps to saturate
            rate = rng.gamma(y_shape, mean / y_shape)  # (n,)
            y = _drawCounts(rate, rng)
        case 'latent_bernoulli':
            # y ~ Bernoulli(expit(eta + σ_c * ε)) with ε standard normal, the shape σ_c
            latent = eta + y_shape * rng.standard_normal(len(eta))  # (n,)
            y = _drawBernoulli(latent, rng)
        case 'gamma':
            # y ~ Gamma(k, mean e^eta), the shape k: the variance is μ² / k
            mean = np.exp(np.clip(eta, -700.0, 700.0))  # (n,)
            draws = rng.gamma(y_shape, mean / y_shape)  # (n,)
            # inside float32's positive range, which a fitter's Gamma likelihood needs
            limits = np.finfo(np.float32)
            y = np.clip(draws, limits.tiny, limits.max)
        case 'beta':
            # y ~ Beta(μ * φ, (1 − μ) * φ), the shape φ, the precision:
            # the mean is μ = expit(eta) and the variance μ * (1 − μ) / (φ + 1)
            mean = np.clip(_invertLogit(eta), MEAN_EDGE, 1 - MEAN_EDGE)  # (n,)
            draws = rng.beta(mean * y_shape, (1 - mean) * y_shape)  # (n,)
            # inside float32's open interval (0, 1), which a fitter's beta likelihood needs
            below_one = 1 - 2.0**-24  # the largest float32 below 1
            y = np.clip(draws, np.finfo(np.float32).tiny, below_one)
    return y.astype(Y_COLUMNS[y_family].dtype)


def _drawBernoulli(eta: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    p = _invertLogit(eta)  # (n,)
    uniforms = rng.random(len(eta))  # (n,)
    return uniforms < p


def _invertLogit(eta: np.ndarray) -> np.ndarray:
    """The inverse logit, without exp's overflow."""
    return 0.5 * (1 + np.tanh(eta / 2))


def _drawCounts(rate: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Poisson counts at a rate per row, saturated at int64's largest above the limit,
    where no int64 holds the count.
    """
    drawn = rate < RATE_LIMIT  # (n,)
    counts = rng.poisson(np.where(drawn, rate, 1.0))  # (n,) int64
    return np.where(drawn, counts, np.iinfo(np.int64).max)
