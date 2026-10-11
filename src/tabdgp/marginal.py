"""marginal.py defines the marginals: the catalogue of families a continuous column is mapped onto, and rank matching.

pipeline:
---------
a family and its resolved parameters
-> a sample of size N                                    FAMILIES
-> placed by the ranks of the node's values              match
"""

from __future__ import annotations

import numpy as np

from tabdgp.hyperprior import Hyperprior

# =====================================================================================
# the catalogue
# =====================================================================================
# scamd's marginal catalogue, narrowed to continuous families: skewed, heavy-tailed, bounded,
# zero-inflated and bimodal, each in its standard form: no field draws a location or scale.


def _normal(n: int, resolved: Hyperprior, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _lognormal(n: int, resolved: Hyperprior, rng: np.random.Generator) -> np.ndarray:
    return rng.lognormal(0, resolved.lognormal_sigma, n)


def _student_t(n: int, resolved: Hyperprior, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_t(resolved.student_df, n)


def _beta(n: int, resolved: Hyperprior, rng: np.random.Generator) -> np.ndarray:
    return rng.beta(resolved.beta_a, resolved.beta_b, n)


def _uniform(n: int, resolved: Hyperprior, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n)


def _zero_inflated_lognormal(n: int, resolved: Hyperprior, rng: np.random.Generator) -> np.ndarray:
    zero = rng.random(n) < resolved.zero_share  # (N,)
    return np.where(zero, 0, _lognormal(n, resolved, rng))


def _normal_mixture(n: int, resolved: Hyperprior, rng: np.random.Generator) -> np.ndarray:
    # the minor component, of weight mixture_weight, sits mixture_gap sds above the major one
    minor = rng.random(n) < resolved.mixture_weight  # (N,)
    return rng.standard_normal(n) + resolved.mixture_gap * minor


FAMILIES = {
    'normal': _normal,
    'lognormal': _lognormal,
    'student_t': _student_t,
    'beta': _beta,
    'uniform': _uniform,
    'zero_inflated_lognormal': _zero_inflated_lognormal,
    'normal_mixture': _normal_mixture,
}


# =====================================================================================
# rank matching
# =====================================================================================


def match(
    values: np.ndarray, family: str, resolved: Hyperprior, rng: np.random.Generator
) -> np.ndarray:
    """The node's values replaced by a sample of the family, keeping their ranks."""
    sample = np.sort(FAMILIES[family](len(values), resolved, rng))  # (N,)
    # Iman and Conover (1982), https://doi.org/10.1080/03610918208812265: the sorted sample
    # placed by the node's ranks is a permutation of it, never a quantile lookup that resamples
    ranks = np.argsort(np.argsort(values))  # (N,)
    return sample[ranks].astype(np.float32)
