"""marginal.py defines the marginals: the families a continuous column's values are sampled from.

pipeline:
---------
a family and its resolved parameters
-> a sample of size N                                    FAMILIES
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from tabdgp.hyperprior import Hyperprior

# =====================================================================================
# the catalogue
# =====================================================================================
# scamd's marginal catalogue, narrowed to continuous families: skewed, heavy-tailed, bounded,
# zero-inflated and bimodal, each in its standard form: no field draws a location or scale.

Family = Literal[
    'normal',
    'lognormal',
    'student_t',
    'beta',
    'uniform',
    'zero_inflated_lognormal',
    'normal_mixture',
]


# every family takes (n, resolved, rng), so FAMILIES calls them alike; normal and uniform have
# no parameter to read from resolved
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

# a family added to marginal_shares or to the catalogue alone fails here, not when drawn
if set(FAMILIES) != set(Hyperprior.categories('marginal_shares')):
    raise ValueError(f'families {list(FAMILIES)} differ from marginal_shares')
