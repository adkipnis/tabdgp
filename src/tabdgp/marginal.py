"""marginal.py defines the marginals: the families a continuous column's values are sampled from.

pipeline:
---------
a family and the column's own parameters
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


# every family takes (n, parameters, rng), so FAMILIES calls them alike; normal and uniform have
# no parameter to read
def _normal(n: int, parameters: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _lognormal(n: int, parameters: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    return rng.lognormal(0, parameters['lognormal_sigma'], n)


def _student_t(n: int, parameters: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    return rng.standard_t(parameters['student_df'], n)


def _beta(n: int, parameters: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    return rng.beta(parameters['beta_a'], parameters['beta_b'], n)


def _uniform(n: int, parameters: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    return rng.random(n)


def _zero_inflated_lognormal(
    n: int, parameters: dict[str, float], rng: np.random.Generator
) -> np.ndarray:
    zero = rng.random(n) < parameters['zero_share']  # (N,)
    return np.where(zero, 0, _lognormal(n, parameters, rng))


def _normal_mixture(n: int, parameters: dict[str, float], rng: np.random.Generator) -> np.ndarray:
    # the minor component, of weight mixture_weight, sits mixture_gap sds above the major one
    minor = rng.random(n) < parameters['mixture_weight']  # (N,)
    return rng.standard_normal(n) + parameters['mixture_gap'] * minor


FAMILIES = {
    'normal': _normal,
    'lognormal': _lognormal,
    'student_t': _student_t,
    'beta': _beta,
    'uniform': _uniform,
    'zero_inflated_lognormal': _zero_inflated_lognormal,
    'normal_mixture': _normal_mixture,
}

# the hyperprior's fields each family reads, one value of each drawn per column
PARAMETERS = {
    'normal': [],
    'lognormal': ['lognormal_sigma'],
    'student_t': ['student_df'],
    'beta': ['beta_a', 'beta_b'],
    'uniform': [],
    'zero_inflated_lognormal': ['lognormal_sigma', 'zero_share'],
    'normal_mixture': ['mixture_gap', 'mixture_weight'],
}

# a family added to marginal_shares, the catalogue or PARAMETERS alone fails here, not when drawn
if not set(FAMILIES) == set(PARAMETERS) == set(Hyperprior.categories('marginal_shares')):
    raise ValueError(f'families {list(FAMILIES)} differ from PARAMETERS or marginal_shares')
