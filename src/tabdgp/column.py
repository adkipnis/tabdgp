"""column.py defines the observed columns: their types and levels, and their values.

pipeline:
---------
the resolved shares and budget
-> one named column after another until the budget is met     plan
-> per column, its type, K and own parameters                  _column
-> per column, a sorted marginal sample or codes               _sample, _codes
-> placed by the node's ranks                                  apply
"""

from __future__ import annotations

from typing import get_args

import numpy as np

from tabdgp.dataset import Column, ColumnType
from tabdgp.hyperprior import MARGINAL_FIELDS, Hyperprior

# =====================================================================================
# the column plan
# =====================================================================================

# a type added to type_shares or to Column alone fails here, not when drawn
if set(get_args(ColumnType)) != set(Hyperprior.categories('type_shares')):
    raise ValueError(f'column types {get_args(ColumnType)} differ from type_shares')


def plan(resolved: Hyperprior, rng: np.random.Generator) -> dict[str, Column]:
    """Columns x0, x1, … drawn one by one until their design columns fill the budget exactly."""
    columns = {}
    remainder = resolved.design_columns
    while remainder > 0:
        column = _column(resolved, remainder, rng)
        columns[f'x{len(columns)}'] = column
        remainder -= column.design_columns
    return columns


def _column(resolved: Hyperprior, remainder: int, rng: np.random.Generator) -> Column:
    type = resolved.type_shares.choose(rng)
    if type == 'continuous':
        family = resolved.marginal_shares.choose(rng)
        parameters = {field: resolved.resolveField(field, rng) for field in MARGINAL_FIELDS[family]}
        return Column(type=type, marginal=family, parameters=parameters)
    levels = 2 if type == 'binary' else 2 + int(rng.geometric(resolved.levels_p))
    if resolved.max_levels is not None:
        levels = min(levels, resolved.max_levels)
    # K − 1 may not overflow the budget; a column clipped to two levels is binary
    levels = min(levels, remainder + 1)
    if levels == 2:
        type = 'binary'
    parameters = {'level_alpha': resolved.resolveField('level_alpha', rng)}
    return Column(type=type, levels=levels, parameters=parameters)


# =====================================================================================
# the values
# =====================================================================================


def apply(node: np.ndarray, column: Column, rng: np.random.Generator) -> np.ndarray:
    """The column's values in its node's order: a marginal sample if continuous, codes if not."""
    if column.type == 'continuous':
        by_rank = np.sort(_sample(len(node), column, rng)).astype(np.float32)  # (N,)
    else:
        by_rank = _codes(len(node), column, rng)  # (N,)
    # Iman and Conover (1982), https://doi.org/10.1080/03610918208812265: the sorted values
    # placed by the node's ranks are a permutation of them, never a quantile lookup that resamples
    ranks = np.argsort(np.argsort(node))  # (N,)
    return by_rank[ranks]


def _sample(n: int, column: Column, rng: np.random.Generator) -> np.ndarray:
    parameters = column.parameters
    match column.marginal:
        case 'normal':
            return rng.standard_normal(n)
        case 'lognormal':
            return rng.lognormal(0, parameters['lognormal_sigma'], n)
        case 'student_t':
            return rng.standard_t(parameters['student_df'], n)
        case 'beta':
            return rng.beta(parameters['beta_a'], parameters['beta_b'], n)
        case 'uniform':
            return rng.random(n)
        case 'zero_inflated_lognormal':
            zero = rng.random(n) < parameters['zero_share']  # (N,)
            return np.where(zero, 0, rng.lognormal(0, parameters['lognormal_sigma'], n))
        case 'normal_mixture':
            minor = rng.random(n) < parameters['mixture_weight']  # (N,)
            return rng.standard_normal(n) + parameters['mixture_gap'] * minor


def _codes(n: int, column: Column, rng: np.random.Generator) -> np.ndarray:
    """Codes 0..K−1 in rank order, cut at drawn level shares, each level present if N ≥ K."""
    k = column.levels
    shares = rng.dirichlet(np.full(k, column.parameters['level_alpha']))  # (K,)
    # one row per level first, the rest split in proportion to the shares
    floor = 1 if n >= k else 0
    edges = np.round(np.cumsum(shares) * (n - floor * k)).astype(np.int64)  # (K,)
    counts = np.diff(edges, prepend=0) + floor  # (K,)
    codes_by_rank = np.repeat(np.arange(k), counts)  # (N,)
    if column.type == 'categorical':
        # nominal levels have no order along the node; permuting the labels is TabICL v1's idea
        codes_by_rank = rng.permutation(k)[codes_by_rank]
    return codes_by_rank
