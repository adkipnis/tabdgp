"""column.py defines the observed columns: their types and levels, and their values.

pipeline:
---------
the resolved shares and budget
-> one named column after another until the budget is met     plan
-> per column, a sorted marginal sample or codes in order      apply, _codes
-> placed by the node's ranks                                  apply
"""

from __future__ import annotations

from typing import get_args

import numpy as np

from tabdgp.dataset import Column, ColumnType
from tabdgp.hyperprior import Hyperprior
from tabdgp.marginal import FAMILIES

# =====================================================================================
# the column plan
# =====================================================================================

# a type added to type_shares or to Column alone fails here, not when drawn
if set(get_args(ColumnType)) != set(Hyperprior.categories('type_shares')):
    raise ValueError(f'column types {get_args(ColumnType)} differ from type_shares')


def plan(resolved: Hyperprior, rng: np.random.Generator) -> dict[str, Column]:
    """Columns x0, x1, … drawn one after another until their design columns fill the budget exactly."""
    columns = {}
    remainder = resolved.design_columns
    while remainder > 0:
        column = _column(resolved, remainder, rng)
        columns[f'x{len(columns)}'] = column
        remainder -= column.width
    return columns


def _column(resolved: Hyperprior, remainder: int, rng: np.random.Generator) -> Column:
    type = resolved.type_shares.choose(rng)
    if type == 'continuous':
        return Column(type=type, marginal=resolved.marginal_shares.choose(rng))
    levels = 2 if type == 'binary' else 2 + int(rng.geometric(resolved.levels_p))
    if resolved.max_levels is not None:
        levels = min(levels, resolved.max_levels)
    # K − 1 may not overflow the budget; a column clipped to two levels is binary
    levels = min(levels, remainder + 1)
    if levels == 2:
        type = 'binary'
    return Column(type=type, levels=levels)


# =====================================================================================
# the values
# =====================================================================================


def apply(
    values: np.ndarray, column: Column, resolved: Hyperprior, rng: np.random.Generator
) -> np.ndarray:
    """The column's values in the node's order: its marginal's sample if continuous, its codes if not."""
    if column.type == 'continuous':
        sample = FAMILIES[column.marginal](len(values), resolved, rng)  # (N,)
        by_rank = np.sort(sample).astype(np.float32)  # (N,)
    else:
        by_rank = _codes(len(values), column, resolved, rng)  # (N,)
    # Iman and Conover (1982), https://doi.org/10.1080/03610918208812265: the sorted values
    # placed by the node's ranks are a permutation of them, never a quantile lookup that resamples
    ranks = np.argsort(np.argsort(values))  # (N,)
    return by_rank[ranks]


def _codes(n: int, column: Column, resolved: Hyperprior, rng: np.random.Generator) -> np.ndarray:
    # codes 0..K−1 in rank order, cut at drawn level shares, every level present if N ≥ K
    k = column.levels
    shares = rng.dirichlet(np.full(k, resolved.level_alpha))  # (K,)
    # one row per level first, the rest split at the shares' quantiles
    floor = 1 if n >= k else 0
    edges = np.round(np.cumsum(shares) * (n - floor * k)).astype(np.int64)  # (K,)
    counts = np.diff(edges, prepend=0) + floor  # (K,)
    codes_by_rank = np.repeat(np.arange(k), counts)  # (N,)
    if column.type == 'categorical':
        # nominal levels have no order along the node; permuting the labels is TabICL v1's idea
        codes_by_rank = rng.permutation(k)[codes_by_rank]
    return codes_by_rank
