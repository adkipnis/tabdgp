"""column.py defines the observed columns: their types and levels under the design-column budget, and their codes.

pipeline:
---------
the resolved shares and budget
-> one column after another until the budget is met            plan
-> per column, codes cut from the node's ranks at level shares    apply
"""

from __future__ import annotations

import numpy as np

from tabdgp.dataset import Column
from tabdgp.hyperprior import Hyperprior

# =====================================================================================
# the column plan
# =====================================================================================


def plan(resolved: Hyperprior, rng: np.random.Generator) -> list[Column]:
    """Columns drawn one after another until their design columns fill the budget exactly."""
    columns = []
    remainder = resolved.design_columns
    while remainder > 0:
        column = _column(resolved, remainder, rng)
        columns.append(column)
        remainder -= 1 if column.levels is None else column.levels - 1
    return columns


def _column(resolved: Hyperprior, remainder: int, rng: np.random.Generator) -> Column:
    type = _choose(resolved.type_shares, rng)
    if type == 'continuous':
        return Column(type=type, marginal=_choose(resolved.marginal_shares, rng))
    levels = 2 if type == 'binary' else 2 + int(rng.geometric(resolved.levels_p))
    if resolved.max_levels is not None:
        levels = min(levels, resolved.max_levels)
    # K − 1 may not overflow the budget; a column clipped to two levels is binary
    levels = min(levels, remainder + 1)
    if levels == 2:
        type = 'binary'
    return Column(type=type, levels=levels)


def _choose(shares: dict[str, float], rng: np.random.Generator) -> str:
    return str(rng.choice(list(shares), p=list(shares.values())))


# =====================================================================================
# the codes
# =====================================================================================


def apply(
    values: np.ndarray, column: Column, resolved: Hyperprior, rng: np.random.Generator
) -> np.ndarray:
    """Codes 0..K−1 cut from the values' ranks at drawn level shares, every level present if N ≥ K."""
    n = len(values)
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
    ranks = np.argsort(np.argsort(values))  # (N,)
    return codes_by_rank[ranks]
