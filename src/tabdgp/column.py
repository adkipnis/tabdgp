"""column.py defines the column types and levels, and the column values.

pipeline:
---------
resolved shares and budget
-> one named column after another until the budget is met     plan
-> per column, its type, K and own parameters                 _resolveColumn
-> per column, a sorted sample                                _sampleMarginal, _sampleCodes
-> sorted by the node's ranks                                 apply
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.hyperprior import MARGINAL_FIELDS, Family, Hyperprior

ColumnType = Literal['continuous', 'binary', 'categorical', 'ordered']

# =====================================================================================
# column metadata
# =====================================================================================


@pydantic_dataclass(frozen=True)
class Column:
    """Metadata of one observed column."""

    type: ColumnType
    levels: int | None = None  # K
    role: Literal['covariate'] = 'covariate'  # extended to grouping factors or outcome later
    marginal: Family | None = None  # for continuous columns
    parameters: dict[str, float] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        # binary is exactly K = 2 and categorical or ordered K ≥ 3, so no two types overlap
        if self.type == 'continuous':
            agree = self.levels is None
        elif self.type == 'binary':
            agree = self.levels == 2
        else:  # categorical | ordered
            agree = self.levels is not None and self.levels >= 3
        if not agree:
            raise ValueError(f'a {self.type} column cannot have {self.levels} levels')
        if (self.type == 'continuous') != (self.marginal is not None):
            raise ValueError(f'a {self.type} column cannot have the marginal {self.marginal}')
        # check that parameters hold exactly the fields this column drew
        if self.type == 'continuous':
            own = MARGINAL_FIELDS[self.marginal]
        else:
            own = ['level_alpha']
        if self.parameters.keys() != set(own):
            raise ValueError(f'a {self.type} column draws {own}, not {list(self.parameters)}')

    @property
    def design_columns(self) -> int:
        return 1 if self.levels is None else self.levels - 1


# =====================================================================================
# the column plan
# =====================================================================================


def plan(resolved: Hyperprior, rng: np.random.Generator) -> dict[str, Column]:
    """Columns x0, x1, … drawn one by one until their design columns fill the budget exactly."""
    columns = {}
    remainder = resolved.design_columns
    while remainder > 0:
        column = _resolveColumn(resolved, remainder, rng)
        columns[f'x{len(columns)}'] = column
        remainder -= column.design_columns
    return columns


def _resolveColumn(resolved: Hyperprior, remainder: int, rng: np.random.Generator) -> Column:
    """One column's type, its K or marginal family (and its own parameters)."""
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
    """The column's values in its node's order: a marginal sample if continuous, level codes otherwise."""
    if column.type == 'continuous':
        by_rank = np.sort(_sampleMarginal(len(node), column, rng)).astype(np.float32)  # (N,)
    else:
        by_rank = _sampleCodes(len(node), column, rng)  # (N,)
    # the sorted values placed by the node's ranks are just a permutation of them (Iman and Conover, 1982)
    ranks = np.argsort(np.argsort(node))  # (N,)
    return by_rank[ranks]


def _sampleMarginal(n: int, column: Column, rng: np.random.Generator) -> np.ndarray:
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


def _sampleCodes(n: int, column: Column, rng: np.random.Generator) -> np.ndarray:
    """N level codes in rank order, in K blocks sized by drawn shares."""
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
