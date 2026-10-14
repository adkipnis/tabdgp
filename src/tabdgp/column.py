"""column.py defines the grouping factors, the column types and levels, and the column values.

pipeline:
---------
resolved sizes, shares and budget
-> per factor, how many coarser factors it nests in           _nestFactors
-> per factor bottom up, its G, sizes and codes               _sampleSizes, assign
-> per column, its type, K and own parameters                 _resolveColumn
-> one named column after another until the budget is met     plan
-> per column, a sorted sample                                _sampleMarginal, _sampleCodes
-> sorted by the node's ranks                                 apply
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field
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
    role: Literal['covariate', 'grouping factor'] = 'covariate'
    nests_in: str | None = None  # the coarser grouping factor of a nested one
    marginal: Family | None = None  # for continuous columns
    parameters: dict[str, float] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        # a grouping factor's codes label its groups, unordered at any G
        if self.role == 'grouping factor' and self.type != 'categorical':
            raise ValueError(f'a grouping factor cannot be {self.type}')
        if self.role == 'covariate' and self.nests_in is not None:
            raise ValueError(f'a covariate cannot nest in {self.nests_in}')
        # binary is exactly K = 2 and categorical or ordered K ≥ 3, so no two types overlap
        if self.type == 'continuous':
            agree = self.levels is None
        elif self.type == 'binary':
            agree = self.levels == 2
        else:  # categorical | ordered; the split from binary is about coding, so not a factor's
            fewest = 2 if self.role == 'grouping factor' else 3
            agree = self.levels is not None and self.levels >= fewest
        if not agree:
            raise ValueError(f'a {self.type} column cannot have {self.levels} levels')
        if (self.type == 'continuous') != (self.marginal is not None):
            raise ValueError(f'a {self.type} column cannot have the marginal {self.marginal}')
        # check that parameters hold exactly the fields this column drew
        if self.role == 'grouping factor':
            # a balanced factor draws nothing, an unbalanced one the concentration of its sizes
            own = ['size_alpha'] if 'size_alpha' in self.parameters else []
        elif self.type == 'continuous':
            own = MARGINAL_FIELDS[self.marginal]
        else:
            own = ['level_alpha']
        if self.parameters.keys() != set(own):
            raise ValueError(f'a {self.type} column draws {own}, not {list(self.parameters)}')

    @property
    def design_columns(self) -> int:
        if self.role == 'grouping factor':
            return 0  # its codes index random effects
        return 1 if self.levels is None else self.levels - 1


# =====================================================================================
# the grouping
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Grouping:
    """A dataset's grouping factors g0, g1, … and the arithmetic of their units.

    - A unit is the row or one factor; a node at a factor's unit has one value per group.
    - A unit's chain is the units a node there may read: the coarser factors it nests in, then itself.
      The row is finer than every factor, so its chain is every factor, then the row.
    """

    metadata: dict[str, Column]
    codes: dict[str, np.ndarray]  # (n,) int64 each, every row's group

    def chain(self, unit: str) -> list[str]:
        if unit == 'row':
            return [*self.metadata, 'row']
        chain = [unit]
        coarser = self.metadata[unit].nests_in
        while coarser is not None:
            chain = [coarser, *chain]
            coarser = self.metadata[coarser].nests_in
        return chain

    def broadcast(self, values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
        """Values at one unit read at a unit of whose chain it is part,
        each group's value repeated over its rows or the finer groups within it.
        """
        if from_unit not in self.chain(to_unit):
            raise ValueError(f'{from_unit} is not in the chain of {to_unit}')
        if from_unit == to_unit:
            return values
        coarser = self.codes[from_unit]  # (n,)
        if to_unit == 'row':
            return values[coarser]
        # every row of a finer group carries its coarser group's code, so any row's will do
        coarser_of = np.empty(self.metadata[to_unit].levels, dtype=np.int64)  # (G,)
        coarser_of[self.codes[to_unit]] = coarser
        return values[coarser_of]


def _nestFactors(resolved: Hyperprior, rng: np.random.Generator) -> list[int]:
    """Each factor's depth, the number of coarser factors it nests in:
    after the first, a factor nests in the previous one with probability nested_share.
    """
    depths = []
    for k in range(resolved.factors):
        nested = k > 0 and rng.random() < resolved.nested_share
        depths.append(depths[-1] + 1 if nested else 0)
    return depths


def _sampleSizes(
    elements: int, groups: int, resolved: Hyperprior, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, float]]:
    """Rule 7's group sizes over the elements, and the parameters they drew."""
    if rng.random() < resolved.balanced_share:
        # n // G each, the remainder one more for the first groups
        sizes = elements // groups + (np.arange(groups) < elements % groups)  # (G,)
        return sizes, {}
    # one element per group, the rest multinomial over Dirichlet(size_alpha) shares
    size_alpha = resolved.resolveField('size_alpha', rng)
    shares = rng.dirichlet(np.full(groups, size_alpha))  # (G,)
    sizes = 1 + rng.multinomial(elements - groups, shares)  # (G,)
    return sizes, {'size_alpha': size_alpha}


def assign(resolved: Hyperprior, rng: np.random.Generator) -> Grouping:
    """The grouping factors g0, g1, …, each row assigned to one group per factor."""
    depths = _nestFactors(resolved, rng)
    # what the next factor's groups split, and its bound:
    # the rows, each row its own group, unless a finer factor nests in it
    unnested = resolved.rows, np.arange(resolved.rows), resolved.rows
    elements, within, high = unnested
    metadata = {}
    codes = {}
    # bottom up, so that a coarser factor splits the groups of the finer one nested in it
    for k in reversed(range(len(depths))):
        # 2 groups and one more per coarser factor above, so that each of them keeps 2
        groups = resolved.resolveField('groups', rng, low=2 + depths[k], high=high)
        sizes, parameters = _sampleSizes(elements, groups, resolved, rng)
        group_of = rng.permutation(np.repeat(np.arange(groups), sizes))  # (elements,)
        codes[f'g{k}'] = group_of[within]  # (n,)
        metadata[f'g{k}'] = Column(
            type='categorical',
            levels=groups,
            role='grouping factor',
            nests_in=f'g{k - 1}' if depths[k] else None,
            parameters=parameters,
        )
        if depths[k]:
            # the previous factor splits this one's groups into fewer
            elements, within, high = groups, codes[f'g{k}'], groups - 1
        else:
            elements, within, high = unnested
    # in index order: coarser before finer along a chain
    return Grouping(metadata=dict(reversed(metadata.items())), codes=dict(reversed(codes.items())))


# =====================================================================================
# the column plan
# =====================================================================================


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


def plan(resolved: Hyperprior, rng: np.random.Generator) -> dict[str, Column]:
    """Columns x0, x1, … drawn one by one until their design columns fill the budget exactly."""
    columns = {}
    remainder = resolved.design_columns
    while remainder > 0:
        column = _resolveColumn(resolved, remainder, rng)
        columns[f'x{len(columns)}'] = column
        remainder -= column.design_columns
    return columns


# =====================================================================================
# the values
# =====================================================================================


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
            zero = rng.random(n) < parameters['zero_share']  # (n,)
            return np.where(zero, 0, rng.lognormal(0, parameters['lognormal_sigma'], n))
        case 'normal_mixture':
            minor = rng.random(n) < parameters['mixture_weight']  # (n,)
            return rng.standard_normal(n) + parameters['mixture_gap'] * minor


def _sampleCodes(n: int, column: Column, rng: np.random.Generator) -> np.ndarray:
    """The n level codes in rank order, in K blocks sized by drawn shares."""
    k = column.levels
    shares = rng.dirichlet(np.full(k, column.parameters['level_alpha']))  # (K,)
    # one row per level first, the rest split in proportion to the shares
    floor = 1 if n >= k else 0
    edges = np.round(np.cumsum(shares) * (n - floor * k)).astype(np.int64)  # (K,)
    counts = np.diff(edges, prepend=0) + floor  # (K,)
    codes_by_rank = np.repeat(np.arange(k), counts)  # (n,)
    if column.type == 'categorical':
        # nominal levels have no order along the node; permuting the labels is TabICL v1's idea
        codes_by_rank = rng.permutation(k)[codes_by_rank]
    return codes_by_rank


def apply(node: np.ndarray, column: Column, rng: np.random.Generator) -> np.ndarray:
    """The column's values in its node's order, one per distinct node value:
    a marginal sample if continuous, level codes otherwise. Tied rows share a value.
    """
    distinct, rank = np.unique(node, return_inverse=True)  # (D,), (n,) each row's dense rank
    if column.type == 'continuous':
        by_rank = np.sort(_sampleMarginal(len(distinct), column, rng)).astype(np.float32)  # (D,)
    else:
        by_rank = _sampleCodes(len(distinct), column, rng)  # (D,)
    # the sorted values placed by the node's ranks are a permutation of them (Iman and Conover, 1982);
    # a node constant within groups, at a factor's unit or at share 1, gives one value per group
    return by_rank[rank]
