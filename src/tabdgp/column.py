"""column.py defines the grouping factors, the column types and levels, and the column values.

pipeline:
---------
resolved sizes, shares and budget
-> per factor (top down), how many factors it nests in           _nestFactors
-> per factor (bottom up), its m groups, their sizes and codes   _sampleSizes, assign
-> per column, its type by its shares, then its k or marginal    plan, _resolveColumn
-> x0, x1, … until the budget is met, then y if the form is scm  plan
-> per column, a sorted sample, one per distinct node value      _sampleMarginal, _sampleCodes
-> placed by the node's ranks, tied values sharing one           apply
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.hyperprior import MARGINAL_FIELDS, Family, Hyperprior

ColumnType = Literal['continuous', 'binary', 'categorical', 'ordered', 'count']
Role = Literal['covariate', 'grouping_factor', 'outcome']

# =====================================================================================
# column metadata
# =====================================================================================


@pydantic_dataclass(frozen=True)
class Column:
    """Metadata of one observed column."""

    type: ColumnType
    levels: int | None = None  # k
    role: Role = 'covariate'
    nests_in: str | None = None  # the superior grouping factor of a nested one
    marginal: Family | None = None  # for continuous and count columns
    parameters: dict[str, float] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        # a grouping factor's codes label its groups, unordered at any m
        if self.role == 'grouping_factor' and self.type != 'categorical':
            raise ValueError(f'a grouping factor cannot be {self.type}')
        if self.role != 'grouping_factor' and self.nests_in is not None:
            raise ValueError(f'a {self.role} cannot nest in {self.nests_in}')
        # a covariate or the outcome is binary at exactly k = 2 and categorical or ordered from k = 3
        if self.type in ('continuous', 'count'):
            agree = self.levels is None
        elif self.type == 'binary':
            agree = self.levels == 2
        else:  # categorical | ordered
            fewest = 2 if self.role == 'grouping_factor' else 3
            agree = self.levels is not None and self.levels >= fewest
        if not agree:
            raise ValueError(f'a {self.type} column cannot have {self.levels} levels')
        # a count is poisson, a continuous column a family of marginal_shares, a coded one none
        if self.type == 'count':
            agree = self.marginal == 'poisson'
        elif self.type == 'continuous':
            agree = self.marginal in Hyperprior.categories('marginal_shares')
        else:
            agree = self.marginal is None
        if not agree:
            raise ValueError(f'a {self.type} column cannot have the marginal {self.marginal}')
        # check that parameters hold exactly the fields this column drew
        if self.role == 'grouping_factor':
            # an unbalanced factor draws the concentration of its sizes, a balanced one nothing
            own = ['size_alpha'] if 'size_alpha' in self.parameters else []
        elif self.marginal is not None:
            own = MARGINAL_FIELDS[self.marginal]
        else:
            own = ['level_alpha']
        if self.parameters.keys() != set(own):
            raise ValueError(f'a {self.type} column draws {own}, not {list(self.parameters)}')

    @property
    def design_columns(self) -> int:
        if self.role != 'covariate':
            return 0  # a factor's codes index random effects, the outcome is what is predicted
        return 1 if self.levels is None else self.levels - 1


# =====================================================================================
# the grouping
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Grouping:
    """A dataset's grouping factors (g0, g1, ...) and the arithmetic of their units.
    A unit (of observation) is the row or one factor;
    a node at a factor's unit has one value per group.
    """

    metadata: dict[str, Column]
    codes: dict[str, np.ndarray]  # (n,) int64 each, every row's group

    def traceChain(self, unit: str) -> list[str]:
        """The units a node at this unit may read, superior first."""
        if unit == 'row':
            return [*self.metadata, 'row']
        chain = [unit]
        superior = self.metadata[unit].nests_in
        while superior is not None:
            chain = [superior, *chain]
            superior = self.metadata[superior].nests_in
        return chain

    def broadcast(self, values: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
        """Copies each group's value at from_unit to every row or subordinate group within it."""
        if from_unit not in self.traceChain(to_unit):
            raise ValueError(f'{from_unit} is not in the chain of {to_unit}')
        if from_unit == to_unit:
            return values
        superior = self.codes[from_unit]  # (n,)
        if to_unit == 'row':
            return values[superior]
        # every row of a subordinate group carries its superior group's code, so any row's will do
        superior_of = np.empty(self.metadata[to_unit].levels, dtype=np.int64)  # (m,)
        superior_of[self.codes[to_unit]] = superior
        return values[superior_of]


def assign(resolved: Hyperprior, rng: np.random.Generator) -> Grouping:
    """The grouping factors (g0, g1, ...), each row assigned to one group per factor."""
    depths = _nestFactors(resolved, rng)
    # what the next factor's groups split, each row's element among them, and its bound:
    # the rows, each row its own element, unless a subordinate factor nests in it
    unnested = resolved.rows, np.arange(resolved.rows), resolved.rows
    elements, element_of, high = unnested
    metadata = {}
    codes = {}
    # bottom up, so that a superior factor splits the groups of the subordinate one nested in it
    for i in reversed(range(len(depths))):
        name = f'g{i}'
        floor = resolved.floorGroups(depths[i])
        groups = resolved.resolveField('groups', rng, low=floor, high=high)
        sizes, parameters = _sampleSizes(elements, groups, resolved, rng)
        sorted_groups = np.repeat(np.arange(groups), sizes)  # (elements,)
        group_of = rng.permutation(sorted_groups)  # (elements,)
        codes[name] = group_of[element_of]  # (n,)
        metadata[name] = Column(
            type='categorical',
            levels=groups,
            role='grouping_factor',
            nests_in=f'g{i - 1}' if depths[i] else None,
            parameters=parameters,
        )
        if depths[i]:
            # the previous factor splits this one's groups into fewer
            elements, element_of, high = groups, codes[name], groups - 1
        else:
            elements, element_of, high = unnested
    # in index order: superior before subordinate along a chain
    return Grouping(metadata=dict(reversed(metadata.items())), codes=dict(reversed(codes.items())))


def _nestFactors(resolved: Hyperprior, rng: np.random.Generator) -> list[int]:
    """Each factor's depth (the number of factors superior to it)."""
    depths = []
    for i in range(resolved.factors):
        nested = i > 0 and rng.random() < resolved.nested_share
        depths.append(depths[-1] + 1 if nested else 0)
    return depths


def _sampleSizes(
    elements: int, groups: int, resolved: Hyperprior, rng: np.random.Generator
) -> tuple[np.ndarray, dict[str, float]]:
    """The group sizes and alphas over the elements, equal or Dirichlet-multinomial."""
    if rng.random() < resolved.balanced_share:
        # e // m each, the remainder one more for the first groups
        sizes = elements // groups + (np.arange(groups) < elements % groups)  # (m,)
        return sizes, {}
    # one element per group, the rest multinomial over Dirichlet(size_alpha) shares
    size_alpha = resolved.resolveField('size_alpha', rng)
    shares = rng.dirichlet(np.full(groups, size_alpha))  # (m,)
    rest = rng.multinomial(elements - groups, shares)  # (m,)
    sizes = 1 + rest  # (m,)
    return sizes, {'size_alpha': size_alpha}


# =====================================================================================
# the column plan
# =====================================================================================


def plan(resolved: Hyperprior, rng: np.random.Generator) -> dict[str, Column]:
    """Columns x0, x1, … drawn one by one until their design columns fill the budget exactly,
    then the outcome y if its form is scm.
    """
    columns = {}
    remainder = resolved.design_columns
    while remainder > 0:
        type = resolved.type_shares.choose(rng)
        # k − 1 may not overflow the budget
        column = _resolveColumn(type, 'covariate', remainder + 1, resolved, rng)
        columns[f'x{len(columns)}'] = column
        remainder -= column.design_columns
    if resolved.outcome == 'scm':
        type = resolved.outcome_type_shares.choose(rng)
        columns['y'] = _resolveColumn(type, 'outcome', resolved.outcome_max_levels, resolved, rng)
    return columns


def _resolveColumn(
    type: ColumnType,
    role: Role,
    most_levels: int,
    resolved: Hyperprior,
    rng: np.random.Generator,
) -> Column:
    """One column's k or marginal family (and its own parameters)."""
    if type in ('continuous', 'count'):
        if type == 'continuous':
            family = resolved.marginal_shares.choose(rng)
        else:
            family = 'poisson'  # a count's marginal by construction, so it takes no share
        parameters = {field: resolved.resolveField(field, rng) for field in MARGINAL_FIELDS[family]}
        return Column(type=type, role=role, marginal=family, parameters=parameters)
    levels = 2
    if type != 'binary':
        extra = rng.geometric(resolved.levels_p)  # 1, 2, …
        levels += int(extra)
    if resolved.max_levels is not None:
        levels = min(levels, resolved.max_levels)
    levels = min(levels, most_levels)
    # a column clipped to two levels is binary
    if levels == 2:
        type = 'binary'
    parameters = {'level_alpha': resolved.resolveField('level_alpha', rng)}
    return Column(type=type, role=role, levels=levels, parameters=parameters)


# =====================================================================================
# the values
# =====================================================================================


def apply(node: np.ndarray, column: Column, rng: np.random.Generator) -> np.ndarray:
    """The column's values in its node's order, one per distinct node value:
    a marginal sample if continuous or a count, level codes otherwise.
    Tied node values have the same column value.
    """
    distinct, rank = np.unique(node, return_inverse=True)  # (u,), (e,) each element's dense rank
    if column.marginal is None:
        by_rank = _sampleCodes(len(distinct), column, rng)  # (u,)
    else:
        sample = _sampleMarginal(len(distinct), column, rng)  # (u,)
        by_rank = np.sort(sample)  # (u,), int64 if a count
        if column.type == 'continuous':
            by_rank = by_rank.astype(np.float32)
    # the sorted values placed by the node's ranks are a permutation of them (Iman and Conover, 1982)
    return by_rank[rank]


def _sampleMarginal(size: int, column: Column, rng: np.random.Generator) -> np.ndarray:
    parameters = column.parameters
    match column.marginal:
        case 'normal':
            return rng.standard_normal(size)
        case 'lognormal':
            return rng.lognormal(0, parameters['lognormal_sigma'], size)
        case 'student_t':
            return rng.standard_t(parameters['student_df'], size)
        case 'beta':
            return rng.beta(parameters['beta_a'], parameters['beta_b'], size)
        case 'uniform':
            return rng.random(size)
        case 'zero_inflated_lognormal':
            zero = rng.random(size) < parameters['zero_share']  # (size,)
            # one positive value at least, as _sampleCodes keeps one value per level
            if zero.all():
                zero[0] = False
            positive = rng.lognormal(0, parameters['lognormal_sigma'], size)  # (size,)
            return np.where(zero, 0, positive)
        case 'normal_mixture':
            minor = rng.random(size) < parameters['mixture_weight']  # (size,)
            normal = rng.standard_normal(size)  # (size,)
            return normal + parameters['mixture_gap'] * minor
        case 'poisson':
            rate = math.exp(parameters['poisson_log_rate'])
            counts = rng.poisson(rate, size)  # (size,)
            # two distinct values at least, as the zero-inflated sample keeps one positive value
            if (counts == counts[0]).all():
                counts[0] += 1
            return counts


def _sampleCodes(size: int, column: Column, rng: np.random.Generator) -> np.ndarray:
    """The level codes of size values in rank order, in k blocks sized by drawn shares."""
    k = column.levels
    shares = rng.dirichlet(np.full(k, column.parameters['level_alpha']))  # (k,)
    # one value per level first, the rest split in proportion to the shares
    floor = 1 if size >= k else 0
    edges = np.round(np.cumsum(shares) * (size - floor * k)).astype(np.int64)  # (k,)
    counts = np.diff(edges, prepend=0) + floor  # (k,)
    codes_by_rank = np.repeat(np.arange(k), counts)  # (size,)
    if column.type == 'categorical':
        # nominal levels have no order along the node; permuting the labels is TabICL v1's idea
        labels = rng.permutation(k)  # (k,)
        codes_by_rank = labels[codes_by_rank]
    return codes_by_rank
