"""Hand-built inputs the module tests share: point-mass shares, continuous columns by name,
a grouping from code lists, covariates from both, a model's y column, a model that draws nothing
and a graph from parent lists.
Also the between-group share the grouped tests measure.
"""

from __future__ import annotations

import numpy as np

from tabdgp.column import Column, Grouping
from tabdgp.field import Config
from tabdgp.graph import Graph
from tabdgp.hyperprior import Hyperprior
from tabdgp.outcome import Covariates

# a model writes its columns without a marginal or parameters
Y = Column(type='continuous', role='outcome')


def point(field: str, category: str, config: type[Config] = Hyperprior) -> dict[str, float]:
    """Shares of a config's field that put all mass on one category."""
    shares = {}
    for name in config.categories(field):
        shares[name] = 1.0 if name == category else 0.0
    return shares


def columnsOf(names: list[str]) -> dict[str, Column]:
    """Continuous columns of these names, a plan whose columns any unit can hold."""
    columns = {}
    for name in names:
        columns[name] = Column(type='continuous', marginal='normal')
    return columns


def groupingOf(codes: list[list[int]], nests_in: list[str | None] | None = None) -> Grouping:
    """A grouping of factors g0, g1, …, codes[i] listing each row's group of factor i.

    Without nests_in, every factor is crossed.
    """
    if nests_in is None:
        nests_in = [None] * len(codes)
    metadata = {}
    arrays = {}
    for i, (codes_of_i, nests_in_of_i) in enumerate(zip(codes, nests_in)):
        metadata[f'g{i}'] = Column(
            type='categorical',
            levels=max(codes_of_i) + 1,
            role='grouping_factor',
            nests_in=nests_in_of_i,
        )
        arrays[f'g{i}'] = np.array(codes_of_i, dtype=np.int64)  # (n,)
    return Grouping(metadata=metadata, codes=arrays)


def covariatesOf(codes: list[list[int]], names: list[str], rows: int | None = None) -> Covariates:
    """Covariates of a grouping from code lists and continuous columns of these names.
    Each column and the score count the rows up from 0.
    A flat table has no codes to count its rows, so it passes rows.
    """
    grouping = groupingOf(codes)
    if rows is None:
        rows = len(codes[0])
    columns = dict(grouping.codes)
    for name in names:
        columns[name] = np.arange(rows, dtype=np.float32)  # (n,)
    return Covariates(
        columns=columns,
        metadata={**grouping.metadata, **columnsOf(names)},
        grouping=grouping,
        score=np.arange(rows, dtype=np.float32),
    )


def drawNothing(covariates: object, rng: np.random.Generator) -> None:
    """A model's signature, for a form that is never called."""


def graphOf(
    parents: list[list[int]],
    mechanism: list[str | None] | None = None,
    activation: str = 'tanh',
    noise_scale: float = 0.1,
    init_scale: float = 1.0,
    unit: list[str] | None = None,
    group_share: float = 0.5,
    group_factor: str | None = None,
) -> Graph:
    """A graph of nodes x0, x1, …, parents[j] listing node j's parents.

    Without a mechanism, a root has none and every other node is linear.
    A node with parents takes the noise scale; an MLP node also the activation and the init scale.
    Without units, every node is at the row and the graph is flat.
    With them, a node j whose unit is g{j} is that grouping factor,
    and every row-unit node takes the group share and the group factor,
    the first factor unless given.
    """
    v = len(parents)
    adjacency = np.zeros((v, v), dtype=bool)  # (v, v)
    for j, parents_of_j in enumerate(parents):
        adjacency[j, parents_of_j] = True
    if mechanism is None:
        mechanism = [None if not parents_of_j else 'linear' for parents_of_j in parents]
    if unit is None:
        unit = ['row'] * v
    mlp = np.array([family == 'mlp' for family in mechanism], dtype=bool)  # (v,)
    root = ~adjacency.any(axis=1)  # (v,)
    names = [f'g{j}' if unit[j] == f'g{j}' else f'x{j}' for j in range(v)]
    factors = [name for name in names if name.startswith('g')]
    mixed = [unit_j == 'row' and bool(factors) for unit_j in unit]  # (v,)
    if group_factor is None and factors:
        group_factor = factors[0]
    return Graph(
        names=names,
        parents=adjacency,
        latent=np.zeros(v, dtype=bool),
        mechanism=mechanism,
        activation=[activation if is_mlp else None for is_mlp in mlp],
        noise_scale=[None if is_root else noise_scale for is_root in root],
        init_scale=[init_scale if is_mlp else None for is_mlp in mlp],
        unit=unit,
        group_share=[group_share if is_mixed else None for is_mixed in mixed],
        group_factor=[group_factor if is_mixed else None for is_mixed in mixed],
    )


def betweenShare(values: np.ndarray, codes: np.ndarray) -> float:
    """The share of an array's variance over rows that lies between the groups of a factor."""
    values = values.astype(np.float64)
    means = np.bincount(codes, weights=values) / np.bincount(codes)  # (m,)
    between = ((means[codes] - values.mean()) ** 2).sum()
    return float(between / ((values - values.mean()) ** 2).sum())
