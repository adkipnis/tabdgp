"""Hand-built inputs the module tests share: point-mass shares, continuous columns by name,
a grouping from code lists and a graph from parent lists.
"""

from __future__ import annotations

import numpy as np

from tabdgp.column import Column, Grouping
from tabdgp.graph import Graph
from tabdgp.hyperprior import Hyperprior


def point(field: str, category: str) -> dict[str, float]:
    """Shares of a hyperprior field that put all mass on one category."""
    shares = {}
    for name in Hyperprior.categories(field):
        shares[name] = 1.0 if name == category else 0.0
    return shares


def columnsOf(names: list[str]) -> dict[str, Column]:
    """Continuous columns of these names, a plan whose columns any unit can hold."""
    columns = {}
    for name in names:
        columns[name] = Column(type='continuous', marginal='normal')
    return columns


def groupingOf(codes: list[list[int]], nests_in: list[str | None] | None = None) -> Grouping:
    """A grouping of factors g0, g1, …, codes[k] listing each row's group of factor k.

    Without nests_in, every factor is crossed.
    """
    if nests_in is None:
        nests_in = [None] * len(codes)
    metadata = {}
    arrays = {}
    for k, (codes_of_k, nests_in_of_k) in enumerate(zip(codes, nests_in)):
        metadata[f'g{k}'] = Column(
            type='categorical',
            levels=max(codes_of_k) + 1,
            role='grouping_factor',
            nests_in=nests_in_of_k,
        )
        arrays[f'g{k}'] = np.array(codes_of_k, dtype=np.int64)  # (n,)
    return Grouping(metadata=metadata, codes=arrays)


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
    d = len(parents)
    adjacency = np.zeros((d, d), dtype=bool)  # (d, d)
    for j, parents_of_j in enumerate(parents):
        adjacency[j, parents_of_j] = True
    if mechanism is None:
        mechanism = [None if not parents_of_j else 'linear' for parents_of_j in parents]
    if unit is None:
        unit = ['row'] * d
    mlp = np.array([family == 'mlp' for family in mechanism], dtype=bool)  # (d,)
    root = ~adjacency.any(axis=1)  # (d,)
    names = [f'g{j}' if unit[j] == f'g{j}' else f'x{j}' for j in range(d)]
    factors = [name for name in names if name.startswith('g')]
    mixed = [unit_j == 'row' and bool(factors) for unit_j in unit]  # (d,)
    if group_factor is None and factors:
        group_factor = factors[0]
    return Graph(
        names=names,
        parents=adjacency,
        latent=np.zeros(d, dtype=bool),
        mechanism=mechanism,
        activation=[activation if is_mlp else None for is_mlp in mlp],
        noise_scale=[None if is_root else noise_scale for is_root in root],
        init_scale=[init_scale if is_mlp else None for is_mlp in mlp],
        unit=unit,
        group_share=[group_share if is_mixed else None for is_mixed in mixed],
        group_factor=[group_factor if is_mixed else None for is_mixed in mixed],
    )
