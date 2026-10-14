"""Hand-built inputs the module tests share: point-mass shares, a grouping from code lists
and a graph from parent lists.
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
            role='grouping factor',
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
) -> Graph:
    """A graph of observed nodes x0, x1, …, parents[j] listing node j's parents.

    Without a mechanism, a root has none and every other node is linear.
    A node with parents takes the noise scale; an MLP node also the activation and the init scale.
    """
    d = len(parents)
    adjacency = np.zeros((d, d), dtype=bool)  # (d, d)
    for j, parents_of_j in enumerate(parents):
        adjacency[j, parents_of_j] = True
    if mechanism is None:
        mechanism = [None if not parents_of_j else 'linear' for parents_of_j in parents]
    mlp = np.array([family == 'mlp' for family in mechanism], dtype=bool)  # (d,)
    root = ~adjacency.any(axis=1)  # (d,)
    names = [f'x{j}' for j in range(d)]
    return Graph(
        names=names,
        parents=adjacency,
        latent=np.zeros(d, dtype=bool),
        mechanism=mechanism,
        activation=[activation if is_mlp else None for is_mlp in mlp],
        noise_scale=[None if is_root else noise_scale for is_root in root],
        init_scale=[init_scale if is_mlp else None for is_mlp in mlp],
    )
