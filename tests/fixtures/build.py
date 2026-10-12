"""Hand-built inputs the module tests share: point-mass shares and a graph from parent lists."""

from __future__ import annotations

import numpy as np

from tabdgp.dataset import Graph
from tabdgp.hyperprior import Hyperprior


def point(field: str, category: str) -> dict[str, float]:
    """Shares of a hyperprior field that put all mass on one category."""
    shares = {}
    for name in Hyperprior.categories(field):
        shares[name] = 1.0 if name == category else 0.0
    return shares


def graphOf(
    parents: list[list[int]],
    mechanism: list[str | None] | None = None,
    activation: str = 'identity',
    noise_scale: float = 0.1,
    init_scale: float = 1.0,
) -> Graph:
    """A graph of observed nodes x0, x1, …, parents[j] listing node j's parents.

    Without a mechanism, a root has none and every other node is linear.
    A node with parents takes the noise scale; an MLP node also the activation and the init scale.
    """
    n = len(parents)
    adjacency = np.zeros((n, n), dtype=bool)  # (n, n)
    for j, parents_of_j in enumerate(parents):
        adjacency[j, parents_of_j] = True
    if mechanism is None:
        mechanism = [None if not parents_of_j else 'linear' for parents_of_j in parents]
    mlp = np.array([family == 'mlp' for family in mechanism], dtype=bool)  # (n,)
    root = ~adjacency.any(axis=1)  # (n,)
    names = [f'x{j}' for j in range(n)]
    return Graph(
        names=names,
        parents=adjacency,
        latent=np.zeros(n, dtype=bool),
        mechanism=mechanism,
        activation=[activation if is_mlp else None for is_mlp in mlp],
        noise_scale=np.where(root, np.nan, noise_scale),
        init_scale=np.where(mlp, init_scale, np.nan),
    )
