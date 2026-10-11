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


def graphOf(parents: list[list[int]], mechanism: list[str] | None = None) -> Graph:
    """A graph of observed nodes x0, x1, …, parents[j] listing node j's parents.

    Without a mechanism, every node is linear.
    """
    n = len(parents)
    adjacency = np.zeros((n, n), dtype=bool)  # (n, n)
    for j, parents_of_j in enumerate(parents):
        adjacency[j, parents_of_j] = True
    if mechanism is None:
        mechanism = ['linear'] * n
    names = [f'x{j}' for j in range(n)]
    return Graph(
        names=names, parents=adjacency, latent=np.zeros(n, dtype=bool), mechanism=mechanism
    )
