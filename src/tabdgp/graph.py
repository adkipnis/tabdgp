"""graph.py defines the generating graph: its nodes, which are latent, and who is whose parent.

pipeline:
---------
the resolved hyperprior and the observed names
-> the names in topological order, latent hubs first     _names
-> the edge logits: Cauchy node terms and a block bonus  _logits
-> the adjacency                                         connect
-> one child for a latent node that drew none            _child
-> each node's own choices: family, noise, activation    _choices
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.hyperprior import Hyperprior

Mechanism = Literal['linear', 'mlp']
Activation = Literal['tanh', 'relu', 'sin', 'identity']

# =====================================================================================
# the generating graph
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Graph:
    """Every node of a dataset in topological order (observed `x`, latent `z`).

    - A root (a node without parents) is just its noise and draws nothing.
    - A linear node draws a noise scale.
    - An MLP node draws a noise scale, an activation and an init scale.
    - None at position i for any field <=> node i does not draw that field.
    """

    names: list[str]
    parents: np.ndarray  # (n, n) bool, parents[i, j] is True if j is a parent of i
    latent: np.ndarray  # (n,) bool
    mechanism: list[Mechanism | None]
    activation: list[Activation | None]
    noise_scale: list[float | None]
    init_scale: list[float | None]

    def __post_init__(self) -> None:
        n = len(self.names)
        shapes = (
            self.parents.shape,
            self.latent.shape,
            len(self.mechanism),
            len(self.activation),
            len(self.noise_scale),
            len(self.init_scale),
        )
        if shapes != ((n, n), (n,), n, n, n, n):
            raise ValueError(f'{n} names, but the per-node fields have shapes {shapes}')
        named_latent = [name.startswith('z') for name in self.names]  # (n,)
        if not np.array_equal(self.latent, named_latent):
            raise ValueError(f'the latent mask {self.latent.tolist()} disagrees with {self.names}')
        if np.triu(self.parents).any():
            raise ValueError('a parent on or above the diagonal breaks topological order')
        # check that each node holds a value for the fields it draws
        root = ~self.parents.any(axis=1)  # (n,)
        not_mlp = np.array([family != 'mlp' for family in self.mechanism])  # (n,)
        unused = {
            'mechanism': (self.mechanism, root),
            'noise_scale': (self.noise_scale, root),
            'activation': (self.activation, not_mlp),
            'init_scale': (self.init_scale, not_mlp),
        }
        for field, (values, unused_by) in unused.items():
            missing = [value is None for value in values]  # (n,)
            if not np.array_equal(missing, unused_by):
                raise ValueError(f'{field} is missing for nodes {missing}, unused by {unused_by}')


# =====================================================================================
# connecting the nodes
# =====================================================================================


def connect(resolved: Hyperprior, observed: list[str], rng: np.random.Generator) -> Graph:
    """The graph over the latent and observed nodes, with each node's own choices; weights are not kept."""
    names = _names(resolved.latent_nodes, observed, rng)
    n = len(names)
    logits = _logits(resolved, n, rng)  # (n, n)
    probability = 0.5 * (1 + np.tanh(logits / 2))  # (n, n), a sigmoid that cannot overflow
    # one uniform per possible edge i -> j, i < j, so the draw count is the edge count
    child, parent = np.tril_indices(n, k=-1)  # (n (n − 1) / 2,) each
    edges = np.zeros((n, n), dtype=bool)  # (n, n), edges[j, i]: i is a parent of j
    edges[child, parent] = rng.uniform(size=len(child)) < probability[child, parent]
    latent = np.arange(n) < resolved.latent_nodes  # (n,)
    parents = _child(edges, latent, rng)  # (n, n)
    mechanism, activation, noise_scale, init_scale = _choices(parents, resolved, rng)
    return Graph(
        names=names,
        parents=parents,
        latent=latent,
        mechanism=mechanism,
        activation=activation,
        noise_scale=noise_scale,
        init_scale=init_scale,
    )


def _names(n_latent: int, observed: list[str], rng: np.random.Generator) -> list[str]:
    # the observed names permute the column plan, so a column's position reveals nothing of its
    # place in the graph
    latent = [f'z{i}' for i in range(n_latent)]
    return latent + [observed[k] for k in rng.permutation(len(observed))]


def _logits(resolved: Hyperprior, n: int, rng: np.random.Generator) -> np.ndarray:
    # TabICL's Cauchy DAG: an edge i -> j has logit offset + out_i + in_j with heavy-tailed node
    # terms, so density runs from near-empty to near-complete and a few nodes become hubs;
    # the latent nodes draw their terms first
    out_term = resolved.hub_scale * rng.standard_cauchy(n)  # (n,)
    in_term = resolved.hub_scale * rng.standard_cauchy(n)  # (n,)
    logits = resolved.edge_offset + in_term[:, None] + out_term[None, :]  # (n, n), [j, i]
    # scamd's block-wise correlation and shared noise, as a bonus for a pair of the same block
    block = rng.integers(resolved.blocks, size=n)  # (n,)
    same_block = block[:, None] == block[None, :]  # (n, n)
    return logits + resolved.block_bonus * same_block


def _child(edges: np.ndarray, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # a latent node is a hub or a confounder, never a dead end: a childless one gets one child
    # drawn uniformly from the nodes after it, if there are any
    n = len(edges)
    childless = np.flatnonzero(latent & ~edges.any(axis=0))  # (c,)
    childless = childless[childless < n - 1]
    children = rng.integers(childless + 1, n)  # (c,)
    forced = edges.copy()
    forced[children, childless] = True
    return forced


def _choices(
    parents: np.ndarray, resolved: Hyperprior, rng: np.random.Generator
) -> tuple[list[Mechanism | None], list[Activation | None], list[float | None], list[float | None]]:
    # a root is its noise and draws nothing; a node with parents draws its family and noise scale,
    # an MLP node also its activation and init scale
    n = len(parents)
    mechanism = [None] * n
    activation = [None] * n
    noise_scale = [None] * n
    init_scale = [None] * n
    for j in np.flatnonzero(parents.any(axis=1)):
        mechanism[j] = resolved.mechanism_shares.choose(rng)
        noise_scale[j] = resolved.resolveField('noise_scale', rng)
        if mechanism[j] == 'mlp':
            activation[j] = resolved.activation_shares.choose(rng)
            init_scale[j] = resolved.resolveField('init_scale', rng)
    return mechanism, activation, noise_scale, init_scale
