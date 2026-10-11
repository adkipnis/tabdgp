"""graph.py defines the generating graph: which nodes a dataset has, which are latent, and who is whose parent.

pipeline:
---------
the resolved hyperprior and the observed node count
-> the names in topological order, latent hubs first     _names
-> the edge logits: Cauchy node terms and a block bonus  _logits
-> the adjacency                                         connect
-> one child for a latent node that drew none            _child
-> the mechanism family per node, noise for a root       _mechanism
"""

from __future__ import annotations

import numpy as np

from tabdgp.dataset import Graph
from tabdgp.hyperprior import Hyperprior

# =====================================================================================
# the generating graph
# =====================================================================================


def connect(resolved: Hyperprior, n_observed: int, rng: np.random.Generator) -> Graph:
    """The graph over the latent and observed nodes; weights are not kept."""
    names = _names(resolved.latent_nodes, n_observed, rng)
    n = len(names)
    logits = _logits(resolved, n, rng)  # (n, n)
    probability = 0.5 * (1 + np.tanh(logits / 2))  # (n, n), a sigmoid that cannot overflow
    # one uniform per possible edge i -> j, i < j, so the draw count is the edge count
    child, parent = np.tril_indices(n, k=-1)  # (n (n − 1) / 2,) each
    edges = np.zeros((n, n), dtype=bool)  # (n, n), edges[j, i]: i is a parent of j
    edges[child, parent] = rng.uniform(size=len(child)) < probability[child, parent]
    latent = np.arange(n) < resolved.latent_nodes  # (n,)
    parents = _child(edges, latent, rng)  # (n, n)
    mechanism = _mechanism(resolved.mechanism_shares, parents, rng)
    return Graph(names=names, parents=parents, latent=latent, mechanism=mechanism)


def _names(n_latent: int, n_observed: int, rng: np.random.Generator) -> list[str]:
    # the observed names permute the column plan, so a column's position reveals nothing of its
    # place in the graph
    latent = [f'z{i}' for i in range(n_latent)]
    observed = [f'x{k}' for k in rng.permutation(n_observed)]
    return latent + observed


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


def _child(parents: np.ndarray, latent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # a latent node is a hub or a confounder, never a dead end: a childless one gets one child
    # drawn uniformly from the nodes after it, if there are any
    n = len(parents)
    childless = np.flatnonzero(latent & ~parents.any(axis=0))  # (c,)
    childless = childless[childless < n - 1]
    children = rng.integers(childless + 1, n)  # (c,)
    forced = parents.copy()
    forced[children, childless] = True
    return forced


def _mechanism(
    shares: dict[str, float], parents: np.ndarray, rng: np.random.Generator
) -> list[str]:
    n = len(parents)
    families = rng.choice(list(shares), size=n, p=list(shares.values()))  # (n,)
    roots = ~parents.any(axis=1)  # (n,)
    return np.where(roots, 'noise', families).tolist()
