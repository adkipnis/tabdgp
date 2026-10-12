"""mechanism.py defines how a node's values are computed from its parents' values + its noise.

- A linear node is the readout of its parents.
- An MLP node puts hidden layers before the same readout.
- MLP width and depth are the dataset's, shared by all its MLP nodes.
- The activation, init scale and noise scale are the node's own (read from the graph).

pipeline:
---------
the generating graph in topological order
-> per node, its parents' values through its family      _mapParents
-> relative noise, z-scored                              _zscore
-> every node's values by name                           propagate
"""

from __future__ import annotations

import numpy as np

from tabdgp.graph import Graph
from tabdgp.hyperprior import Hyperprior

# =====================================================================================
# activations
# =====================================================================================
# the three an MLP node draws from: finite, growing at most linearly, and non-linear,
# since an MLP of identities is a linear node.


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


ACTIVATIONS = {'tanh': np.tanh, 'relu': _relu, 'sin': np.sin}


# =====================================================================================
# propagation
# =====================================================================================


def _mapParents(
    nodes: dict[str, np.ndarray],
    j: int,
    graph: Graph,
    resolved: Hyperprior,
    rng: np.random.Generator,
) -> np.ndarray:
    layers = resolved.layers if graph.mechanism[j] == 'mlp' else 0
    parents = [nodes[graph.names[i]] for i in np.flatnonzero(graph.parents[j])]
    # weights N(0, init_scale² / fan_in) keep a layer's scale independent of its width (LeCun et al. 1998)
    hidden = np.stack(parents, axis=1)  # (n, p)
    for _ in range(layers):
        fan_in = hidden.shape[1]
        scale = graph.init_scale[j] / fan_in**0.5
        weights = scale * rng.standard_normal((fan_in, resolved.hidden), dtype=np.float32)
        # a bias per unit moves the activation off the parents' mean, after scamd's RandomScale
        bias = rng.standard_normal(resolved.hidden, dtype=np.float32)  # (hidden,)
        hidden = ACTIVATIONS[graph.activation[j]](hidden @ weights + bias)  # (n, hidden)
    # the readout's scale vanishes in the z-score that follows; only the relative weights count
    readout = rng.standard_normal(hidden.shape[1], dtype=np.float32)  # (hidden,) or (p,)
    return hidden @ readout


def _zscore(x: np.ndarray) -> np.ndarray:
    # a constant (a dead relu, or the layer after one) z-scores to zeros
    centred = x - x.mean()
    std = centred.std()
    residue = 1000 * np.finfo(x.dtype).eps * np.abs(x).max()
    if std <= residue:
        return np.zeros_like(x)
    return centred / std


def propagate(
    graph: Graph, resolved: Hyperprior, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Every node's values by name in the graph's order, each z-scored and its own array."""
    nodes: dict[str, np.ndarray] = {}
    for j, name in enumerate(graph.names):
        noise = rng.standard_normal(resolved.rows, dtype=np.float32)  # (n,)
        if not graph.parents[j].any():  # a root is its noise
            nodes[name] = _zscore(noise)
            continue
        output = _mapParents(nodes, j, graph, resolved, rng)  # (n,)
        # noise scale is relative to the z-scored output
        nodes[name] = _zscore(_zscore(output) + graph.noise_scale[j] * noise)
    return nodes
