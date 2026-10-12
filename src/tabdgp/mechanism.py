"""mechanism.py defines how a node's values follow from its parents' values and its noise.

A linear node is the readout of its parents; an MLP node puts hidden layers before the same
readout. Width and depth are the dataset's; the activation, init scale and noise scale are the
node's own, read from the graph.

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
    values: np.ndarray, j: int, graph: Graph, resolved: Hyperprior, rng: np.random.Generator
) -> np.ndarray:
    layers = resolved.layers if graph.mechanism[j] == 'mlp' else 0
    # weights N(0, init_scale² / fan_in) keep a layer's scale independent of its width
    # (LeCun et al. 1998); a bias per unit moves relu kinks and tanh and sin centres off the
    # parents' mean, as scamd's RandomScale did
    hidden = values[:, graph.parents[j]]  # (N, p)
    for _ in range(layers):
        fan_in = hidden.shape[1]
        scale = graph.init_scale[j] / fan_in**0.5
        weights = scale * rng.standard_normal((fan_in, resolved.hidden), dtype=np.float32)
        bias = rng.standard_normal(resolved.hidden, dtype=np.float32)  # (hidden,)
        hidden = ACTIVATIONS[graph.activation[j]](hidden @ weights + bias)  # (N, hidden)
    # the readout's scale vanishes in the z-score that follows; only the relative weights count
    readout = rng.standard_normal(hidden.shape[1], dtype=np.float32)  # (hidden,) or (p,)
    return hidden @ readout


def _zscore(x: np.ndarray) -> np.ndarray:
    # a constant (a dead relu, or the layer after one) z-scores to zeros, never to NaN nor to
    # the mean's rounding residue scaled up to unit variance
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
    n = len(graph.names)
    rows = resolved.rows
    values = np.empty((rows, n), dtype=np.float32)  # (N, n)
    for j in range(n):
        noise = rng.standard_normal(rows, dtype=np.float32)  # (N,)
        if not graph.parents[j].any():  # a root is its noise
            values[:, j] = _zscore(noise)
            continue
        output = _mapParents(values, j, graph, resolved, rng)  # (N,)
        # relative noise: its scale is relative to the z-scored output, so scales never run
        # away through depth
        values[:, j] = _zscore(_zscore(output) + graph.noise_scale[j] * noise)
    return {name: values[:, j].copy() for j, name in enumerate(graph.names)}  # (N,) each
