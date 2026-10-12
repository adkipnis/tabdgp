"""mechanism.py defines how a node's values follow from its parents' values and its noise.

A linear node is the readout of its parents; an MLP node puts hidden layers before the same
readout. Width and depth are the dataset's; the activation, init scale and noise scale are the
node's own, read from the graph.

pipeline:
---------
the generating graph in topological order
-> per node, its parents' values through its family      propagate, _mlp
-> relative noise, z-scored                              _zscore
"""

from __future__ import annotations

from typing import get_args

import numpy as np

from tabdgp.dataset import Activation, Graph, Mechanism
from tabdgp.hyperprior import Hyperprior

# =====================================================================================
# activations
# =====================================================================================
# the four an MLP node draws from: finite, and growing at most linearly.


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


ACTIVATIONS = {'tanh': np.tanh, 'relu': _relu, 'sin': np.sin, 'identity': _identity}

# a category added to the hyperprior's shares or to a catalogue alone fails here, not when drawn
if set(ACTIVATIONS) != set(Hyperprior.categories('activation_shares')):
    raise ValueError(f'activations {list(ACTIVATIONS)} differ from activation_shares')
if set(get_args(Activation)) != set(ACTIVATIONS):
    raise ValueError(f'activations {get_args(Activation)} differ from the catalogue')
if set(get_args(Mechanism)) != set(Hyperprior.categories('mechanism_shares')):
    raise ValueError(f'mechanisms {get_args(Mechanism)} differ from mechanism_shares')


# =====================================================================================
# propagation
# =====================================================================================


def propagate(
    graph: Graph, resolved: Hyperprior, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Every node's values by name in the graph's order, each z-scored and its own array."""
    n = len(graph.names)
    n_obs = resolved.n_obs
    values = np.empty((n_obs, n), dtype=np.float32)  # (N, n)
    for j in range(n):
        noise = rng.standard_normal(n_obs, dtype=np.float32)  # (N,)
        if not graph.parents[j].any():  # a root is its noise
            values[:, j] = _zscore(noise)
            continue
        layers = resolved.layers if graph.mechanism[j] == 'mlp' else 0
        inputs = values[:, graph.parents[j]]  # (N, p)
        # python floats, since a numpy float64 scalar would promote the float32 arithmetic
        init_scale, noise_scale = float(graph.init_scale[j]), float(graph.noise_scale[j])
        output = _mlp(inputs, layers, graph.activation[j], init_scale, resolved, rng)  # (N,)
        # relative noise: its scale is relative to the z-scored output, so scales never run
        # away through depth
        values[:, j] = _zscore(_zscore(output) + noise_scale * noise)
    return {name: values[:, j].copy() for j, name in enumerate(graph.names)}  # (N,) each


def _mlp(
    inputs: np.ndarray,
    layers: int,
    activation: Activation | None,
    init_scale: float,
    resolved: Hyperprior,
    rng: np.random.Generator,
) -> np.ndarray:
    # fan-in scaled weights keep every hidden layer bounded on z-scored inputs, whatever the
    # width (scamd's unscaled MLP reached 1e15); a bias per unit moves relu kinks and tanh and
    # sin centres off the parents' mean, as scamd's RandomScale did
    hidden = inputs  # (N, p)
    for _ in range(layers):
        fan_in = hidden.shape[1]
        scale = init_scale / fan_in**0.5
        weights = scale * rng.standard_normal((fan_in, resolved.hidden), dtype=np.float32)
        bias = rng.standard_normal(resolved.hidden, dtype=np.float32)  # (hidden,)
        hidden = ACTIVATIONS[activation](hidden @ weights + bias)  # (N, hidden)
    # the readout's scale vanishes in the z-score that follows; only the relative weights count
    readout = rng.standard_normal(hidden.shape[1], dtype=np.float32)  # (hidden,) or (p,)
    return hidden @ readout


def _zscore(x: np.ndarray) -> np.ndarray:
    # a constant (a dead relu, or the layer after one) z-scores to zeros, never to NaN nor to
    # the mean's rounding residue scaled up to unit variance
    centred = x - x.mean()
    std = centred.std()
    if std <= 1000 * np.finfo(x.dtype).eps * np.abs(x).max():
        return np.zeros_like(x)
    return centred / std
