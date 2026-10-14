"""mechanism.py defines how a node's values are computed from its parents' values + its noise.

- A linear node is the readout of its parents.
- An MLP node puts hidden layers before the same readout.
- MLP width and depth are the dataset's, shared by all its MLP nodes.
- The activation, init scale and noise scale are the node's own (read from the graph).
- A node is computed at a unit: its parents are read there, one noise per element,
  and the result is z-scored at the node's own unit.
- A row-unit node of a grouped dataset is two parts: a group part over the parents in its
  group factor's chain, computed at that factor, and a row part over every other parent,
  mixed by its group share, the group part's share of its variance.

pipeline:
---------
the generating graph in topological order, the grouping
-> per node, its parents' values at a unit through its family   _mapParents
-> relative noise, z-scored at the node's unit                   _runMechanism
-> a row-unit node's group part and row part, mixed by its share _mix
-> every node's values by name, at its unit                      propagate
"""

from __future__ import annotations

import math

import numpy as np

from tabdgp.column import Grouping
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


def _isConstant(x: np.ndarray) -> bool:
    # the same on every row; the first column settles most layers without reading the rest
    return bool((x[:, 0] == x[0, 0]).all() and (x == x[0]).all())


def _mapParents(
    hidden: np.ndarray, j: int, graph: Graph, resolved: Hyperprior, rng: np.random.Generator
) -> np.ndarray:
    """Node j's family over its parents' values, with fresh weights."""
    layers = resolved.layers if graph.mechanism[j] == 'mlp' else 0
    # weights N(0, init_scale² / fan_in) keep a layer's scale independent of its width
    # (LeCun et al. 1998)
    constant = _isConstant(hidden)
    for _ in range(layers):
        fan_in = hidden.shape[1]
        scale = graph.init_scale[j] / fan_in**0.5
        normal = rng.standard_normal((fan_in, resolved.hidden), dtype=np.float32)
        weights = scale * normal  # (fan_in, hidden)
        # a bias per neuron moves the activation off the parents' mean, after scamd's RandomScale
        bias = rng.standard_normal(resolved.hidden, dtype=np.float32)  # (hidden,)
        hidden = ACTIVATIONS[graph.activation[j]](hidden @ weights + bias)  # (m, hidden)
        constant = constant or _isConstant(hidden)
    # the readout's scale vanishes in the z-score that follows; only the relative weights count
    readout = rng.standard_normal(hidden.shape[1], dtype=np.float32)  # (hidden,) or (p,)
    if constant:  # a matrix product may since have spread it over the last bits
        return np.zeros(len(hidden), dtype=np.float32)  # (m,)
    return hidden @ readout  # (m,)


def _zscore(x: np.ndarray) -> np.ndarray:
    # a constant node (a dead relu, or the layer after one) arrives as zeros and stays zeros
    centred = x - x.mean()
    std = centred.std()
    if std == 0:
        return np.zeros_like(x)
    return centred / std


def _countElements(unit: str, grouping: Grouping, resolved: Hyperprior) -> int:
    """How many elements a unit has: the groups of a factor, the rows of the row."""
    if unit == 'row':
        return resolved.rows
    return grouping.metadata[unit].levels


def _runMechanism(
    parents: list[int],
    j: int,
    at: str,
    nodes: dict[str, np.ndarray],
    graph: Graph,
    grouping: Grouping,
    resolved: Hyperprior,
    rng: np.random.Generator,
) -> np.ndarray:
    """Node j's mechanism over these parents, run at the unit `at` and z-scored at j's own unit.

    The parents are read at `at` and the matrix products run at its size, with one noise per
    element; output and noise are broadcast to j's unit before each z-score, so that the
    noise scale and the group share are shares of variance over j's unit.
    """
    to = graph.unit[j]
    m = _countElements(at, grouping, resolved)
    noise = rng.standard_normal(m, dtype=np.float32)  # (m,)
    noise = grouping.broadcast(noise, at, to)
    if not parents:  # a root is its noise
        return _zscore(noise)
    parent_values = [grouping.broadcast(nodes[graph.names[i]], graph.unit[i], at) for i in parents]
    hidden = np.stack(parent_values, axis=1)  # (m, p)
    output = _mapParents(hidden, j, graph, resolved, rng)  # (m,)
    output = grouping.broadcast(output, at, to)
    # noise scale is relative to the z-scored output
    return _zscore(_zscore(output) + graph.noise_scale[j] * noise)


def _mix(
    parents: list[int],
    j: int,
    nodes: dict[str, np.ndarray],
    graph: Graph,
    grouping: Grouping,
    resolved: Hyperprior,
    rng: np.random.Generator,
) -> np.ndarray:
    """The two-part mechanism of row-unit node j: z(sqrt(s) · group + sqrt(1 − s) · row).

    The group part draws before the row part; this order is fixed.
    """
    factor = graph.group_factor[j]
    chain = grouping.traceChain(factor)
    group_parents = [i for i in parents if graph.unit[i] in chain]
    row_parents = [i for i in parents if graph.unit[i] not in chain]
    group = _runMechanism(group_parents, j, factor, nodes, graph, grouping, resolved, rng)  # (n,)
    row = _runMechanism(row_parents, j, 'row', nodes, graph, grouping, resolved, rng)  # (n,)
    share = graph.group_share[j]
    return _zscore(math.sqrt(share) * group + math.sqrt(1 - share) * row)  # (n,)


def propagate(
    graph: Graph, resolved: Hyperprior, grouping: Grouping, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Every node's values by name in the graph's order, each at its unit, z-scored, its own array.

    The grouping factors are skipped: their values are the codes the grouping holds.
    """
    nodes: dict[str, np.ndarray] = {}
    for j, name in enumerate(graph.names):
        if name in grouping.metadata:
            continue
        parents = np.flatnonzero(graph.parents[j]).tolist()
        if graph.group_factor[j] is None:  # a group-level node, or any node of a flat dataset
            nodes[name] = _runMechanism(
                parents, j, graph.unit[j], nodes, graph, grouping, resolved, rng
            )
        else:
            nodes[name] = _mix(parents, j, nodes, graph, grouping, resolved, rng)
    return nodes
