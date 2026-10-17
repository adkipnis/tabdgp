"""mechanism.py defines how a node's values are computed from its parents' values + its noise.

- A linear node is the readout of its parents.
- An MLP node puts hidden layers before the same readout.
- MLP width and depth are the dataset's, shared by all its MLP nodes.
- The activation, init scale and noise scale are the node's own (read from the graph).
- A node at a grouping factor's unit has one value per group, a node at the row unit one per row.
  Each value is computed from its parents' values for the same group or row, plus its own noise.
  The node is then z-scored over its own values, a group-level node over its groups.
- A row-unit node of a grouped dataset mixes a group part and a row part by its group share.
  The group part is computed at its group factor, over the parents in that factor's chain.
  The row part is computed at the row, over every other parent.

pipeline:
---------
the generating graph in topological order, the grouping
-> per node, its parents' values at a unit through its family     _mapParents
-> relative noise, z-scored at the node's unit                    _runMechanism
-> a row-unit node's group part and row part, mixed by its share  _mixParts
-> every node's values by name, at its unit                       propagate
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
            nodes[name] = _mixParts(parents, j, nodes, graph, grouping, resolved, rng)
    return nodes


def _runMechanism(
    parents: list[int],
    j: int,
    run_unit: str,
    nodes: dict[str, np.ndarray],
    graph: Graph,
    grouping: Grouping,
    resolved: Hyperprior,
    rng: np.random.Generator,
) -> np.ndarray:
    """Node j's mechanism over these parents, run at `run_unit` and z-scored at j's own unit."""
    unit_j = graph.unit[j]
    e = _countElements(run_unit, grouping, resolved)
    noise = rng.standard_normal(e, dtype=np.float32)  # (e,)
    # output and noise are broadcast to j's unit before each z-score,
    # so that the noise scale and the group share are shares of variance over j's unit
    noise = grouping.broadcast(noise, run_unit, unit_j)
    if not parents:  # a root is its noise
        return _zscore(noise)
    parent_values = [
        grouping.broadcast(nodes[graph.names[i]], graph.unit[i], run_unit) for i in parents
    ]
    hidden = np.stack(parent_values, axis=1)  # (e, c)
    output = _mapParents(hidden, j, graph, resolved, rng)  # (e,)
    output = grouping.broadcast(output, run_unit, unit_j)
    # noise scale is relative to the z-scored output
    return _zscore(_zscore(output) + graph.noise_scale[j] * noise)


def _countElements(unit: str, grouping: Grouping, resolved: Hyperprior) -> int:
    """How many elements a unit has: the groups of a factor, the rows of the row."""
    if unit == 'row':
        return resolved.rows
    return grouping.metadata[unit].levels


def _mapParents(
    hidden: np.ndarray, j: int, graph: Graph, resolved: Hyperprior, rng: np.random.Generator
) -> np.ndarray:
    """Node j's family over its parents' values, with fresh weights."""
    layers = resolved.layers if graph.mechanism[j] == 'mlp' else 0
    constant = _isConstant(hidden)
    for _ in range(layers):
        fan_in = hidden.shape[1]
        # weights N(0, init_scale² / fan_in) keep a layer's scale independent of its width
        # (LeCun et al. 1998)
        scale = graph.init_scale[j] / fan_in**0.5
        normal = rng.standard_normal((fan_in, resolved.hidden), dtype=np.float32)
        weights = scale * normal  # (fan_in, hidden)
        # a bias per neuron moves the activation off the parents' mean, after scamd's RandomScale
        bias = rng.standard_normal(resolved.hidden, dtype=np.float32)  # (hidden,)
        hidden = ACTIVATIONS[graph.activation[j]](hidden @ weights + bias)  # (e, hidden)
        constant = constant or _isConstant(hidden)
    # the readout's scale vanishes in the z-score that follows; only the relative weights count
    readout = rng.standard_normal(hidden.shape[1], dtype=np.float32)  # (hidden,) or (c,)
    if constant:  # a matrix product may since have spread it over the last bits
        return np.zeros(len(hidden), dtype=np.float32)  # (e,)
    return hidden @ readout  # (e,)


def _isConstant(x: np.ndarray) -> bool:
    # the same in every element; the first column settles most layers without reading the rest
    return bool((x[:, 0] == x[0, 0]).all() and (x == x[0]).all())


def _zscore(x: np.ndarray) -> np.ndarray:
    # a constant node (a dead relu, or the layer after one) arrives as zeros and stays zeros
    centered = x - x.mean()
    std = centered.std()
    if std == 0:
        return np.zeros_like(x)
    return centered / std


def _mixParts(
    parents: list[int],
    j: int,
    nodes: dict[str, np.ndarray],
    graph: Graph,
    grouping: Grouping,
    resolved: Hyperprior,
    rng: np.random.Generator,
) -> np.ndarray:
    """The two-part mechanism of row-unit node j: z(sqrt(s) · group + sqrt(1 − s) · row)."""
    factor = graph.group_factor[j]
    chain = grouping.traceChain(factor)
    group_parents = [i for i in parents if graph.unit[i] in chain]
    row_parents = [i for i in parents if graph.unit[i] not in chain]
    # the group part draws first, the row part second: swapping them changes what a key draws
    group = _runMechanism(group_parents, j, factor, nodes, graph, grouping, resolved, rng)  # (n,)
    row = _runMechanism(row_parents, j, 'row', nodes, graph, grouping, resolved, rng)  # (n,)
    share = graph.group_share[j]
    return _zscore(math.sqrt(share) * group + math.sqrt(1 - share) * row)  # (n,)
