"""graph.py defines the nodes, their structure and what each node draws.

pipeline:
---------
the resolved hyperprior, the grouping and the observed columns
-> each node's unit                                                   _sampleUnits
-> names and units in topological order, the factors first            _orderNodes
-> edges: Cauchy node terms and a block bonus                         _sampleEdges
-> the edges a child may read                                         _maskEdges
-> each node's own field draws: family, scales, activation            _resolveNodes
-> each row-unit node's group share and group factor                  _resolveParts
-> the Graph                                                          connect
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping
from tabdgp.hyperprior import Hyperprior

Mechanism = Literal['linear', 'mlp']
Activation = Literal['tanh', 'relu', 'sin']

# =====================================================================================
# the generating graph
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Graph:
    """Every node of a dataset in topological order (factors `g`, latent `z`, observed `x`).

    - A grouping factor comes first at its own unit: its values are its group codes.
    - Every other node is at a unit: the units in the factors' order,
      the row last, latent before observed within a unit.
      A node reads parents at its own unit or any unit superior to it,
      so a latent row-unit node may have an observed group-level parent.
    - A root (a node without parents) is just its noise and draws no family or scale.
    - A linear node draws a noise scale.
    - An MLP node draws a noise scale, an activation and an init scale.
    - A row-unit node of a grouped dataset, roots included, draws a group share and a group factor.
    - None at position i for any field <=> node i does not draw that field.
    - The weights are drawn in propagation and not kept.
    """

    names: list[str]
    parents: np.ndarray  # (v, v) bool, parents[j, i]: i is a parent of j
    latent: np.ndarray  # (v,) bool
    mechanism: list[Mechanism | None]
    activation: list[Activation | None]
    noise_scale: list[float | None]
    init_scale: list[float | None]
    unit: list[str]  # 'row' or a grouping factor's name
    group_share: list[float | None]
    group_factor: list[str | None]  # the factor of the group part

    def __post_init__(self) -> None:
        v = len(self.names)
        shapes = (
            self.parents.shape,
            self.latent.shape,
            len(self.mechanism),
            len(self.activation),
            len(self.noise_scale),
            len(self.init_scale),
            len(self.unit),
            len(self.group_share),
            len(self.group_factor),
        )
        if shapes != ((v, v), (v,), v, v, v, v, v, v, v):
            raise ValueError(f'{v} names, but the per-node fields have shapes {shapes}')
        named_latent = [name.startswith('z') for name in self.names]  # (v,)
        if not np.array_equal(self.latent, named_latent):
            raise ValueError(f'the latent mask {self.latent.tolist()} disagrees with {self.names}')
        if np.triu(self.parents).any():
            raise ValueError('a parent on or above the diagonal breaks topological order')
        # check that the grouping factors come first, without edges, each at its own unit
        factors = [name for name in self.names if name.startswith('g')]
        f = len(factors)
        if self.names[:f] != factors:
            raise ValueError(f'the grouping factors {factors} do not come first in {self.names}')
        if self.parents[:f].any() or self.parents[:, :f].any():
            raise ValueError(f'the grouping factors {factors} take no edge')
        if self.unit[:f] != factors:
            raise ValueError(f'the grouping factors {factors} are at the units {self.unit[:f]}')
        # check that each unit and group factor is known, the units in the factors' order, the row last
        units = [*factors, 'row']
        unknown = (set(self.unit) - set(units)) | (set(self.group_factor) - {*factors, None})
        if unknown:
            raise ValueError(f'{sorted(unknown)} name no grouping factor among {factors}')
        ranks = [units.index(unit) for unit in self.unit[f:]]  # (v − f,)
        if ranks != sorted(ranks):
            raise ValueError(f'the units {self.unit[f:]} are not in the order {units}')
        # check that each node holds a value for the fields it draws
        root = ~self.parents.any(axis=1)  # (v,)
        not_mlp = np.array([family != 'mlp' for family in self.mechanism])  # (v,)
        # only a row-unit node of a grouped graph mixes a group part and a row part
        at_row = np.array([unit == 'row' for unit in self.unit], dtype=bool)  # (v,)
        unmixed = ~(at_row & bool(factors))  # (v,)
        unused = {
            'mechanism': (self.mechanism, root),
            'noise_scale': (self.noise_scale, root),
            'activation': (self.activation, not_mlp),
            'init_scale': (self.init_scale, not_mlp),
            'group_share': (self.group_share, unmixed),
            'group_factor': (self.group_factor, unmixed),
        }
        for field, (values, unused_by) in unused.items():
            missing = [value is None for value in values]  # (v,)
            if not np.array_equal(missing, unused_by):
                raise ValueError(f'{field} is missing for nodes {missing}, unused by {unused_by}')


# =====================================================================================
# connecting the nodes
# =====================================================================================


def _sampleUnits(resolved: Hyperprior, f: int, nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Each node's unit as an index into the f factors, f for the row."""
    if f == 0:
        return np.zeros(nodes, dtype=np.int64)  # a flat table draws nothing
    group_level = rng.random(nodes) < resolved.group_level_share  # (nodes,)
    factor = rng.integers(f, size=nodes)  # (nodes,), uniform among the factors
    return np.where(group_level, factor, f)  # (nodes,)


def _keepEveryLevel(unit: np.ndarray, columns: list[Column], grouping: Grouping) -> np.ndarray:
    """Each column's unit, the row for one with more levels than its factor has groups:
    a column at a factor's unit takes one value per group, so it shows at most m levels.
    """
    groups = [factor.levels for factor in grouping.metadata.values()]
    most_levels = np.array([*groups, math.inf])  # (f + 1,), the row is the fallback
    levels = np.array([column.levels or 0 for column in columns])  # (d,), 0 if continuous
    return np.where(levels > most_levels[unit], len(groups), unit)  # (d,)


def _orderNodes(
    resolved: Hyperprior,
    grouping: Grouping,
    columns: dict[str, Column],
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """The names and units in topological order: the factors, then the other nodes
    by unit in the factors' order and the row last, latent before observed within a unit.
    """
    factors = list(grouping.metadata)
    latent_nodes = resolved.latent_nodes
    # the observed nodes take the plan's names in a random order:
    # a column's position is independent of its place in the graph
    permuted = rng.permutation(list(columns))  # (d,)
    observed = permuted.tolist()
    latent_unit = _sampleUnits(resolved, len(factors), latent_nodes, rng)  # (v − f − d,)
    # sorted, so that z0, z1, … follow the order
    latent_unit = np.sort(latent_unit)
    observed_unit = _sampleUnits(resolved, len(factors), len(observed), rng)  # (d,)
    observed_columns = [columns[name] for name in observed]
    observed_unit = _keepEveryLevel(observed_unit, observed_columns, grouping)
    nodes = [f'z{i}' for i in range(latent_nodes)] + observed
    unit_index = np.concatenate([latent_unit, observed_unit])  # (v − f,)
    order = np.argsort(unit_index, kind='stable')  # (v − f,), latent before observed within a unit
    units = grouping.traceChain('row')
    names = factors + [nodes[i] for i in order]
    unit = factors + [units[unit_index[i]] for i in order]
    return names, unit


def _sampleEdges(resolved: Hyperprior, v: int, rng: np.random.Generator) -> np.ndarray:
    # TabICL's Cauchy DAG: an edge i -> j has logit offset + out_i + in_j with heavy-tailed node terms;
    # this way, graph density varies and a few nodes become hubs
    out_cauchy = rng.standard_cauchy(v)  # (v,)
    in_cauchy = rng.standard_cauchy(v)  # (v,)
    out_term = resolved.hub_scale * out_cauchy  # (v,)
    in_term = resolved.hub_scale * in_cauchy  # (v,)
    logits = resolved.edge_offset + in_term[:, None] + out_term[None, :]  # (v, v), [j, i]
    # scamd's block-wise correlation and shared noise
    block = rng.integers(resolved.blocks, size=v)  # (v,)
    same_block = block[:, None] == block[None, :]  # (v, v)
    logits = logits + resolved.block_bonus * same_block  # (v, v)
    probability = 0.5 * (1.0 + np.tanh(logits / 2.0))  # (v, v), a sigmoid that cannot overflow
    # one uniform per possible edge i -> j, i < j, so the draw count is the edge count
    child, parent = np.tril_indices(v, k=-1)  # (v (v − 1) / 2,) each
    edges = np.zeros((v, v), dtype=bool)  # (v, v), edges[j, i]: i is a parent of j
    edges[child, parent] = rng.uniform(size=len(child)) < probability[child, parent]
    return edges


def _maskEdges(edges: np.ndarray, unit: list[str], grouping: Grouping) -> np.ndarray:
    """Drops each edge a child cannot read."""
    units = grouping.traceChain('row')
    # a child reads a parent at its own unit or a superior one of its chain
    reads = np.array(
        [[parent in grouping.traceChain(child) for parent in units] for child in units]
    )  # (f + 1, f + 1), [child's unit, parent's unit]
    index = [units.index(unit_j) for unit_j in unit]  # (v,)
    kept = reads[np.ix_(index, index)]  # (v, v), [j, i]
    # a grouping factor's values are its codes: it reads no parent and is read by none
    f = len(grouping.metadata)
    kept[:f] = False
    kept[:, :f] = False
    return edges & kept


def _resolveNodes(
    parents: np.ndarray, resolved: Hyperprior, rng: np.random.Generator
) -> tuple[list[Mechanism | None], list[Activation | None], list[float | None], list[float | None]]:
    """Each node's own draws, None where it draws nothing."""
    v = len(parents)
    mechanism = [None] * v
    activation = [None] * v
    noise_scale = [None] * v
    init_scale = [None] * v
    for j in np.flatnonzero(parents.any(axis=1)):
        mechanism[j] = resolved.mechanism_shares.choose(rng)
        noise_scale[j] = resolved.resolveField('noise_scale', rng)
        if mechanism[j] == 'mlp':
            activation[j] = resolved.activation_shares.choose(rng)
            init_scale[j] = resolved.resolveField('init_scale', rng)
    return mechanism, activation, noise_scale, init_scale


def _resolveParts(
    unit: list[str], grouping: Grouping, resolved: Hyperprior, rng: np.random.Generator
) -> tuple[list[float | None], list[str | None]]:
    """Each row-unit node's group share and group factor, None elsewhere and in a flat table."""
    factors = list(grouping.metadata)
    group_share = [None] * len(unit)
    group_factor = [None] * len(unit)
    for j, unit_j in enumerate(unit):
        if unit_j == 'row' and factors:
            group_share[j] = resolved.resolveField('group_share', rng)
            index = rng.integers(len(factors))
            group_factor[j] = factors[index]
    return group_share, group_factor


def connect(
    resolved: Hyperprior,
    grouping: Grouping,
    columns: dict[str, Column],
    rng: np.random.Generator,
) -> Graph:
    """The graph over every node (the grouping factors included) and what each node draws."""
    names, unit = _orderNodes(resolved, grouping, columns, rng)
    latent = np.array([name.startswith('z') for name in names], dtype=bool)  # (v,)
    edges = _sampleEdges(resolved, len(names), rng)  # (v, v)
    parents = _maskEdges(edges, unit, grouping)  # (v, v)
    mechanism, activation, noise_scale, init_scale = _resolveNodes(parents, resolved, rng)
    group_share, group_factor = _resolveParts(unit, grouping, resolved, rng)
    return Graph(
        names=names,
        parents=parents,
        latent=latent,
        mechanism=mechanism,
        activation=activation,
        noise_scale=noise_scale,
        init_scale=init_scale,
        unit=unit,
        group_share=group_share,
        group_factor=group_factor,
    )
