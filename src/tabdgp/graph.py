"""graph.py defines the nodes, their structure and what each node draws.

pipeline:
---------
the resolved SCM hyperprior, the grouping, and the covariate columns
-> each node's unit                                                                    _sampleUnits
-> names and units in topological order, the grouping factors first, y last if drawn   _orderNodes
-> edges: Cauchy node terms and a block bonus                                          _sampleEdges
-> the edges a child may read                                                          _maskEdges
-> each node's own field draws: family, scales, activation                             _resolveNodes
-> each row-unit node's group share and group factor                                   _resolveParts
-> the Graph                                                                           connect
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping
from tabdgp.scm import Scm

Mechanism = Literal['linear', 'mlp']
Activation = Literal['tanh', 'relu', 'sin']

# =====================================================================================
# the generating graph
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Graph:
    """Every node of a dataset in topological order
    (factors `g`, latent `z`, observed `x`, outcome `y`).

    - A grouping factor comes first at its own unit: its values are its group codes.
    - Every other node is at a unit: the units in the factors' order,
      the row last, latent before observed within a unit.
      A node reads parents at its own unit or any unit superior to it,
      so a latent row-unit node may have an observed group-level parent.
    - The outcome y (when present) comes last at the row, i.e. no node reads it.
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
        # every field holds one entry per node
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

        # the latent mask marks the z... names, and every parent comes before its child
        named_latent = [name.startswith('z') for name in self.names]  # (v,)
        if not np.array_equal(self.latent, named_latent):
            raise ValueError(f'the latent mask {self.latent.tolist()} disagrees with {self.names}')
        if np.triu(self.parents).any():
            raise ValueError('a parent on or above the diagonal breaks topological order')

        # the grouping factors come first, without edges, each at its own unit
        factors = [name for name in self.names if name.startswith('g')]
        f = len(factors)
        if self.names[:f] != factors:
            raise ValueError(f'the grouping factors {factors} do not come first in {self.names}')
        if self.parents[:f].any() or self.parents[:, :f].any():
            raise ValueError(f'the grouping factors {factors} take no edge')
        if self.unit[:f] != factors:
            raise ValueError(f'the grouping factors {factors} are at the units {self.unit[:f]}')

        # each unit and group factor is known, the units in the factors' order and the row last
        units = [*factors, 'row']
        unknown = (set(self.unit) - set(units)) | (set(self.group_factor) - {*factors, None})
        if unknown:
            raise ValueError(f'{sorted(unknown)} name no grouping factor among {factors}')
        ranks = [units.index(unit) for unit in self.unit[f:]]  # (v − f,)
        if ranks != sorted(ranks):
            raise ValueError(f'the units {self.unit[f:]} are not in the order {units}')

        # the outcome, when present, is last, so that no node reads it, and at the row
        if 'y' in self.names:
            if self.names[-1] != 'y':
                raise ValueError(f'the outcome y is not last in {self.names}')
            if self.unit[-1] != 'row':
                raise ValueError(f'the outcome y is at the unit {self.unit[-1]}, not the row')

        # each node holds a value exactly for the fields it draws
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


def connect(
    resolved: Scm,
    grouping: Grouping,
    columns: dict[str, Column],
    rng: np.random.Generator,
) -> Graph:
    """The graph over every node (the grouping factors included) and what each node draws."""
    names, unit = _orderNodes(resolved, grouping, columns, rng)
    latent = np.array([name.startswith('z') for name in names], dtype=bool)  # (v,)
    edges = _sampleEdges(resolved, len(names), rng)  # (v, v)
    parents = _maskEdges(edges, unit, grouping)  # (v, v)
    nodes = _resolveNodes(names, parents, resolved, rng)
    group_share, group_factor = _resolveParts(unit, grouping, resolved, rng)
    return Graph(
        names=names,
        parents=parents,
        latent=latent,
        unit=unit,
        group_share=group_share,
        group_factor=group_factor,
        **nodes,
    )


def _orderNodes(
    resolved: Scm,
    grouping: Grouping,
    columns: dict[str, Column],
    rng: np.random.Generator,
) -> tuple[list[str], list[str]]:
    """The names and units in topological order (the factors first, y last if drawn)."""
    factors = list(grouping.metadata)
    latent_nodes = resolved.latent_nodes

    # the observed nodes take the plan's names in a random order:
    # a column's position is independent of its place in the graph
    permuted = rng.permutation(list(columns))  # (d,)
    observed = permuted.tolist()

    # each node's unit
    latent_unit = _sampleUnits(resolved, len(factors), latent_nodes, rng)  # (latent_nodes,)
    # sorted, so that z0, z1, ... follow the order
    latent_unit = np.sort(latent_unit)
    observed_unit = _sampleUnits(resolved, len(factors), len(observed), rng)  # (d,)
    observed_columns = [columns[name] for name in observed]
    observed_unit = _keepEveryLevel(observed_unit, observed_columns, grouping)

    # the nodes by unit in the factors' order, the row last
    nodes = [f'z{i}' for i in range(latent_nodes)] + observed
    unit_index = np.concatenate([latent_unit, observed_unit])  # (latent_nodes + d,)
    # latent before observed within a unit
    order = np.argsort(unit_index, kind='stable')  # (latent_nodes + d,)
    units = grouping.traceChain('row')
    names = factors + [nodes[i] for i in order]
    unit = factors + [units[unit_index[i]] for i in order]

    # the SCM turns y's node into y
    if resolved.outcome_source != 'none':
        names.append('y')
        unit.append('row')
    return names, unit


def _sampleUnits(resolved: Scm, f: int, nodes: int, rng: np.random.Generator) -> np.ndarray:
    """Each node's unit as an index into the f factors, f for the row."""
    if f == 0:
        return np.zeros(nodes, dtype=np.int64)  # a flat table draws nothing
    group_level = rng.random(nodes) < resolved.group_level_share  # (nodes,)
    factor = rng.integers(f, size=nodes)  # (nodes,), uniform among the factors
    return np.where(group_level, factor, f)  # (nodes,)


def _keepEveryLevel(unit: np.ndarray, columns: list[Column], grouping: Grouping) -> np.ndarray:
    """Each column's unit, the row for one with more levels than its factor has groups."""
    groups = [factor.levels for factor in grouping.metadata.values()]
    most_levels = np.array([*groups, math.inf])  # (f + 1,), the row is the fallback
    levels = np.array([column.levels or 0 for column in columns])  # (d,), 0 without levels
    # a column at a factor's unit takes one value per group, so it shows at most m levels
    return np.where(levels > most_levels[unit], len(groups), unit)  # (d,)


def _sampleEdges(resolved: Scm, v: int, rng: np.random.Generator) -> np.ndarray:
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
    reads = np.zeros((len(units), len(units)), dtype=bool)  # (f + 1, f + 1), [child, parent]
    for i, child in enumerate(units):
        chain = grouping.traceChain(child)
        reads[i] = [parent in chain for parent in units]
    index = [units.index(unit_j) for unit_j in unit]  # (v,)
    kept = reads[np.ix_(index, index)]  # (v, v), [j, i]
    # a grouping factor's values are its codes: it reads no parent and is read by none
    f = len(grouping.metadata)
    kept[:f] = False
    kept[:, :f] = False
    return edges & kept


def _resolveNodes(
    names: list[str], parents: np.ndarray, resolved: Scm, rng: np.random.Generator
) -> dict[str, list]:
    """Each node's own draws by field, None where it draws nothing."""
    v = len(parents)
    mechanism: list[Mechanism | None] = [None] * v
    activation: list[Activation | None] = [None] * v
    noise_scale: list[float | None] = [None] * v
    init_scale: list[float | None] = [None] * v
    for j in np.flatnonzero(parents.any(axis=1)):
        mechanism[j] = resolved.mechanism_shares.choose(rng)
        # the outcome's share of unexplained variance reaches far above a covariate's
        noise_field = 'outcome_noise_scale' if names[j] == 'y' else 'noise_scale'
        noise_scale[j] = resolved.resolveField(noise_field, rng)
        if mechanism[j] == 'mlp':
            activation[j] = resolved.activation_shares.choose(rng)
            init_scale[j] = resolved.resolveField('init_scale', rng)
    return {
        'mechanism': mechanism,
        'activation': activation,
        'noise_scale': noise_scale,
        'init_scale': init_scale,
    }


def _resolveParts(
    unit: list[str], grouping: Grouping, resolved: Scm, rng: np.random.Generator
) -> tuple[list[float | None], list[str | None]]:
    """Each row-unit node's group share and group factor, None elsewhere and in a flat table."""
    factors = list(grouping.metadata)
    group_share: list[float | None] = [None] * len(unit)
    group_factor: list[str | None] = [None] * len(unit)
    for j, unit_j in enumerate(unit):
        if unit_j == 'row' and factors:
            group_share[j] = resolved.resolveField('group_share', rng)
            index = rng.integers(len(factors))
            group_factor[j] = factors[index]
    return group_share, group_factor
