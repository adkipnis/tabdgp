"""draw.py defines the pipeline and its output: one dataset as a pure function of its key
and a preset.

One generator is threaded through the steps in a fixed order; a change of that order changes
every dataset, so it is a version change. Every value is computed at its node's unit and
broadcast to the rows only in the last step.

pipeline:
---------
the key and the hyperprior
-> one generator from the key                                    default_rng
-> the resolved hyperprior                                       Hyperprior.resolveDataset
-> the grouping factors, their codes and metadata                assign
-> the column plan                                               plan
-> the graph, each node at a unit of the grouping                connect
-> the node values, each at its unit                             propagate
-> the typed columns at the rows, joined to their nodes by name  _applyColumns
-> the latent nodes at the rows                                  _broadcastLatent
-> the Dataset, the grouping factors first                       draw
"""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping, apply, assign, plan
from tabdgp.graph import Graph, connect
from tabdgp.hyperprior import Hyperprior
from tabdgp.mechanism import propagate

VERSION = version('tabdgp')
NUMPY_VERSION = version('numpy')

# =====================================================================================
# the dataset
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Dataset:
    """A dataset draw in its white-box format.
    Each array is stored as drawn (not copied), so a consumer can share its memory.
    """

    columns: dict[str, np.ndarray]  # (n,) each, float32 if continuous, else int64 codes 0..K−1
    metadata: dict[str, Column]
    graph: Graph
    latent: dict[str, np.ndarray]  # (n,) float32 each, z-scored at its node's unit
    hyperparameters: Hyperprior  # resolved, except the fields each column, node or factor draws
    seed: int  # the corpus
    index: int  # the dataset within its corpus
    version: str  # of TabDGP; a key reproduces a dataset only within one version
    numpy_version: str  # its generator's streams may change between numpy releases


# =====================================================================================
# the draw
# =====================================================================================


def _applyColumns(
    nodes: dict[str, np.ndarray],
    metadata: dict[str, Column],
    graph: Graph,
    grouping: Grouping,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    unit = dict(zip(graph.names, graph.unit))
    columns = {}
    for name, column in metadata.items():
        values = apply(nodes[name], column, rng)  # placed at the node's unit, then broadcast
        columns[name] = grouping.broadcast(values, unit[name], 'row')  # (n,)
    return columns


def _broadcastLatent(
    nodes: dict[str, np.ndarray], graph: Graph, grouping: Grouping
) -> dict[str, np.ndarray]:
    latent = {}
    for name, unit, is_latent in zip(graph.names, graph.unit, graph.latent):
        if is_latent:
            latent[name] = grouping.broadcast(nodes[name], unit, 'row')  # (n,)
    return latent


def draw(seed: int, index: int, hyperprior: Hyperprior) -> Dataset:
    """Dataset `index` of the corpus `seed`, the same bits whatever else is drawn."""
    rng = np.random.default_rng([seed, index])  # the key's generator
    resolved = hyperprior.resolveDataset(rng)
    # the grouping before the graph, which draws each node's unit among its factors
    grouping = assign(resolved, rng)
    metadata = plan(resolved, rng)
    graph = connect(resolved, grouping, metadata, rng)
    nodes = propagate(graph, resolved, grouping, rng)  # each at its unit
    columns = _applyColumns(nodes, metadata, graph, grouping, rng)
    latent = _broadcastLatent(nodes, graph, grouping)
    return Dataset(
        columns={**grouping.codes, **columns},
        metadata={**grouping.metadata, **metadata},
        graph=graph,
        latent=latent,
        hyperparameters=resolved,
        seed=seed,
        index=index,
        version=VERSION,
        numpy_version=NUMPY_VERSION,
    )
