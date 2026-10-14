"""draw.py defines the pipeline and its output: one dataset as a pure function of its key.

One generator is threaded through the steps in a fixed order; a change of that order changes
every dataset, so it is a version change.

pipeline:
---------
the key
-> one generator from the key                        default_rng
-> the resolved hyperprior                           Hyperprior.resolveDataset
-> the column plan                                   plan
-> the graph                                         connect
-> the node values                                   propagate
-> the typed columns, joined to their nodes by name  apply
-> the Dataset                                       draw
"""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping, apply, plan
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
    latent: dict[str, np.ndarray]  # (n,) float32 each, z-scored
    hyperparameters: Hyperprior  # resolved, except the fields each column or node draws
    seed: int  # the corpus
    index: int  # the dataset within its corpus
    version: str  # of TabDGP; a key reproduces a dataset only within one version
    numpy_version: str  # its generator's streams may change between numpy releases


# =====================================================================================
# the draw
# =====================================================================================


def draw(seed: int, index: int, hyperprior: Hyperprior | None = None) -> Dataset:
    """Dataset `index` of the corpus `seed`, the same bits whatever else is drawn."""
    rng = np.random.default_rng([seed, index])  # the key's generator
    resolved = (hyperprior or Hyperprior()).resolveDataset(rng)
    metadata = plan(resolved, rng)
    grouping = Grouping(metadata={}, codes={})
    graph = connect(resolved, grouping, list(metadata), rng)
    nodes = propagate(graph, resolved, grouping, rng)
    columns = {name: apply(nodes[name], column, rng) for name, column in metadata.items()}
    latent = {name: nodes[name] for name, is_latent in zip(graph.names, graph.latent) if is_latent}
    return Dataset(
        columns=columns,
        metadata=metadata,
        graph=graph,
        latent=latent,
        hyperparameters=resolved,
        seed=seed,
        index=index,
        version=VERSION,
        numpy_version=NUMPY_VERSION,
    )
