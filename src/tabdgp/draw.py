"""draw.py defines the key and the pipeline: one dataset as a pure function of its seed and index.

One generator is threaded through the steps in a fixed order; a change of that order changes
every dataset, so it is a version change.

pipeline:
---------
the key
-> one generator from default_rng([seed, index])     draw
-> the resolved hyperprior                           Hyperprior.resolve
-> the column plan                                   plan
-> the graph                                         connect
-> the node values                                   propagate
-> the typed columns, in plan order                  _columns
-> the Dataset                                       draw
"""

from __future__ import annotations

from importlib.metadata import version

import numpy as np

from tabdgp.column import apply, plan
from tabdgp.dataset import Column, Dataset, Graph, Key
from tabdgp.graph import connect
from tabdgp.hyperprior import Hyperprior
from tabdgp.marginal import match
from tabdgp.mechanism import propagate

VERSION = version('tabdgp')


# =====================================================================================
# the draw
# =====================================================================================


def draw(seed: int, index: int, hyperprior: Hyperprior | None = None) -> Dataset:
    """Dataset `index` of the corpus `seed`, the same bits whatever else is drawn."""
    key = Key(seed=seed, index=index)
    rng = np.random.default_rng([seed, index])
    resolved = (hyperprior or Hyperprior()).resolve(rng)
    metadata = plan(resolved, rng)
    graph = connect(resolved, len(metadata), rng)
    values = propagate(graph, resolved, resolved.n_obs, rng)  # (N, n)
    nodes = dict(zip(graph.names, values.T))
    latent = {
        name: np.ascontiguousarray(nodes[name])  # its own array, not a view of every node's
        for name, is_latent in zip(graph.names, graph.latent)
        if is_latent
    }
    return Dataset(
        columns=_columns(nodes, metadata, resolved, rng),
        metadata={f'x{k}': column for k, column in enumerate(metadata)},
        graph=graph,
        latent=latent,
        hyperparameters=resolved,
        key=key,
        version=VERSION,
    )


def _columns(
    nodes: dict[str, np.ndarray],
    metadata: list[Column],
    resolved: Hyperprior,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    # column k is node x{k}: continuous columns take their marginal, the rest their codes
    columns = {}
    for k, column in enumerate(metadata):
        values = nodes[f'x{k}']  # (N,)
        if column.type == 'continuous':
            columns[f'x{k}'] = match(values, column.marginal, resolved, rng)
        else:
            columns[f'x{k}'] = apply(values, column, resolved, rng)
    return columns
