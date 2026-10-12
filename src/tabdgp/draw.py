"""draw.py defines the pipeline: one dataset as a pure function of its seed and index.

One generator is threaded through the steps in a fixed order; a change of that order changes
every dataset, so it is a version change.

pipeline:
---------
the key
-> one generator from the key                        Key.generator
-> the resolved hyperprior                           Hyperprior.resolve
-> the column plan                                   plan
-> the graph                                         connect
-> the node values                                   propagate
-> the typed columns, joined to their nodes by name  apply
-> the Dataset                                       draw
"""

from __future__ import annotations

from importlib.metadata import version

from tabdgp.column import apply, plan
from tabdgp.dataset import Dataset, Key
from tabdgp.graph import connect
from tabdgp.hyperprior import Hyperprior
from tabdgp.mechanism import propagate

VERSION = version('tabdgp')


# =====================================================================================
# the draw
# =====================================================================================


def draw(seed: int, index: int, hyperprior: Hyperprior | None = None) -> Dataset:
    """Dataset `index` of the corpus `seed`, the same bits whatever else is drawn."""
    key = Key(seed=seed, index=index)
    rng = key.generator()
    resolved = (hyperprior or Hyperprior()).resolve(rng)
    metadata = plan(resolved, rng)
    graph = connect(resolved, list(metadata), rng)
    nodes = propagate(graph, resolved, rng)
    columns = {name: apply(nodes[name], column, rng) for name, column in metadata.items()}
    latent = {name: nodes[name] for name, is_latent in zip(graph.names, graph.latent) if is_latent}
    return Dataset(
        columns=columns,
        metadata=metadata,
        graph=graph,
        latent=latent,
        hyperparameters=resolved,
        key=key,
        version=VERSION,
    )
