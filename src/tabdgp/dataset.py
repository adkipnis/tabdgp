"""dataset.py defines the pipeline and its output:
one dataset as a pure function of its key and its SCM hyperprior.

pipeline:
---------
the key and the SCM hyperprior
-> one generator from the key's seed sequence                    _seedSequence
-> the resolved SCM hyperprior                                   Scm.resolveDataset
-> the grouping factors, their codes and metadata                assign
-> the column plan, y's column last under the SCM                plan
-> the graph, each node at a unit of the grouping, y last        connect
-> the node values, each at its unit, y's the score              propagate
-> the covariate columns at the rows, joined to their nodes      _rankMatchColumns
-> the latent nodes at the rows                                  _broadcastLatent
-> the covariates, their metadata, the grouping and the score    Covariates
-> the SCM's outcome, y rank-matched to the score                rankMatch
-> the Dataset, the grouping factors first, the outcome last     draw

a key, covariates (drawn or a table's) and a model
-> the key's first child stream, then the model's outcome        modelOutcome
"""

from __future__ import annotations

from importlib.metadata import version

import numpy as np
import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping, assign, plan, rankMatch
from tabdgp.covariates import Covariates
from tabdgp.graph import Graph, connect
from tabdgp.mechanism import propagate
from tabdgp.outcome import Model, Outcome, runModel
from tabdgp.scm import Scm

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

    covariates: Covariates  # the grouping factors and the covariates, the score under 'scm'
    graph: Graph
    latent: dict[str, np.ndarray]  # (n,) each, z-scored over its groups if group-level
    outcome: Outcome | None  # y under 'scm', None under 'none'
    hyperparameters: Scm  # after resolution
    seed: int  # the corpus
    index: int  # the dataset within its corpus
    version: str  # of TabDGP; a key reproduces a dataset only within one version
    numpy_version: str  # its generator's streams may change between numpy releases

    @property
    def columns(self) -> dict[str, np.ndarray]:
        """The covariates' columns, then the outcome's (y last)."""
        if self.outcome is None:
            return self.covariates.columns
        return {**self.covariates.columns, **self.outcome.columns}  # (n,) each

    @property
    def metadata(self) -> dict[str, Column]:
        if self.outcome is None:
            return self.covariates.metadata
        return {**self.covariates.metadata, **self.outcome.metadata}

    def toPandas(self) -> pd.DataFrame:
        """The columns as a table (the latent nodes left out)."""
        table: dict[str, np.ndarray | pd.Categorical] = {}
        for name, column in self.metadata.items():
            values = self.columns[name]
            # a coded column is a categorical over its codes 0..k − 1, ordered if its type is
            if column.levels is not None:
                ordered = column.type == 'ordered'
                values = pd.Categorical.from_codes(values, range(column.levels), ordered=ordered)
            table[name] = values
        return pd.DataFrame(table)


# =====================================================================================
# the draw
# =====================================================================================


def draw(seed: int, index: int, hyperprior: Scm) -> Dataset:
    """Dataset `index` of the corpus `seed`, the same bits whatever else is drawn."""
    rng = np.random.default_rng(_seedSequence(seed, index))
    resolved = hyperprior.resolveDataset(rng)

    # the grouping before the graph, which draws each node's unit among the grouping factors
    grouping = assign(resolved, rng)
    planned = plan(resolved, rng)  # the covariate columns, then y's under the SCM
    # y's planned column is the outcome's, the rest the graph's and the covariates'
    covariate_metadata = {name: column for name, column in planned.items() if name != 'y'}
    graph = connect(resolved, grouping, covariate_metadata, rng)
    nodes = propagate(graph, resolved, grouping, rng)  # each at its unit
    covariate_columns = _rankMatchColumns(nodes, covariate_metadata, graph, grouping, rng)
    latent = _broadcastLatent(nodes, graph, grouping)
    covariates = Covariates(
        columns={**grouping.codes, **covariate_columns},
        metadata={**grouping.metadata, **covariate_metadata},
        grouping=grouping,
        y_score=nodes.get('y'),  # (n,) as y's node is at the row, None under 'none'
    )

    # under the SCM, y rank-matched onto its marginal at the score the covariates carry
    outcome = None
    if resolved.outcome_source == 'scm':
        y = rankMatch(covariates.y_score, planned['y'], rng)  # (n,)
        outcome = Outcome(columns={'y': y}, metadata={'y': planned['y']})
    return Dataset(
        covariates=covariates,
        graph=graph,
        latent=latent,
        outcome=outcome,
        hyperparameters=resolved,
        seed=seed,
        index=index,
        version=VERSION,
        numpy_version=NUMPY_VERSION,
    )


def _seedSequence(seed: int, index: int) -> np.random.SeedSequence:
    """The key's seed sequence, the one source of every draw keyed by seed and index."""
    return np.random.SeedSequence([seed, index])


def _rankMatchColumns(
    nodes: dict[str, np.ndarray],
    metadata: dict[str, Column],
    graph: Graph,
    grouping: Grouping,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    unit = dict(zip(graph.names, graph.unit))
    columns: dict[str, np.ndarray] = {}
    for name, column in metadata.items():
        values = rankMatch(nodes[name], column, rng)  # (e,) one per group of its unit, or per row
        columns[name] = grouping.broadcast(values, unit[name], 'row')  # (n,)
    return columns


def _broadcastLatent(
    nodes: dict[str, np.ndarray], graph: Graph, grouping: Grouping
) -> dict[str, np.ndarray]:
    latent: dict[str, np.ndarray] = {}
    for name, unit, is_latent in zip(graph.names, graph.unit, graph.latent):
        if is_latent:
            latent[name] = grouping.broadcast(nodes[name], unit, 'row')  # (n,)
    return latent


def modelOutcome(seed: int, index: int, covariates: Covariates, model: Model) -> Outcome:
    """A model's outcome on drawn or given covariates, keyed like a dataset."""
    # the key's first child stream, since the draw reads the root's numbers
    [child] = _seedSequence(seed, index).spawn(1)
    return runModel(model, covariates, np.random.default_rng(child))
