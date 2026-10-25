"""draw.py defines the pipeline and its output:
one dataset as a pure function of its key and its hyperprior.

pipeline:
---------
the key and the hyperprior
-> one generator from the key                                    _seedGenerator
-> the resolved hyperprior                                       Hyperprior.resolveDataset
-> the grouping factors, their codes and metadata                assign
-> the column plan, y's column last under the scm form           plan
-> the graph, each node at a unit of the grouping, y last        connect
-> the node values, each at its unit, y's the score              propagate
-> the covariates at the rows, joined to their nodes by name     _applyColumns
-> the latent nodes at the rows                                  _broadcastLatent
-> the covariates, their metadata, the grouping and the score    Covariates
-> the form's outcome: none, y rank-matched to the score,
   or a model's                                                  _drawOutcome
-> the Dataset, the grouping factors first, the outcome last     draw

a key, covariates (a table's) and a model
-> the key's generator, then the model's outcome                 drawOutcome
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping, apply, assign, plan
from tabdgp.graph import Graph, connect
from tabdgp.hyperprior import Hyperprior, Model, OutcomeForm
from tabdgp.mechanism import propagate
from tabdgp.outcome import Covariates, Design, Outcome, ScmOutcome, codeDesign, run

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

    columns: dict[str, np.ndarray]  # (n,) each, float32 if continuous, int64 counts or codes 0..k−1
    metadata: dict[str, Column]
    graph: Graph
    latent: dict[str, np.ndarray]  # (n,) float32 each, z-scored over its groups if group-level
    outcome: Outcome | None  # the form's columns and truth, None under 'none'
    hyperparameters: Hyperprior  # resolved, except the fields each column, node or factor draws
    seed: int  # the corpus
    index: int  # the dataset within its corpus
    version: str  # of TabDGP; a key reproduces a dataset only within one version
    numpy_version: str  # its generator's streams may change between numpy releases

    def toPandas(self) -> pd.DataFrame:
        """The columns as a table, the latent nodes left out.
        A coded column is a categorical over its codes 0..k − 1, ordered if its type is.
        """
        table = {}
        for name, column in self.metadata.items():
            values = self.columns[name]
            if column.levels is not None:
                ordered = column.type == 'ordered'
                values = pd.Categorical.from_codes(values, range(column.levels), ordered=ordered)
            table[name] = values
        return pd.DataFrame(table)

    def toDesign(self) -> Design:
        """The design matrix of the covariates, by the coder a GLM's or GLMM's truth refers to."""
        return codeDesign(self.columns, self.metadata, self.hyperparameters.rows)

    def toDict(self) -> dict[str, Any]:
        """The dataset in JSON's types, lossless but for a study's model, recorded by its name.
        An array records its dtype and values, an object its class and fields.
        """
        record = self._recordValue(self)
        if self.outcome is not None:
            # the outcome's columns are the dataset's, under the same names, so recorded once
            del record['outcome']['columns']
            del record['outcome']['metadata']
        return record

    @staticmethod
    def _recordValue(value: Any) -> Any:
        match value:
            case np.ndarray():
                return {'dtype': value.dtype.name, 'values': value.tolist()}
            case dict():
                return {key: Dataset._recordValue(item) for key, item in value.items()}
            case list() | tuple():
                return [Dataset._recordValue(item) for item in value]
            case None | str() | int() | float():
                return value
            case BaseModel():
                return Dataset._recordFields(value, list(type(value).model_fields))
            case _ if dataclasses.is_dataclass(value):
                names = [field.name for field in dataclasses.fields(value)]
                return Dataset._recordFields(value, names)
            case functools.partial():
                return Dataset._nameModel(value.func)  # its bound settings are the study's
            case _ if callable(value):
                return Dataset._nameModel(value)
            case _:
                raise TypeError(f'{value!r}, a {type(value).__name__}, has no JSON form')

    @staticmethod
    def _recordFields(value: Any, names: list[str]) -> dict[str, Any]:
        record = {'class': type(value).__name__}
        for name in names:
            record[name] = Dataset._recordValue(getattr(value, name))
        return record

    @staticmethod
    def _nameModel(model: Callable) -> str:
        """A function's qualified name, or an object's class's."""
        named = model if hasattr(model, '__qualname__') else type(model)
        return f'{named.__module__}.{named.__qualname__}'


# =====================================================================================
# the draw
# =====================================================================================


def draw(seed: int, index: int, hyperprior: Hyperprior) -> Dataset:
    """Dataset `index` of the corpus `seed`, the same bits whatever else is drawn."""
    rng = _seedGenerator(seed, index)
    resolved = hyperprior.resolveDataset(rng)
    # the grouping before the graph, which draws each node's unit among its factors
    grouping = assign(resolved, rng)
    planned = plan(resolved, rng)  # the covariates' columns, then y's under the scm form
    graph = connect(resolved, grouping, planned, rng)
    nodes = propagate(graph, resolved, grouping, rng)  # each at its unit
    covariate_metadata = {name: column for name, column in planned.items() if name != 'y'}
    covariate_columns = _applyColumns(nodes, covariate_metadata, graph, grouping, rng)
    latent = _broadcastLatent(nodes, graph, grouping)
    covariates = Covariates(
        columns={**grouping.codes, **covariate_columns},
        metadata={**grouping.metadata, **covariate_metadata},
        grouping=grouping,
        score=nodes.get('y'),  # (n,), y is at the row; None under 'none'
    )
    outcome = _drawOutcome(resolved.outcome_form, covariates, planned, rng)
    columns = covariates.columns
    metadata = covariates.metadata
    if outcome is not None:
        # the form's columns after the covariates', so that y stays last
        columns = {**columns, **outcome.columns}
        metadata = {**metadata, **outcome.metadata}
    return Dataset(
        columns=columns,
        metadata=metadata,
        graph=graph,
        latent=latent,
        outcome=outcome,
        hyperparameters=resolved,
        seed=seed,
        index=index,
        version=VERSION,
        numpy_version=NUMPY_VERSION,
    )


def _seedGenerator(seed: int, index: int) -> np.random.Generator:
    """The key's generator, the one source of every draw keyed by seed and index."""
    return np.random.default_rng([seed, index])


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
        values = apply(nodes[name], column, rng)  # (e,) one per group of its unit, or per row
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


def _drawOutcome(
    form: OutcomeForm | Model,
    covariates: Covariates,
    planned: dict[str, Column],
    rng: np.random.Generator,
) -> Outcome | None:
    """The form's outcome from the covariates and the score, None under 'none'."""
    match form:
        case 'none':
            return None
        case 'scm':
            # y's column rank-matched to the score, the last draw, as under 0.3.0
            y = apply(covariates.score, planned['y'], rng)  # (n,)
            metadata = {'y': planned['y']}
            return ScmOutcome(columns={'y': y}, metadata=metadata, score=covariates.score)
        case _:
            return run(form, covariates, rng)


# -------------------------------------------------------------------------------------
# a resampled outcome: the key -> the model's outcome on given covariates
# -------------------------------------------------------------------------------------


def drawOutcome(seed: int, index: int, covariates: Covariates, model: Model) -> Outcome:
    """The outcome of a model on given covariates, keyed like a dataset."""
    rng = _seedGenerator(seed, index)
    return run(model, covariates, rng)
