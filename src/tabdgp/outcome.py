"""outcome.py defines what a form reads and what it returns.

A form reads the covariates: the observed columns at the rows, their metadata,
the grouping and the score (the outcome node's z-scored value in the SCM, None without one).

pipeline:
---------
the covariates at the rows, their metadata, the grouping and the score  Covariates
-> the columns a form writes, their metadata and its truth              Outcome
-> a model's outcome, checked at the covariates' rows                   run
"""

from __future__ import annotations

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping
from tabdgp.hyperprior import Model

# =====================================================================================
# the checks
# =====================================================================================
# the covariates, an outcome and a model's result each hold columns at one set of rows.


def _checkOneLength(arrays: list[np.ndarray | None], of: str) -> None:
    lengths = {len(array) for array in arrays if array is not None}
    if len(lengths) > 1:
        raise ValueError(f'the arrays of {of} have the lengths {sorted(lengths)}, not one')


def _checkNames(columns: dict[str, np.ndarray], metadata: dict[str, Column]) -> None:
    if columns.keys() != metadata.keys():
        raise ValueError(f'the columns {list(columns)} have metadata {list(metadata)}')


# =====================================================================================
# what a form reads
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Covariates:
    """The observed columns a form reads: the grouping factors' codes and the covariates."""

    columns: dict[str, np.ndarray]  # (n,) each
    metadata: dict[str, Column]
    grouping: Grouping
    score: np.ndarray | None  # (n,) float32, None for a table without an SCM

    def __post_init__(self) -> None:
        # check that every array has one length, each column its metadata, and none is y
        _checkOneLength([*self.columns.values(), self.score], 'the covariates')
        _checkNames(self.columns, self.metadata)
        if 'y' in self.columns:
            raise ValueError("a covariate cannot be named 'y', the outcome's name")


# =====================================================================================
# what a form returns
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Outcome:
    """The columns a form writes (y today) and their metadata.
    A form's own subclass adds its truth.
    """

    columns: dict[str, np.ndarray]  # (n,) each, in its type's dtype
    metadata: dict[str, Column]

    def __post_init__(self) -> None:
        # check that each column has its metadata, the role outcome, its type's dtype and one length
        _checkNames(self.columns, self.metadata)
        for name, column in self.metadata.items():
            if column.role != 'outcome':
                raise ValueError(f'{name} has the role {column.role}, not outcome')
            dtype = self.columns[name].dtype
            if dtype != column.dtype:
                raise ValueError(
                    f'the {column.type} column {name} is {dtype}, not {column.dtype.__name__}'
                )
        _checkOneLength(list(self.columns.values()), 'the outcome')


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ScmOutcome(Outcome):
    """The scm form's y and its truth, the score it rank-matched onto y's marginal."""

    score: np.ndarray  # (n,) float32

    def __post_init__(self) -> None:
        super().__post_init__()
        _checkOneLength([*self.columns.values(), self.score], 'the outcome')


# =====================================================================================
# the model's call
# =====================================================================================


def run(model: Model, covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """Calls a model on the covariates and the generator.
    What it returns must be an Outcome at the covariates' rows; a failure names the model.
    """
    outcome = model(covariates, rng)
    if not isinstance(outcome, Outcome):
        raise TypeError(f'{model!r} returned a {type(outcome).__name__}, not an Outcome')
    arrays = [*covariates.columns.values(), covariates.score, *outcome.columns.values()]
    _checkOneLength(arrays, f'{model!r} and its covariates')
    # the dataset holds both, so a shared name would replace the covariate
    shared = covariates.columns.keys() & outcome.columns.keys()
    if shared:
        raise ValueError(f'{model!r} wrote the covariates {sorted(shared)}')
    return outcome
