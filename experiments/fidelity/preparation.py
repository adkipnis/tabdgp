"""preparation.py prepares a real table for its measure: its covariates and y,
or the reason it is left out.

A real table arrives as its source stores it: with row labels, constant columns, missing
values and a response among its columns. `Covariates.fromPandas` refuses most of these,
so the columns and rows it cannot read are dropped first.
Each rule was read off mixedbench 1.0.15's flat sample (#93).

pipeline:
---------
a table, its response and the type mixedbench read for it
-> string names, a blank label as a missing value              _blankToMissing
-> without the covariates missing in most rows                 _dropSparse
-> without the constant columns and the row labels             _dropUninformative
-> its complete rows, the columns checked once more
-> y in its type, its levels read off its values               _readOutcome
-> the covariates, each type inferred                          _castLabels, Covariates.fromPandas
-> the prepared table, or the reason it is left out            prepare
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, ColumnType
from tabdgp.covariates import Covariates
from tabdgp.outcome import Outcome

# =====================================================================================
# the prepared table
# =====================================================================================

ResponseType = Literal['continuous', 'count', 'binomial', 'binary', 'ordinal', 'categorical']
Reason = Literal['no complete row', 'no covariate']
# y's column type by the type mixedbench read:
# tabdgp has no binomial column, so its successes enter as a count
OUTCOME_TYPES: dict[ResponseType, ColumnType] = {
    'continuous': 'continuous',
    'count': 'count',
    'binomial': 'count',
    'binary': 'binary',
    'ordinal': 'ordered',
    'categorical': 'categorical',
}
# a covariate missing in a larger share of the rows is dropped before the rows are:
# a column that is mostly missing costs more rows than it tells
MOSTLY_MISSING = 0.5


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Prepared:
    """A real table as its measure reads it, or the reason it is left out."""

    covariates: Covariates | None
    outcome: Outcome | None  # None too without a response, or with a constant one
    reason: Reason | None


# =====================================================================================
# the preparation
# =====================================================================================


def prepare(
    table: pd.DataFrame, response: str | None, response_type: ResponseType | None
) -> Prepared:
    """The table's covariates and y, or the reason the rules leave it without rows or covariates."""
    table = table.rename(columns=str)
    table = _blankToMissing(table)
    table = _dropSparse(table, response)
    # before the rows, so that a row label's missing value costs no row
    table = _dropUninformative(table, response)
    table = table.dropna()
    if len(table) == 0:
        return Prepared(covariates=None, outcome=None, reason='no complete row')
    # again, since a column may be constant over the rows that are left
    table = _dropUninformative(table, response)

    outcome = None
    if response is not None:
        outcome = _readOutcome(table[response], response_type)
        table = table.drop(columns=response)
    if table.shape[1] == 0:
        return Prepared(covariates=None, outcome=None, reason='no covariate')
    table = _castLabels(table)
    table = table.rename(columns={'y': 'y_covariate'})  # y names the outcome
    covariates = Covariates.fromPandas(table.reset_index(drop=True), factors=[])
    return Prepared(covariates=covariates, outcome=outcome, reason=None)


# -------------------------------------------------------------------------------------
# a table -> its readable columns and complete rows
# -------------------------------------------------------------------------------------


def _isLabel(values: pd.Series) -> bool:
    """Whether a column holds labels: a factor, strings or objects."""
    dtype = values.dtype
    factor = isinstance(dtype, pd.CategoricalDtype)
    return factor or pd.api.types.is_string_dtype(dtype) or dtype == object


def _blankToMissing(table: pd.DataFrame) -> pd.DataFrame:
    """A label of whitespace alone read as a missing value (R's factor level '')."""
    cleaned = {}
    for name in table.columns:
        values = table[name]
        if not _isLabel(values):
            continue
        if isinstance(values.dtype, pd.CategoricalDtype):
            blank = [level for level in values.cat.categories if not str(level).strip()]
            cleaned[name] = values.cat.remove_categories(blank)
        else:
            blank = values.astype(str).str.strip() == ''
            cleaned[name] = values.mask(blank)
    return table.assign(**cleaned)


def _dropSparse(table: pd.DataFrame, response: str | None) -> pd.DataFrame:
    """The table without the covariates that are missing in most rows. The response stays."""
    kept = []
    for name in table.columns:
        missing_share = table[name].isna().mean()
        if name == response or missing_share <= MOSTLY_MISSING:
            kept.append(name)
    return table[kept]


def _dropUninformative(table: pd.DataFrame, response: str | None) -> pd.DataFrame:
    """The table without its constant covariates and its row labels
    (a label column whose levels have fewer than two rows on average: a name, a word, a date).
    """
    kept = []
    for name in table.columns:
        present = table[name].dropna()
        distinct = _countDistinct(present)
        constant = distinct < 2
        row_label = _isLabel(present) and distinct > len(present) / 2
        if name == response or not (constant or row_label):
            kept.append(name)
    return table[kept]


def _countDistinct(values: pd.Series) -> int:
    """A column's distinct values, numbers counted in float32, as the covariates store them."""
    dtype = values.dtype
    if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
        stored = values.to_numpy(dtype=np.float32)  # (n,)
        return len(np.unique(stored))
    return values.nunique()


def _castLabels(table: pd.DataFrame) -> pd.DataFrame:
    """An object column without strings (a date per row) cast to its labels."""
    labels = {}
    for name in table.columns:
        values = table[name]
        if values.dtype == object and not pd.api.types.is_string_dtype(values):
            labels[name] = values.astype(str)
    return table.assign(**labels)


# -------------------------------------------------------------------------------------
# the response -> y
# -------------------------------------------------------------------------------------


def _readOutcome(values: pd.Series, response_type: ResponseType) -> Outcome | None:
    """y in the column type of its response type, binary at two values, None if constant."""
    type = OUTCOME_TYPES[response_type]
    # a number as y stores it, so that its levels are counted on the stored values
    if type == 'continuous':
        stored = values.to_numpy(dtype=np.float32)  # (n,)
    elif type == 'count':
        stored = values.to_numpy(dtype=np.int64)  # (n,)
        if (stored != values.to_numpy()).any():
            raise ValueError(f'the count response {values.name} holds a fraction')
    else:
        stored = values
    # the codes follow the sorted values, or a factor's own order
    codes, levels = pd.factorize(stored, sort=True)  # (n,), (k,)
    k = len(levels)
    if k < 2:
        return None
    if k == 2:
        column = Column(type='binary', levels=2, role='outcome')
    elif type in ['categorical', 'ordered']:
        column = Column(type=type, levels=k, role='outcome')
    else:
        column = Column(type=type, role='outcome')
    y = stored if column.levels is None else codes.astype(column.dtype)
    return Outcome(columns={'y': y}, metadata={'y': column})
