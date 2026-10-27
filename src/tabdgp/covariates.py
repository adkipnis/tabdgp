"""covariates.py has utility functions to prepare covariates for a model-based outcome.

The covariates are the observed columns, the grouping, and their metadata.
If available, the SCM's z-scored node values are also included.
Alternatively, this module can also take a ready external tabular dataset.

pipeline:
---------
a table, its grouping factors and optionally metadata
-> every observed column, a value in each row        _checkValues
-> the metadata, either declared by Column          _validateMetadata
   or inferred: from their codes and dtype                     _inferMetadata
-> the factors' codes, checked against their levels and nesting       _readGrouping
-> each covariate's values in its type's dtype    _readValues
-> the covariates                                    Covariates.fromPandas
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import ConfigDict, TypeAdapter
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping

# =====================================================================================
# the checks
# =====================================================================================

def checkOneLength(arrays: list[np.ndarray | None], of: str) -> None:
    lengths: set[int] = {len(array) for array in arrays if array is not None}
    if len(lengths) > 1:
        raise ValueError(f'the arrays of {of} have the lengths {sorted(lengths)}, not one')


def checkNames(columns: dict[str, np.ndarray], metadata: dict[str, Column]) -> None:
    if columns.keys() != metadata.keys():
        raise ValueError(f'the columns {list(columns)} have metadata {list(metadata)}')


# TODO: use some other name instead of `form`, form is semantically meaningless here
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
    METADATA: ClassVar[TypeAdapter] = TypeAdapter(dict[str, Column]) # TODO: is this really useful? can we simplify this or use it in-line only?
    RESERVED: ClassVar[dict[str, str]] = {
        'y': "the outcome's name",
        'intercept': "the design matrix's first term",
        'fixed': "the fixed effects' block",
    }

    def __post_init__(self) -> None:
        # check that every array has one length, each column its metadata, and none a reserved name
        checkOneLength([*self.columns.values(), self.score], 'the covariates')
        checkNames(self.columns, self.metadata)
        # TODO: rewrite this with match-case instead, RESERVED is not used anywhere else
        for name, reason in self.RESERVED.items():
            if name in self.columns:
                raise ValueError(f'a column cannot be named {name!r}, {reason}: rename it')

    @property
    # TODO: smells like overengineering, where is this useful?
    def rows(self) -> int:
        """n, the number of observations."""
        for array in [*self.columns.values(), self.score]:
            if array is not None:
                return len(array)
        raise ValueError('covariates without a column or a score have no rows')

    @classmethod
    def fromPandas(
        cls,
        table: pd.DataFrame,
        factors: list[str] | None = None,
        metadata: dict[str, Column | dict[str, Any]] | None = None,
    ) -> Covariates:
        """The covariates of a supplied table, each column read by its metadata and checked against it."""
        if (factors is None) == (metadata is None):
            raise ValueError('name the grouping factors to infer the metadata, or declare it')
        if metadata is None:
            cls._checkValues(table)
            metadata = cls._inferMetadata(table, factors)
        else:
            metadata = cls._validateMetadata(table, metadata)
            observed = [name for name, column in metadata.items() if column.role != 'outcome']
            cls._checkValues(table[observed])
        grouping = cls._readGrouping(table, metadata)
        columns = dict(grouping.codes)
        for name, column in metadata.items():
            if column.role == 'covariate':
                columns[name] = cls._readValues(table[name], column)
        observed_metadata = {name: metadata[name] for name in columns}
        return cls(columns=columns, metadata=observed_metadata, grouping=grouping, score=None)

    @classmethod
    def _validateMetadata(
        cls, table: pd.DataFrame, metadata: dict[str, Column | dict[str, Any]]
    ) -> dict[str, Column]:
        """The declared metadata as Columns (one per column)."""
        declared = cls.METADATA.validate_python(metadata)
        if declared.keys() != set(table.columns):
            raise ValueError(
                f'the table has the columns {list(table.columns)}, the metadata {list(declared)}'
            )
        return declared

    @staticmethod
    def _checkValues(table: pd.DataFrame) -> None:
        """Checks that every column has a value in every row and at least two distinct ones."""
        for name in table.columns:
            values = table[name]
            if values.isna().any():
                raise ValueError(f'the column {name} has a missing value')
            if values.nunique() < 2:
                raise ValueError(
                    f'the column {name} takes one value only, '
                    "so no fit tells its effect from the intercept's: drop it"
                )

    @classmethod
    def _inferMetadata(cls, table: pd.DataFrame, factors: list[str]) -> dict[str, Column]:
        """The metadata inferred from `factors` codes and dtype."""
        codes = {}
        levels = {}
        for name in factors:
            codes[name], levels[name] = cls._codeLevels(table[name])
        metadata = {}
        for name in factors:
            nests_in = cls._findSuperior(name, codes, levels)
            metadata[name] = Column(
                type='categorical', levels=levels[name], role='grouping_factor', nests_in=nests_in
            )
        for name in table.columns:
            if name not in factors:
                metadata[name] = cls._inferCovariate(table[name])
        return metadata

    @staticmethod
    def _codeLevels(values: pd.Series) -> tuple[np.ndarray, int]:
        """A column's codes and it's levels k."""
        if not isinstance(values.dtype, pd.CategoricalDtype):
            values = values.astype('category')  # the categories are the sorted values
        present = values.cat.remove_unused_categories()
        codes = present.cat.codes.to_numpy(dtype=np.int64)  # (n,)
        return codes, len(present.cat.categories)

    @classmethod
    def _findSuperior(
        cls, factor: str, codes: dict[str, np.ndarray], levels: dict[str, int]
    ) -> str | None:
        """Infer the grouping factor's immediate superior based on relative group counts."""
        superior = None
        for other, other_codes in codes.items():
            fewer = levels[other] < levels[factor]
            if not fewer or not cls._nests(codes[factor], other_codes):
                continue
            if superior is None or levels[other] > levels[superior]:
                superior = other
        return superior

    @staticmethod
    def _nests(subordinate: np.ndarray, superior: np.ndarray) -> bool:
        """Check if each group of the subordinate factor lies within one group of the superior."""
        pairs = np.unique(np.stack([subordinate, superior], axis=1), axis=0)  # (u, 2)
        return len(pairs) == np.unique(subordinate).size

    @staticmethod
    def _inferCovariate(values: pd.Series) -> Column:
        dtype = values.dtype
        categorical = isinstance(dtype, pd.CategoricalDtype)
        k = values.nunique()
        if k == 2:
            return Column(type='binary', levels=2)
        if categorical and dtype.ordered:
            return Column(type='ordered', levels=k)
        if categorical or pd.api.types.is_string_dtype(values):
            return Column(type='categorical', levels=k)
        if pd.api.types.is_integer_dtype(dtype):
            if (values < 0).any():
                raise ValueError(
                    f'the integer column {values.name} has a negative value, so it is no count: '
                    'cast it to float if it is continuous, '
                    'or drop the rows where a negative value marks a missing one'
                )
            return Column(type='count')
        if pd.api.types.is_float_dtype(dtype):
            return Column(type='continuous')
        raise ValueError(
            f'the column {values.name} has the dtype {dtype}, which states no column type: '
            'cast it to float, int, bool, str or category'
        )

    @classmethod
    def _readGrouping(cls, table: pd.DataFrame, metadata: dict[str, Column]) -> Grouping:
        """TODO: rewrite in the same voice and detail as the other docstrings"""
        factors = {}
        codes = {}
        for name, column in metadata.items():
            if column.role == 'grouping_factor':
                factors[name] = column
                codes[name] = cls._readCodes(table[name], column)
        names = list(factors)
        for i, name in enumerate(names):
            superior = factors[name].nests_in
            if superior is not None and superior not in factors:
                raise ValueError(
                    f'the factor {name} nests in {superior}, which is no grouping factor'
                )
            if superior is not None and not cls._nests(codes[name], codes[superior]):
                raise ValueError(f'the factor {name} has a group across two groups of {superior}')
            for other in names[i + 1 :]:
                within = cls._nests(codes[name], codes[other])
                around = cls._nests(codes[other], codes[name])
                if within and around:
                    raise ValueError(
                        f'the factors {name} and {other} group the rows alike, '
                        'one factor named twice'
                    )
        return Grouping(metadata=factors, codes=codes)

    @classmethod
    def _readCodes(cls, values: pd.Series, column: Column) -> np.ndarray:
        """A column's codes (checked against its k levels)."""
        codes, k = cls._codeLevels(values)
        if k != column.levels:
            raise ValueError(
                f'the column {values.name} has {k} distinct values, '
                f'its metadata {column.levels} levels'
            )
        return codes

    @classmethod
    def _readValues(cls, values: pd.Series, column: Column) -> np.ndarray:
        """A covariate's values in its type's dtype (checked against its type):
        one of its k levels if coded, a number if continuous, a non-negative integer if a count.
        """
        if column.levels is not None:
            return cls._readCodes(values, column)
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise ValueError(
                f'the {column.type} column {values.name} holds {values.dtype}, not numbers'
            )
        if column.type == 'continuous':
            return values.to_numpy(dtype=column.dtype)
        counts = values.to_numpy(dtype=column.dtype)  # (n,)
        if (counts < 0).any() or (counts != values.to_numpy()).any():
            raise ValueError(f'the count column {values.name} holds a value that is no count')
        return counts
