"""summary.py measures a table into its summary vector, a real table like a generated one.

pipeline:
---------
the covariates and the outcome
-> the sizes: rows, design columns, columns, type shares, levels   _measureSizes
-> a statistic over one type's columns as its median and maximum   _aggregate
-> each covariate's shape statistics, by its type                  _measureShape, _measureColumn
   (a coded column's level shares, another column's moments)       _measureLevels, _measureMarginal
-> each covariate coded for its relations                          _codeColumn, _rankAverage
-> each covariate's largest relation with another one              _measureRelations, _relatePair
-> the condition index of the design, each covariate as its span   _measureConditionIndex
-> the leverage: Mardia's kurtosis of the normal scores            _measureLeverage, _scoreNormal
-> y's shape statistics and its largest relation with a covariate  _measureOutcome
-> the vector in AXES' order, NaN where the table lacks an axis    measure
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist
from typing import get_args

import numpy as np

from tabdgp.column import Column, ColumnType, MarginalType
from tabdgp.covariates import Covariates
from tabdgp.outcome import Outcome
from tabdgp.scm import LeveledType

TYPES: tuple[ColumnType, ...] = get_args(ColumnType)

# a continuous or count column's shape statistics
MARGINAL_STATISTICS = [
    'skewness',
    'excess_kurtosis',
    'outlier_share',
    'zero_share',
    'distinct_share',
    'mode_share',
    'bound_excess',
]
# a column's shape statistics by its type
STATISTICS: dict[ColumnType, list[str]] = {
    'continuous': MARGINAL_STATISTICS,
    'count': [*MARGINAL_STATISTICS, 'dispersion'],
    # a binary column's largest level share is its majority, whose maximum finds a rare indicator
    'binary': ['largest_level_share'],
    'categorical': ['largest_level_share', 'smallest_level_share'],
    'ordered': ['largest_level_share', 'smallest_level_share'],
}
# a signed statistic's maximum is its largest absolute value
SIGNED = ['skewness', 'excess_kurtosis']

# =====================================================================================
# the axes
# =====================================================================================


def _nameAxes() -> tuple[str, ...]:
    """The axes' names in the vector's order, block by block."""
    axes = ['rows', 'design_columns', 'columns']
    for type in TYPES:
        axes.append(f'{type}_share')
    axes.extend(['levels_median', 'levels_max'])
    for type, statistics in STATISTICS.items():
        for statistic in statistics:
            axes.extend([f'{type}_{statistic}_median', f'{type}_{statistic}_max'])
    for type in TYPES:
        axes.extend([f'{type}_relation_median', f'{type}_relation_max'])
    axes.extend(['condition_index', 'leverage'])
    for type, statistics in STATISTICS.items():
        for statistic in statistics:
            axes.append(f'y_{type}_{statistic}')
    axes.append('y_relation')
    return tuple(axes)


AXES = _nameAxes()

# =====================================================================================
# the measure
# =====================================================================================


def measure(covariates: Covariates, outcome: Outcome | None) -> np.ndarray:
    """The table's summary vector in AXES' order, NaN where the table cannot show an axis."""
    # the grouping factors are left out
    metadata: dict[str, Column] = {}
    for name, column in covariates.metadata.items():
        if column.role == 'covariate':
            metadata[name] = column
    columns = {name: covariates.columns[name] for name in metadata}
    coded = [_codeColumn(columns[name], column) for name, column in metadata.items()]
    measured = {
        **_measureSizes(covariates.rows, metadata),
        **_measureShape(columns, metadata),
        **_measureRelations(coded, metadata),
        'condition_index': _measureConditionIndex(covariates),
        **_measureLeverage(columns, metadata),
    }
    if outcome is not None:
        measured.update(_measureOutcome(outcome, coded))
    vector = np.full(len(AXES), math.nan)  # (p,)
    for axis, value in measured.items():
        vector[AXES.index(axis)] = value
    return vector


# -------------------------------------------------------------------------------------
# the sizes and the shape
# -------------------------------------------------------------------------------------


def _measureSizes(rows: int, metadata: dict[str, Column]) -> dict[str, float]:
    sizes: dict[str, float] = {
        'rows': rows,
        'design_columns': sum(column.design_columns for column in metadata.values()),
        'columns': len(metadata),
    }
    types = [column.type for column in metadata.values()]
    for type in TYPES:
        sizes[f'{type}_share'] = types.count(type) / len(types)
    # binary's k = 2 is its type, so the levels are read over the categorical and ordered columns
    levels = [column.levels for column in metadata.values() if column.type in get_args(LeveledType)]
    sizes.update(_aggregate('levels', levels))
    return sizes


def _aggregate(name: str, values: list[float], signed: bool = False) -> dict[str, float]:
    """A statistic over one type's columns: its median and its maximum, none without a column."""
    if not values:
        return {}
    largest = np.max(np.abs(values)) if signed else np.max(values)
    return {f'{name}_median': float(np.median(values)), f'{name}_max': float(largest)}


def _measureShape(columns: dict[str, np.ndarray], metadata: dict[str, Column]) -> dict[str, float]:
    """Each type's shape statistics over its columns."""
    shape: dict[str, float] = {}
    for type, statistics in STATISTICS.items():
        measured: list[dict[str, float]] = []
        for name, column in metadata.items():
            if column.type == type:
                measured.append(_measureColumn(columns[name], column))
        for statistic in statistics:
            values = [column_statistics[statistic] for column_statistics in measured]
            shape.update(_aggregate(f'{type}_{statistic}', values, statistic in SIGNED))
    return shape


def _measureColumn(values: np.ndarray, column: Column) -> dict[str, float]:
    if column.levels is None:
        return _measureMarginal(values, column.type)
    return _measureLevels(values, column.levels)


def _measureLevels(codes: np.ndarray, levels: int) -> dict[str, float]:
    shares = np.bincount(codes, minlength=levels) / len(codes)  # (k,)
    return {'largest_level_share': float(shares.max()), 'smallest_level_share': float(shares.min())}


def _measureMarginal(values: np.ndarray, type: ColumnType) -> dict[str, float]:
    """A continuous or count column's moments, tails, ties and pile at a bound,
    a count's dispersion.
    """
    values = values.astype(np.float64)  # (n,)
    rows = len(values)
    mean = values.mean()
    deviations = values - mean  # (n,)
    # the population's central moments
    m2 = np.mean(deviations**2)
    m3 = np.mean(deviations**3)
    m4 = np.mean(deviations**4)
    # Tukey's fences at numpy's default quartiles (R's type 7)
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    outside = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)  # (n,)
    _, counts = np.unique(values, return_counts=True)  # (u,) from the smallest value up
    # the larger end's share beyond the 1/u of a typical value, so that ties alone leave no pile
    end_share = int(max(counts[0], counts[-1])) / rows
    statistics = {
        'skewness': float(m3 / m2**1.5),
        'excess_kurtosis': float(m4 / m2**2 - 3),
        'outlier_share': float(outside.mean()),
        'zero_share': float(np.mean(values == 0)),
        'distinct_share': len(counts) / rows,
        'mode_share': int(counts.max()) / rows,
        'bound_excess': max(end_share - 1 / len(counts), 0.0),
    }
    if type == 'count':
        statistics['dispersion'] = float(m2 / mean)
    return statistics


# -------------------------------------------------------------------------------------
# the relations
# -------------------------------------------------------------------------------------


def _codeColumn(values: np.ndarray, column: Column) -> np.ndarray:
    """A categorical column one-hot over its present levels, another by its z-scored ranks."""
    if column.type == 'categorical':
        _, codes = np.unique(values, return_inverse=True)  # (n,)
        return np.eye(codes.max() + 1)[codes]  # (n, k)
    ranks = _rankAverage(values)  # (n,)
    return (ranks - ranks.mean()) / ranks.std()


def _rankAverage(values: np.ndarray) -> np.ndarray:
    """The ranks 1 to n, tied values at their average rank."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)  # (n,), (u,)
    last = np.cumsum(counts)  # (u,) each distinct value's highest rank
    return (last - (counts - 1) / 2)[inverse]  # (n,)


def _measureRelations(coded: list[np.ndarray], metadata: dict[str, Column]) -> dict[str, float]:
    """Each type's relations over its columns, a column's relation its largest with another one."""
    x = len(coded)
    # a single covariate relates to no other
    if x < 2:
        return {}
    relations = np.full((x, x), math.nan)  # (x, x), NaN on the diagonal
    for i, j in combinations(range(x), 2):
        relation = _relatePair(coded[i], coded[j])
        relations[i, j] = relation
        relations[j, i] = relation
    largest = np.nanmax(relations, axis=1)  # (x,)
    measured: dict[str, float] = {}
    for type in TYPES:
        values: list[float] = []
        for i, column in enumerate(metadata.values()):
            if column.type == type:
                values.append(float(largest[i]))
        measured.update(_aggregate(f'{type}_relation', values))
    return measured


def _relatePair(a: np.ndarray, b: np.ndarray) -> float:
    """|ρ| of two ranked columns,
    the correlation ratio η of a ranked column's ranks over a categorical's levels,
    and Cramér's V of two categoricals: each |ρ| at two levels.
    """
    rows = len(a)
    match a.ndim, b.ndim:
        case (1, 1):
            return abs(float(np.mean(a * b)))
        case (2, 1) | (1, 2):
            # η² is the share of the ranks' variance (1 after the z-score) between the levels
            levels, ranks = (a, b) if a.ndim == 2 else (b, a)
            sums = levels.T @ ranks  # (k,)
            between = np.sum(sums**2 / levels.sum(axis=0)) / rows
            return math.sqrt(between)
        case _:
            # χ²/n from the contingency table, scaled by its fewer levels less one (Cramér 1946)
            table = a.T @ b  # (k_a, k_b)
            expected = np.outer(table.sum(axis=1), table.sum(axis=0))  # (k_a, k_b), times n
            # independent columns (a crossed design) leave rounding error below 0
            phi_squared = max(np.sum(table**2 / expected) - 1, 0.0)
            fewer = min(table.shape) - 1
            return math.sqrt(phi_squared / fewer)


def _measureConditionIndex(covariates: Covariates) -> float:
    """The design's largest over its smallest singular value,
    each covariate's columns centered and replaced by an orthonormal basis of their span.
    """
    # the basis makes the index blind to a column's scale and to which level the dummies drop
    design = covariates.toDesign()
    bases: list[np.ndarray] = []
    for positions in design.positions.values():
        block = design.matrix[:, positions]  # (n, size)
        basis, _ = np.linalg.qr(block - block.mean(axis=0))  # (n, size)
        bases.append(basis)
    spans = np.hstack(bases)  # (n, d − 1)
    singular = np.linalg.svd(spans, compute_uv=False)  # (d − 1,)
    # a rank-deficient design (e.g. covariates at a factor of few groups) has no finite index
    if singular[-1] <= _findTolerance(singular, spans.shape):
        return math.inf
    return float(singular[0] / singular[-1])


def _findTolerance(singular: np.ndarray, shape: tuple[int, ...]) -> float:
    """The tolerance of numpy's matrix_rank: a singular value at or below it counts as zero."""
    return float(singular[0] * max(shape) * np.finfo(np.float64).eps)


def _measureLeverage(
    columns: dict[str, np.ndarray], metadata: dict[str, Column]
) -> dict[str, float]:
    """Mardia's kurtosis of the continuous and count columns' normal scores,
    over its normal value r(r + 2) at the scores' rank r:
    how far a few rows lie from the rest jointly.
    """
    # the normal scores leave out each column's marginal, which its own statistics measure
    score_columns: list[np.ndarray] = []
    for name, column in metadata.items():
        if column.type in get_args(MarginalType):
            score_columns.append(_scoreNormal(columns[name]))
    # a single column has no joint extremes
    if len(score_columns) < 2:
        return {}
    scores = np.stack(score_columns, axis=1)  # (n, x)
    centered = scores - scores.mean(axis=0)  # (n, x)
    basis, singular, _ = np.linalg.svd(centered, full_matrices=False)  # (n, x), (x,)
    # a monotone function of another column has its scores and adds no rank
    basis = basis[:, singular > _findTolerance(singular, centered.shape)]  # (n, r)
    rank = basis.shape[1]
    # each row's squared Mahalanobis distance under the covariance by n (Mardia 1970, b₂,ₚ)
    distances = len(scores) * np.sum(basis**2, axis=1)  # (n,)
    return {'leverage': float(np.mean(distances**2) / (rank * (rank + 2)))}


def _scoreNormal(values: np.ndarray) -> np.ndarray:
    """Each value's normal score Φ⁻¹((rank − 1/2) / n), tied values at their average rank."""
    ranks = _rankAverage(values)  # (n,)
    distinct, inverse = np.unique(ranks, return_inverse=True)  # (u,), (n,)
    quantile = NormalDist().inv_cdf
    scores = [quantile((rank - 0.5) / len(values)) for rank in distinct.tolist()]
    return np.array(scores)[inverse]  # (n,)


# -------------------------------------------------------------------------------------
# y
# -------------------------------------------------------------------------------------


def _measureOutcome(outcome: Outcome, coded: list[np.ndarray]) -> dict[str, float]:
    """y's shape statistics, once, and its largest relation with a covariate."""
    values = outcome.columns['y']  # (n,)
    column = outcome.metadata['y']
    statistics = _measureColumn(values, column)
    measured: dict[str, float] = {}
    for statistic in STATISTICS[column.type]:
        measured[f'y_{column.type}_{statistic}'] = statistics[statistic]
    y = _codeColumn(values, column)
    measured['y_relation'] = max(_relatePair(y, covariate) for covariate in coded)
    return measured
