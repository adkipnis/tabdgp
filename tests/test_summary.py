"""The summary module's seam: measure, a table's covariates and outcome as one vector named by AXES,
each statistic worked by hand on a table small enough to check by eye.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from tabdgp.column import Column, Grouping
from tabdgp.covariates import Covariates
from tabdgp.dataset import Dataset, drawDataset
from tabdgp.outcome import Outcome
from tabdgp.scm import Scm
from tabdgp.summary import AXES, measure
from tests.fixtures.build import groupingOf


def _flatCovariates(columns: dict[str, tuple[Column, list[float]]]) -> Covariates:
    """Flat covariates of hand-written columns, each in its type's dtype."""
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Column] = {}
    for name, (column, values) in columns.items():
        arrays[name] = np.array(values, dtype=column.dtype)  # (n,)
        metadata[name] = column
    return Covariates(columns=arrays, metadata=metadata, grouping=groupingOf([]), y_score=None)


def _measured(covariates: Covariates, outcome: Outcome | None) -> dict[str, float]:
    """The summary vector by axis name."""
    return dict(zip(AXES, measure(covariates, outcome).tolist()))


def test_sizes_count_the_rows_the_design_columns_the_columns_and_the_levels() -> None:
    """A coded column counts its k − 1 dummies as design columns.
    The levels are read over the categorical and ordered columns, since binary's k = 2 is its type.
    """
    covariates = _flatCovariates(
        {
            'x0': (Column(type='continuous'), [0.0, 1.0, 2.0, 3.0]),
            'x1': (Column(type='binary', levels=2), [0, 1, 0, 1]),
            'x2': (Column(type='categorical', levels=3), [0, 1, 2, 0]),
            'x3': (Column(type='ordered', levels=4), [0, 1, 2, 3]),
        }
    )
    expected = {
        'rows': 4,
        'design_columns': 7,
        'columns': 4,
        'continuous_share': 0.25,
        'count_share': 0.0,
        'binary_share': 0.25,
        'categorical_share': 0.25,
        'ordered_share': 0.25,
        'levels_median': 3.5,
        'levels_max': 4,
    }
    measured = _measured(covariates, None)
    assert {axis: measured[axis] for axis in expected} == expected


# central moments of [0, 0, 1, 2, 7] about its mean 2, worked by hand
M2, M3, M4 = 34 / 5, 108 / 5, 658 / 5
SKEWNESS = M3 / M2**1.5
EXCESS_KURTOSIS = M4 / M2**2 - 3  # −0.15, so its maximum is its absolute value


def test_a_continuous_or_count_column_enters_by_its_moments_tails_ties_and_dispersion() -> None:
    """x1 mirrors x0, so the continuous skewness has median 0 and the absolute value as maximum.
    Quartiles 0 and 2 put the fences at −3 and 5, so 7 is x0's one outlier, and 0 is x1's.
    The count's dispersion is its variance 34/5 over its mean 2.
    """
    covariates = _flatCovariates(
        {
            'x0': (Column(type='continuous'), [0.0, 0.0, 1.0, 2.0, 7.0]),
            'x1': (Column(type='continuous'), [7.0, 7.0, 6.0, 5.0, 0.0]),
            'x2': (Column(type='count'), [0, 0, 1, 2, 7]),
        }
    )
    expected = {
        'continuous_skewness_median': 0.0,
        'continuous_skewness_max': SKEWNESS,
        'continuous_excess_kurtosis_median': EXCESS_KURTOSIS,
        'continuous_excess_kurtosis_max': -EXCESS_KURTOSIS,
        'continuous_outlier_share_median': 0.2,
        'continuous_outlier_share_max': 0.2,
        'continuous_zero_share_median': 0.3,
        'continuous_zero_share_max': 0.4,
        'continuous_distinct_share_median': 0.8,
        'continuous_distinct_share_max': 0.8,
        'continuous_mode_share_median': 0.4,
        'continuous_mode_share_max': 0.4,
        'count_skewness_median': SKEWNESS,
        'count_skewness_max': SKEWNESS,
        'count_excess_kurtosis_median': EXCESS_KURTOSIS,
        'count_excess_kurtosis_max': -EXCESS_KURTOSIS,
        'count_outlier_share_median': 0.2,
        'count_outlier_share_max': 0.2,
        'count_zero_share_median': 0.4,
        'count_zero_share_max': 0.4,
        'count_distinct_share_median': 0.8,
        'count_distinct_share_max': 0.8,
        'count_mode_share_median': 0.4,
        'count_mode_share_max': 0.4,
        'count_dispersion_median': 3.4,
        'count_dispersion_max': 3.4,
    }
    measured = _measured(covariates, None)
    assert {axis: measured[axis] for axis in expected} == pytest.approx(expected)


def test_a_coded_column_enters_by_its_largest_and_smallest_level_share() -> None:
    """A binary column enters by its largest level share alone, its majority,
    so the maximum over several binary columns finds the rarest indicator.
    """
    covariates = _flatCovariates(
        {
            'x0': (Column(type='binary', levels=2), [0, 0, 0, 1]),
            'x1': (Column(type='binary', levels=2), [0, 1, 0, 1]),
            'x2': (Column(type='categorical', levels=3), [2, 0, 1, 2]),
            'x3': (Column(type='ordered', levels=4), [0, 1, 2, 3]),
        }
    )
    expected = {
        'binary_largest_level_share_median': 0.625,
        'binary_largest_level_share_max': 0.75,
        'categorical_largest_level_share_median': 0.5,
        'categorical_largest_level_share_max': 0.5,
        'categorical_smallest_level_share_median': 0.25,
        'categorical_smallest_level_share_max': 0.25,
        'ordered_largest_level_share_median': 0.25,
        'ordered_largest_level_share_max': 0.25,
        'ordered_smallest_level_share_median': 0.25,
        'ordered_smallest_level_share_max': 0.25,
    }
    measured = _measured(covariates, None)
    assert {axis: measured[axis] for axis in expected} == expected


CONTINUOUS = Column(type='continuous')
BINARY = Column(type='binary', levels=2)
CATEGORICAL = Column(type='categorical', levels=3)

# worked by hand on four rows; a column's relation is its largest with another column
RELATIONS = [
    pytest.param(
        {'x0': (CONTINUOUS, [1.0, 2.0, 3.0, 4.0]), 'x1': (CONTINUOUS, [1.0, 2.0, 4.0, 3.0])},
        {'continuous_relation_median': 0.8, 'continuous_relation_max': 0.8},
        id='two ranked columns relate by |ρ|, 1 − 6 · 2/60',
    ),
    pytest.param(
        {
            'x0': (CONTINUOUS, [1.0, 2.0, 3.0, 4.0]),
            'x1': (CONTINUOUS, [1.0, 2.0, 4.0, 3.0]),
            'x2': (BINARY, [0, 1, 0, 1]),
        },
        {
            'continuous_relation_max': 0.8,
            'binary_relation_median': 0.2**0.5,
            'binary_relation_max': 0.2**0.5,
        },
        id='a binary column ranked with ties, |ρ| √0.2 with x0 and 0 with x1',
    ),
    pytest.param(
        {'x0': (CONTINUOUS, [1.0, 2.0, 3.0, 4.0]), 'x1': (CATEGORICAL, [0, 0, 1, 2])},
        {'continuous_relation_max': 0.9**0.5, 'categorical_relation_max': 0.9**0.5},
        id='a categorical column and a ranked one relate by η, 4.5 of 5 between its levels',
    ),
    pytest.param(
        {'x0': (CATEGORICAL, [0, 0, 1, 2]), 'x1': (CATEGORICAL, [0, 1, 1, 2])},
        {'categorical_relation_max': 0.625**0.5},
        id="two categorical columns relate by Cramér's V, χ² 5 at n 4 and k 3",
    ),
]


@pytest.mark.parametrize(('columns', 'expected'), RELATIONS)
def test_a_column_enters_by_its_largest_relation_with_another_column(
    columns: dict[str, tuple[Column, list[float]]], expected: dict[str, float]
) -> None:
    """Each measure is |ρ| when both columns have two levels, so the three share one scale."""
    measured = _measured(_flatCovariates(columns), None)
    assert {axis: measured[axis] for axis in expected} == pytest.approx(expected)


# two unit blocks at canonical correlation r have the singular values √(1 ± r),
# and 1 for the rest
CONDITION_INDICES = [
    pytest.param(
        {'x0': (CONTINUOUS, [1.0, 2.0, 3.0, 4.0]), 'x1': (CONTINUOUS, [1.0, 2.0, 4.0, 3.0])},
        3.0,
        id='two continuous columns at r 0.8, √(1.8/0.2)',
    ),
    pytest.param(
        {'x0': (CONTINUOUS, [1.0, 2.0, 3.0, 4.0]), 'x1': (CATEGORICAL, [0, 0, 1, 2])},
        ((1 + 0.9**0.5) / (1 - 0.9**0.5)) ** 0.5,
        id="a categorical column's dummies enter as their span, at η √0.9 with x0",
    ),
    pytest.param(
        {'x0': (CONTINUOUS, [1.0, 2.0, 3.0, 4.0]), 'x1': (CONTINUOUS, [2.0, 4.0, 6.0, 8.0])},
        float('inf'),
        id='a column and its multiple leave the design rank-deficient, without a finite index',
    ),
]


@pytest.mark.parametrize(('columns', 'expected'), CONDITION_INDICES)
def test_the_condition_index_reads_the_relations_between_covariates_alone(
    columns: dict[str, tuple[Column, list[float]]], expected: float
) -> None:
    """Each covariate's design columns are centered
    and replaced by an orthonormal basis of their span,
    so neither a column's scale nor a categorical's dropped level moves the index.
    """
    measured = _measured(_flatCovariates(columns), None)
    assert measured['condition_index'] == pytest.approx(expected)


def _continuousOutcome(values: list[float]) -> Outcome:
    """A model's continuous y."""
    y = Column(type='continuous', role='outcome')
    return Outcome(columns={'y': np.array(values, dtype=np.float32)}, metadata={'y': y})


def test_y_enters_by_its_shape_once_and_its_largest_relation_with_a_covariate() -> None:
    """y's ranks relate to x0's by |ρ| √0.95 and to x1's by 0.67, so y's relation is √0.95.
    x0 and x1 relate by 0.6 alone, since a covariate's relations leave y out.
    """
    covariates = _flatCovariates(
        {
            'x0': (CONTINUOUS, [1.0, 2.0, 3.0, 4.0, 5.0]),
            'x1': (CONTINUOUS, [2.0, 1.0, 5.0, 3.0, 4.0]),
        }
    )
    expected = {
        'continuous_relation_max': 0.6,
        'y_continuous_skewness': SKEWNESS,
        'y_continuous_excess_kurtosis': EXCESS_KURTOSIS,
        'y_continuous_outlier_share': 0.2,
        'y_continuous_zero_share': 0.4,
        'y_continuous_distinct_share': 0.8,
        'y_continuous_mode_share': 0.4,
        'y_relation': 0.95**0.5,
    }
    measured = _measured(covariates, _continuousOutcome([0.0, 0.0, 1.0, 2.0, 7.0]))
    assert {axis: measured[axis] for axis in expected} == pytest.approx(expected)
    assert np.isnan(measured['y_count_dispersion'])


def _axesOf(prefix: str) -> list[str]:
    """The axes whose names start with prefix."""
    return [axis for axis in AXES if axis.startswith(prefix)]


TWO_CONTINUOUS = {
    'x0': (CONTINUOUS, [1.0, 2.0, 3.0, 4.0]),
    'x1': (CONTINUOUS, [1.0, 2.0, 4.0, 3.0]),
}
# what a table cannot show is missing, never 0, which a table could show
MISSING = [
    pytest.param(TWO_CONTINUOUS, None, _axesOf('y_'), id='without y, every axis of y'),
    pytest.param(
        TWO_CONTINUOUS,
        _continuousOutcome([0.0, 0.0, 1.0, 2.0]),
        [axis for axis in _axesOf('count_') if axis != 'count_share'],
        id='without a count column, every count axis but its share',
    ),
    pytest.param(
        {'x0': TWO_CONTINUOUS['x0']},
        None,
        ['continuous_relation_median', 'continuous_relation_max'],
        id='with one covariate, its relation to another',
    ),
]


@pytest.mark.parametrize(('columns', 'outcome', 'missing'), MISSING)
def test_an_axis_the_table_cannot_show_is_missing(
    columns: dict[str, tuple[Column, list[float]]], outcome: Outcome | None, missing: list[str]
) -> None:
    measured = _measured(_flatCovariates(columns), outcome)
    assert all(np.isnan(measured[axis]) for axis in missing)


def _permute(dataset: Dataset, rng: np.random.Generator) -> tuple[Covariates, Outcome]:
    """The dataset's covariates and y with the rows shuffled, the columns in reverse order,
    and each binary or categorical column's labels permuted.
    """
    rows = rng.permutation(dataset.covariates.rows)  # (n,)
    columns: dict[str, np.ndarray] = {}
    metadata: dict[str, Column] = {}
    for name, column in reversed(dataset.metadata.items()):
        values = dataset.columns[name][rows]  # (n,)
        if column.type in ['binary', 'categorical'] and column.role != 'grouping_factor':
            labels = rng.permutation(column.levels)  # (k,)
            values = labels[values]
        columns[name] = values
        metadata[name] = column
    grouping = dataset.covariates.grouping
    codes = {name: factor_codes[rows] for name, factor_codes in grouping.codes.items()}
    covariates = Covariates(
        columns={name: values for name, values in columns.items() if name != 'y'},
        metadata={name: column for name, column in metadata.items() if name != 'y'},
        grouping=Grouping(metadata=grouping.metadata, codes=codes),
        y_score=None,
    )
    outcome = Outcome(columns={'y': columns['y']}, metadata={'y': metadata['y']})
    return covariates, outcome


@pytest.mark.parametrize('index', range(5))
@pytest.mark.parametrize('preset', ['flat', 'grouped'])
def test_the_vector_is_blind_to_the_order_of_rows_and_columns_and_to_nominal_labels(
    preset: str, index: int
) -> None:
    """A table's rows and columns have no order and a nominal column's labels name no quantity,
    so a summary that moved with them would tell two copies of one table apart.
    """
    dataset = drawDataset(0, index, getattr(Scm, preset)())
    permuted = _permute(dataset, np.random.default_rng(index))
    np.testing.assert_allclose(
        measure(*permuted), measure(dataset.covariates, dataset.outcome), rtol=1e-9
    )


def _readOutcome(table: pd.DataFrame, column: Column) -> Outcome:
    """y read back from a table, a coded y by its categorical's codes."""
    values = table['y']
    if column.levels is not None:
        values = values.cat.codes
    return Outcome(columns={'y': values.to_numpy(dtype=column.dtype)}, metadata={'y': column})


@pytest.mark.parametrize('index', range(3))
@pytest.mark.parametrize('preset', ['flat', 'grouped'])
def test_a_drawn_dataset_and_its_table_read_back_give_one_vector(preset: str, index: int) -> None:
    """The proofs measure real tables through fromPandas and draws as they come,
    so the two routes have to agree bit for bit.
    """
    dataset = drawDataset(0, index, getattr(Scm, preset)())
    table = dataset.toPandas()
    covariates = Covariates.fromPandas(table, metadata=dataset.metadata)
    outcome = _readOutcome(table, dataset.metadata['y'])
    np.testing.assert_array_equal(
        measure(covariates, outcome), measure(dataset.covariates, dataset.outcome)
    )


def test_a_table_of_15000_rows_and_19_design_columns_is_measured_in_under_a_second() -> None:
    """The largest table Scm.flat() draws, for the content proof's thousands of draws."""
    dataset = drawDataset(0, 0, Scm.flat(rows=15000, design_columns=19))
    start = time.perf_counter()
    measure(dataset.covariates, dataset.outcome)
    assert time.perf_counter() - start < 1
