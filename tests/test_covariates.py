"""The covariates module's seams: what Covariates guarantee on their own,
and fromPandas, a table read by its declared or inferred metadata and checked against its values.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tabdgp.column import Column
from tabdgp.covariates import Covariates
from tests.fixtures.build import columnsOf, groupingOf

# four rows in two groups of g0, one continuous covariate x0 and the score
GROUPING = groupingOf([[0, 0, 1, 1]])
FIELDS = {
    'columns': {**GROUPING.codes, 'x0': np.zeros(4, dtype=np.float32)},
    'metadata': {**GROUPING.metadata, **columnsOf(['x0'])},
    'grouping': GROUPING,
    'y_score': np.zeros(4, dtype=np.float32),
}


def test_covariates_that_meet_every_check_construct_with_or_without_a_score() -> None:
    """The control for the checks below; a real table has no SCM and so no score."""
    Covariates(**FIELDS)
    Covariates(**FIELDS | {'y_score': None})


def _withColumn(name: str) -> dict:
    """FIELDS' columns and metadata with a continuous covariate of this name added."""
    return {
        'columns': FIELDS['columns'] | {name: np.zeros(4, dtype=np.float32)},
        'metadata': FIELDS['metadata'] | {name: Column(type='continuous')},
    }


UNFIT = [
    pytest.param(
        {'columns': {**GROUPING.codes, 'x0': np.zeros(3, dtype=np.float32)}},
        'length',
        id='a covariate one row short',
    ),
    pytest.param({'y_score': np.zeros(5, dtype=np.float32)}, 'length', id='a score one row long'),
    pytest.param({'metadata': GROUPING.metadata}, 'metadata', id='a column without metadata'),
    pytest.param(
        {'metadata': {**GROUPING.metadata, **columnsOf(['x0', 'x1'])}},
        'metadata',
        id='metadata without a column',
    ),
    pytest.param(_withColumn('y'), "'y'.*rename it", id='a column named y'),
    pytest.param(
        _withColumn('intercept'),
        "'intercept'.*rename it",
        id="a column named the design's first term",
    ),
    pytest.param(_withColumn('fixed'), "'fixed'.*rename it", id='a column named the fixed block'),
]


@pytest.mark.parametrize(('change', 'match'), UNFIT)
def test_covariates_refuse_arrays_of_two_lengths_a_name_without_metadata_and_a_reserved_name(
    change: dict, match: str
) -> None:
    """A model reads every array at the same rows and looks a column up by name in both dicts.
    y is the name the outcome writes, so a covariate under it would be overwritten in the dataset.
    intercept is the design matrix's first term, so a covariate under it would repeat the term,
    and fixed is the fixed block's name among the factors' blocks, so a factor under it would
    replace the fixed block's share and scale.
    """
    with pytest.raises(ValidationError, match=match):
        Covariates(**FIELDS | change)


def test_covariates_without_a_column_count_their_rows_by_the_score() -> None:
    """A flat draw without covariates has only the score,
    and an intercept-only GLM on it still needs n.
    """
    covariates = Covariates(
        columns={}, metadata={}, grouping=groupingOf([]), y_score=np.zeros(5, dtype=np.float32)
    )
    assert covariates.rows == 5


def _covariatesOf(columns: dict[str, np.ndarray], metadata: dict[str, Column]) -> Covariates:
    """Covariates of these columns, without a grouping factor or a score."""
    return Covariates(columns=columns, metadata=metadata, grouping=groupingOf([]), y_score=None)


def test_the_design_puts_the_intercept_first_z_scores_a_continuous_column_and_records_its_mean_and_sd() -> (
    None
):
    """The glossary's design matrix: the intercept, then each continuous column z-scored by the
    population sd over the n rows, in float64.
    Its mean and sd map a standardized β back to the raw column.
    """
    columns = {'x0': np.array([1, 2, 3, 6], dtype=np.float32)}
    design = _covariatesOf(columns, columnsOf(['x0'])).toDesign()
    # mean 3, deviations −2, −1, 0 and 3, population sd √(14 / 4)
    sd = math.sqrt(3.5)
    expected = np.array([[1, -2 / sd], [1, -1 / sd], [1, 0], [1, 3 / sd]])  # (n, d)
    assert design.names == ['intercept', 'x0']
    assert design.matrix.dtype == np.float64
    np.testing.assert_allclose(design.matrix, expected)
    assert design.means == {'x0': 3.0}
    assert design.sds == {'x0': pytest.approx(sd)}


def test_the_design_codes_a_binary_0_1_a_categoricals_k_minus_1_dummies_dropping_the_first_level_and_a_count_as_continuous() -> (
    None
):
    """Treatment coding against the first level in the codes' order, as R's default contrasts do,
    an ordered column like a categorical; a count is z-scored like any continuous column.
    Only the z-scored columns have a mean and an sd to record.
    Each covariate's positions are its columns of the matrix, which a random slope takes whole.
    """
    columns = {
        'b': np.array([0, 1, 1, 0]),
        'c': np.array([0, 1, 2, 1]),
        'o': np.array([2, 0, 1, 1]),
        'n': np.array([0, 2, 4, 2]),
    }
    metadata = {
        'b': Column(type='binary', levels=2),
        'c': Column(type='categorical', levels=3),
        'o': Column(type='ordered', levels=3),
        'n': Column(type='count'),
    }
    design = _covariatesOf(columns, metadata).toDesign()
    # the count: mean 2, deviations −2, 0, 2 and 0, population sd √2
    root2 = math.sqrt(2)
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 1, -root2],
            [1, 1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0, root2],
            [1, 0, 1, 0, 1, 0, 0],
        ]
    )  # (n, d)
    assert design.names == ['intercept', 'b', 'c=1', 'c=2', 'o=1', 'o=2', 'n']
    assert design.positions == {'b': [1], 'c': [2, 3], 'o': [4, 5], 'n': [6]}
    np.testing.assert_allclose(design.matrix, expected)
    assert design.means == {'n': 2.0}
    assert design.sds == {'n': pytest.approx(root2)}


def test_the_design_leaves_out_a_grouping_factor_by_role() -> None:
    """A factor's codes index random effects (the glossary), so they are no term,
    whatever the factor's type; the GLM on grouped covariates is then pooled.
    """
    columns = {**GROUPING.codes, 'x0': np.array([1, 2, 3, 6], dtype=np.float32)}
    metadata = {**GROUPING.metadata, **columnsOf(['x0'])}
    covariates = Covariates(columns=columns, metadata=metadata, grouping=GROUPING, y_score=None)
    assert covariates.toDesign().names == ['intercept', 'x0']


# eight rows: classes coded as integers nested in two schools named after them,
# raters crossed with both, and one covariate of each dtype
TABLE = pd.DataFrame(
    {
        'class': [3, 3, 1, 1, 7, 7, 5, 5],
        'school': ['north'] * 4 + ['south'] * 4,
        'rater': pd.Categorical(['r1', 'r2'] * 4),
        'x_float': [0.5, -1.0, 2.0, 0.0, 1.5, 3.0, -2.5, 1.0],
        'x_int': [0, 2, 5, 1, 0, 3, 4, 2],
        'x_bool': [True, False, False, True, True, False, True, True],
        'x_two': pd.Categorical(['hi', 'lo', 'lo', 'hi', 'lo', 'hi', 'hi', 'lo'], ['lo', 'hi']),
        'x_ordered': pd.Categorical(
            ['mid', 'low', 'high', 'mid', 'low', 'high', 'mid', 'low'],
            ['low', 'mid', 'high'],
            ordered=True,
        ),
        'x_string': ['c', 'a', 'b', 'a', 'c', 'b', 'a', 'c'],
        'x_dummy': [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        'x_sex': [2, 1, 1, 2, 2, 2, 1, 1],
    }
)


def test_from_pandas_reads_types_levels_codes_and_nesting_from_a_data_frame() -> None:
    """The dtype is the table's statement of a covariate's type, a factor is categorical by role.
    A column of two values is binary whatever its dtype, as mixedbench reads one:
    a 0/1 dummy stored as float, a sex coded 1/2.
    Codes follow the sorted values, or a categorical's own order.
    Nesting is read from the codes, whichever factor is named first:
    real tables list a subordinate before its superior too.
    A pupil per row lies within every factor's groups; its immediate superior is the class.
    """
    table = TABLE.assign(pupil=range(8))
    covariates = Covariates.fromPandas(table, ['class', 'school', 'rater', 'pupil'])
    pupil = Column('categorical', 8, 'grouping_factor', 'class')
    expected = {
        'class': ([1, 1, 0, 0, 3, 3, 2, 2], Column('categorical', 4, 'grouping_factor', 'school')),
        'school': ([0, 0, 0, 0, 1, 1, 1, 1], Column('categorical', 2, 'grouping_factor')),
        'rater': ([0, 1, 0, 1, 0, 1, 0, 1], Column('categorical', 2, 'grouping_factor')),
        'pupil': (list(range(8)), pupil),
        'x_int': ([0, 2, 5, 1, 0, 3, 4, 2], Column('count')),
        'x_bool': ([1, 0, 0, 1, 1, 0, 1, 1], Column('binary', 2)),
        'x_two': ([1, 0, 0, 1, 0, 1, 1, 0], Column('binary', 2)),
        'x_ordered': ([1, 0, 2, 1, 0, 2, 1, 0], Column('ordered', 3)),
        'x_string': ([2, 0, 1, 0, 2, 1, 0, 2], Column('categorical', 3)),
        'x_dummy': ([0, 1, 1, 0, 0, 1, 0, 1], Column('binary', 2)),
        'x_sex': ([1, 0, 0, 1, 1, 1, 0, 0], Column('binary', 2)),
    }
    factors = ['class', 'school', 'rater', 'pupil']
    names = ['x_float', 'x_int', 'x_bool', 'x_two', 'x_ordered', 'x_string', 'x_dummy', 'x_sex']
    assert list(covariates.columns) == [*factors, *names]
    for name, (codes, column) in expected.items():
        assert covariates.columns[name].dtype == np.int64
        assert covariates.columns[name].tolist() == codes
        assert covariates.metadata[name] == column
    x_float = np.array(TABLE['x_float'], dtype=np.float32)
    assert np.array_equal(covariates.columns['x_float'], x_float)
    assert covariates.columns['x_float'].dtype == np.float32
    assert covariates.metadata['x_float'] == Column('continuous')
    assert list(covariates.grouping.metadata) == factors


# a pupil per row lies within the four classes and the four shifts, which cross
TIES = [
    pytest.param(['class', 'shift', 'pupil'], 'class', id='class named first'),
    pytest.param(['shift', 'class', 'pupil'], 'shift', id='shift named first'),
]


@pytest.mark.parametrize(('factors', 'superior'), TIES)
def test_from_pandas_nests_a_factor_in_the_first_named_of_two_superiors_with_as_many_groups(
    factors: list[str], superior: str
) -> None:
    """The codes cannot tell which of two crossed factors of four groups is a pupil's superior,
    so the order the factors are named in decides.
    """
    table = TABLE.assign(pupil=range(8), shift=[0, 1, 2, 3] * 2)
    covariates = Covariates.fromPandas(table, factors)
    assert covariates.metadata['pupil'].nests_in == superior


def test_from_pandas_gives_no_score() -> None:
    """A table has no SCM, so a GLM or GLMM drawn on it has no deviation."""
    assert Covariates.fromPandas(TABLE, ['class']).y_score is None


def test_from_pandas_makes_a_factors_codes_dense() -> None:
    """A category no row has would be a group without rows, or a dummy column of zeros,
    so the codes skip it, for a factor and a coded covariate alike.
    """
    table = pd.DataFrame(
        {
            'g': pd.Categorical(['a', 'c', 'c', 'a'], ['a', 'b', 'c']),
            'x': pd.Categorical(['u', 'w', 'v', 'u'], ['u', 'v', 'z', 'w']),
        }
    )
    covariates = Covariates.fromPandas(table, ['g'])
    assert covariates.columns['g'].tolist() == [0, 1, 1, 0]
    assert covariates.metadata['g'].levels == 2
    assert covariates.columns['x'].tolist() == [0, 2, 1, 0]
    assert covariates.metadata['x'].levels == 3


# one column of TABLE replaced, and what the refusal names
UNREADABLE = [
    pytest.param(
        'x_float', [None, -1.0, 2.0, 0.0, 1.5, 3.0, -2.5, 1.0], 'x_float', id='a missing value'
    ),
    pytest.param(
        'school', ['north'] * 3 + [None] + ['south'] * 4, 'school', id="a factor's missing value"
    ),
    pytest.param('x_int', [1] * 8, 'x_int takes one value.*drop it', id='a constant covariate'),
    pytest.param('rater', ['r1'] * 8, 'rater takes one value.*drop it', id='a factor of one group'),
    pytest.param(
        'rater',
        ['c', 'c', 'a', 'a', 'd', 'd', 'b', 'b'],
        'class and rater',
        id='a factor grouping the rows as another does',
    ),
    pytest.param(
        'x_int',
        [0, -1, 5, 1, 0, 3, 4, 2],
        'x_int.*read it as continuous',
        id='an integer column with a negative value',
    ),
    pytest.param(
        'x_float', pd.date_range('2026-01-01', periods=8), 'x_float', id='a datetime column'
    ),
]


@pytest.mark.parametrize(('name', 'values', 'match'), UNREADABLE)
def test_from_pandas_refuses_a_column_it_cannot_read_naming_it(
    name: str, values: list, match: str
) -> None:
    """The coder never meets a missing value or a constant column, a z-score of 0 over 0.
    Two factors that group the rows alike are one factor under two names.
    A negative integer may be a code for a missing value, so it is no count,
    and a dtype outside the five states no column type.
    A real table meets the constant column most (an empty interaction cell, a fixed setting)
    and the negative code for a missing value too, so both refusals say what to do.
    """
    table = TABLE.assign(**{name: values})
    with pytest.raises(ValueError, match=match):
        Covariates.fromPandas(table, ['class', 'school', 'rater'])


# TABLE's metadata with one column's replaced (None drops it), and what the refusal names
CONTRADICTED = [
    pytest.param('x_string', Column('categorical', 4), 'x_string', id='a level no row has'),
    pytest.param(
        'rater',
        Column('categorical', 2, 'grouping_factor', 'school'),
        'rater',
        id='a nesting the codes do not show',
    ),
    pytest.param(
        'rater',
        Column('categorical', 2, 'grouping_factor', 'district'),
        'rater.*district',
        id='a nesting in no factor of the table',
    ),
    pytest.param('x_int', None, 'x_int', id='a column without metadata'),
    pytest.param('x_float', Column('count'), 'x_float', id='a count of negative fractions'),
    pytest.param('x_string', Column('continuous'), 'x_string', id='a continuous column of strings'),
]


@pytest.mark.parametrize(('name', 'column', 'match'), CONTRADICTED)
def test_from_pandas_refuses_a_declared_metadata_the_values_contradict_naming_the_column(
    name: str, column: Column | None, match: str
) -> None:
    """A declaration is checked against the values once, at construction,
    so a wrong one fails there and not as a wrong design matrix or grouping.
    It names every column, so none is read by a guess.
    """
    metadata = dict(Covariates.fromPandas(TABLE, ['class', 'school', 'rater']).metadata)
    if column is None:
        del metadata[name]
    else:
        metadata[name] = column
    with pytest.raises(ValueError, match=match):
        Covariates.fromPandas(TABLE, metadata=metadata)


def test_from_pandas_takes_the_factors_or_the_metadata_not_both() -> None:
    """The factors name the roles inference needs, a declaration names them itself,
    so passing both would say one thing twice.
    """
    metadata = Covariates.fromPandas(TABLE, ['class']).metadata
    with pytest.raises(ValueError, match='or declare it'):
        Covariates.fromPandas(TABLE, ['class'], metadata)
