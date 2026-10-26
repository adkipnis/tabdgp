"""The outcome module's seams: what Covariates and an Outcome guarantee on their own,
and run, a model's call and the check of its result.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from tabdgp.column import Column
from tabdgp.outcome import (
    FAMILIES,
    Bernoulli,
    Beta,
    Covariates,
    Gamma,
    GlmmOutcome,
    GlmOutcome,
    LatentBernoulli,
    NegativeBinomial,
    Normal,
    Outcome,
    Poisson,
    ScmOutcome,
    StudentT,
    codeDesign,
    run,
)
from tabdgp.prior import Glm, Glmm
from tests.fixtures.build import Y, columnsOf, covariatesOf, groupingOf, point

# four rows in two groups of g0, one continuous covariate x0 and the score
GROUPING = groupingOf([[0, 0, 1, 1]])
FIELDS = {
    'columns': {**GROUPING.codes, 'x0': np.zeros(4, dtype=np.float32)},
    'metadata': {**GROUPING.metadata, **columnsOf(['x0'])},
    'grouping': GROUPING,
    'score': np.zeros(4, dtype=np.float32),
}


def test_covariates_that_meet_every_check_construct_with_or_without_a_score() -> None:
    """The control for the checks below; a real table has no SCM and so no score."""
    Covariates(**FIELDS)
    Covariates(**FIELDS | {'score': None})


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
    pytest.param({'score': np.zeros(5, dtype=np.float32)}, 'length', id='a score one row long'),
    pytest.param({'metadata': GROUPING.metadata}, 'metadata', id='a column without metadata'),
    pytest.param(
        {'metadata': {**GROUPING.metadata, **columnsOf(['x0', 'x1'])}},
        'metadata',
        id='metadata without a column',
    ),
    pytest.param(
        {
            'columns': FIELDS['columns'] | {'y': np.zeros(4, dtype=np.float32)},
            'metadata': FIELDS['metadata'] | {'y': Y},
        },
        "'y'.*rename it",
        id='a column named y',
    ),
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
    y is the name the form writes, so a covariate under it would be overwritten in the dataset.
    intercept is the design matrix's first term, so a covariate under it would repeat the term,
    and fixed is the fixed block's name among the factors' blocks, so a factor under it would
    replace the fixed block's share and scale.
    """
    with pytest.raises(ValidationError, match=match):
        Covariates(**FIELDS | change)


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


def test_from_pandas_gives_no_score() -> None:
    """A table has no SCM, so a GLM or GLMM drawn on it has no deviation."""
    assert Covariates.fromPandas(TABLE, ['class']).score is None


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
        'x_int', [0, -1, 5, 1, 0, 3, 4, 2], 'x_int', id='an integer column with a negative value'
    ),
    pytest.param(
        'x_float', pd.date_range('2026-01-01', periods=8), 'x_float', id='a datetime column'
    ),
]


@pytest.mark.parametrize(('name', 'values', 'match'), UNREADABLE)
def test_from_pandas_refuses_a_missing_value_a_constant_column_an_identical_partition_a_negative_integer_or_another_dtype_naming_the_column(
    name: str, values: list, match: str
) -> None:
    """The coder never meets a missing value or a constant column, a z-score of 0 over 0.
    Two factors that group the rows alike are one factor under two names.
    A negative integer may be a code for a missing value, so it is no count,
    and a dtype outside the five states no column type.
    A real table meets the constant column most (an empty interaction cell, a fixed setting),
    so its refusal says what to do.
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


# a continuous y, a count and a binary column, each in its type's dtype
COUNT = Column(type='count', role='outcome')
BINARY = Column(type='binary', levels=2, role='outcome')
WRITTEN = {
    'columns': {
        'y': np.zeros(4, dtype=np.float32),
        'count': np.zeros(4, dtype=np.int64),
        'binary': np.zeros(4, dtype=np.int64),
    },
    'metadata': {'y': Y, 'count': COUNT, 'binary': BINARY},
}


def test_an_outcome_that_meets_every_check_constructs() -> None:
    """The control for the checks below."""
    Outcome(**WRITTEN)


def _written(**arrays: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """WRITTEN's columns with these arrays in place of its own."""
    return {'columns': WRITTEN['columns'] | arrays}


COVARIATE_Y = Column(type='continuous')
MISWRITTEN = [
    pytest.param(
        _written(count=np.zeros(3, dtype=np.int64)), 'length', id='a column one row short'
    ),
    pytest.param(_written(y=np.zeros(4)), 'float32', id='a continuous column in float64'),
    pytest.param(_written(count=np.zeros(4, dtype=np.float32)), 'int64', id='a count in float32'),
    pytest.param(_written(binary=np.zeros(4, dtype=np.int32)), 'int64', id='codes in int32'),
    pytest.param(
        {'metadata': {'y': Y, 'count': COUNT}}, 'metadata', id='a column without metadata'
    ),
    pytest.param(
        {'metadata': WRITTEN['metadata'] | {'y': COVARIATE_Y}},
        'role',
        id='a column of another role',
    ),
]


@pytest.mark.parametrize(('change', 'match'), MISWRITTEN)
def test_an_outcome_refuses_two_lengths_a_dtype_other_than_its_types_a_name_without_metadata_or_another_role(
    change: dict, match: str
) -> None:
    """A consumer stacks the outcome's columns at the rows by their type's dtype,
    looks each up by name, and finds the outcome by its role.
    """
    with pytest.raises(ValidationError, match=match):
        Outcome(**WRITTEN | change)


def test_a_scm_outcome_holds_its_score_as_given_and_refuses_one_of_another_length() -> None:
    """The scm form's truth is the score it rank-matched onto y's marginal, at y's rows."""
    score = np.zeros(4, dtype=np.float32)
    outcome = ScmOutcome(**WRITTEN, score=score)
    assert outcome.score is score
    with pytest.raises(ValidationError, match='length'):
        ScmOutcome(**WRITTEN, score=np.zeros(5, dtype=np.float32))


def _noisyScore(covariates: Covariates, rng: np.random.Generator, shift: float = 0.0) -> Outcome:
    """A model whose y is the score plus the generator's standard normals and a shift."""
    noise = rng.standard_normal(len(covariates.score), dtype=np.float32)  # (n,)
    y = covariates.score + noise + shift  # (n,)
    return Outcome(columns={'y': y}, metadata={'y': Y})


MODELS = [
    pytest.param(_noisyScore, 0.0, id='a function'),
    pytest.param(functools.partial(_noisyScore, shift=2.0), 2.0, id='a partial with its settings'),
]


@pytest.mark.parametrize(('model', 'shift'), MODELS)
def test_run_calls_a_model_with_the_covariates_and_the_generator_and_returns_its_outcome(
    model: object, shift: float
) -> None:
    """A study's model is any callable of the covariates and the key's generator,
    its settings bound by a partial; y must be its draw from that generator.
    """
    covariates = covariatesOf([[0, 0, 1, 1]], ['x0'])
    outcome = run(model, covariates, np.random.default_rng(0))
    noise = np.random.default_rng(0).standard_normal(4, dtype=np.float32)  # (n,)
    assert np.array_equal(outcome.columns['y'], covariates.score + noise + shift)


def _shortY(covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """A model whose y is one row short of the covariates."""
    y = np.zeros(len(covariates.score) - 1, dtype=np.float32)  # (n − 1,)
    return Outcome(columns={'y': y}, metadata={'y': Y})


def _bareY(covariates: Covariates, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A model that returns its y without an Outcome around it."""
    return {'y': covariates.score}


def _x0(covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """A model that writes its outcome under a covariate's name."""
    return Outcome(columns={'x0': covariates.score}, metadata={'x0': Y})


MISFIT = [
    pytest.param(_shortY, ValueError, 'length', id='a y one row short'),
    pytest.param(_bareY, TypeError, 'not an Outcome', id='a dict, not an Outcome'),
    pytest.param(_x0, ValueError, 'x0', id="a column under a covariate's name"),
]


@pytest.mark.parametrize(('model', 'error', 'match'), MISFIT)
def test_run_refuses_a_models_result_of_the_wrong_length_or_type_or_a_covariates_name_naming_the_model(
    model: object, error: type[Exception], match: str
) -> None:
    """A user's model is checked only on what it returns, so the failure must say whose it is.
    Its dtypes are checked by its own Outcome(...) call, inside the model.
    A column under a covariate's name would replace that covariate in the dataset.
    """
    with pytest.raises(error, match=match) as raised:
        run(model, covariatesOf([[0, 0, 1, 1]], ['x0']), np.random.default_rng(0))
    assert model.__name__ in str(raised.value)


def test_the_coder_puts_the_intercept_first_z_scores_a_continuous_column_and_records_its_mean_and_sd() -> (
    None
):
    """The glossary's design matrix: the intercept, then each continuous column z-scored by the
    population sd over the n rows, in float64.
    Its mean and sd map a standardized β back to the raw column.
    """
    columns = {'x0': np.array([1, 2, 3, 6], dtype=np.float32)}
    design = codeDesign(columns, columnsOf(['x0']), rows=4)
    # mean 3, deviations −2, −1, 0 and 3, population sd √(14 / 4)
    sd = math.sqrt(3.5)
    expected = np.array([[1, -2 / sd], [1, -1 / sd], [1, 0], [1, 3 / sd]])  # (n, d)
    assert design.names == ['intercept', 'x0']
    assert design.matrix.dtype == np.float64
    np.testing.assert_allclose(design.matrix, expected)
    assert design.means == {'x0': 3.0}
    assert design.sds == {'x0': pytest.approx(sd)}


def test_the_coder_codes_a_binary_0_1_a_categoricals_k_minus_1_dummies_dropping_the_first_level_and_a_count_as_continuous() -> (
    None
):
    """Treatment coding against the first level in the codes' order, as R's default contrasts do,
    an ordered column like a categorical; a count is z-scored like any continuous column.
    Only the z-scored columns have a mean and an sd to record.
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
    design = codeDesign(columns, metadata, rows=4)
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
    np.testing.assert_allclose(design.matrix, expected)
    assert design.means == {'n': 2.0}
    assert design.sds == {'n': pytest.approx(root2)}


def test_the_coder_leaves_out_a_grouping_factor_and_y_by_role() -> None:
    """A factor's codes index random effects and y is what is predicted (the glossary),
    so neither is a term, whatever its type; the GLM on grouped covariates is then pooled.
    """
    columns = {
        **GROUPING.codes,
        'x0': np.array([1, 2, 3, 6], dtype=np.float32),
        'y': np.array([0, 1, 1, 0]),
    }
    metadata = {**GROUPING.metadata, **columnsOf(['x0']), 'y': BINARY}
    assert codeDesign(columns, metadata, rows=4).names == ['intercept', 'x0']


def test_the_families_are_exactly_the_categories_of_glms_family_shares() -> None:
    """A family with a share but no class would fail only when a dataset chose it."""
    assert list(FAMILIES) == Glm.categories('family_shares')


FLOAT32 = np.finfo(np.float32)
BELOW_ONE = 1 - 2.0**-24  # the largest float32 below 1
INT64_MAX = np.iinfo(np.int64).max
# a family, y's dtype and its range, and a shape that suits it (None for the base three)
RANGES = [
    pytest.param(Normal, np.float32, -np.inf, np.inf, None, id='normal'),
    pytest.param(Bernoulli, np.int64, 0, 1, None, id='bernoulli'),
    pytest.param(Poisson, np.int64, 0, np.inf, None, id='poisson'),
    pytest.param(StudentT, np.float32, -np.inf, np.inf, 3.0, id='student_t'),
    pytest.param(NegativeBinomial, np.int64, 0, INT64_MAX, 1.0, id='negative_binomial'),
    pytest.param(LatentBernoulli, np.int64, 0, 1, 2.0, id='latent_bernoulli'),
    pytest.param(Gamma, np.float32, FLOAT32.tiny, FLOAT32.max, 1.0, id='gamma'),
    pytest.param(Beta, np.float32, FLOAT32.tiny, BELOW_ONE, 2.0, id='beta'),
]
# η over what a wide prior reaches, and far beyond it, where float64 and float32 run out
ETAS = [
    pytest.param(np.linspace(-40, 40, 1001), id='wide prior'),
    pytest.param(np.array([-800.0, -100.0, -10.0, 0.0, 10.0, 100.0, 800.0]), id='beyond'),
]


@pytest.mark.parametrize(('family', 'dtype', 'low', 'high', 'shape'), RANGES)
@pytest.mark.parametrize('eta', ETAS)
def test_each_familys_draw_is_in_range_and_of_its_dtype_without_a_warning(
    family: type, dtype: type, low: float, high: float, shape: float | None, eta: np.ndarray
) -> None:
    """y's column holds the family's draw in its type's dtype, at any η:
    a Gamma y is positive and a beta y inside (0, 1) as float32 holds them, so neither is
    a bound the fitter's likelihood would reject; a count saturates at int64's largest.
    """
    scale = np.full(len(eta), 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = family.draw(np.random.default_rng(0), eta, scale, shape)
    assert y.dtype == family.COLUMN.dtype == dtype
    assert y.shape == eta.shape
    assert np.isfinite(y).all()
    assert low <= y.min() and y.max() <= high


def _latentMean(eta: float, scale: float) -> float:
    """E[expit(η + σ ε)] for ε standard normal, by Gauss-Hermite quadrature."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    return float((weights / math.sqrt(2 * math.pi) * _expit(eta + scale * nodes)).sum())


def _expit(x: np.ndarray | float) -> np.ndarray | float:
    return 1 / (1 + np.exp(-x))


# η, the shape and y's mean and variance in closed form: the identity, the logit and the log of η
# at which the means are round (1.5, logit⁻¹(log 3) = 0.75, e^(log 4) = 4), the variance of each
# family's own noise at scale 1
LAWS = [
    pytest.param(Normal, 1.5, None, 1.5, 1.0, id='normal: the identity'),
    pytest.param(Bernoulli, math.log(3), None, 0.75, 0.75 * 0.25, id='bernoulli: the logit'),
    pytest.param(Poisson, math.log(4), None, 4.0, 4.0, id='poisson: the log'),
    pytest.param(StudentT, 1.5, 10.0, 1.5, 10 / 8, id='student_t: ν / (ν − 2) at σ_ε = 1'),
    pytest.param(
        NegativeBinomial,
        math.log(4),
        2.0,
        4.0,
        4 * (1 + 4 / 2),
        id='negative_binomial: μ (1 + μ / φ)',
    ),
    pytest.param(
        LatentBernoulli,
        math.log(3),
        1.5,
        _latentMean(math.log(3), 1.5),
        _latentMean(math.log(3), 1.5) * (1 - _latentMean(math.log(3), 1.5)),
        id='latent_bernoulli: the logit of η plus noise, attenuated toward ½',
    ),
    pytest.param(Gamma, math.log(2), 4.0, 2.0, 2.0**2 / 4, id='gamma: the log, μ² / shape'),
    pytest.param(
        Beta,
        math.log(3),
        4.0,
        0.75,
        0.75 * 0.25 / (4 + 1),
        id='beta: the logit, μ (1 − μ) / (φ + 1)',
    ),
]


@pytest.mark.parametrize(('family', 'eta', 'shape', 'mean', 'variance'), LAWS)
def test_each_familys_draw_has_the_mean_and_variance_of_its_law_at_its_shape(
    family: type, eta: float, shape: float | None, mean: float, variance: float
) -> None:
    """Each family has its canonical link (ADR 0006), which a fitter of that family assumes,
    and its own noise at the shape: the draw's moments are the law's, not numpy's by construction.
    """
    rows = 400_000
    y = family.draw(np.random.default_rng(0), np.full(rows, eta), np.ones(rows), shape)
    assert y.mean() == pytest.approx(mean, rel=0.02)
    assert y.var() == pytest.approx(variance, rel=0.05)


def test_a_latent_bernoullis_mean_is_attenuated_by_its_noise() -> None:
    """The latent noise makes y misspecified for a Bernoulli fitter: its mean is pulled to ½."""
    assert 0.5 < _latentMean(math.log(3), 1.5) < 0.7


def test_a_negative_binomial_count_above_numpys_limit_saturates_as_the_poisson_does() -> None:
    """numpy's own negative_binomial raises when its gamma rate passes the limit,
    which happens at a mean far below it when φ is small, so y saturates instead.
    """
    eta = np.array([0.0, 800.0])  # (n,)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = NegativeBinomial.draw(np.random.default_rng(0), eta, np.ones(2), 1.0)
    assert y[0] < INT64_MAX
    assert y[1] == INT64_MAX


def test_a_poisson_rate_above_numpys_limit_draws_without_failing() -> None:
    """A log link at a wide scale reaches rates numpy refuses (above about 9.2e18, η ≈ 43.7),
    and int64 holds no count above that, so y saturates at int64's largest:
    the draw never fails or warns, and the rate below the limit is drawn as is.
    """
    eta = np.array([0.0, 43.0, 44.0, 800.0])  # (n,)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = Poisson.draw(np.random.default_rng(0), eta, np.ones(4), None)
    largest = np.iinfo(np.int64).max
    assert y.dtype == np.int64
    assert y[1] < largest
    assert y[1] == pytest.approx(math.exp(43), rel=1e-6)
    assert y[2:].tolist() == [largest, largest]


RUNS = [
    pytest.param('normal', 'identity', 'continuous', id='normal'),
    pytest.param('bernoulli', 'logit', 'binary', id='bernoulli'),
    pytest.param('poisson', 'log', 'count', id='poisson'),
    pytest.param('student_t', 'identity', 'continuous', id='student_t'),
    pytest.param('negative_binomial', 'log', 'count', id='negative_binomial'),
    pytest.param('latent_bernoulli', 'logit', 'binary', id='latent_bernoulli'),
    pytest.param('gamma', 'log', 'continuous', id='gamma'),
    pytest.param('beta', 'logit', 'continuous', id='beta'),
]
# every shape field a point, so the shape a run records is known
SHAPES = {
    'residual_df': 4.0,
    'dispersion': 4.0,
    'latent_scale': 4.0,
    'gamma_shape': 4.0,
    'beta_precision': 4.0,
}


@pytest.mark.parametrize(('family', 'link', 'column_type'), RUNS)
@pytest.mark.parametrize(
    'covariates',
    [
        pytest.param(covariatesOf([], ['x0', 'x1'], rows=6), id='flat'),
        pytest.param(covariatesOf([[0, 0, 1, 1, 2, 2]], ['x0', 'x1']), id='grouped, pooled'),
    ],
)
def test_run_on_a_glm_returns_a_glm_outcome_without_per_factor_fields(
    family: str, link: str, column_type: str, covariates: Covariates
) -> None:
    """A Glm is TabDGP's own model: run codes the covariates, draws the prior and writes y.
    On grouped covariates it is the pooled truth: the factors' codes are no term,
    and the context has the one block fixed.
    Only the Normal and Student-t families have a σ_ε, only the contaminated families have a
    shape, and the deviation is zero at λ = 0.
    """
    glm = Glm(family_shares=point('family_shares', family, Glm), **SHAPES)
    outcome = run(glm, covariates, np.random.default_rng(0))
    assert type(outcome) is GlmOutcome
    assert (outcome.family, outcome.link) == (family, link)
    assert list(outcome.columns) == list(outcome.metadata) == ['y']
    assert (outcome.metadata['y'].type, outcome.metadata['y'].role) == (column_type, 'outcome')
    assert outcome.columns['y'].shape == (6,)
    assert outcome.fixed_terms == ['intercept', 'x0', 'x1']
    assert list(outcome.context.split) == ['fixed']
    assert (outcome.sigma_eps is None) == (family not in ['normal', 'student_t'])
    assert outcome.shape == (None if family in ['normal', 'bernoulli', 'poisson'] else 4.0)
    assert outcome.context.shape == outcome.shape
    assert outcome.deviation.tolist() == [0.0] * 6
    assert outcome.deviation_scale == 0.0


MISSHAPEN = [
    pytest.param({'beta': np.zeros(1)}, 'beta', id='beta one term short'),
    pytest.param({'beta': np.zeros((2, 1))}, 'beta', id='beta as a column'),
    pytest.param({'deviation': np.zeros(3)}, 'length', id='a deviation one row short'),
]


@pytest.mark.parametrize(('change', 'match'), MISSHAPEN)
def test_a_glm_outcome_checks_its_shapes_against_n_and_d(change: dict, match: str) -> None:
    """A consumer pairs beta with the terms by position and adds the deviation to η row by row,
    so a shape that disagrees fails where the outcome is built.
    """
    outcome = run(Glm(), covariatesOf([[0, 0, 1, 1]], ['x0']), np.random.default_rng(0))
    with pytest.raises(ValidationError, match=match):
        dataclasses.replace(outcome, **change)


def _covariance(sd: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Σ = diag(sd) corr diag(sd)."""
    return corr * np.outer(sd, sd)  # (q, q)


def test_the_raw_scale_parameters_on_the_raw_columns_give_the_standardized_eta_up_to_float32() -> (
    None
):
    """A user fitting the raw columns (lme4 on a table) compares against the raw-scale truth:
    a z-scored column's β over its sd, the intercept absorbing each one's β · mean / sd,
    a 0/1 column's β as drawn, and each group's random effects alike.
    Σ on the raw columns must give each row's random part the variance Σ gives it on the coded
    ones, which scaling Σ by the sds alone would miss in the intercept's row and column.
    The raw matrix here is built by hand, not by the coder.
    Both scales are float32, so they agree up to its rounding.
    """
    columns = {
        'x0': np.array([1.5, -2, 3, 0.25, 7, 4], dtype=np.float32),
        'b': np.array([0, 1, 1, 0, 1, 0]),
        'c': np.array([0, 1, 2, 2, 1, 0]),
        'n': np.array([3, 0, 5, 1, 2, 9]),
    }
    metadata = {
        'x0': Column(type='continuous'),
        'b': Column(type='binary', levels=2),
        'c': Column(type='categorical', levels=3),
        'n': Column(type='count'),
    }
    grouping = groupingOf([[0, 0, 1, 1, 2, 2]])
    covariates = Covariates(
        columns={**grouping.codes, **columns},
        metadata={**grouping.metadata, **metadata},
        grouping=grouping,
        score=None,
    )
    # at a budget of d every term is random too, the effects correlated
    glmm = Glmm(random_effects=6, correlated_share=1.0)
    outcome = run(glmm, covariates, np.random.default_rng(0))
    raw = np.stack(
        [
            np.ones(6),
            columns['x0'].astype(np.float64),
            columns['b'],
            columns['c'] == 1,
            columns['c'] == 2,
            columns['n'],
        ],
        axis=1,
    ).astype(np.float64)  # (n, d)
    standardized = codeDesign(columns, metadata, rows=6).matrix  # (n, d)
    np.testing.assert_allclose(raw @ outcome.beta_raw, standardized @ outcome.beta, rtol=1e-6)
    assert outcome.beta_raw[2:5].tolist() == outcome.beta[2:5].tolist()
    assert outcome.random_terms['g0'] == outcome.fixed_terms
    codes = grouping.codes['g0']  # (n,)
    random_raw = (raw * outcome.alpha_raw['g0'][codes]).sum(axis=1)  # (n,)
    random_standardized = (standardized * outcome.alpha['g0'][codes]).sum(axis=1)  # (n,)
    np.testing.assert_allclose(random_raw, random_standardized, rtol=1e-6, atol=1e-6)
    sigma = _covariance(outcome.sigma_alpha['g0'], outcome.corr_alpha['g0'])  # (q, q)
    sigma_raw = _covariance(outcome.sigma_alpha_raw['g0'], outcome.corr_alpha_raw['g0'])  # (q, q)
    variance_raw = ((raw @ sigma_raw) * raw).sum(axis=1)  # (n,)
    variance_standardized = ((standardized @ sigma) * standardized).sum(axis=1)  # (n,)
    np.testing.assert_allclose(variance_raw, variance_standardized, rtol=1e-6, atol=1e-6)


def test_the_truths_arrays_are_float32_on_both_scales() -> None:
    """The truth goes to torch with the columns, so it has their precision, float32,
    on the raw columns too.
    """
    covariates = covariatesOf([[0, 0, 1, 1, 2, 2]], ['x0', 'x1'])
    outcome = run(Glmm(), covariates, np.random.default_rng(0))
    arrays = {'beta': outcome.beta, 'beta_raw': outcome.beta_raw, 'deviation': outcome.deviation}
    for field in [
        'sigma_alpha',
        'corr_alpha',
        'alpha',
        'sigma_alpha_raw',
        'corr_alpha_raw',
        'alpha_raw',
    ]:
        arrays[field] = getattr(outcome, field)['g0']
    for field, array in arrays.items():
        assert array.dtype == np.float32, field


def test_run_on_a_glmm_returns_a_glmm_outcome_with_one_block_per_factor() -> None:
    """A Glmm draws one block of random effects per grouping factor, crossed or nested alike.
    Each factor's terms are design terms led by its intercept,
    and its effects have one row per group; GlmmOutcome checks their q itself.
    The context names the blocks fixed, then the factors in order.
    """
    covariates = covariatesOf([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], ['x0', 'x1'])
    outcome = run(Glmm(), covariates, np.random.default_rng(0))
    assert type(outcome) is GlmmOutcome
    assert list(outcome.context.split) == ['fixed', 'g0', 'g1']
    assert list(outcome.random_terms) == ['g0', 'g1']
    for factor, groups in [('g0', 3), ('g1', 2)]:
        terms = outcome.random_terms[factor]
        assert terms[0] == 'intercept'
        assert set(terms) <= set(outcome.fixed_terms)
        assert len(outcome.alpha[factor]) == groups


# six rows in three groups of g0, and covariates of design width 1, 2 and 3
SLOPE_GROUPING = groupingOf([[0, 0, 1, 1, 2, 2]])
SLOPE_COLUMNS = {
    'x': np.array([1.5, -2, 3, 0.25, 7, 4], dtype=np.float32),
    'c': np.array([0, 1, 2, 2, 1, 0]),
    'k': np.array([0, 1, 2, 3, 1, 0]),
}
SLOPE_METADATA = {
    'x': Column(type='continuous'),
    'c': Column(type='categorical', levels=3),
    'k': Column(type='categorical', levels=4),
}


def _slopeCovariates(names: list[str]) -> Covariates:
    """The grouping and these covariates of SLOPE_COLUMNS."""
    columns = dict(SLOPE_GROUPING.codes)
    metadata = dict(SLOPE_GROUPING.metadata)
    for name in names:
        columns[name] = SLOPE_COLUMNS[name]
        metadata[name] = SLOPE_METADATA[name]
    return Covariates(columns=columns, metadata=metadata, grouping=SLOPE_GROUPING, score=None)


def _randomTerms(names: list[str], random_effects: int, seed: int) -> list[str]:
    """g0's random terms under a budget, on the grouping and these covariates."""
    glmm = Glmm(random_effects=random_effects)
    outcome = run(glmm, _slopeCovariates(names), np.random.default_rng(seed))
    return outcome.random_terms['g0']


BUDGETS = [
    pytest.param(['x', 'c'], 1, ['intercept'], id='a budget of 1: the intercept alone'),
    pytest.param(['x', 'c'], 2, ['intercept', 'x'], id="a budget of 2: c's two dummies never fit"),
    pytest.param(['k'], 3, ['intercept'], id="nothing fits k's three dummies: q = 1"),
    pytest.param(['x', 'c'], 4, ['intercept', 'x', 'c=1', 'c=2'], id='a budget of d: every term'),
]


@pytest.mark.parametrize(('names', 'random_effects', 'terms'), BUDGETS)
def test_the_random_slopes_fit_the_budget_and_q_is_1_when_nothing_fits(
    names: list[str], random_effects: int, terms: list[str]
) -> None:
    """ADR 0006: the intercept always, then whole covariates, each taken while it fits
    what remains of the budget q_k, so a factor with no room keeps its intercept alone.
    Whatever order the covariates are drawn in, these budgets admit one set of terms.
    """
    for seed in range(10):
        assert _randomTerms(names, random_effects, seed) == terms


def test_a_slope_on_a_categorical_column_takes_its_k_minus_1_dummies() -> None:
    """A random slope on a categorical column is one per dummy (the glossary's random effect),
    in the design's order, so q_k counts design columns.
    """
    assert _randomTerms(['c'], 3, seed=0) == ['intercept', 'c=1', 'c=2']


def test_the_covariates_that_take_the_slopes_vary_with_the_draw() -> None:
    """The covariates are tried in a random order: at a budget of 3, x then c gives x alone
    and c then x gives c's two dummies, so both sets occur over the draws.
    """
    drawn = set()
    for seed in range(20):
        drawn.add(tuple(_randomTerms(['x', 'c'], 3, seed)))
    assert drawn == {('intercept', 'x'), ('intercept', 'c=1', 'c=2')}


def test_a_random_effects_point_above_d_fails_naming_the_field() -> None:
    """A random term needs its fixed one (every random slope in the registry has it),
    so q_k cannot exceed d, and a point above d fails like any point above its bound.
    """
    with pytest.raises(ValueError, match='random_effects'):
        _randomTerms(['x'], 3, seed=0)


def test_run_refuses_a_glmm_on_covariates_without_a_factor_naming_the_model() -> None:
    """A GLMM without a grouping factor would split var(η) over no random block
    and draw a GLM at a fraction of its scale, silently; draw's hyperprior refuses it earlier,
    so this guards a table read without factors.
    """
    covariates = covariatesOf([], ['x0'], rows=4)
    with pytest.raises(ValueError, match='Glmm'):
        run(Glmm(), covariates, np.random.default_rng(0))


MISSHAPEN_RANDOM = [
    pytest.param({'sigma_alpha': {'g0': np.ones(1)}}, 'sigma_alpha', id='sds one term short'),
    pytest.param({'corr_alpha': {'g0': np.eye(3)}}, 'corr_alpha', id='a correlation one too wide'),
    pytest.param({'alpha': {'g0': np.zeros((2, 1))}}, 'alpha', id='effects one term short'),
    pytest.param({'alpha': {'g0': np.zeros(2)}}, 'alpha', id='effects as one row'),
    pytest.param({'sigma_alpha': {}}, 'sigma_alpha', id='sds without the factor'),
]


@pytest.mark.parametrize(('change', 'match'), MISSHAPEN_RANDOM)
def test_a_glmm_outcome_checks_each_factors_shapes_against_its_random_terms(
    change: dict, match: str
) -> None:
    """A consumer pairs each factor's effects with its terms by position and its groups by row,
    so a shape that disagrees with the terms fails where the outcome is built.
    The rows are the groups, which the outcome does not hold; the draw sizes them.
    """
    covariates = covariatesOf([[0, 0, 1, 1]], ['x0'])
    outcome = run(Glmm(random_effects=2), covariates, np.random.default_rng(0))
    assert outcome.random_terms == {'g0': ['intercept', 'x0']}
    with pytest.raises(ValidationError, match=match):
        dataclasses.replace(outcome, **change)


def test_the_deviation_is_lambda_times_the_sd_of_x_beta_plus_z_alpha_times_the_score() -> None:
    """The deviation is the score at λ · sd(η_lin), η_lin = Xβ + Zα the linear part,
    the sd over the n rows, so that a fitted GLMM is misspecified by exactly this term.
    The score counts up from 0, so a deviation that z-scored it again would differ.
    The design matrix here is built by hand, not by the coder.
    """
    covariates = covariatesOf([[0, 0, 1, 1, 2, 2]], ['x0'])
    glmm = Glmm(random_effects=2, deviation_scale=0.5)
    outcome = run(glmm, covariates, np.random.default_rng(0))
    x0 = np.arange(6.0)  # (n,)
    design = np.stack([np.ones(6), (x0 - x0.mean()) / x0.std()], axis=1)  # (n, d)
    codes = covariates.grouping.codes['g0']  # (n,)
    linear = design @ outcome.beta + (design * outcome.alpha['g0'][codes]).sum(axis=1)  # (n,)
    assert outcome.random_terms['g0'] == ['intercept', 'x0']
    assert outcome.deviation_scale == 0.5
    np.testing.assert_allclose(outcome.deviation, 0.5 * linear.std() * x0, rtol=1e-6)


WITHOUT_DEVIATION = [
    pytest.param(False, 0.5, id='no score, as in a table'),
    pytest.param(True, 0.0, id='lambda 0, the default'),
]


@pytest.mark.parametrize(('scored', 'deviation_scale'), WITHOUT_DEVIATION)
def test_the_deviation_is_zeros_without_a_score_or_at_lambda_0_and_y_is_the_linear_parts(
    scored: bool, deviation_scale: float
) -> None:
    """Either way the deviation is zeros and λ is recorded as given,
    so that a GlmOutcome has the same fields whether the covariates come from a draw or a table.
    y is then bitwise the y drawn as if neither the score nor λ existed.
    """
    scored_covariates = covariatesOf([[0, 0, 1, 1, 2, 2]], ['x0'])
    plain = dataclasses.replace(scored_covariates, score=None)
    covariates = scored_covariates if scored else plain
    normal = point('family_shares', 'normal', Glm)
    glmm = Glmm(family_shares=normal, deviation_scale=deviation_scale)
    outcome = run(glmm, covariates, np.random.default_rng(0))
    reference = run(Glmm(family_shares=normal), plain, np.random.default_rng(0))
    assert outcome.deviation.tolist() == [0.0] * 6
    assert outcome.deviation.dtype == np.float32
    assert outcome.deviation_scale == deviation_scale
    assert outcome.columns['y'].tobytes() == reference.columns['y'].tobytes()


@pytest.mark.parametrize(('family', 'link', 'column_type'), RUNS)
def test_run_on_a_glmm_records_the_shape_beside_the_random_effects(
    family: str, link: str, column_type: str
) -> None:
    """The contaminated families draw on a GLMM as on a GLM: η holds the random effects,
    and the shape is the family's own, not a group's.
    """
    glmm = Glmm(family_shares=point('family_shares', family, Glm), **SHAPES)
    covariates = covariatesOf([[0, 0, 1, 1, 2, 2]], ['x0', 'x1'])
    outcome = run(glmm, covariates, np.random.default_rng(0))
    assert type(outcome) is GlmmOutcome
    assert (outcome.family, outcome.link, outcome.metadata['y'].type) == (family, link, column_type)
    assert outcome.shape == (None if family in ['normal', 'bernoulli', 'poisson'] else 4.0)
    assert list(outcome.alpha) == ['g0']
