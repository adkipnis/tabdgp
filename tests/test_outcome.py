"""The outcome module's seams: what Covariates and an Outcome guarantee on their own,
and run, a model's call and the check of its result.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.column import Column
from tabdgp.outcome import (
    FAMILIES,
    Bernoulli,
    Covariates,
    GlmOutcome,
    Normal,
    Outcome,
    Poisson,
    ScmOutcome,
    codeDesign,
    run,
)
from tabdgp.prior import Glm
from tests.fixtures.build import columnsOf, covariatesOf, groupingOf, point

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


# a model writes its columns without a marginal or parameters
Y = Column(type='continuous', role='outcome')
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
        "'y'",
        id='a column named y',
    ),
]


@pytest.mark.parametrize(('change', 'match'), UNFIT)
def test_covariates_refuse_arrays_of_two_lengths_a_name_without_metadata_and_a_column_named_y(
    change: dict, match: str
) -> None:
    """A model reads every array at the same rows and looks a column up by name in both dicts.
    y is the name the form writes, so a covariate under it would be overwritten in the dataset.
    """
    with pytest.raises(ValidationError, match=match):
        Covariates(**FIELDS | change)


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


RANGES = [
    pytest.param(Normal, np.float32, -np.inf, np.inf, id='normal'),
    pytest.param(Bernoulli, np.int64, 0, 1, id='bernoulli'),
    pytest.param(Poisson, np.int64, 0, np.inf, id='poisson'),
]


@pytest.mark.parametrize(('family', 'dtype', 'low', 'high'), RANGES)
def test_each_familys_draw_is_in_range_and_of_its_dtype(
    family: type, dtype: type, low: float, high: float
) -> None:
    """y's column holds the family's draw in its type's dtype, over the η a wide prior reaches:
    a Normal y any float32, a Bernoulli y the codes 0 and 1, a Poisson y a count.
    """
    eta = np.linspace(-40, 40, 1001)  # (n,)
    y = family.draw(np.random.default_rng(0), eta, np.full(1001, 0.5))
    assert y.dtype == family.COLUMN.dtype == dtype
    assert y.shape == eta.shape
    assert np.isfinite(y).all()
    assert low <= y.min() and y.max() <= high


# η at which each family's mean is a round number: 1.5, logit⁻¹(log 3) = 0.75, exp(log 4) = 4
MEANS = [
    pytest.param(Normal, 1.5, 1.5, id='normal: the identity'),
    pytest.param(Bernoulli, math.log(3), 0.75, id='bernoulli: the logit'),
    pytest.param(Poisson, math.log(4), 4.0, id='poisson: the log'),
]


@pytest.mark.parametrize(('family', 'eta', 'mean'), MEANS)
def test_each_familys_mean_is_its_inverse_link_of_eta(
    family: type, eta: float, mean: float
) -> None:
    """Each family has its canonical link (ADR 0006), which a fitter of that family assumes."""
    rows = 20000
    y = family.draw(np.random.default_rng(0), np.full(rows, eta), np.ones(rows))
    assert y.mean() == pytest.approx(mean, rel=0.02)


def test_a_poisson_rate_above_numpys_limit_draws_without_failing() -> None:
    """A log link at a wide scale reaches rates numpy refuses (above about 9.2e18, η ≈ 43.7),
    and int64 holds no count above that, so y saturates at int64's largest:
    the draw never fails or warns, and the rate below the limit is drawn as is.
    """
    eta = np.array([0.0, 43.0, 44.0, 800.0])  # (n,)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = Poisson.draw(np.random.default_rng(0), eta, np.ones(4))
    largest = np.iinfo(np.int64).max
    assert y.dtype == np.int64
    assert y[1] < largest
    assert y[1] == pytest.approx(math.exp(43), rel=1e-6)
    assert y[2:].tolist() == [largest, largest]


RUNS = [
    pytest.param('normal', 'identity', 'continuous', id='normal'),
    pytest.param('bernoulli', 'logit', 'binary', id='bernoulli'),
    pytest.param('poisson', 'log', 'count', id='poisson'),
]


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
    Only the Normal family has a σ_ε, and the deviation is zero at λ = 0.
    """
    glm = Glm(family_shares=point('family_shares', family, Glm))
    outcome = run(glm, covariates, np.random.default_rng(0))
    assert type(outcome) is GlmOutcome
    assert (outcome.family, outcome.link) == (family, link)
    assert list(outcome.columns) == list(outcome.metadata) == ['y']
    assert (outcome.metadata['y'].type, outcome.metadata['y'].role) == (column_type, 'outcome')
    assert outcome.columns['y'].shape == (6,)
    assert outcome.fixed_terms == ['intercept', 'x0', 'x1']
    assert outcome.beta.shape == (3,)
    assert list(outcome.context.scales) == ['fixed']
    assert (outcome.sigma_eps is None) == (family != 'normal')
    assert outcome.deviation.tolist() == [0.0] * 6
    assert outcome.deviation_scale == 0.0


def _fieldsOf(outcome: GlmOutcome) -> dict:
    """The fields a GlmOutcome was built from, to build another."""
    fields = {}
    for field in dataclasses.fields(outcome):
        fields[field.name] = getattr(outcome, field.name)
    return fields


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
    fields = _fieldsOf(outcome)
    GlmOutcome(**fields)
    with pytest.raises(ValidationError, match=match):
        GlmOutcome(**fields | change)


def test_the_raw_scale_parameters_on_the_raw_columns_give_the_standardized_eta_exactly() -> None:
    """A user fitting the raw columns (lme4 on a table) compares against beta_raw:
    a z-scored column's β over its sd, the intercept absorbing each one's β · mean / sd,
    a 0/1 column's β as drawn. The raw matrix here is built by hand, not by the coder.
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
    covariates = Covariates(columns=columns, metadata=metadata, grouping=groupingOf([]), score=None)
    outcome = run(Glm(), covariates, np.random.default_rng(0))
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
    np.testing.assert_allclose(raw @ outcome.beta_raw, standardized @ outcome.beta, rtol=1e-12)
    assert outcome.beta_raw[2:5].tolist() == outcome.beta[2:5].tolist()
