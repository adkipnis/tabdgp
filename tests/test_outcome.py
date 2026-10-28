"""The outcome module's seams: what an Outcome guarantees on its own,
and runModel, a model's call and the check of its result.
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.column import Column
from tabdgp.covariates import Covariates
from tabdgp.outcome import (
    GlmmOutcome,
    GlmOutcome,
    Outcome,
    runModel,
)
from tabdgp.prior import Glm, Glmm
from tests.fixtures.build import Y, covariatesOf, groupingOf

# four rows in two groups of g0
GROUPING = groupingOf([[0, 0, 1, 1]])


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


def _noisyScore(covariates: Covariates, rng: np.random.Generator, shift: float = 0.0) -> Outcome:
    """A model whose y is the score plus the generator's standard normals and a shift."""
    noise = rng.standard_normal(len(covariates.y_score), dtype=np.float32)  # (n,)
    y = covariates.y_score + noise + shift  # (n,)
    return Outcome(columns={'y': y}, metadata={'y': Y})


MODELS = [
    pytest.param(_noisyScore, 0.0, id='a function'),
    pytest.param(functools.partial(_noisyScore, shift=2.0), 2.0, id='a partial with its settings'),
]


@pytest.mark.parametrize(('model', 'shift'), MODELS)
def test_run_model_calls_a_model_with_the_covariates_and_the_generator_and_returns_its_outcome(
    model: object, shift: float
) -> None:
    """A study's model is any callable of the covariates and the key's generator,
    its settings bound by a partial; y must be its draw from that generator.
    """
    covariates = covariatesOf([[0, 0, 1, 1]], ['x0'])
    outcome = runModel(model, covariates, np.random.default_rng(0))
    noise = np.random.default_rng(0).standard_normal(4, dtype=np.float32)  # (n,)
    assert np.array_equal(outcome.columns['y'], covariates.y_score + noise + shift)


def _shortY(covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """A model whose y is one row short of the covariates."""
    y = np.zeros(len(covariates.y_score) - 1, dtype=np.float32)  # (n − 1,)
    return Outcome(columns={'y': y}, metadata={'y': Y})


def _bareY(covariates: Covariates, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """A model that returns its y without an Outcome around it."""
    return {'y': covariates.y_score}


def _x0(covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """A model that writes its outcome under a covariate's name."""
    return Outcome(columns={'x0': covariates.y_score}, metadata={'x0': Y})


MISFIT = [
    pytest.param(_shortY, ValueError, 'length', id='a y one row short'),
    pytest.param(_bareY, TypeError, 'not an Outcome', id='a dict, not an Outcome'),
    pytest.param(_x0, ValueError, 'x0', id="a column under a covariate's name"),
]


@pytest.mark.parametrize(('model', 'error', 'match'), MISFIT)
def test_run_model_refuses_a_models_result_of_the_wrong_length_or_type_or_a_covariates_name_naming_the_model(
    model: object, error: type[Exception], match: str
) -> None:
    """A user's model is checked only on what it returns, so the failure must say whose it is.
    Its dtypes are checked by its own Outcome(...) call, inside the model.
    A column under a covariate's name would replace that covariate in the dataset.
    """
    with pytest.raises(error, match=match) as raised:
        runModel(model, covariatesOf([[0, 0, 1, 1]], ['x0']), np.random.default_rng(0))
    assert model.__name__ in str(raised.value)


# each family with its link, y's type and a shape as a point, so the shape a run records is known
RUNS = [
    pytest.param('gaussian', 'identity', 'continuous', None, id='gaussian'),
    pytest.param('bernoulli', 'logit', 'binary', None, id='bernoulli'),
    pytest.param('poisson', 'log', 'count', None, id='poisson'),
    pytest.param('student_t', 'identity', 'continuous', 4.0, id='student_t'),
    pytest.param('negative_binomial', 'log', 'count', 4.0, id='negative_binomial'),
    pytest.param('latent_bernoulli', 'logit', 'binary', 4.0, id='latent_bernoulli'),
    pytest.param('gamma', 'log', 'continuous', 4.0, id='gamma'),
    pytest.param('beta', 'logit', 'continuous', 4.0, id='beta'),
]


@pytest.mark.parametrize(('family', 'link', 'column_type', 'shape'), RUNS)
@pytest.mark.parametrize(
    'covariates',
    [
        pytest.param(covariatesOf([], ['x0', 'x1'], rows=6), id='flat'),
        pytest.param(covariatesOf([[0, 0, 1, 1, 2, 2]], ['x0', 'x1']), id='grouped, pooled'),
    ],
)
def test_run_model_on_a_glm_returns_a_glm_outcome_without_per_factor_fields(
    family: str, link: str, column_type: str, shape: float | None, covariates: Covariates
) -> None:
    """A Glm is TabDGP's own model: runModel codes the covariates, draws the prior and writes y.
    On grouped covariates it is the pooled ground truth: the factors' codes are no term,
    and the prior has the one block fixed.
    Only the Gaussian and Student-t families have a σ_ε, only the contaminated families have a
    shape, and the deviation is zero at λ = 0.
    """
    glm = Glm(family=family, shape=shape)
    outcome = runModel(glm, covariates, np.random.default_rng(0))
    assert type(outcome) is GlmOutcome
    assert (outcome.prior.family, outcome.link) == (family, link)
    assert list(outcome.columns) == list(outcome.metadata) == ['y']
    assert (outcome.metadata['y'].type, outcome.metadata['y'].role) == (column_type, 'outcome')
    assert outcome.columns['y'].shape == (6,)
    assert outcome.fixed_terms == ['intercept', 'x0', 'x1']
    assert list(outcome.prior.split) == ['fixed']
    assert (outcome.parameters.sigma_eps is None) == (family not in ['gaussian', 'student_t'])
    assert outcome.prior.shape == shape
    assert outcome.deviation.tolist() == [0.0] * 6
    assert outcome.deviation_scale == 0.0


MISSHAPEN = [
    pytest.param({'parameters': {'beta': np.zeros(1)}}, 'beta', id='beta one term short'),
    pytest.param({'parameters': {'beta': np.zeros((2, 1))}}, 'beta', id='beta as a column'),
    pytest.param({'deviation': np.zeros(3)}, 'length', id='a deviation one row short'),
]


@pytest.mark.parametrize(('change', 'match'), MISSHAPEN)
def test_a_glm_outcome_checks_its_shapes_against_n_and_d(change: dict, match: str) -> None:
    """A consumer pairs beta with the terms by position and adds the deviation to η row by row,
    so a shape that disagrees fails where the outcome is built.
    """
    outcome = runModel(
        Glm(family='gaussian'), covariatesOf([[0, 0, 1, 1]], ['x0']), np.random.default_rng(0)
    )
    with pytest.raises(ValidationError, match=match):
        _replaced(outcome, change)


def _replaced(outcome: GlmOutcome, change: dict) -> GlmOutcome:
    """The outcome with some fields changed, a dict under 'parameters' changing theirs."""
    fields = dict(change)
    if 'parameters' in fields:
        fields['parameters'] = dataclasses.replace(outcome.parameters, **fields['parameters'])
    return dataclasses.replace(outcome, **fields)


def _covariance(sd: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Σ = diag(sd) corr diag(sd)."""
    return corr * np.outer(sd, sd)  # (q, q)


# six rows in three groups of g0 and a covariate of each type, of design width 1 but c's 2 and k's 3
TYPED_GROUPING = groupingOf([[0, 0, 1, 1, 2, 2]])
TYPED_COLUMNS = {
    'x': np.array([1.5, -2, 3, 0.25, 7, 4], dtype=np.float32),
    'b': np.array([0, 1, 1, 0, 1, 0]),
    'c': np.array([0, 1, 2, 2, 1, 0]),
    'k': np.array([0, 1, 2, 3, 1, 0]),
    'n': np.array([3, 0, 5, 1, 2, 9]),
}
TYPED_METADATA = {
    'x': Column(type='continuous'),
    'b': Column(type='binary', levels=2),
    'c': Column(type='categorical', levels=3),
    'k': Column(type='categorical', levels=4),
    'n': Column(type='count'),
}


def _typedCovariates(names: list[str]) -> Covariates:
    """The grouping and these covariates of TYPED_COLUMNS, in this order."""
    columns = dict(TYPED_GROUPING.codes)
    metadata = dict(TYPED_GROUPING.metadata)
    for name in names:
        columns[name] = TYPED_COLUMNS[name]
        metadata[name] = TYPED_METADATA[name]
    return Covariates(columns=columns, metadata=metadata, grouping=TYPED_GROUPING, y_score=None)


def test_the_raw_scale_parameters_on_the_raw_columns_give_the_standardized_eta_up_to_float32() -> (
    None
):
    """A user fitting the raw columns (lme4 on a table) compares against the raw-scale ground truth:
    a z-scored column's β over its sd, the intercept absorbing each one's β · mean / sd,
    a 0/1 column's β as drawn, and each group's random effects alike.
    Σ on the raw columns must give each row's random part the variance Σ gives it on the coded
    ones, which scaling Σ by the sds alone would miss in the intercept's row and column.
    The raw matrix here is built by hand, not by the coder.
    Both scales are float32, so they agree up to its rounding.
    """
    covariates = _typedCovariates(['x', 'b', 'c', 'n'])
    # at a budget of d every term is random too, the effects correlated
    glmm = Glmm(family='gaussian', random_effects=6, correlated_share=1.0)
    outcome = runModel(glmm, covariates, np.random.default_rng(0))
    parameters = outcome.parameters
    raw = np.stack(
        [
            np.ones(6),
            TYPED_COLUMNS['x'].astype(np.float64),
            TYPED_COLUMNS['b'],
            TYPED_COLUMNS['c'] == 1,
            TYPED_COLUMNS['c'] == 2,
            TYPED_COLUMNS['n'],
        ],
        axis=1,
    ).astype(np.float64)  # (n, d)
    standardized = covariates.toDesign().matrix  # (n, d)
    np.testing.assert_allclose(raw @ outcome.beta_raw, standardized @ parameters.beta, rtol=1e-6)
    assert outcome.beta_raw[2:5].tolist() == parameters.beta[2:5].tolist()
    assert outcome.random_terms['g0'] == outcome.fixed_terms
    codes = TYPED_GROUPING.codes['g0']  # (n,)
    random_raw = (raw * outcome.alpha_raw['g0'][codes]).sum(axis=1)  # (n,)
    random_standardized = (standardized * parameters.alpha['g0'][codes]).sum(axis=1)  # (n,)
    np.testing.assert_allclose(random_raw, random_standardized, rtol=1e-6, atol=1e-6)
    sigma = _covariance(parameters.sigma_alpha['g0'], parameters.corr_alpha['g0'])  # (q, q)
    sigma_raw = _covariance(outcome.sigma_alpha_raw['g0'], outcome.corr_alpha_raw['g0'])  # (q, q)
    variance_raw = ((raw @ sigma_raw) * raw).sum(axis=1)  # (n,)
    variance_standardized = ((standardized @ sigma) * standardized).sum(axis=1)  # (n,)
    np.testing.assert_allclose(variance_raw, variance_standardized, rtol=1e-6, atol=1e-6)


def test_the_ground_truths_arrays_are_float32_on_both_scales() -> None:
    """The ground truth goes to torch with the columns, so it has their precision, float32,
    on the raw columns too.
    """
    covariates = covariatesOf([[0, 0, 1, 1, 2, 2]], ['x0', 'x1'])
    outcome = runModel(Glmm(family='gaussian'), covariates, np.random.default_rng(0))
    parameters = outcome.parameters
    arrays = {
        'beta': parameters.beta,
        'beta_raw': outcome.beta_raw,
        'deviation': outcome.deviation,
        'sigma_alpha': parameters.sigma_alpha['g0'],
        'corr_alpha': parameters.corr_alpha['g0'],
        'alpha': parameters.alpha['g0'],
        'sigma_alpha_raw': outcome.sigma_alpha_raw['g0'],
        'corr_alpha_raw': outcome.corr_alpha_raw['g0'],
        'alpha_raw': outcome.alpha_raw['g0'],
    }
    for field, array in arrays.items():
        assert array.dtype == np.float32, field


def test_run_model_on_a_glmm_returns_a_glmm_outcome_with_one_block_per_factor() -> None:
    """A Glmm draws one block of random effects per grouping factor, crossed or nested alike.
    Each factor's terms are design terms led by its intercept,
    and its effects have one row per group; GlmmOutcome checks their q itself.
    The prior names the blocks fixed, then the factors in order.
    """
    covariates = covariatesOf([[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]], ['x0', 'x1'])
    outcome = runModel(Glmm(family='gaussian'), covariates, np.random.default_rng(0))
    assert type(outcome) is GlmmOutcome
    assert list(outcome.prior.split) == ['fixed', 'g0', 'g1']
    assert list(outcome.random_terms) == ['g0', 'g1']
    for factor, groups in [('g0', 3), ('g1', 2)]:
        terms = outcome.random_terms[factor]
        assert terms[0] == 'intercept'
        assert set(terms) <= set(outcome.fixed_terms)
        assert len(outcome.parameters.alpha[factor]) == groups


def _randomTerms(names: list[str], random_effects: int, seed: int) -> list[str]:
    """g0's random terms under a budget, on the grouping and these covariates."""
    glmm = Glmm(family='gaussian', random_effects=random_effects)
    outcome = runModel(glmm, _typedCovariates(names), np.random.default_rng(seed))
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
    """ADR 0006: the intercept always, then whole covariates, each taken if it still fits
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


def test_run_model_refuses_a_glmm_on_covariates_without_a_factor_naming_the_model() -> None:
    """A GLMM without a grouping factor would split var(η) over no random block
    and draw a GLM at a fraction of its scale, silently; draw's hyperprior refuses it earlier,
    so this guards a table read without factors.
    """
    covariates = covariatesOf([], ['x0'], rows=4)
    with pytest.raises(ValueError, match='Glmm'):
        runModel(Glmm(family='gaussian'), covariates, np.random.default_rng(0))


# changes of the parameters, which a GLMM's outcome checks against its random terms
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
    outcome = runModel(
        Glmm(family='gaussian', random_effects=2), covariates, np.random.default_rng(0)
    )
    assert outcome.random_terms == {'g0': ['intercept', 'x0']}
    with pytest.raises(ValidationError, match=match):
        _replaced(outcome, {'parameters': change})


def test_the_deviation_is_lambda_times_the_sd_of_x_beta_plus_z_alpha_times_the_score() -> None:
    """The deviation is the score at λ · sd(η_lin), η_lin = Xβ + Zα the linear part,
    the sd over the n rows, so that a fitted GLMM is misspecified by exactly this term.
    The score counts up from 0, so a deviation that z-scored it again would differ.
    The design matrix here is built by hand, not by the coder.
    """
    covariates = covariatesOf([[0, 0, 1, 1, 2, 2]], ['x0'])
    glmm = Glmm(family='gaussian', random_effects=2, deviation_scale=0.5)
    outcome = runModel(glmm, covariates, np.random.default_rng(0))
    x0 = np.arange(6.0)  # (n,)
    design = np.stack([np.ones(6), (x0 - x0.mean()) / x0.std()], axis=1)  # (n, d)
    codes = covariates.grouping.codes['g0']  # (n,)
    linear = design @ outcome.parameters.beta + (
        design * outcome.parameters.alpha['g0'][codes]
    ).sum(axis=1)  # (n,)
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
    plain = dataclasses.replace(scored_covariates, y_score=None)
    covariates = scored_covariates if scored else plain
    glmm = Glmm(family='gaussian', deviation_scale=deviation_scale)
    outcome = runModel(glmm, covariates, np.random.default_rng(0))
    reference = runModel(Glmm(family='gaussian'), plain, np.random.default_rng(0))
    assert outcome.deviation.tolist() == [0.0] * 6
    assert outcome.deviation.dtype == np.float32
    assert outcome.deviation_scale == deviation_scale
    assert outcome.columns['y'].tobytes() == reference.columns['y'].tobytes()


@pytest.mark.parametrize(('family', 'link', 'column_type', 'shape'), RUNS)
def test_run_model_on_a_glmm_records_the_shape_beside_the_random_effects(
    family: str, link: str, column_type: str, shape: float | None
) -> None:
    """The contaminated families draw on a GLMM as on a GLM: η holds the random effects,
    and the shape is the family's own, not a group's.
    """
    glmm = Glmm(family=family, shape=shape)
    covariates = covariatesOf([[0, 0, 1, 1, 2, 2]], ['x0', 'x1'])
    outcome = runModel(glmm, covariates, np.random.default_rng(0))
    assert type(outcome) is GlmmOutcome
    assert (outcome.prior.family, outcome.link, outcome.metadata['y'].type) == (
        family,
        link,
        column_type,
    )
    assert outcome.prior.shape == shape
    assert list(outcome.parameters.alpha) == ['g0']
