"""The outcome module's seams: what Covariates and an Outcome guarantee on their own,
and run, a model's call and the check of its result.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.column import Column
from tabdgp.outcome import Covariates, Outcome, ScmOutcome, run
from tests.fixtures.build import columnsOf, covariatesOf, groupingOf

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
