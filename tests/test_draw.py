"""The draw module's seams: what a Dataset guarantees on its own,
and draw(seed, index, hyperprior), one dataset per key.
"""

from __future__ import annotations

import functools
import hashlib
import json
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tabdgp.column import Column
from tabdgp.draw import Dataset, draw, drawOutcome
from tabdgp.hyperprior import Hyperprior, UniformInt
from tabdgp.outcome import (
    Covariates,
    GlmmOutcome,
    GlmOutcome,
    Outcome,
    ScmOutcome,
    run,
)
from tabdgp.prior import Glm, Glmm
from tests.fixtures.build import Y, betweenShare, graphOf, point


def test_a_dataset_holds_its_arrays_without_copying() -> None:
    """metabeta hands the columns to torch.from_numpy; a copy per draw would cost throughput."""
    values = np.zeros(5, dtype=np.float32)
    latent = np.ones(5, dtype=np.float32)
    y = np.ones(5, dtype=np.float32)
    dataset = Dataset(
        columns={'x0': values, 'y': y},
        metadata={'x0': Column(type='continuous', marginal='normal'), 'y': Y},
        graph=graphOf([[], []]),
        latent={'z0': latent},
        outcome=Outcome(columns={'y': y}, metadata={'y': Y}),
        hyperparameters=Hyperprior().resolveDataset(np.random.default_rng(0)),
        seed=0,
        index=0,
        version='0.0.1',
        numpy_version='2.3.0',
    )
    assert dataset.columns['x0'] is values and dataset.latent['z0'] is latent
    assert dataset.outcome.columns['y'] is y


def _assertIdentical(a: Dataset, b: Dataset) -> None:
    """Two datasets agree bit for bit: every array in dtype and bytes, everything else by ==."""
    for name in ['columns', 'latent']:
        arrays_a, arrays_b = getattr(a, name), getattr(b, name)
        assert list(arrays_a) == list(arrays_b)
        for key in arrays_a:
            assert arrays_a[key].dtype == arrays_b[key].dtype
            assert arrays_a[key].tobytes() == arrays_b[key].tobytes()
    assert a.metadata == b.metadata
    assert a.graph.names == b.graph.names
    assert (a.graph.mechanism, a.graph.activation) == (b.graph.mechanism, b.graph.activation)
    assert np.array_equal(a.graph.parents, b.graph.parents)
    assert np.array_equal(a.graph.latent, b.graph.latent)
    assert (a.graph.noise_scale, a.graph.init_scale) == (b.graph.noise_scale, b.graph.init_scale)
    assert (a.graph.unit, a.graph.group_share) == (b.graph.unit, b.graph.group_share)
    assert a.graph.group_factor == b.graph.group_factor
    assert (a.seed, a.index, a.version) == (b.seed, b.index, b.version)
    assert a.numpy_version == b.numpy_version
    assert a.hyperparameters == b.hyperparameters


@pytest.mark.parametrize(('seed', 'index'), [(-1, 0), (0, -1)])
def test_a_negative_seed_or_index_fails(seed: int, index: int) -> None:
    """default_rng([seed, index]) rejects negative entries, so a bad key fails before any draw."""
    with pytest.raises(ValueError):
        draw(seed, index, Hyperprior.grouped())


SEEDS = st.integers(0, 2**32)
INDICES = st.integers(0, 1000)


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, other_seed=SEEDS, index=INDICES, others=st.lists(INDICES, max_size=3))
def test_a_dataset_is_bitwise_identical_alone_inside_two_index_sets_and_under_a_second_seed_order(
    seed: int, other_seed: int, index: int, others: list[int]
) -> None:
    """The seed is the corpus:
    dataset i may not depend on what else a worker, batch or shard draws.
    On the grouped preset, so the grouping, the units, the two parts and y are covered.
    """
    grouped = Hyperprior.grouped()
    alone = draw(seed, index, grouped)
    first_set = [draw(seed, i, grouped) for i in [*others, index]][-1]
    second_set = [draw(seed, i, grouped) for i in [index, *reversed(others), index + 1]][0]
    other_first = [draw(other_seed, index, grouped), draw(seed, index, grouped)][1]
    for dataset in [first_set, second_set, other_first]:
        _assertIdentical(alone, dataset)


GRAPH_LISTS = [
    'names',
    'mechanism',
    'activation',
    'noise_scale',
    'init_scale',
    'unit',
    'group_share',
    'group_factor',
]


def _digest(datasets: list[Dataset]) -> str:
    """A sha256 over the datasets' arrays in dtype and bytes, their metadata and their graphs."""
    digest = hashlib.sha256()
    for dataset in datasets:
        for arrays in [dataset.columns, dataset.latent]:
            for name, values in arrays.items():
                digest.update(name.encode())
                digest.update(values.dtype.str.encode())
                digest.update(values.tobytes())
        digest.update(repr(dataset.metadata).encode())
        digest.update(dataset.graph.parents.tobytes())
        digest.update(dataset.graph.latent.tobytes())
        for field in GRAPH_LISTS:
            digest.update(repr(getattr(dataset.graph, field)).encode())
    return digest.hexdigest()


def test_scm_and_none_draw_0_3_0s_bits() -> None:
    """A refactor or a new form may not change a corpus the scm or none form drew under 0.3.0.
    The digest holds for the numpy that uv.lock pins, as the README's quickstart does.
    The resolved hyperparameters stay out: their repr is pydantic's.
    """
    presets = [Hyperprior.grouped(), Hyperprior.flat(), Hyperprior.grouped(outcome_form='none')]
    datasets = []
    for hyperprior in presets:
        for index in range(5):
            datasets.append(draw(3, index, hyperprior))
    assert _digest(datasets) == 'b749a156036d5573f7430318ac8928eca50cc121b334e51c21ce7af78f1e5aff'


def test_columns_metadata_graph_and_latent_agree_on_names_with_the_grouping_factors_first() -> None:
    """A consumer looks a column up by name in columns, metadata and the graph alike,
    and finds the grouping factors before the covariates.
    """
    for index in range(20):
        dataset = draw(0, index, Hyperprior.grouped(latent_nodes=2, factors=2))
        graph = dataset.graph
        observed = [name for name, latent in zip(graph.names, graph.latent) if not latent]
        latent = [name for name, latent in zip(graph.names, graph.latent) if latent]
        assert list(dataset.columns) == list(dataset.metadata)
        assert list(dataset.columns)[:2] == ['g0', 'g1']
        assert sorted(dataset.columns) == sorted(observed)
        assert list(dataset.latent) == latent


@pytest.mark.parametrize('preset', [Hyperprior.grouped, Hyperprior.flat], ids=['grouped', 'flat'])
def test_y_is_the_last_column_and_the_last_node_with_role_outcome_under_grouped_and_flat(
    preset: Callable[[], Hyperprior],
) -> None:
    """A consumer takes the last column as the outcome and the rest as its predictors,
    and no node reads y, so it is last in the graph too.
    """
    for index in range(20):
        dataset = draw(0, index, preset())
        assert list(dataset.columns)[-1] == 'y'
        assert list(dataset.metadata)[-1] == 'y'
        assert dataset.metadata['y'].role == 'outcome'
        assert dataset.graph.names[-1] == 'y'
        roles = [column.role for column in dataset.metadata.values()]
        assert roles.count('outcome') == 1


def test_outcome_none_gives_no_y() -> None:
    """A covariate-only table, spec #22's, is one override, and it has no outcome to hold."""
    for index in range(20):
        dataset = draw(0, index, Hyperprior.grouped(outcome_form='none'))
        assert dataset.outcome is None
        assert 'y' not in dataset.columns
        assert 'y' not in dataset.metadata
        assert 'y' not in dataset.graph.names
        assert all(column.role != 'outcome' for column in dataset.metadata.values())


@pytest.mark.parametrize('preset', [Hyperprior.grouped, Hyperprior.flat], ids=['grouped', 'flat'])
def test_the_scm_form_returns_a_scm_outcome_whose_y_is_the_datasets_and_whose_score_is_z_scored(
    preset: Callable[[], Hyperprior],
) -> None:
    """The scm form's truth is y's node before rank matching, z-scored over the rows like every
    row-unit node; the dataset and its outcome hold one y, not two copies.
    """
    for index in range(10):
        dataset = draw(0, index, preset())
        outcome = dataset.outcome
        assert isinstance(outcome, ScmOutcome)
        assert outcome.columns['y'] is dataset.columns['y']
        assert outcome.metadata['y'] is dataset.metadata['y']
        assert outcome.score.dtype == np.float32
        np.testing.assert_allclose(outcome.score.mean(), 0, atol=1e-5)
        np.testing.assert_allclose(outcome.score.std(), 1, atol=1e-5)


class _ScoreAsY:
    """A model that keeps the covariates it was last called with and returns their score as y."""

    def __call__(self, covariates: Covariates, rng: np.random.Generator) -> Outcome:
        self.covariates = covariates
        return Outcome(columns={'y': covariates.score}, metadata={'y': Y})


@pytest.mark.parametrize('preset', [Hyperprior.grouped, Hyperprior.flat], ids=['grouped', 'flat'])
def test_a_model_gets_the_covariates_the_grouping_and_the_score_and_its_y_is_the_last_column(
    preset: Callable[..., Hyperprior],
) -> None:
    """A study's model reads every observed column but y, the grouping and the score,
    each the dataset's own array, and its y joins the dataset after them, as the scm form's does.
    """
    model = _ScoreAsY()
    for index in range(10):
        dataset = draw(0, index, preset(outcome_form=model))
        covariates = model.covariates
        assert list(dataset.columns) == [*covariates.columns, 'y']
        for name, values in covariates.columns.items():
            assert dataset.columns[name] is values
        for name, codes in covariates.grouping.codes.items():
            assert dataset.columns[name] is codes
        assert dataset.columns['y'] is covariates.score
        assert dataset.metadata['y'].role == 'outcome'
        assert dataset.graph.names[-1] == 'y'


def test_the_datasets_outcome_form_records_the_model() -> None:
    """A dataset names what drew it; a user's model is part of that, as the object passed."""
    model = _ScoreAsY()
    dataset = draw(0, 0, Hyperprior.grouped(outcome_form=model))
    assert dataset.hyperparameters.outcome_form is model


@pytest.mark.parametrize('preset', [Hyperprior.grouped, Hyperprior.flat], ids=['grouped', 'flat'])
def test_glm_returns_y_last_with_a_glm_outcome_and_on_grouped_the_pooled_truth(
    preset: Callable[..., Hyperprior],
) -> None:
    """A GLM corpus needs no grouping (flat), and on grouped covariates it is the pooled truth:
    the factors' codes stay in the dataset but are no term, so a fitter's random-effect scale
    should go to zero. One term per design column besides the intercept.
    """
    for index in range(10):
        dataset = draw(0, index, preset(outcome_form='glm'))
        outcome = dataset.outcome
        assert type(outcome) is GlmOutcome
        assert list(dataset.columns)[-1] == 'y'
        assert dataset.graph.names[-1] == 'y'
        assert outcome.columns['y'] is dataset.columns['y']
        roles = [column.role for column in dataset.metadata.values()]
        assert roles.count('grouping_factor') == dataset.hyperparameters.factors
        assert len(outcome.fixed_terms) == 1 + dataset.hyperparameters.design_columns
        assert list(outcome.prior.split) == ['fixed']


def test_grouped_with_glmm_returns_y_last_with_a_glmm_outcome_of_one_block_per_factor() -> None:
    """metabeta's one call (ADR 0006): y last, after the grouping factors and the covariates,
    and one block of random effects per grouping factor, a row per group,
    each block within its budget of 6 and within the design.
    """
    for index in range(10):
        dataset = draw(0, index, Hyperprior.grouped(outcome_form='glmm'))
        outcome = dataset.outcome
        assert type(outcome) is GlmmOutcome
        assert list(dataset.columns)[-1] == 'y'
        assert outcome.columns['y'] is dataset.columns['y']
        factors = [
            name for name, column in dataset.metadata.items() if column.role == 'grouping_factor'
        ]
        assert list(outcome.random_terms) == factors
        assert list(outcome.prior.split) == ['fixed', *factors]
        for factor in factors:
            q = len(outcome.random_terms[factor])
            assert q <= min(6, len(outcome.fixed_terms))
            assert outcome.parameters.alpha[factor].shape == (dataset.metadata[factor].levels, q)


GLM_FAMILIES = [
    pytest.param('normal', 'continuous', np.float32, id='normal'),
    pytest.param('bernoulli', 'binary', np.int64, id='bernoulli'),
    pytest.param('poisson', 'count', np.int64, id='poisson'),
]


@pytest.mark.parametrize(
    ('preset', 'config'),
    [(Hyperprior.flat, Glm), (Hyperprior.grouped, Glmm)],
    ids=['glm on flat', 'glmm on grouped'],
)
@pytest.mark.parametrize(('family', 'column_type', 'dtype'), GLM_FAMILIES)
def test_glm_or_glmm_with_one_family_draws_that_family_only_and_ys_dtype_follows_it(
    family: str,
    column_type: str,
    dtype: type,
    preset: Callable[..., Hyperprior],
    config: type[Glm],
) -> None:
    """A corpus of one family is one override (ADR 0006),
    and a consumer stacks y by its type: a Normal y float32, a Bernoulli y codes, a count int64.
    """
    hyperprior = preset(outcome_form=config(y_family_shares=point('y_family_shares', family, Glm)))
    for index in range(10):
        dataset = draw(0, index, hyperprior)
        assert dataset.outcome.prior.y_family == family
        assert dataset.metadata['y'].type == column_type
        assert dataset.columns['y'].dtype == dtype


def test_a_normal_glm_at_large_n_and_small_sigma_eps_is_recovered_by_least_squares_on_the_coded_columns() -> (
    None
):
    """The truth refers to the design matrix a user fits on: least squares on the coded columns
    returns beta. At r = 0.9999 and a half-normal F, σ_ε is about 0.01.
    """
    glm = Glm(
        y_family_shares=point('y_family_shares', 'normal', Glm),
        sigma_eps_family_shares=point('sigma_eps_family_shares', 'half_normal', Glm),
        signal_share=0.9999,
    )
    hyperprior = Hyperprior.flat(rows=2000, outcome_form=glm)
    for index in range(5):
        dataset = draw(0, index, hyperprior)
        design = dataset.toDesign()
        y = dataset.columns['y'].astype(np.float64)  # (n,)
        estimate, *_ = np.linalg.lstsq(design.matrix, y, rcond=None)
        np.testing.assert_allclose(estimate, dataset.outcome.parameters.beta, atol=0.01)


def _linearPart(dataset: Dataset) -> np.ndarray:
    """Xβ + Σ_k Z_k α_k[codes_k] on the coded columns, a GLMM's η without the deviation."""
    outcome = dataset.outcome
    design = dataset.toDesign()
    linear = design.matrix @ outcome.parameters.beta  # (n,)
    for factor, terms in outcome.random_terms.items():
        positions = [design.names.index(term) for term in terms]
        z = design.matrix[:, positions]  # (n, q)
        alpha = outcome.parameters.alpha[factor][dataset.columns[factor]]  # (n, q)
        linear = linear + (z * alpha).sum(axis=1)  # (n,)
    return linear


@pytest.mark.parametrize('deviation_scale', [0.0, 0.3], ids=['lambda 0', 'lambda 0.3'])
def test_a_normal_glmms_y_minus_x_beta_every_factors_z_alpha_and_the_deviation_is_noise_at_sigma_eps(
    deviation_scale: float,
) -> None:
    """The truth refers to the design matrix a user fits on:
    η = Xβ + Σ_k Z_k α_k[codes_k] + the deviation, Z_k the design's columns of the factor's
    random terms, so what y leaves after them is the Normal noise, sd σ_ε, up to y's float32
    rounding. At r = 0.9999 and a half-normal F, σ_ε is about 0.01, too small to hide a wrong
    Zα, or at λ = 0.3 a deviation that is not the one which entered η.
    """
    glmm = Glmm(
        y_family_shares=point('y_family_shares', 'normal', Glm),
        sigma_eps_family_shares=point('sigma_eps_family_shares', 'half_normal', Glm),
        signal_share=0.9999,
        correlated_share=1.0,
        deviation_scale=deviation_scale,
    )
    hyperprior = Hyperprior.grouped(rows=2000, outcome_form=glmm)
    for index in range(5):
        dataset = draw(0, index, hyperprior)
        outcome = dataset.outcome
        y = dataset.columns['y'].astype(np.float64)  # (n,)
        residual = y - _linearPart(dataset) - outcome.deviation  # (n,)
        sigma_eps = outcome.parameters.sigma_eps
        assert abs(residual.std() - sigma_eps) < 0.1 * sigma_eps + 1e-4


def test_glmm_with_a_deviation_scale_returns_a_deviation_of_the_stated_sd_and_the_same_beta_and_alpha_as_without_it_under_one_key() -> (
    None
):
    """The white-box misspecified corpus differs from metabeta's only by the deviation:
    λ enters after the parameter draw, so under one key β, α and the prior do not move.
    The score is z-scored over the rows, so the deviation's sd is λ times the linear part's.
    """
    for index in range(10):
        plain = draw(0, index, Hyperprior.grouped(outcome_form='glmm'))
        deviated = draw(0, index, Hyperprior.grouped(outcome_form=Glmm(deviation_scale=0.3)))
        outcome = deviated.outcome
        assert outcome.deviation_scale == 0.3
        assert outcome.prior == plain.outcome.prior
        assert outcome.parameters.beta.tobytes() == plain.outcome.parameters.beta.tobytes()
        assert outcome.random_terms == plain.outcome.random_terms
        for factor in outcome.random_terms:
            assert (
                outcome.parameters.alpha[factor].tobytes()
                == plain.outcome.parameters.alpha[factor].tobytes()
            )
        linear_sd = _linearPart(deviated).std()
        assert outcome.deviation.std() == pytest.approx(0.3 * linear_sd, rel=1e-4)


# a real table's shape: sixty pupils in ten classes nested in five schools, three covariates
PUPILS = np.random.default_rng(0)
TABLE = pd.DataFrame(
    {
        'school': np.repeat(np.arange(5), 12),
        'class': np.repeat(np.arange(10), 6),
        'age': PUPILS.normal(10, 2, size=60),
        'girl': PUPILS.random(60) < 0.5,
        'track': pd.Categorical(PUPILS.choice(['a', 'b', 'c'], size=60)),
    }
)
TABLE_COVARIATES = Covariates.fromPandas(TABLE, ['school', 'class'])


def test_draw_outcome_on_from_pandas_is_deterministic_per_key_and_differs_between_keys() -> None:
    """A resampled outcome is keyed like a dataset, default_rng([seed, index]),
    so the oracle set a study builds downstream from a table is reproducible per key.
    """
    model = Glmm()
    first = drawOutcome(3, 1, TABLE_COVARIATES, model)
    again = drawOutcome(3, 1, TABLE_COVARIATES, model)
    other = drawOutcome(3, 2, TABLE_COVARIATES, model)
    keyed = run(model, TABLE_COVARIATES, np.random.default_rng([3, 1]))
    for outcome in [again, keyed]:
        assert np.array_equal(outcome.columns['y'], first.columns['y'])
        assert np.array_equal(outcome.parameters.beta, first.parameters.beta)
    assert not np.array_equal(other.parameters.beta, first.parameters.beta)


def test_a_resampled_glmm_outcome_has_as_many_groups_per_factor_as_the_table() -> None:
    """A random effect per group of each factor the table has, a row of alpha per school
    and per class, so a study reads a group's truth at its code.
    """
    for index in range(5):
        outcome = drawOutcome(0, index, TABLE_COVARIATES, Glmm())
        assert list(outcome.parameters.alpha) == ['school', 'class']
        assert outcome.parameters.alpha['school'].shape[0] == 5
        assert outcome.parameters.alpha['class'].shape[0] == 10


def test_to_pandas_leaves_the_latent_nodes_out() -> None:
    """A table holds what was observed, in the dataset's column order;
    the latent nodes stay on the dataset, which keeps them white-box.
    """
    for index in range(5):
        dataset = draw(0, index, Hyperprior.grouped(latent_nodes=2))
        table = dataset.toPandas()
        assert list(table.columns) == list(dataset.columns)
        assert len(table) == dataset.hyperparameters.rows
        assert not set(dataset.latent) & set(table.columns)


# every covariate type in reach, with room for a dozen design columns
EVERY_TYPE = {'continuous': 0.25, 'binary': 0.25, 'categorical': 0.25, 'ordered': 0.25}


def test_to_pandas_codes_categoricals_ordered_when_ordered_continuous_float32_and_counts_int64() -> (
    None
):
    """The column type picks the pandas dtype: a coded column is a categorical over its codes,
    so a pandas-based tool sees levels, not numbers, and a count stays an integer.
    A Poisson y is the dataset's count.
    """
    poisson = Glm(y_family_shares=point('y_family_shares', 'poisson', Glm))
    hyperprior = Hyperprior.grouped(
        rows=200, design_columns=12, type_shares=EVERY_TYPE, outcome_form=poisson
    )
    types = set()
    for index in range(5):
        dataset = draw(0, index, hyperprior)
        table = dataset.toPandas()
        for name, column in dataset.metadata.items():
            values = table[name]
            types.add(column.type)
            if column.levels is None:
                assert values.dtype == column.dtype
                assert np.array_equal(values.to_numpy(), dataset.columns[name])
            else:
                assert list(values.cat.categories) == list(range(column.levels))
                assert values.cat.ordered == (column.type == 'ordered')
                assert np.array_equal(values.cat.codes, dataset.columns[name])
    assert types == {'continuous', 'binary', 'categorical', 'ordered', 'count'}


def test_from_pandas_reads_a_datasets_table_with_its_metadata_into_its_columns_and_metadata() -> (
    None
):
    """A drawn table and its metadata come back as they went, the outcome y identified by its
    role and left out: each covariate's and factor's values, dtype and metadata as drawn.
    Key (2, 90) draws a crossed factor with a group per row, which lies within every other's
    groups, so only its declared metadata tells that the draw did not nest it.
    """
    hyperprior = Hyperprior.grouped(rows=200, design_columns=12, type_shares=EVERY_TYPE)
    datasets = [draw(0, index, hyperprior) for index in range(10)]
    datasets.append(draw(2, 90, Hyperprior.grouped()))
    for dataset in datasets:
        covariates = Covariates.fromPandas(dataset.toPandas(), metadata=dataset.metadata)
        observed = [name for name, column in dataset.metadata.items() if column.role != 'outcome']
        assert list(covariates.columns) == observed
        for name in observed:
            assert covariates.metadata[name] == dataset.metadata[name]
            assert covariates.columns[name].dtype == dataset.columns[name].dtype
            assert np.array_equal(covariates.columns[name], dataset.columns[name])


def test_to_design_is_the_coding_the_glm_used() -> None:
    """The truth refers to the design matrix a user fits on, which toDesign returns.
    At r = 1, τ_ε and so σ_ε are 0, and y is η = Xβ in float32, exactly.
    On grouped covariates the factors are no term.
    """
    glm = Glm(y_family_shares=point('y_family_shares', 'normal', Glm), signal_share=1.0)
    for index in range(5):
        dataset = draw(0, index, Hyperprior.grouped(outcome_form=glm))
        design = dataset.toDesign()
        outcome = dataset.outcome
        assert design.names == outcome.fixed_terms
        assert (design.means, design.sds) == (outcome.means, outcome.sds)
        eta = design.matrix @ outcome.parameters.beta  # (n,)
        assert np.array_equal(dataset.columns['y'], eta.astype(np.float32))


def _shiftedScore(covariates: Covariates, rng: np.random.Generator, shift: float) -> Outcome:
    """A model whose y is the score plus a shift, its setting bound by a partial."""
    return Outcome(columns={'y': covariates.score + np.float32(shift)}, metadata={'y': Y})


def _throughJson(dataset: Dataset) -> dict:
    """The dataset's dict after a trip through strict JSON."""
    return json.loads(json.dumps(dataset.toDict(), allow_nan=False))


def _restored(record: dict) -> np.ndarray:
    """An array from its dict, its values in its dtype."""
    return np.array(record['values'], dtype=record['dtype'])


def test_to_dict_is_json_serializable_and_records_dtypes_and_a_users_model_by_name() -> None:
    """A dataset survives a trip through JSON:
    each array comes back bit for bit from its dtype and values, the truth's float32 too,
    and each object and distribution keeps its class.
    y is recorded once, among the dataset's columns: the outcome's record holds its truth.
    A study's model, which JSON cannot hold, is recorded by its qualified name.
    """
    dataset = draw(0, 0, Hyperprior.grouped(outcome_form='glmm'))
    record = _throughJson(dataset)
    arrays = {
        **{('columns', name): values for name, values in dataset.columns.items()},
        **{('latent', name): values for name, values in dataset.latent.items()},
        ('graph', 'parents'): dataset.graph.parents,
        ('outcome', 'parameters', 'beta'): dataset.outcome.parameters.beta,
    }
    for factor, alpha in dataset.outcome.parameters.alpha.items():
        arrays['outcome', 'parameters', 'alpha', factor] = alpha
    for path, values in arrays.items():
        stored = record
        for key in path:
            stored = stored[key]
        restored = _restored(stored)
        assert restored.dtype == values.dtype
        assert restored.tobytes() == values.tobytes()
    assert record['outcome']['class'] == 'GlmmOutcome'
    assert not {'columns', 'metadata'} & set(record['outcome'])
    assert record['hyperparameters']['outcome_form']['class'] == 'Glmm'
    level_alpha = {'class': 'LogUniform', 'low': 0.5, 'high': 5.0}
    assert record['hyperparameters']['level_alpha'] == level_alpha
    models = {'_ScoreAsY': _ScoreAsY(), '_shiftedScore': functools.partial(_shiftedScore, shift=1)}
    for name, model in models.items():
        record = _throughJson(draw(0, 0, Hyperprior.grouped(outcome_form=model)))
        assert record['hyperparameters']['outcome_form'].endswith(f'test_draw.{name}')


def test_from_pandas_reads_the_metadata_to_dict_wrote_after_a_trip_through_json() -> None:
    """A table saved beside its metadata file comes back whole:
    the metadata as toDict writes it, each Column's fields and its class, is a declaration.
    """
    for index in range(5):
        dataset = draw(0, index, Hyperprior.grouped())
        metadata = _throughJson(dataset)['metadata']
        covariates = Covariates.fromPandas(dataset.toPandas(), metadata=metadata)
        for name, column in covariates.metadata.items():
            assert column == dataset.metadata[name]


@pytest.mark.parametrize(
    ('outcome_type', 'dtype'), [('count', np.int64), ('continuous', np.float32)]
)
def test_a_count_y_is_int64_and_a_continuous_y_float32(outcome_type: str, dtype: type) -> None:
    """A consumer stacks y by its type: a count is an integer a Poisson likelihood reads,
    not a code 0..k − 1, and a continuous y shares the covariates' dtype.
    """
    shares = point('outcome_type_shares', outcome_type)
    hyperprior = Hyperprior.grouped(outcome_type_shares=shares)
    for index in range(5):
        assert draw(0, index, hyperprior).columns['y'].dtype == dtype


# one latent root z0 and a sure linear edge from it to the one other node, in a flat table
SURE_EDGE = {
    'rows': 1000,
    'latent_nodes': 1,
    'mechanism_shares': point('mechanism_shares', 'linear'),
    'noise_scale': 0.01,
    'edge_offset': 30.0,
    'hub_scale': 0.0,
    'block_bonus': 0.0,
}


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ranks_a = np.argsort(np.argsort(a))
    ranks_b = np.argsort(np.argsort(b))
    return float(np.corrcoef(ranks_a, ranks_b)[0, 1])


@pytest.mark.parametrize(
    ('outcome_noise_scale', 'low', 'high'),
    [(0.01, 0.95, 1.0), (100.0, 0.0, 0.2)],
    ids=['learnable', 'hopeless'],
)
def test_y_depends_on_its_parents_at_low_outcome_noise_and_is_near_pure_noise_at_a_high_one(
    outcome_noise_scale: float, low: float, high: float
) -> None:
    """The outcome's noise scale is its own:
    a learnable y at a low one, a hopeless one at a high one, whatever noise_scale says.
    With y the one node after z0, the correlation between them is the signal's share,
    since the noise scale is relative to the z-scored output.
    A tie-free marginal, since a zero-inflated y's ties cost rank correlation on their own.
    """
    hyperprior = Hyperprior.flat(
        design_columns=0,
        outcome_noise_scale=outcome_noise_scale,
        outcome_type_shares=point('outcome_type_shares', 'continuous'),
        marginal_shares=point('marginal_shares', 'normal'),
        **SURE_EDGE,
    )
    for index in range(10):
        dataset = draw(0, index, hyperprior)
        assert dataset.graph.parents.tolist() == [[False, False], [True, False]]
        correlation = abs(_spearman(dataset.columns['y'], dataset.latent['z0']))
        assert low <= correlation <= high


def _exceedsAFactorsGroups(dataset: Dataset, name: str) -> bool:
    """Whether a column has more levels than some factor has groups,
    the one case in which a column at group_level_share = 1 stays at the row.
    """
    column = dataset.metadata.get(name)
    groups = [
        factor.levels for factor in dataset.metadata.values() if factor.role == 'grouping_factor'
    ]
    return column is not None and column.levels is not None and column.levels > min(groups)


GLM_OVERRIDE = Glm(intercept_scale=0.5, deviation_scale=0.3)  # τ_0 in the prior, λ beside it
GLMM_OVERRIDE = Glmm(random_effects=1)  # every factor's intercept alone


@settings(max_examples=25, deadline=None)
@given(
    rows=st.integers(1, 300),
    design_columns=st.integers(0, 19),
    latent_nodes=st.integers(0, 9),
    max_levels=st.integers(3, 6),
    factors=st.integers(0, 3),
    groups=st.integers(3, 8),
    nested=st.booleans(),
    balanced=st.booleans(),
    group_level=st.booleans(),
    group_share=st.sampled_from([0.0, 0.5, 1.0]),
    form=st.sampled_from(['scm', 'none', GLM_OVERRIDE, GLMM_OVERRIDE]),
    outcome_type=st.sampled_from(Hyperprior.categories('outcome_type_shares')),
)
def test_every_override_is_honored(
    rows: int,
    design_columns: int,
    latent_nodes: int,
    max_levels: int,
    factors: int,
    groups: int,
    nested: bool,
    balanced: bool,
    group_level: bool,
    group_share: float,
    form: str | Glm | Glmm,
    outcome_type: str,
) -> None:
    """metabeta passes its sizes as points; the dataset must have exactly those sizes.

    The fields without a visible size are honored through hyperparameters, which must hold
    every point override as given, a field drawn per node through the graph, one drawn per
    factor through the factor's column, the scm form's through y's column, a GLM's through
    its outcome's prior and a GLMM's through each factor's random terms. A point groups
    cannot nest, so a chain draws its groups from a range below the point.
    """
    # a point above its bound fails instead: rows ≥ 2d, m ≤ n and a chain of f factors in f + 1 rows
    assume(rows >= max(2 * design_columns, groups, factors + 1))
    assume(groups >= factors + 1)  # the bottom factor of a chain of f needs f + 1 groups
    assume(form != GLMM_OVERRIDE or factors > 0)  # a GLMM needs a factor
    hyperprior = Hyperprior.grouped(
        rows=rows,
        design_columns=design_columns,
        latent_nodes=latent_nodes,
        max_levels=max_levels,
        noise_scale=0.5,
        outcome_noise_scale=0.5,
        factors=factors,
        groups=UniformInt(2, groups) if nested else groups,
        nested_share=1.0 if nested else 0.0,
        balanced_share=1.0 if balanced else 0.0,
        size_alpha=2.0,
        group_level_share=1.0 if group_level else 0.0,
        group_share=group_share,
        outcome_form=form,
        outcome_type_shares=point('outcome_type_shares', outcome_type),
        poisson_log_rate=2.0,
    )
    dataset = draw(0, 0, hyperprior)
    metadata = dataset.metadata
    assert sum(column.design_columns for column in metadata.values()) == design_columns
    for array in [*dataset.columns.values(), *dataset.latent.values()]:
        assert array.shape == (rows,)
    assert len(dataset.latent) == latent_nodes
    # the grouping factors, each recording its size draw
    factor_names = [name for name, column in metadata.items() if column.role == 'grouping_factor']
    assert factor_names == [f'g{i}' for i in range(factors)]
    for i, name in enumerate(factor_names):
        column = metadata[name]
        assert 2 <= column.levels <= groups if nested else column.levels == groups
        assert column.nests_in == (f'g{i - 1}' if nested and i > 0 else None)
        assert column.parameters == ({} if balanced else {'size_alpha': 2.0})
    for name, column in metadata.items():
        if column.role == 'covariate':
            assert column.levels is None or column.levels <= max_levels
    # the outcome's fields through y's column
    assert ('y' in metadata) == (form != 'none')
    if form == 'scm':
        assert metadata['y'].type == outcome_type
        if outcome_type == 'count':
            assert metadata['y'].parameters == {'poisson_log_rate': 2.0}
    if form == GLM_OVERRIDE:
        assert dataset.outcome.prior.beta_scales[0] == 0.5
        assert dataset.outcome.deviation_scale == 0.3
    if form == GLMM_OVERRIDE:
        assert list(dataset.outcome.random_terms) == factor_names
        for name in factor_names:
            assert dataset.outcome.random_terms[name] == ['intercept']
    # the per-node fields through the graph
    graph = dataset.graph
    for name, unit in zip(graph.names, graph.unit):
        if name not in [*factor_names, 'y'] and group_level and factors > 0:
            assert unit != 'row' or _exceedsAFactorsGroups(dataset, name)
        elif name not in factor_names:
            assert unit == 'row'
    assert set(graph.noise_scale) <= {None, 0.5}
    assert set(graph.group_share) <= {None, group_share}
    # the per-dataset fields through hyperparameters, the per-factor and per-node ones as given
    resolved = dataset.hyperparameters
    assert (resolved.rows, resolved.design_columns, resolved.latent_nodes) == (
        rows,
        design_columns,
        latent_nodes,
    )
    assert (resolved.max_levels, resolved.noise_scale, resolved.factors) == (
        max_levels,
        0.5,
        factors,
    )
    assert (resolved.balanced_share, resolved.nested_share, resolved.group_level_share) == (
        float(balanced),
        float(nested),
        float(group_level),
    )
    assert (resolved.groups, resolved.size_alpha, resolved.group_share) == (
        hyperprior.groups,
        2.0,
        group_share,
    )
    assert (resolved.outcome_form, resolved.outcome_noise_scale, resolved.poisson_log_rate) == (
        form,
        0.5,
        2.0,
    )


def test_a_distribution_override_is_honored() -> None:
    """A simulation study narrows one axis by a distribution; every draw stays inside it."""
    for index in range(20):
        hyperprior = Hyperprior.flat(rows=UniformInt(30, 40), design_columns=1, outcome_form='none')
        dataset = draw(0, index, hyperprior)
        [values] = dataset.columns.values()
        assert 30 <= len(values) <= 40


def test_a_budget_of_0_gives_no_columns() -> None:
    """metabeta's budget may be 0: a table of rows and no covariates, whose graph is latent only."""
    for index in range(10):
        hyperprior = Hyperprior.flat(design_columns=0, latent_nodes=2, rows=50, outcome_form='none')
        dataset = draw(0, index, hyperprior)
        assert dataset.columns == {}
        assert dataset.metadata == {}
        assert dataset.graph.latent.all()
        assert len(dataset.latent['z0']) == 50


def test_every_column_and_latent_node_is_finite() -> None:
    """scamd's repair and retry hid broken draws; here a broken one would reach the consumer.

    Zero noise lets a dead relu leave a node constant, the case most likely to break.
    """
    for noise_scale in [0.0, 0.5]:
        hyperprior = Hyperprior(noise_scale=noise_scale, outcome_noise_scale=noise_scale)
        for index in range(50):
            dataset = draw(0, index, hyperprior)
            for values in [*dataset.columns.values(), *dataset.latent.values()]:
                assert np.isfinite(values).all()


def test_every_continuous_column_is_non_constant() -> None:
    """A consumer z-scores a continuous column, so a constant one divides by zero.

    At the default noise scales: an override noise_scale = 0 asks for a noiseless SCM,
    in which a dead relu is the constant function and its column is constant.
    """
    for index in range(100):
        dataset = draw(0, index, Hyperprior.grouped())
        for name, values in dataset.columns.items():
            if dataset.metadata[name].type == 'continuous':
                assert values.std() > 0


def _neighborAgreement(codes: np.ndarray, parent: np.ndarray) -> float:
    """The share of rows whose code equals the next row's in the parent's order.

    A column cut from its parent changes code only at its k − 1 thresholds, whatever its sign
    or label permutation; an independent one agrees only by chance, about the sum of squared
    level shares.
    """
    ordered = codes[np.argsort(parent)]  # (n,)
    return float((ordered[1:] == ordered[:-1]).mean())


def test_each_binary_categorical_and_ordered_column_depends_on_its_parents_at_low_noise() -> None:
    """scamd's categoricals were nearly independent of their parents, and only shapes were tested.

    One latent root, one typed column of k ≤ 3 and a sure edge between them, so the column must
    follow the root's values; flat, so that the edge is the graph's only one.
    """
    for type, design_columns in [('binary', 1), ('categorical', 2), ('ordered', 2)]:
        hyperprior = Hyperprior.flat(
            design_columns=design_columns,
            type_shares=point('type_shares', type),
            max_levels=3,
            level_alpha=5.0,
            outcome_form='none',
            **SURE_EDGE,
        )
        for index in range(10):
            dataset = draw(0, index, hyperprior)
            assert dataset.graph.parents.tolist() == [[False, False], [True, False]]
            agreement = _neighborAgreement(dataset.columns['x0'], dataset.latent['z0'])
            assert agreement > 0.95


PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'


def test_the_dataset_carries_the_installed_version() -> None:
    """A key reproduces a dataset only within one version, so the dataset names its own."""
    expected = tomllib.loads(PYPROJECT.read_text())['project']['version']
    assert draw(0, 0, Hyperprior.grouped()).version == expected


def test_the_dataset_carries_the_installed_numpy_version() -> None:
    """numpy promises no stable streams across its releases, so the dataset names numpy's too."""
    assert draw(0, 0, Hyperprior.grouped()).numpy_version == np.__version__


def test_every_array_has_length_n() -> None:
    """A group-level column or latent node has one value per group until draw copies it to the rows,
    so a consumer stacks every array without looking at the units.
    """
    hyperprior = Hyperprior.grouped(rows=80, group_level_share=0.8, latent_nodes=3)
    for index in range(20):
        dataset = draw(0, index, hyperprior)
        assert 'y' in dataset.columns
        outcome = dataset.outcome
        arrays = [*dataset.columns.values(), *dataset.latent.values(), *outcome.columns.values()]
        for array in [*arrays, outcome.score]:
            assert array.shape == (80,)


def test_each_g_column_is_int64_codes_0_to_g_minus_1_with_every_group_present() -> None:
    """metabeta indexes a random effect by the codes; a missing group would give an empty one."""
    for index in range(20):
        dataset = draw(0, index, Hyperprior.grouped(factors=3))
        factors = [
            name for name, column in dataset.metadata.items() if column.role == 'grouping_factor'
        ]
        assert factors == ['g0', 'g1', 'g2']
        for name in factors:
            codes = dataset.columns[name]
            assert codes.dtype == np.int64
            assert np.array_equal(np.unique(codes), np.arange(dataset.metadata[name].levels))


def test_every_level_of_a_covariate_appears() -> None:
    """A level no row has would be an all-zero dummy column in metabeta's design matrix;
    a group-level column shows at most m levels, so one with more stays at the row.
    An override group_share = 1 is the known exception (ADR 0003), so the test draws the defaults.
    """
    for index in range(100):
        dataset = draw(0, index, Hyperprior.grouped())
        for name, column in dataset.metadata.items():
            if column.role == 'covariate' and column.levels is not None:
                codes = dataset.columns[name]
                assert np.array_equal(np.unique(codes), np.arange(column.levels))


def _unitOf(dataset: Dataset) -> dict[str, str]:
    return dict(zip(dataset.graph.names, dataset.graph.unit))


def _isConstantWithinGroups(values: np.ndarray, codes: np.ndarray) -> bool:
    first_rows = np.unique(codes, return_index=True)[1]  # (m,)
    return bool(np.array_equal(values, values[first_rows][codes]))


def test_a_group_level_column_or_latent_node_is_constant_within_its_groups() -> None:
    """A school's size is one value per school: a column or latent node at a factor's unit
    repeats its group's value over the group's rows, whatever its type.
    A broadcast through another factor's codes would break it.
    """
    hyperprior = Hyperprior.grouped(rows=200, group_level_share=1.0, latent_nodes=3)
    for index in range(20):
        dataset = draw(0, index, hyperprior)
        unit = _unitOf(dataset)
        for name, column in dataset.metadata.items():
            if column.role == 'covariate' and unit[name] == 'row':
                assert _exceedsAFactorsGroups(dataset, name)
            elif column.role == 'covariate':
                assert _isConstantWithinGroups(dataset.columns[name], dataset.columns[unit[name]])
        for name, values in dataset.latent.items():
            assert _isConstantWithinGroups(values, dataset.columns[unit[name]])


def test_a_group_level_column_takes_its_marginals_values_at_g_distinct_ranks() -> None:
    """Its marginal is sampled at m values, one per group, not at n values with ties;
    a tie-free family makes the count exact.
    """
    hyperprior = Hyperprior.grouped(
        rows=200,
        group_level_share=1.0,
        type_shares=point('type_shares', 'continuous'),
        marginal_shares=point('marginal_shares', 'normal'),
    )
    for index in range(20):
        dataset = draw(0, index, hyperprior)
        unit = _unitOf(dataset)
        for name, column in dataset.metadata.items():
            if column.role == 'covariate':
                levels = dataset.metadata[unit[name]].levels
                assert len(np.unique(dataset.columns[name])) == levels


def test_group_share_0_under_one_factor_gives_every_within_group_covariate_a_between_group_share_at_chance_level() -> (
    None
):
    """The ablation #22 asks for switches group confounding off by this one override.
    Under one factor every parent lies in the group factor's chain, so a share of 0 drops the
    group part and every row-unit node, a root or a child of a group-level parent, is i.i.d.
    over rows; the chance level is about (m − 1) / (n − 1), here 0.01.
    """
    hyperprior = Hyperprior.grouped(rows=2000, factors=1, groups=20, group_share=0.0)
    for index in range(20):
        dataset = draw(0, index, hyperprior)
        unit = _unitOf(dataset)
        codes = dataset.columns['g0']
        within_group = [
            name
            for name, column in dataset.metadata.items()
            if column.role == 'covariate' and unit[name] == 'row'
        ]
        for name in within_group:
            assert betweenShare(dataset.columns[name], codes) < 0.03


def test_flat_gives_no_g_column_every_unit_row_and_no_two_part_field() -> None:
    """A flat study changes nothing: spec #1's table, with the grouping fields idle."""
    for index in range(20):
        dataset = draw(0, index, Hyperprior.flat())
        graph = dataset.graph
        assert all(column.role != 'grouping_factor' for column in dataset.metadata.values())
        assert not any(name.startswith('g') for name in graph.names)
        assert set(graph.unit) <= {'row'}
        assert set(graph.group_share) <= {None}
        assert set(graph.group_factor) <= {None}
        assert dataset.hyperparameters.factors == 0


def test_draw_without_a_hyperprior_fails() -> None:
    """No preset is the default: a caller names the kind of table it draws."""
    with pytest.raises(TypeError):
        draw(0, 0)


def test_group_share_1_gives_every_row_unit_column_constant_within_its_groups() -> None:
    """At share 1 the row part has weight 0, so a row-unit node is one value per group of its
    group factor; the column keeps its ties, so it is too, whatever its type.
    y is one of them, so group confounding reaches the outcome.
    """
    hyperprior = Hyperprior.grouped(rows=600, factors=2, group_share=1.0, group_level_share=0.0)
    for index in range(20):
        dataset = draw(0, index, hyperprior)
        group_factor = dict(zip(dataset.graph.names, dataset.graph.group_factor))
        for name, column in dataset.metadata.items():
            if column.role != 'grouping_factor':
                codes = dataset.columns[group_factor[name]]
                assert _isConstantWithinGroups(dataset.columns[name], codes)
        for name, values in dataset.latent.items():
            assert _isConstantWithinGroups(values, dataset.columns[group_factor[name]])
