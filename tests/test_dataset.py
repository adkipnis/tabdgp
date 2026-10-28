"""The dataset module's seams: what a Dataset guarantees on its own,
and drawDataset(seed, index, scm), one dataset per key.
"""

from __future__ import annotations

import hashlib
import platform
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tabdgp.column import Column
from tabdgp.covariates import Covariates
from tabdgp.dataset import Dataset, drawDataset, modelOutcome
from tabdgp.outcome import (
    GlmmOutcome,
    GlmOutcome,
    Outcome,
)
from tabdgp.prior import Glm, Glmm
from tabdgp.scm import Scm, UniformInt
from tests.fixtures.build import Y, betweenShare, graphOf, groupingOf, point


def test_a_dataset_holds_its_arrays_without_copying() -> None:
    """metabeta hands the columns to torch.from_numpy; a copy per draw would cost throughput.
    The columns are the covariates' arrays, then the outcome's.
    """
    values = np.zeros(5, dtype=np.float32)
    latent = np.ones(5, dtype=np.float32)
    y = np.ones(5, dtype=np.float32)
    covariates = Covariates(
        columns={'x0': values},
        metadata={'x0': Column(type='continuous', marginal='normal')},
        grouping=groupingOf([]),
        y_score=None,
    )
    dataset = Dataset(
        covariates=covariates,
        graph=graphOf([[], []]),
        latent={'z0': latent},
        outcome=Outcome(columns={'y': y}, metadata={'y': Y}),
        hyperparameters=Scm().resolveDataset(np.random.default_rng(0)),
        seed=0,
        index=0,
        version='0.0.1',
        numpy_version='2.3.0',
    )
    assert list(dataset.columns) == list(dataset.metadata) == ['x0', 'y']
    assert dataset.columns['x0'] is values and dataset.latent['z0'] is latent
    assert dataset.columns['y'] is y


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


def _assertIdentical(a: Dataset, b: Dataset) -> None:
    """Two datasets agree bit for bit: the digest's arrays, metadata and graph, the rest by ==."""
    assert _digest([a]) == _digest([b])
    assert (a.seed, a.index, a.version, a.numpy_version) == (
        b.seed,
        b.index,
        b.version,
        b.numpy_version,
    )
    assert a.hyperparameters == b.hyperparameters


@pytest.mark.parametrize(('seed', 'index'), [(-1, 0), (0, -1)])
def test_a_negative_seed_or_index_fails(seed: int, index: int) -> None:
    """default_rng([seed, index]) rejects negative entries, so a bad key fails before any draw."""
    with pytest.raises(ValueError):
        drawDataset(seed, index, Scm.grouped())


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
    grouped = Scm.grouped()
    alone = drawDataset(seed, index, grouped)
    first_set = [drawDataset(seed, i, grouped) for i in [*others, index]][-1]
    second_set = [drawDataset(seed, i, grouped) for i in [index, *reversed(others), index + 1]][0]
    other_first = [drawDataset(other_seed, index, grouped), drawDataset(seed, index, grouped)][1]
    for dataset in [first_set, second_set, other_first]:
        _assertIdentical(alone, dataset)


def _factors(dataset: Dataset) -> list[str]:
    """The grouping factors' names, in the dataset's order."""
    return [name for name, column in dataset.metadata.items() if column.role == 'grouping_factor']


# bits hold per CPU type and numpy build: elsewhere numpy's float draws (uniform, beta, gamma)
# and functions (exp, tanh, sin, matmul) differ in the last bit, and a rejection shifts the stream
@pytest.mark.skipif(
    (sys.platform, platform.machine()) != ('darwin', 'arm64'),
    reason='the digest is pinned on macOS arm64',
)
def test_scm_and_none_draw_0_3_0s_bits() -> None:
    """A refactor or a new model may not change a corpus drawn under the SCM or none in 0.3.0.
    The digest holds for the numpy that uv.lock pins, as the README's quickstart does.
    The resolved hyperparameters stay out: their repr is pydantic's.
    """
    presets = [Scm.grouped(), Scm.flat(), Scm.grouped(outcome_source='none')]
    datasets = []
    for scm in presets:
        for index in range(5):
            datasets.append(drawDataset(3, index, scm))
    assert _digest(datasets) == 'b749a156036d5573f7430318ac8928eca50cc121b334e51c21ce7af78f1e5aff'


def test_columns_metadata_graph_and_latent_agree_on_names_with_the_grouping_factors_first() -> None:
    """A consumer looks a column up by name in columns, metadata and the graph alike,
    and finds the grouping factors before the covariates.
    """
    for index in range(20):
        dataset = drawDataset(0, index, Scm.grouped(latent_nodes=2, factors=2))
        graph = dataset.graph
        observed = [name for name, latent in zip(graph.names, graph.latent) if not latent]
        latent = [name for name, latent in zip(graph.names, graph.latent) if latent]
        assert list(dataset.columns) == list(dataset.metadata)
        assert list(dataset.columns)[:2] == ['g0', 'g1']
        assert sorted(dataset.columns) == sorted(observed)
        assert list(dataset.latent) == latent


def test_outcome_none_gives_no_y_and_no_score() -> None:
    """A covariate-only table, spec #22's, is one override, and it has no outcome to hold.
    It never draws y's node, so its covariates have no score.
    """
    for index in range(20):
        dataset = drawDataset(0, index, Scm.grouped(outcome_source='none'))
        assert dataset.outcome is None
        assert dataset.covariates.y_score is None
        assert 'y' not in dataset.columns
        assert 'y' not in dataset.metadata
        assert 'y' not in dataset.graph.names
        assert all(column.role != 'outcome' for column in dataset.metadata.values())


@pytest.mark.parametrize('preset', [Scm.grouped, Scm.flat], ids=['grouped', 'flat'])
def test_under_the_scm_the_covariates_carry_ys_score_z_scored(
    preset: Callable[[], Scm],
) -> None:
    """The SCM's ground truth is y's node before rank matching,
    z-scored over the rows like every row-unit node, and a model's deviation reads it.
    """
    for index in range(10):
        score = drawDataset(0, index, preset()).covariates.y_score
        assert score.dtype == np.float32
        np.testing.assert_allclose(score.mean(), 0, atol=1e-5)
        np.testing.assert_allclose(score.std(), 1, atol=1e-5)


@pytest.mark.parametrize('preset', [Scm.grouped, Scm.flat], ids=['grouped', 'flat'])
def test_y_is_the_last_column_and_node_with_role_outcome_and_one_array_with_the_outcomes(
    preset: Callable[[], Scm],
) -> None:
    """A consumer takes the last column as the outcome and the rest as its predictors,
    and no node reads y, so it is last in the graph too.
    The dataset and its outcome hold one y, not two copies.
    """
    for index in range(10):
        dataset = drawDataset(0, index, preset())
        assert list(dataset.columns)[-1] == 'y'
        assert list(dataset.metadata)[-1] == 'y'
        assert dataset.graph.names[-1] == 'y'
        roles = [column.role for column in dataset.metadata.values()]
        assert roles.count('outcome') == 1
        assert dataset.metadata['y'].role == 'outcome'
        assert dataset.outcome.columns['y'] is dataset.columns['y']
        assert dataset.outcome.metadata['y'] is dataset.metadata['y']


@pytest.mark.parametrize('preset', [Scm.grouped, Scm.flat], ids=['grouped', 'flat'])
def test_glm_returns_a_glm_outcome_and_on_grouped_the_pooled_ground_truth(
    preset: Callable[[], Scm],
) -> None:
    """A GLM corpus needs no grouping (flat),
    and on grouped covariates it is the pooled ground truth:
    the factors' codes stay in the covariates but are no term, so a fitter's random-effect scale
    should go to zero. One term per design column besides the intercept.
    """
    glm = Glm(family='gaussian')
    for index in range(10):
        dataset = drawDataset(0, index, preset())
        outcome = modelOutcome(0, index, dataset.covariates, glm)
        assert type(outcome) is GlmOutcome
        assert len(_factors(dataset)) == dataset.hyperparameters.factors
        assert len(outcome.fixed_terms) == 1 + dataset.hyperparameters.design_columns
        assert list(outcome.prior.split) == ['fixed']


def test_a_glmm_on_grouped_covariates_returns_a_glmm_outcome_of_one_block_per_factor() -> None:
    """metabeta's one call (ADR 0006): one block of random effects per grouping factor,
    a row per group, each block within its budget of 6 and within the design.
    """
    glmm = Glmm(family='gaussian')
    for index in range(10):
        dataset = drawDataset(0, index, Scm.grouped())
        outcome = modelOutcome(0, index, dataset.covariates, glmm)
        assert type(outcome) is GlmmOutcome
        factors = _factors(dataset)
        assert list(outcome.random_terms) == factors
        assert list(outcome.prior.split) == ['fixed', *factors]
        for factor in factors:
            q = len(outcome.random_terms[factor])
            assert q <= min(6, len(outcome.fixed_terms))
            assert outcome.parameters.alpha[factor].shape == (dataset.metadata[factor].levels, q)


GLM_FAMILIES = [
    pytest.param('gaussian', 'continuous', np.float32, id='gaussian'),
    pytest.param('bernoulli', 'binary', np.int64, id='bernoulli'),
    pytest.param('poisson', 'count', np.int64, id='poisson'),
]


@pytest.mark.parametrize(
    ('preset', 'config'),
    [(Scm.flat, Glm), (Scm.grouped, Glmm)],
    ids=['glm on flat', 'glmm on grouped'],
)
@pytest.mark.parametrize(('family', 'column_type', 'dtype'), GLM_FAMILIES)
def test_glm_or_glmm_with_one_family_draws_that_family_only_and_ys_dtype_follows_it(
    family: str,
    column_type: str,
    dtype: type,
    preset: Callable[[], Scm],
    config: type[Glm],
) -> None:
    """A corpus of one family is one model (ADR 0006),
    and a consumer stacks y by its type: a Gaussian y float32, a Bernoulli y codes, a count int64.
    """
    model = config(family=family)
    for index in range(10):
        dataset = drawDataset(0, index, preset())
        outcome = modelOutcome(0, index, dataset.covariates, model)
        assert outcome.family == family
        assert outcome.metadata['y'].type == column_type
        assert outcome.columns['y'].dtype == dtype


def test_a_normal_glm_at_large_n_and_small_sigma_eps_is_recovered_by_least_squares_on_the_coded_columns() -> (
    None
):
    """The ground truth refers to the design matrix a user fits on:
    least squares on the coded columns returns beta.
    At r = 0.9999 and a half-normal F, σ_ε is about 0.01.
    """
    glm = Glm(
        family='gaussian',
        sigma_eps_family_shares=point('sigma_eps_family_shares', 'half_normal', Glm),
        signal_share=0.9999,
    )
    for index in range(5):
        dataset = drawDataset(0, index, Scm.flat(rows=2000))
        outcome = modelOutcome(0, index, dataset.covariates, glm)
        design = dataset.covariates.toDesign()
        y = outcome.columns['y'].astype(np.float64)  # (n,)
        estimate, *_ = np.linalg.lstsq(design.matrix, y, rcond=None)
        np.testing.assert_allclose(estimate, outcome.parameters.beta, atol=0.01)


def _linearPart(dataset: Dataset, outcome: GlmmOutcome) -> np.ndarray:
    """Xβ + Σ_k Z_k α_k[codes_k] on the coded columns, a GLMM's η without the deviation."""
    design = dataset.covariates.toDesign()
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
    """The ground truth refers to the design matrix a user fits on:
    η = Xβ + Σ_k Z_k α_k[codes_k] + the deviation, Z_k the design's columns of the factor's
    random terms, so what y leaves after them is the Gaussian noise, sd σ_ε, up to y's float32
    rounding. At r = 0.9999 and a half-normal F, σ_ε is about 0.01, too small to hide a wrong
    Zα, or at λ = 0.3 a deviation that is not the one which entered η.
    """
    glmm = Glmm(
        family='gaussian',
        sigma_eps_family_shares=point('sigma_eps_family_shares', 'half_normal', Glm),
        signal_share=0.9999,
        correlated_share=1.0,
        deviation_scale=deviation_scale,
    )
    for index in range(5):
        dataset = drawDataset(0, index, Scm.grouped(rows=2000))
        outcome = modelOutcome(0, index, dataset.covariates, glmm)
        y = outcome.columns['y'].astype(np.float64)  # (n,)
        residual = y - _linearPart(dataset, outcome) - outcome.deviation  # (n,)
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
        dataset = drawDataset(0, index, Scm.grouped())
        plain = modelOutcome(0, index, dataset.covariates, Glmm(family='gaussian'))
        deviated = Glmm(family='gaussian', deviation_scale=0.3)
        outcome = modelOutcome(0, index, dataset.covariates, deviated)
        assert outcome.deviation_scale == 0.3
        assert outcome.prior == plain.prior
        assert outcome.parameters.beta.tobytes() == plain.parameters.beta.tobytes()
        assert outcome.random_terms == plain.random_terms
        for factor in outcome.random_terms:
            assert (
                outcome.parameters.alpha[factor].tobytes()
                == plain.parameters.alpha[factor].tobytes()
            )
        linear_sd = _linearPart(dataset, outcome).std()
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


def test_model_outcome_on_from_pandas_is_deterministic_per_key_and_differs_between_keys() -> None:
    """A model's outcome is keyed like a dataset,
    so the oracle set a study builds downstream from a table is reproducible per key.
    """
    model = Glmm(family='gaussian')
    first = modelOutcome(3, 1, TABLE_COVARIATES, model)
    again = modelOutcome(3, 1, TABLE_COVARIATES, model)
    other = modelOutcome(3, 2, TABLE_COVARIATES, model)
    assert np.array_equal(again.columns['y'], first.columns['y'])
    assert np.array_equal(again.parameters.beta, first.parameters.beta)
    assert not np.array_equal(other.parameters.beta, first.parameters.beta)


def _uniformY(covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """A model whose y is the first uniforms of the generator it is handed."""
    y = rng.random(covariates.rows, dtype=np.float32)  # (n,)
    return Outcome(columns={'y': y}, metadata={'y': Y})


def test_model_outcome_at_a_datasets_own_key_reads_other_numbers_than_the_draw() -> None:
    """A model on a dataset's covariates at the dataset's own key must not replay the numbers
    its covariates were drawn from (ρ = 1.00 between rows and ν_0 over 200 keys when it did),
    so it draws from a stream of the key's own, not the draw's generator default_rng([seed, index]).
    """
    y = modelOutcome(3, 1, TABLE_COVARIATES, _uniformY).columns['y']
    root = np.random.default_rng([3, 1]).random(len(y), dtype=np.float32)  # (n,)
    assert not np.array_equal(y, root)


def test_a_model_outcome_on_a_table_has_as_many_groups_per_factor_as_the_table() -> None:
    """A random effect per group of each factor the table has, a row of alpha per school
    and per class, so a study reads a group's ground truth at its code.
    """
    for index in range(5):
        outcome = modelOutcome(0, index, TABLE_COVARIATES, Glmm(family='gaussian'))
        assert list(outcome.parameters.alpha) == ['school', 'class']
        assert outcome.parameters.alpha['school'].shape[0] == 5
        assert outcome.parameters.alpha['class'].shape[0] == 10


def test_to_pandas_leaves_the_latent_nodes_out() -> None:
    """A table holds what was observed, in the dataset's column order;
    the latent nodes stay on the dataset, which keeps them white-box.
    """
    for index in range(5):
        dataset = drawDataset(0, index, Scm.grouped(latent_nodes=2))
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
    A count y, the SCM's, is the dataset's count.
    """
    count = point('outcome_type_shares', 'count')
    scm = Scm.grouped(
        rows=200, design_columns=12, type_shares=EVERY_TYPE, outcome_type_shares=count
    )
    types = set()
    for index in range(5):
        dataset = drawDataset(0, index, scm)
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
    scm = Scm.grouped(rows=200, design_columns=12, type_shares=EVERY_TYPE)
    datasets = [drawDataset(0, index, scm) for index in range(10)]
    datasets.append(drawDataset(2, 90, Scm.grouped()))
    for dataset in datasets:
        covariates = Covariates.fromPandas(dataset.toPandas(), metadata=dataset.metadata)
        observed = [name for name, column in dataset.metadata.items() if column.role != 'outcome']
        assert list(covariates.columns) == observed
        for name in observed:
            assert covariates.metadata[name] == dataset.metadata[name]
            assert covariates.columns[name].dtype == dataset.columns[name].dtype
            assert np.array_equal(covariates.columns[name], dataset.columns[name])


def test_to_design_is_the_coding_the_glm_used() -> None:
    """The ground truth refers to the design matrix a user fits on, which toDesign returns.
    At r = 1, τ_ε and so σ_ε are 0, and y is η = Xβ in float32, exactly.
    On grouped covariates the factors are no term.
    """
    glm = Glm(family='gaussian', signal_share=1.0)
    for index in range(5):
        dataset = drawDataset(0, index, Scm.grouped())
        outcome = modelOutcome(0, index, dataset.covariates, glm)
        design = dataset.covariates.toDesign()
        assert design.names == outcome.fixed_terms
        assert (design.means, design.sds) == (outcome.means, outcome.sds)
        eta = design.matrix @ outcome.parameters.beta  # (n,)
        assert np.array_equal(outcome.columns['y'], eta.astype(np.float32))


@pytest.mark.parametrize(
    ('outcome_type', 'dtype'), [('count', np.int64), ('continuous', np.float32)]
)
def test_a_count_y_is_int64_and_a_continuous_y_float32(outcome_type: str, dtype: type) -> None:
    """A consumer stacks y by its type: a count is an integer a Poisson likelihood reads,
    not a code 0..k − 1, and a continuous y shares the covariates' dtype.
    """
    shares = point('outcome_type_shares', outcome_type)
    scm = Scm.grouped(outcome_type_shares=shares)
    for index in range(5):
        assert drawDataset(0, index, scm).columns['y'].dtype == dtype


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
    scm = Scm.flat(
        design_columns=0,
        outcome_noise_scale=outcome_noise_scale,
        outcome_type_shares=point('outcome_type_shares', 'continuous'),
        marginal_shares=point('marginal_shares', 'normal'),
        **SURE_EDGE,
    )
    for index in range(10):
        dataset = drawDataset(0, index, scm)
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


GLM_OVERRIDE = Glm(
    family='gaussian', intercept_scale=0.5, deviation_scale=0.3
)  # τ_0 in the prior, λ beside it
GLMM_OVERRIDE = Glmm(family='gaussian', random_effects=1)  # every factor's intercept alone


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
    model=st.sampled_from(['scm', 'none', GLM_OVERRIDE, GLMM_OVERRIDE]),
    outcome_type=st.sampled_from(Scm.categories('outcome_type_shares')),
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
    model: str | Glm | Glmm,
    outcome_type: str,
) -> None:
    """metabeta passes its sizes as points; the dataset must have exactly those sizes.

    The fields without a visible size are honored through hyperparameters, which must hold
    every point override as given, a field drawn per node through the graph, one drawn per
    factor through the factor's column, the SCM's through y's column, a GLM's through
    its outcome's prior and a GLMM's through each factor's random terms. A point groups
    cannot nest, so a chain draws its groups from a range below the point.
    """
    # a point above its bound fails instead: rows >= 2d, m <= n and a chain of f factors in f + 1 rows
    assume(rows >= max(2 * design_columns, groups, factors + 1))
    assume(groups >= factors + 1)  # the bottom factor of a chain of f needs f + 1 groups
    assume(model != GLMM_OVERRIDE or factors > 0)  # a GLMM needs a factor
    # a model draws on the SCM's covariates, whose score its deviation reads
    source = model if isinstance(model, str) else 'scm'
    scm = Scm.grouped(
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
        outcome_source=source,
        outcome_type_shares=point('outcome_type_shares', outcome_type),
        poisson_log_rate=2.0,
    )
    dataset = drawDataset(0, 0, scm)
    metadata = dataset.metadata
    assert sum(column.design_columns for column in metadata.values()) == design_columns
    for array in [*dataset.columns.values(), *dataset.latent.values()]:
        assert array.shape == (rows,)
    assert len(dataset.latent) == latent_nodes
    # the grouping factors, each recording its size draw
    factor_names = _factors(dataset)
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
    assert ('y' in metadata) == (source != 'none')
    if source == 'scm':
        assert metadata['y'].type == outcome_type
        if outcome_type == 'count':
            assert metadata['y'].parameters == {'poisson_log_rate': 2.0}
    if model == GLM_OVERRIDE:
        outcome = modelOutcome(0, 0, dataset.covariates, model)
        assert outcome.prior.beta_scales[0] == 0.5
        assert outcome.deviation_scale == 0.3
    if model == GLMM_OVERRIDE:
        outcome = modelOutcome(0, 0, dataset.covariates, model)
        assert list(outcome.random_terms) == factor_names
        for name in factor_names:
            assert outcome.random_terms[name] == ['intercept']
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
        scm.groups,
        2.0,
        group_share,
    )
    assert (resolved.outcome_source, resolved.outcome_noise_scale, resolved.poisson_log_rate) == (
        source,
        0.5,
        2.0,
    )


def test_a_distribution_override_is_honored() -> None:
    """A simulation study narrows one axis by a distribution; every draw stays inside it."""
    for index in range(20):
        scm = Scm.flat(rows=UniformInt(30, 40), design_columns=1, outcome_source='none')
        dataset = drawDataset(0, index, scm)
        [values] = dataset.columns.values()
        assert 30 <= len(values) <= 40


def test_a_budget_of_0_gives_no_columns() -> None:
    """metabeta's budget may be 0: a table of rows and no covariates, whose graph is latent only."""
    for index in range(10):
        scm = Scm.flat(design_columns=0, latent_nodes=2, rows=50, outcome_source='none')
        dataset = drawDataset(0, index, scm)
        assert dataset.columns == {}
        assert dataset.metadata == {}
        assert dataset.graph.latent.all()
        assert len(dataset.latent['z0']) == 50


def test_every_column_and_latent_node_is_finite() -> None:
    """scamd's repair and retry hid broken draws; here a broken one would reach the consumer.

    Zero noise lets a dead relu leave a node constant, the case most likely to break.
    """
    for noise_scale in [0.0, 0.5]:
        scm = Scm(noise_scale=noise_scale, outcome_noise_scale=noise_scale)
        for index in range(50):
            dataset = drawDataset(0, index, scm)
            for values in [*dataset.columns.values(), *dataset.latent.values()]:
                assert np.isfinite(values).all()


def test_every_continuous_column_is_non_constant() -> None:
    """A consumer z-scores a continuous column, so a constant one divides by zero.

    At the default noise scales: an override noise_scale = 0 asks for a noiseless SCM,
    in which a dead relu is the constant function and its column is constant.
    """
    for index in range(100):
        dataset = drawDataset(0, index, Scm.grouped())
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

    One latent root, one typed column of k <= 3 and a sure edge between them, so the column must
    follow the root's values; flat, so that the edge is the graph's only one.
    """
    for type, design_columns in [('binary', 1), ('categorical', 2), ('ordered', 2)]:
        scm = Scm.flat(
            design_columns=design_columns,
            type_shares=point('type_shares', type),
            max_levels=3,
            level_alpha=5.0,
            outcome_source='none',
            **SURE_EDGE,
        )
        for index in range(10):
            dataset = drawDataset(0, index, scm)
            assert dataset.graph.parents.tolist() == [[False, False], [True, False]]
            agreement = _neighborAgreement(dataset.columns['x0'], dataset.latent['z0'])
            assert agreement > 0.95


PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'


def test_the_dataset_carries_the_installed_version() -> None:
    """A key reproduces a dataset only within one version, so the dataset names its own."""
    expected = tomllib.loads(PYPROJECT.read_text())['project']['version']
    assert drawDataset(0, 0, Scm.grouped()).version == expected


def test_the_dataset_carries_the_installed_numpy_version() -> None:
    """numpy promises no stable streams across its releases, so the dataset names numpy's too."""
    assert drawDataset(0, 0, Scm.grouped()).numpy_version == np.__version__


def test_every_array_has_length_n() -> None:
    """A group-level column or latent node has one value per group until draw copies it to the rows,
    so a consumer stacks every array without looking at the units.
    """
    scm = Scm.grouped(rows=80, group_level_share=0.8, latent_nodes=3)
    for index in range(20):
        dataset = drawDataset(0, index, scm)
        assert 'y' in dataset.columns
        for array in [*dataset.columns.values(), *dataset.latent.values()]:
            assert array.shape == (80,)


def test_each_g_column_is_int64_codes_0_to_g_minus_1_with_every_group_present() -> None:
    """metabeta indexes a random effect by the codes; a missing group would give an empty one."""
    for index in range(20):
        dataset = drawDataset(0, index, Scm.grouped(factors=3))
        assert _factors(dataset) == ['g0', 'g1', 'g2']
        for name in _factors(dataset):
            codes = dataset.columns[name]
            assert codes.dtype == np.int64
            assert np.array_equal(np.unique(codes), np.arange(dataset.metadata[name].levels))


def test_every_level_of_a_covariate_appears() -> None:
    """A level no row has would be an all-zero dummy column in metabeta's design matrix;
    a group-level column shows at most m levels, so one with more stays at the row.
    An override group_share = 1 is the known exception (ADR 0003), so the test draws the defaults.
    """
    for index in range(100):
        dataset = drawDataset(0, index, Scm.grouped())
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
    scm = Scm.grouped(rows=200, group_level_share=1.0, latent_nodes=3)
    for index in range(20):
        dataset = drawDataset(0, index, scm)
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
    scm = Scm.grouped(
        rows=200,
        group_level_share=1.0,
        type_shares=point('type_shares', 'continuous'),
        marginal_shares=point('marginal_shares', 'normal'),
    )
    for index in range(20):
        dataset = drawDataset(0, index, scm)
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
    scm = Scm.grouped(rows=2000, factors=1, groups=20, group_share=0.0)
    for index in range(20):
        dataset = drawDataset(0, index, scm)
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
        dataset = drawDataset(0, index, Scm.flat())
        graph = dataset.graph
        assert all(column.role != 'grouping_factor' for column in dataset.metadata.values())
        assert not any(name.startswith('g') for name in graph.names)
        assert set(graph.unit) <= {'row'}
        assert set(graph.group_share) <= {None}
        assert set(graph.group_factor) <= {None}
        assert dataset.hyperparameters.factors == 0


def test_a_draw_without_an_scm_fails() -> None:
    """No preset is the default: a caller names the kind of table it draws."""
    with pytest.raises(TypeError):
        drawDataset(0, 0)


def test_group_share_1_gives_every_row_unit_column_constant_within_its_groups() -> None:
    """At share 1 the row part has weight 0, so a row-unit node is one value per group of its
    group factor; the column keeps its ties, so it is too, whatever its type.
    y is one of them, so group confounding reaches the outcome.
    """
    scm = Scm.grouped(rows=600, factors=2, group_share=1.0, group_level_share=0.0)
    for index in range(20):
        dataset = drawDataset(0, index, scm)
        group_factor = dict(zip(dataset.graph.names, dataset.graph.group_factor))
        for name, column in dataset.metadata.items():
            if column.role != 'grouping_factor':
                codes = dataset.columns[group_factor[name]]
                assert _isConstantWithinGroups(dataset.columns[name], codes)
        for name, values in dataset.latent.items():
            assert _isConstantWithinGroups(values, dataset.columns[group_factor[name]])
