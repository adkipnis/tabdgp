"""The draw module's seams: what a Dataset guarantees on its own,
and draw(seed, index, hyperprior), one dataset per key.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from tabdgp.column import Column
from tabdgp.draw import Dataset, draw
from tabdgp.hyperprior import Hyperprior, UniformInt
from tests.fixtures.build import graphOf, point


def test_a_dataset_holds_its_arrays_without_copying() -> None:
    """metabeta hands the columns to torch.from_numpy; a copy per draw would cost throughput."""
    values = np.zeros(5, dtype=np.float32)
    latent = np.ones(5, dtype=np.float32)
    dataset = Dataset(
        columns={'x0': values},
        metadata={'x0': Column(type='continuous', marginal='normal')},
        graph=graphOf([[], []]),
        latent={'z0': latent},
        hyperparameters=Hyperprior().resolveDataset(np.random.default_rng(0)),
        seed=0,
        index=0,
        version='0.0.1',
        numpy_version='2.3.0',
    )
    assert dataset.columns['x0'] is values and dataset.latent['z0'] is latent


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
    assert (a.seed, a.index, a.version) == (b.seed, b.index, b.version)
    assert a.numpy_version == b.numpy_version
    assert a.hyperparameters == b.hyperparameters


@pytest.mark.parametrize(('seed', 'index'), [(-1, 0), (0, -1)])
def test_a_negative_seed_or_index_fails(seed: int, index: int) -> None:
    """default_rng([seed, index]) rejects negative entries, so a bad key fails before any draw."""
    with pytest.raises(ValueError):
        draw(seed, index)


SEEDS = st.integers(0, 2**32)
INDICES = st.integers(0, 1000)


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, other_seed=SEEDS, index=INDICES, others=st.lists(INDICES, max_size=3))
def test_a_dataset_is_bitwise_identical_alone_inside_two_index_sets_and_under_a_second_seed_order(
    seed: int, other_seed: int, index: int, others: list[int]
) -> None:
    """The seed is the corpus:
    dataset i may not depend on what else a worker, batch or shard draws.
    """
    alone = draw(seed, index)
    first_set = [draw(seed, i) for i in [*others, index]][-1]
    second_set = [draw(seed, i) for i in [index, *reversed(others), index + 1]][0]
    other_first = [draw(other_seed, index), draw(seed, index)][1]
    for dataset in [first_set, second_set, other_first]:
        _assertIdentical(alone, dataset)


def test_columns_metadata_graph_and_latent_agree_on_names() -> None:
    """A consumer looks a column up by name in columns, metadata and the graph alike."""
    for index in range(20):
        dataset = draw(0, index, Hyperprior(latent_nodes=2))
        graph = dataset.graph
        observed = [name for name, latent in zip(graph.names, graph.latent) if not latent]
        latent = [name for name, latent in zip(graph.names, graph.latent) if latent]
        assert list(dataset.columns) == list(dataset.metadata)
        assert sorted(dataset.columns) == sorted(observed)
        assert list(dataset.latent) == latent


@settings(max_examples=25, deadline=None)
@given(
    rows=st.integers(1, 300),
    design_columns=st.integers(0, 19),
    latent_nodes=st.integers(0, 9),
    max_levels=st.integers(3, 6),
)
def test_every_override_is_honoured(
    rows: int, design_columns: int, latent_nodes: int, max_levels: int
) -> None:
    """metabeta passes its sizes as points; the dataset must have exactly those sizes.

    The fields without a visible size are honoured through hyperparameters, which must hold
    every point override as given, and a field drawn per node through the graph.
    """
    assume(rows >= 2 * design_columns)  # a point above its bound fails instead
    hyperprior = Hyperprior(
        rows=rows,
        design_columns=design_columns,
        latent_nodes=latent_nodes,
        max_levels=max_levels,
        noise_scale=0.5,
    )
    dataset = draw(0, 0, hyperprior)
    metadata = dataset.metadata.values()
    assert sum(column.design_columns for column in metadata) == design_columns
    for column in metadata:
        assert column.levels is None or column.levels <= max_levels
    assert len(dataset.latent) == latent_nodes
    for array in [*dataset.columns.values(), *dataset.latent.values()]:
        assert array.shape == (rows,)
    resolved = dataset.hyperparameters
    assert (resolved.rows, resolved.design_columns, resolved.latent_nodes) == (
        rows,
        design_columns,
        latent_nodes,
    )
    assert (resolved.max_levels, resolved.noise_scale) == (max_levels, 0.5)
    assert set(dataset.graph.noise_scale) <= {None, 0.5}


def test_a_distribution_override_is_honoured() -> None:
    """A simulation study narrows one axis by a distribution; every draw stays inside it."""
    for index in range(20):
        dataset = draw(0, index, Hyperprior(rows=UniformInt(30, 40), design_columns=1))
        [values] = dataset.columns.values()
        assert 30 <= len(values) <= 40


def test_a_budget_of_0_gives_no_columns() -> None:
    """metabeta's budget may be 0: a table of rows and no covariates, whose graph is latent only."""
    for index in range(10):
        dataset = draw(0, index, Hyperprior(design_columns=0, latent_nodes=2, rows=50))
        assert dataset.columns == {}
        assert dataset.metadata == {}
        assert dataset.graph.latent.all()
        assert len(dataset.latent['z0']) == 50


def test_every_column_is_finite_and_every_continuous_column_non_constant() -> None:
    """scamd's repair and retry hid broken draws; here a broken one would reach the consumer.

    Zero noise lets a dead relu leave a node constant, the case most likely to break.
    """
    for noise_scale in [0.0, 0.5]:
        for index in range(50):
            dataset = draw(0, index, Hyperprior(noise_scale=noise_scale))
            for name, values in dataset.columns.items():
                assert np.isfinite(values).all()
                if dataset.metadata[name].type == 'continuous':
                    assert values.std() > 0
            for values in dataset.latent.values():
                assert np.isfinite(values).all()


def _neighbourAgreement(codes: np.ndarray, parent: np.ndarray) -> float:
    """The share of rows whose code equals the next row's in the parent's order.

    A column cut from its parent changes code only at its K − 1 thresholds, whatever its sign
    or label permutation; an independent one agrees only by chance, about the sum of squared
    level shares.
    """
    ordered = codes[np.argsort(parent)]  # (n,)
    return float((ordered[1:] == ordered[:-1]).mean())


def test_each_binary_categorical_and_ordered_column_depends_on_its_parents_at_low_noise() -> None:
    """scamd's categoricals were nearly independent of their parents, and only shapes were tested.

    One latent root, one typed column of K ≤ 3 and a sure edge between them, so the column must
    follow the root's values.
    """
    dense = {'edge_offset': 30.0, 'hub_scale': 0.0, 'block_bonus': 0.0}
    for type, design_columns in [('binary', 1), ('categorical', 2), ('ordered', 2)]:
        hyperprior = Hyperprior(
            rows=1000,
            design_columns=design_columns,
            type_shares=point('type_shares', type),
            max_levels=3,
            latent_nodes=1,
            mechanism_shares={'linear': 1.0, 'mlp': 0.0},
            noise_scale=0.01,
            level_alpha=5.0,
            **dense,
        )
        for index in range(10):
            dataset = draw(0, index, hyperprior)
            assert dataset.graph.parents.tolist() == [[False, False], [True, False]]
            agreement = _neighbourAgreement(dataset.columns['x0'], dataset.latent['z0'])
            assert agreement > 0.95


PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'


def test_the_dataset_carries_the_installed_version() -> None:
    """A key reproduces a dataset only within one version, so the dataset names its own."""
    expected = tomllib.loads(PYPROJECT.read_text())['project']['version']
    assert draw(0, 0).version == expected


def test_the_dataset_carries_the_installed_numpy_version() -> None:
    """numpy promises no stable streams across its releases, so the dataset names numpy's too."""
    assert draw(0, 0).numpy_version == np.__version__
