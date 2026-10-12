"""The draw module's seam: draw(seed, index, hyperprior), one dataset per key."""

from __future__ import annotations

import tomllib
from pathlib import Path

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from tabdgp.dataset import Dataset
from tabdgp.draw import draw
from tabdgp.hyperprior import Hyperprior, UniformInt
from tests.fixtures.build import point


def _assert_identical(a: Dataset, b: Dataset) -> None:
    """Two datasets agree bit for bit: every array in dtype and bytes, everything else by ==."""
    for name in ['columns', 'latent']:
        arrays_a, arrays_b = getattr(a, name), getattr(b, name)
        assert list(arrays_a) == list(arrays_b)
        for key in arrays_a:
            assert arrays_a[key].dtype == arrays_b[key].dtype
            assert arrays_a[key].tobytes() == arrays_b[key].tobytes()
    assert a.metadata == b.metadata
    assert a.graph.names == b.graph.names
    assert a.graph.mechanism == b.graph.mechanism
    assert np.array_equal(a.graph.parents, b.graph.parents)
    assert np.array_equal(a.graph.latent, b.graph.latent)
    assert (a.hyperparameters, a.key, a.version) == (b.hyperparameters, b.key, b.version)


SEEDS = st.integers(0, 2**32)
INDICES = st.integers(0, 1000)


@settings(max_examples=25, deadline=None)
@given(seed=SEEDS, other_seed=SEEDS, index=INDICES, others=st.lists(INDICES, max_size=3))
def test_a_dataset_is_bitwise_identical_alone_inside_two_index_sets_and_under_a_second_seed_order(
    seed: int, other_seed: int, index: int, others: list[int]
) -> None:
    """The seed is the corpus: dataset i may not depend on what else a worker, batch or shard draws."""
    alone = draw(seed, index)
    first_set = [draw(seed, i) for i in [*others, index]][-1]
    second_set = [draw(seed, i) for i in [index, *reversed(others), index + 1]][0]
    other_first = [draw(other_seed, index), draw(seed, index)][1]
    for dataset in [first_set, second_set, other_first]:
        _assert_identical(alone, dataset)


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
    n_obs=st.integers(1, 300),
    design_columns=st.integers(0, 19),
    latent_nodes=st.integers(0, 9),
    max_levels=st.integers(3, 6),
)
def test_every_override_is_honoured(
    n_obs: int, design_columns: int, latent_nodes: int, max_levels: int
) -> None:
    """metabeta passes its sizes as points; the dataset must have exactly those sizes.

    The fields without a visible size are honoured through hyperparameters, which must hold
    every point override as given.
    """
    hyperprior = Hyperprior(
        n_obs=n_obs,
        design_columns=design_columns,
        latent_nodes=latent_nodes,
        max_levels=max_levels,
        noise_scale=0.5,
    )
    dataset = draw(0, 0, hyperprior)
    metadata = dataset.metadata.values()
    assert sum(column.width for column in metadata) == design_columns
    for column in metadata:
        assert column.levels is None or column.levels <= max_levels
    assert len(dataset.latent) == latent_nodes
    for array in [*dataset.columns.values(), *dataset.latent.values()]:
        assert array.shape == (n_obs,)
    resolved = dataset.hyperparameters
    assert (resolved.n_obs, resolved.design_columns, resolved.latent_nodes) == (
        n_obs,
        design_columns,
        latent_nodes,
    )
    assert (resolved.max_levels, resolved.noise_scale) == (max_levels, 0.5)


def test_a_distribution_override_is_honoured() -> None:
    """A simulation study narrows one axis by a distribution; every draw stays inside it."""
    for index in range(20):
        dataset = draw(0, index, Hyperprior(n_obs=UniformInt(30, 40), design_columns=1))
        [values] = dataset.columns.values()
        assert 30 <= len(values) <= 40


def test_a_budget_of_0_gives_no_columns() -> None:
    """metabeta's d − 1 may be 0: a table of rows and no covariates, whose graph is latent only."""
    for index in range(10):
        dataset = draw(0, index, Hyperprior(design_columns=0, latent_nodes=2, n_obs=50))
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


def _neighbour_agreement(codes: np.ndarray, parent: np.ndarray) -> float:
    """The share of rows whose code equals the next row's in the parent's order.

    A column cut from its parent changes code only at its K − 1 thresholds, whatever its sign
    or label permutation; an independent one agrees only by chance, about the sum of squared
    level shares.
    """
    ordered = codes[np.argsort(parent)]  # (N,)
    return float((ordered[1:] == ordered[:-1]).mean())


def test_each_binary_categorical_and_ordered_column_depends_on_its_parents_at_low_noise() -> None:
    """scamd's categoricals were nearly independent of their parents, and only shapes were tested.

    One latent root and no drawn edges leave the root's forced child as the only node with a
    parent, so the typed column must follow that root's values.
    """
    isolated = {'edge_offset': -30.0, 'hub_scale': 0.0, 'block_bonus': 0.0}
    for type in ['binary', 'categorical', 'ordered']:
        hyperprior = Hyperprior(
            n_obs=1000,
            design_columns=4,
            type_shares=point('type_shares', type),
            latent_nodes=1,
            mechanism_shares={'linear': 1.0, 'mlp': 0.0},
            noise_scale=0.01,
            level_alpha=5.0,
            **isolated,
        )
        for index in range(10):
            dataset = draw(0, index, hyperprior)
            graph = dataset.graph
            child = graph.names[int(np.flatnonzero(graph.parents[:, 0])[0])]
            agreement = _neighbour_agreement(dataset.columns[child], dataset.latent['z0'])
            assert agreement > 0.95


PYPROJECT = Path(__file__).parents[1] / 'pyproject.toml'


def test_the_dataset_carries_the_installed_version() -> None:
    """A key reproduces a dataset only within one version, so the dataset names its own."""
    expected = tomllib.loads(PYPROJECT.read_text())['project']['version']
    assert draw(0, 0).version == expected
