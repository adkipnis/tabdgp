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
    On the grouped preset, so the grouping, the units and the two parts are covered.
    """
    grouped = Hyperprior.grouped()
    alone = draw(seed, index, grouped)
    first_set = [draw(seed, i, grouped) for i in [*others, index]][-1]
    second_set = [draw(seed, i, grouped) for i in [index, *reversed(others), index + 1]][0]
    other_first = [draw(other_seed, index, grouped), draw(seed, index, grouped)][1]
    for dataset in [first_set, second_set, other_first]:
        _assertIdentical(alone, dataset)


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


def _exceedsAFactorsGroups(dataset: Dataset, name: str) -> bool:
    """Whether a column has more levels than some factor has groups,
    the one case in which a column at group_level_share = 1 stays at the row.
    """
    column = dataset.metadata.get(name)
    groups = [
        factor.levels for factor in dataset.metadata.values() if factor.role == 'grouping_factor'
    ]
    return column is not None and column.levels is not None and column.levels > min(groups)


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
)
def test_every_override_is_honoured(
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
) -> None:
    """metabeta passes its sizes as points; the dataset must have exactly those sizes.

    The fields without a visible size are honoured through hyperparameters, which must hold
    every point override as given, a field drawn per node through the graph and one drawn per
    factor through the factor's column. A point groups cannot nest, so a chain draws its groups
    from a range below the point.
    """
    # a point above its bound fails instead: rows ≥ 2d, G ≤ n and a chain of L factors in L + 1 rows
    assume(rows >= max(2 * design_columns, groups, factors + 1))
    assume(groups >= factors + 1)  # the bottom factor of a chain of L needs L + 1 groups
    hyperprior = Hyperprior.grouped(
        rows=rows,
        design_columns=design_columns,
        latent_nodes=latent_nodes,
        max_levels=max_levels,
        noise_scale=0.5,
        factors=factors,
        groups=UniformInt(2, groups) if nested else groups,
        nested_share=1.0 if nested else 0.0,
        balanced_share=1.0 if balanced else 0.0,
        size_alpha=2.0,
        group_level_share=1.0 if group_level else 0.0,
        group_share=group_share,
    )
    dataset = draw(0, 0, hyperprior)
    metadata = dataset.metadata
    assert sum(column.design_columns for column in metadata.values()) == design_columns
    for array in [*dataset.columns.values(), *dataset.latent.values()]:
        assert array.shape == (rows,)
    assert len(dataset.latent) == latent_nodes
    # the grouping factors, each recording its size draw
    factor_names = [name for name, column in metadata.items() if column.role == 'grouping_factor']
    assert factor_names == [f'g{k}' for k in range(factors)]
    for k, name in enumerate(factor_names):
        column = metadata[name]
        assert 2 <= column.levels <= groups if nested else column.levels == groups
        assert column.nests_in == (f'g{k - 1}' if nested and k > 0 else None)
        assert column.parameters == ({} if balanced else {'size_alpha': 2.0})
    for name, column in metadata.items():
        if column.role == 'covariate':
            assert column.levels is None or column.levels <= max_levels
    # the per-node fields through the graph
    graph = dataset.graph
    for name, unit in zip(graph.names, graph.unit):
        if name not in factor_names and group_level and factors > 0:
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


def test_a_distribution_override_is_honoured() -> None:
    """A simulation study narrows one axis by a distribution; every draw stays inside it."""
    for index in range(20):
        dataset = draw(0, index, Hyperprior.flat(rows=UniformInt(30, 40), design_columns=1))
        [values] = dataset.columns.values()
        assert 30 <= len(values) <= 40


def test_a_budget_of_0_gives_no_columns() -> None:
    """metabeta's budget may be 0: a table of rows and no covariates, whose graph is latent only."""
    for index in range(10):
        dataset = draw(0, index, Hyperprior.flat(design_columns=0, latent_nodes=2, rows=50))
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
    follow the root's values; flat, so that the edge is the graph's only one.
    """
    dense = {'edge_offset': 30.0, 'hub_scale': 0.0, 'block_bonus': 0.0}
    for type, design_columns in [('binary', 1), ('categorical', 2), ('ordered', 2)]:
        hyperprior = Hyperprior.flat(
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
        for array in [*dataset.columns.values(), *dataset.latent.values()]:
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
    a group-level column shows at most G levels, so one with more stays at the row.
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
    first_rows = np.unique(codes, return_index=True)[1]  # (G,)
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
    """Its marginal is sampled at G values, one per group, not at n values with ties;
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


def _betweenShare(values: np.ndarray, codes: np.ndarray) -> float:
    """The share of a column's variance over rows that lies between the groups of a factor."""
    values = values.astype(np.float64)
    means = np.bincount(codes, weights=values) / np.bincount(codes)  # (G,)
    between = ((means[codes] - values.mean()) ** 2).sum()
    return float(between / ((values - values.mean()) ** 2).sum())


def test_group_share_0_under_one_factor_gives_every_within_group_covariate_a_between_group_share_at_chance_level() -> (
    None
):
    """Item 9's ablation switches group confounding off by this one override.
    Under one factor every parent lies in the group factor's chain, so a share of 0 drops the
    group part and every row-unit node, a root or a child of a group-level parent, is i.i.d.
    over rows; the chance level is about (G − 1) / (n − 1), here 0.01.
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
            assert _betweenShare(dataset.columns[name], codes) < 0.03


def test_flat_gives_no_g_column_every_unit_row_and_no_two_part_field() -> None:
    """A flat study changes nothing: item 1's table, with the grouping fields idle."""
    for index in range(20):
        dataset = draw(0, index, Hyperprior.flat())
        graph = dataset.graph
        assert all(column.role == 'covariate' for column in dataset.metadata.values())
        assert not any(name.startswith('g') for name in graph.names)
        assert set(graph.unit) <= {'row'}
        assert set(graph.group_share) <= {None}
        assert set(graph.group_factor) <= {None}
        assert dataset.hyperparameters.factors == 0


def test_draw_without_a_hyperprior_fails() -> None:
    """No preset is the default: a caller names the kind of table it draws."""
    with pytest.raises(TypeError):
        draw(0, 0)


def test_group_share_1_gives_every_within_group_covariate_constant_within_its_groups() -> None:
    """At share 1 the row part has weight 0, so a row-unit node is one value per group of its
    group factor; the column keeps its ties, so it is too, whatever its type.
    """
    hyperprior = Hyperprior.grouped(rows=600, factors=2, group_share=1.0, group_level_share=0.0)
    for index in range(20):
        dataset = draw(0, index, hyperprior)
        group_factor = dict(zip(dataset.graph.names, dataset.graph.group_factor))
        for name, column in dataset.metadata.items():
            if column.role == 'covariate':
                codes = dataset.columns[group_factor[name]]
                assert _isConstantWithinGroups(dataset.columns[name], codes)
        for name, values in dataset.latent.items():
            assert _isConstantWithinGroups(values, dataset.columns[group_factor[name]])
