"""The output contract's seams: what Column, Key, Graph and Dataset guarantee on their own."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.dataset import Column, Dataset, Graph, Key
from tabdgp.hyperprior import Hyperprior
from tests.fixtures.build import graphOf


def test_a_column_type_outside_the_four_fails() -> None:
    """metabeta's encoder branches on the type, so a fifth would pass through unencoded."""
    with pytest.raises(ValidationError):
        Column(type='count')


CONTRADICTING_LEVELS = [
    pytest.param('continuous', 3, id='a continuous column with levels'),
    pytest.param('binary', 3, id='a binary column of three levels'),
    pytest.param('binary', None, id='a binary column without levels'),
    pytest.param('categorical', 2, id='a categorical column of two levels'),
    pytest.param('ordered', 2, id='an ordered column of two levels'),
    pytest.param('categorical', None, id='a categorical column without levels'),
]


@pytest.mark.parametrize(('type', 'levels'), CONTRADICTING_LEVELS)
def test_levels_that_contradict_the_type_fail(type: str, levels: int | None) -> None:
    """Binary is exactly K = 2 and the other discrete types K ≥ 3, so the types never overlap."""
    with pytest.raises(ValidationError):
        Column(type=type, levels=levels)


def test_a_marginal_on_a_discrete_column_fails() -> None:
    """Only continuous columns are rank matched; a discrete column's values are its codes."""
    with pytest.raises(ValidationError):
        Column(type='binary', levels=2, marginal='lognormal')


def test_a_continuous_column_without_a_marginal_fails() -> None:
    """Its parameters are its family's, so a continuous column names the family it was drawn from."""
    with pytest.raises(ValidationError):
        Column(type='continuous')


OWN_PARAMETERS = [
    pytest.param(Column(type='continuous', marginal='normal'), {}, id='a normal column has none'),
    pytest.param(
        Column(
            type='continuous',
            marginal='zero_inflated_lognormal',
            parameters={'lognormal_sigma': 0.5, 'zero_share': 0.2},
        ),
        {'lognormal_sigma': 0.5, 'zero_share': 0.2},
        id='a zero-inflated lognormal column has two',
    ),
    pytest.param(
        Column(type='ordered', levels=4, parameters={'level_alpha': 2.0}),
        {'level_alpha': 2.0},
        id='a coded column has its concentration',
    ),
]


@pytest.mark.parametrize(('column', 'parameters'), OWN_PARAMETERS)
def test_a_column_records_the_values_it_was_drawn_with(column: Column, parameters: dict) -> None:
    """Each column draws its own marginal parameters or level concentration;
    keeping them keeps the dataset white-box.
    """
    assert column.parameters == parameters


FOREIGN_PARAMETERS = [
    pytest.param(
        {'type': 'continuous', 'marginal': 'lognormal'}, {}, id='a lognormal without its sigma'
    ),
    pytest.param(
        {'type': 'continuous', 'marginal': 'normal'},
        {'lognormal_sigma': 1.0},
        id='a normal with a lognormal sigma',
    ),
    pytest.param({'type': 'binary', 'levels': 2}, {}, id='a binary without its concentration'),
    pytest.param(
        {'type': 'categorical', 'levels': 3},
        {'level_alpha': 1.0, 'beta_a': 2.0},
        id='a categorical with a beta shape',
    ),
]


@pytest.mark.parametrize(('fields', 'parameters'), FOREIGN_PARAMETERS)
def test_parameters_other_than_the_columns_own_fail(fields: dict, parameters: dict) -> None:
    """A missing value hides a draw; an extra one claims a draw that played no part."""
    with pytest.raises(ValidationError):
        Column(**fields, parameters=parameters)


def test_an_unknown_marginal_family_fails_at_construction() -> None:
    """A family outside the catalogue would otherwise fail only when the column is matched."""
    with pytest.raises(ValidationError):
        Column(type='continuous', marginal='gamma')


ALPHA = {'level_alpha': 1.0}
DESIGN_COLUMNS = [
    pytest.param(Column(type='continuous', marginal='normal'), 1, id='continuous'),
    pytest.param(Column(type='binary', levels=2, parameters=ALPHA), 1, id='binary'),
    pytest.param(Column(type='categorical', levels=5, parameters=ALPHA), 4, id='categorical'),
    pytest.param(Column(type='ordered', levels=3, parameters=ALPHA), 2, id='ordered'),
]


@pytest.mark.parametrize(('column', 'design_columns'), DESIGN_COLUMNS)
def test_a_column_is_one_design_column_wide_if_continuous_and_k_minus_1_if_not(
    column: Column, design_columns: int
) -> None:
    """The design-column budget counts metabeta's dummy coding: K levels take K − 1 columns."""
    assert column.design_columns == design_columns


@pytest.mark.parametrize(('seed', 'index'), [(-1, 0), (0, -1)])
def test_a_negative_seed_or_index_fails(seed: int, index: int) -> None:
    """default_rng([seed, index]) rejects negative entries, so the key fails where it is written."""
    with pytest.raises(ValidationError):
        Key(seed=seed, index=index)


def test_a_key_generator_is_default_rng_of_seed_and_index() -> None:
    """The key-to-generator rule is the corpus's identity; a change to it redraws every dataset."""
    key = Key(seed=3, index=5)
    expected = np.random.default_rng([3, 5]).bit_generator.state
    assert key.generator().bit_generator.state == expected


@pytest.mark.parametrize(('child', 'parent'), [(0, 1), (1, 1)], ids=['above', 'on'])
def test_a_parent_on_or_above_the_diagonal_fails(child: int, parent: int) -> None:
    """parents[j, i] says i is a parent of j; only i < j keeps the names in topological order."""
    parents = [[], []]
    parents[child].append(parent)
    with pytest.raises(ValidationError):
        graphOf(parents)


# x0 -> x1 -> x2: a root, a linear node and an MLP node, each with exactly the choices it uses
CHAIN = {
    'names': ['x0', 'x1', 'x2'],
    'parents': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=bool),
    'latent': np.zeros(3, dtype=bool),
    'mechanism': [None, 'linear', 'mlp'],
    'activation': [None, None, 'tanh'],
    'noise_scale': np.array([np.nan, 0.1, 0.2]),
    'init_scale': np.array([np.nan, np.nan, 1.5]),
}

MISMATCHED = [
    pytest.param({'names': ['x0', 'x1']}, id='names one short'),
    pytest.param({'latent': np.zeros(2, dtype=bool)}, id='latent one short'),
    pytest.param({'mechanism': [None, 'linear']}, id='mechanism one short'),
    pytest.param({'parents': np.zeros((3, 2), dtype=bool)}, id='parents not square'),
    pytest.param({'activation': [None, None]}, id='activation one short'),
    pytest.param({'noise_scale': np.array([np.nan, 0.1])}, id='noise scale one short'),
    pytest.param({'init_scale': np.array([np.nan, np.nan])}, id='init scale one short'),
]


@pytest.mark.parametrize('change', MISMATCHED)
def test_graph_fields_of_different_lengths_fail(change: dict) -> None:
    """Every field indexes the same nodes; a mismatch would misalign names and mechanisms."""
    with pytest.raises(ValidationError):
        Graph(**CHAIN | change)


MISLABELLED = [
    pytest.param({'latent': np.array([True, False, False])}, id='an x node marked latent'),
    pytest.param({'names': ['z0', 'x1', 'x2']}, id='a z node not marked latent'),
]


@pytest.mark.parametrize('change', MISLABELLED)
def test_a_latent_mask_that_disagrees_with_the_names_fails(change: dict) -> None:
    """The mask and the x…/z… names both tell latent from observed nodes; they may not disagree."""
    with pytest.raises(ValidationError):
        Graph(**CHAIN | change)


def test_a_graph_records_each_nodes_own_choices() -> None:
    """Each node draws its own family, noise scale, activation and init scale;
    keeping them keeps the dataset white-box.
    """
    graph = Graph(**CHAIN)
    assert graph.mechanism == [None, 'linear', 'mlp']
    assert graph.activation == [None, None, 'tanh']
    np.testing.assert_array_equal(graph.noise_scale, [np.nan, 0.1, 0.2])
    np.testing.assert_array_equal(graph.init_scale, [np.nan, np.nan, 1.5])


UNUSED_OR_MISSING = [
    pytest.param({'mechanism': ['linear', 'linear', 'mlp']}, id='a root with a family'),
    pytest.param({'mechanism': [None, None, 'mlp']}, id='a node with parents without a family'),
    pytest.param({'noise_scale': np.array([0.1, 0.1, 0.2])}, id='a root with a noise scale'),
    pytest.param(
        {'noise_scale': np.array([np.nan, np.nan, 0.2])},
        id='a node with parents without a noise scale',
    ),
    pytest.param({'activation': [None, 'tanh', 'tanh']}, id='a linear node with an activation'),
    pytest.param(
        {'init_scale': np.array([np.nan, 1.0, 1.5])}, id='a linear node with an init scale'
    ),
    pytest.param({'activation': [None, None, None]}, id='an mlp node without an activation'),
    pytest.param(
        {'init_scale': np.array([np.nan, np.nan, np.nan])}, id='an mlp node without an init scale'
    ),
]


@pytest.mark.parametrize('change', UNUSED_OR_MISSING)
def test_a_choice_a_node_does_not_use_or_lacks_fails(change: dict) -> None:
    """A recorded choice that played no part would claim it did; a missing one hides a draw."""
    with pytest.raises(ValidationError):
        Graph(**CHAIN | change)


def test_an_empty_graph_is_valid() -> None:
    """A budget of 0 with no latent nodes draws a dataset of rows and no nodes at all."""
    graphOf([])


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
        key=Key(seed=0, index=0),
        version='0.0.1',
    )
    assert dataset.columns['x0'] is values and dataset.latent['z0'] is latent
