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


def test_an_unknown_marginal_family_fails_at_construction() -> None:
    """A family outside the catalogue would otherwise fail only when the column is matched."""
    with pytest.raises(ValidationError):
        Column(type='continuous', marginal='gamma')


WIDTHS = [
    pytest.param(Column(type='continuous', marginal='beta'), 1, id='continuous'),
    pytest.param(Column(type='binary', levels=2), 1, id='binary'),
    pytest.param(Column(type='categorical', levels=5), 4, id='categorical'),
    pytest.param(Column(type='ordered', levels=3), 2, id='ordered'),
]


@pytest.mark.parametrize(('column', 'width'), WIDTHS)
def test_a_column_is_one_design_column_wide_if_continuous_and_k_minus_1_if_not(
    column: Column, width: int
) -> None:
    """The design-column budget counts metabeta's dummy coding: K levels take K − 1 columns."""
    assert column.width == width


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


MISMATCHED = [
    pytest.param({'names': ['x0', 'x1']}, id='names one short'),
    pytest.param({'latent': np.zeros(2, dtype=bool)}, id='latent one short'),
    pytest.param({'mechanism': ['linear', 'linear']}, id='mechanism one short'),
    pytest.param({'parents': np.zeros((3, 2), dtype=bool)}, id='parents not square'),
]


@pytest.mark.parametrize('change', MISMATCHED)
def test_graph_fields_of_different_lengths_fail(change: dict) -> None:
    """Every field indexes the same nodes; a mismatch would misalign names and mechanisms."""
    fields = {
        'names': ['x0', 'x1', 'x2'],
        'parents': np.zeros((3, 3), dtype=bool),
        'latent': np.zeros(3, dtype=bool),
        'mechanism': ['linear', 'linear', 'linear'],
    }
    with pytest.raises(ValidationError):
        Graph(**fields | change)


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
        hyperparameters=Hyperprior().resolve(np.random.default_rng(0)),
        key=Key(seed=0, index=0),
        version='0.0.1',
    )
    assert dataset.columns['x0'] is values and dataset.latent['z0'] is latent
