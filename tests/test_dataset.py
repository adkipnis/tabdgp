"""The output contract's seams: what Column, Key, Graph and Dataset guarantee on their own."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.dataset import Column, Dataset, Graph, Key
from tabdgp.hyperprior import Hyperprior


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


@pytest.mark.parametrize(('seed', 'index'), [(-1, 0), (0, -1)])
def test_a_negative_seed_or_index_fails(seed: int, index: int) -> None:
    """default_rng([seed, index]) rejects negative entries, so the key fails where it is written."""
    with pytest.raises(ValidationError):
        Key(seed=seed, index=index)


def _graph(parents: np.ndarray) -> Graph:
    """A graph of observed nodes over the given adjacency, roots with the noise mechanism."""
    n = len(parents)
    roots = ~parents.any(axis=1)
    return Graph(
        names=[f'x{i}' for i in range(n)],
        parents=parents,
        latent=np.zeros(n, dtype=bool),
        mechanism=['noise' if root else 'linear' for root in roots],
    )


@pytest.mark.parametrize(('child', 'parent'), [(0, 1), (1, 1)], ids=['above', 'on'])
def test_a_parent_on_or_above_the_diagonal_fails(child: int, parent: int) -> None:
    """parents[j, i] says i is a parent of j; only i < j keeps the names in topological order."""
    parents = np.zeros((2, 2), dtype=bool)
    parents[child, parent] = True
    with pytest.raises(ValidationError):
        _graph(parents)


MISMATCHED = [
    pytest.param({'names': ['x0', 'x1']}, id='names one short'),
    pytest.param({'latent': np.zeros(2, dtype=bool)}, id='latent one short'),
    pytest.param({'mechanism': ['noise', 'noise']}, id='mechanism one short'),
    pytest.param({'parents': np.zeros((3, 2), dtype=bool)}, id='parents not square'),
]


@pytest.mark.parametrize('change', MISMATCHED)
def test_graph_fields_of_different_lengths_fail(change: dict) -> None:
    """Every field indexes the same nodes; a mismatch would misalign names and mechanisms."""
    fields = {
        'names': ['x0', 'x1', 'x2'],
        'parents': np.zeros((3, 3), dtype=bool),
        'latent': np.zeros(3, dtype=bool),
        'mechanism': ['noise', 'noise', 'noise'],
    }
    with pytest.raises(ValidationError):
        Graph(**fields | change)


@pytest.mark.parametrize('mechanism', [['linear', 'mlp'], ['noise', 'noise']])
def test_noise_is_the_mechanism_of_exactly_the_roots(mechanism: list[str]) -> None:
    """A root's mechanism is its noise; a node with parents computes from them."""
    parents = np.array([[False, False], [True, False]])
    with pytest.raises(ValidationError):
        Graph(
            names=['x0', 'x1'], parents=parents, latent=np.zeros(2, dtype=bool), mechanism=mechanism
        )


def test_an_empty_graph_is_valid() -> None:
    """A budget of 0 with no latent nodes draws a dataset of rows and no nodes at all."""
    _graph(np.zeros((0, 0), dtype=bool))


def test_a_dataset_holds_its_arrays_without_copying() -> None:
    """metabeta hands the columns to torch.from_numpy; a copy per draw would cost throughput."""
    values = np.zeros(5, dtype=np.float32)
    latent = np.ones(5, dtype=np.float32)
    dataset = Dataset(
        columns={'x0': values},
        metadata={'x0': Column(type='continuous', marginal='normal')},
        graph=_graph(np.zeros((2, 2), dtype=bool)),
        latent={'z0': latent},
        hyperparameters=Hyperprior().resolve(np.random.default_rng(0)),
        key=Key(seed=0, index=0),
        version='0.0.1',
    )
    assert dataset.columns['x0'] is values and dataset.latent['z0'] is latent
