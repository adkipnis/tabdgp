"""The graph module's seam: connect, the Cauchy generating graph over observed and latent nodes."""

from __future__ import annotations

import numpy as np
import pytest

from tabdgp.dataset import Graph
from tabdgp.graph import connect
from tabdgp.hyperprior import Hyperprior
from tests.fixtures.build import point

# connect carries whatever names the plan gives it; letters show that it mints none of its own
OBSERVED = list('abcdefghij')


def _connect(seed: int, n_observed: int, **overrides: object) -> Graph:
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(**overrides).resolve(rng)
    return connect(resolved, OBSERVED[:n_observed], rng)


def test_zero_observed_and_zero_latent_nodes_give_an_empty_graph() -> None:
    """A budget of 0 with no latent nodes is a dataset of rows alone; the graph must still exist."""
    graph = _connect(0, 0, latent_nodes=0)
    assert graph.names == []
    assert graph.parents.shape == (0, 0)


def test_the_adjacency_is_strictly_lower_triangular() -> None:
    """parents[j, i] with i < j only: the names are in topological order, so propagation is one pass."""
    for seed in range(20):
        graph = _connect(seed, 10)
        assert graph.parents.dtype == bool
        assert not np.triu(graph.parents).any()


def test_latent_nodes_precede_observed_ones() -> None:
    """A latent node is a hub or a confounder, never a sink, so it sits before every observed node."""
    graph = _connect(0, 5, latent_nodes=3)
    assert graph.latent.tolist() == [True] * 3 + [False] * 5
    assert graph.names[:3] == ['z0', 'z1', 'z2']
    assert sorted(graph.names[3:]) == ['a', 'b', 'c', 'd', 'e']


@pytest.mark.parametrize('family', ['linear', 'mlp'])
def test_every_node_draws_its_family_from_the_mechanism_shares(family: str) -> None:
    """The mechanism shares are honoured for every node, roots included."""
    for seed in range(20):
        graph = _connect(seed, 8, mechanism_shares=point('mechanism_shares', family))
        assert graph.mechanism == [family] * len(graph.names)


def _density(offset: float, **overrides: object) -> float:
    """The share of possible edges present, averaged over seeds."""
    densities = []
    for seed in range(50):
        graph = _connect(seed, 10, edge_offset=offset, latent_nodes=0, **overrides)
        n = len(graph.names)
        densities.append(graph.parents.sum() / (n * (n - 1) / 2))
    return float(np.mean(densities))


def test_density_is_monotone_in_the_offset() -> None:
    """The offset is the one knob that moves the graph from near-empty to near-complete."""
    no_hubs_no_blocks = {'hub_scale': 0.0, 'block_bonus': 0.0}
    densities = [_density(offset, **no_hubs_no_blocks) for offset in [-8, -4, 0, 4, 8]]
    assert densities[0] < 0.01
    assert densities[-1] > 0.99
    assert densities == sorted(densities)


def _cliques(parents: np.ndarray) -> np.ndarray:
    """Whether the undirected skeleton is a disjoint union of cliques: its reachability is itself."""
    adjacent = parents | parents.T | np.eye(len(parents), dtype=bool)  # (n, n)
    return np.array_equal(adjacent.astype(int) @ adjacent.astype(int) > 0, adjacent)


def test_a_block_bonus_raises_within_block_density() -> None:
    """A bonus that dwarfs a far-negative offset connects every pair within a block and none across.

    The skeleton is then a disjoint union of cliques, one per non-empty block, and with two
    blocks some seed splits the nodes.
    """
    two_blocks = {'edge_offset': -30.0, 'block_bonus': 60.0, 'hub_scale': 0.0, 'blocks': 2}
    graphs = [_connect(seed, 8, latent_nodes=0, **two_blocks) for seed in range(20)]
    assert all(_cliques(graph.parents) for graph in graphs)
    assert any(not graph.parents[1:, 0].all() for graph in graphs)
    assert all(graph.parents.any() for graph in graphs)


def test_the_observed_names_are_a_random_permutation_of_the_plan() -> None:
    """Column position must reveal nothing of a node's place in the graph, scamd's weakness."""
    orders = {tuple(_connect(seed, 6, latent_nodes=0).names) for seed in range(20)}
    assert len(orders) > 1


def test_connect_is_deterministic_under_a_fixed_generator() -> None:
    """A key reproduces its dataset bitwise only if every step draws from the generator alone."""
    first, second = _connect(7, 9), _connect(7, 9)
    assert first.names == second.names
    assert np.array_equal(first.parents, second.parents)
    assert first.mechanism == second.mechanism


def test_every_latent_node_has_a_child_when_a_later_node_exists() -> None:
    """A latent node is a hub or a confounder; one with no children would be a dead latent column."""
    sparse = {'edge_offset': -30.0, 'hub_scale': 0.0, 'block_bonus': 0.0}
    for seed in range(20):
        graph = _connect(seed, 5, latent_nodes=3, **sparse)
        has_child = graph.parents.any(axis=0)  # (n,)
        assert has_child[graph.latent].all()
    last = _connect(0, 0, latent_nodes=1, **sparse)
    assert not last.parents.any()


def _degree_spread(hub_scale: float) -> float:
    """The mean over 50 graphs of 10 nodes of the standard deviation of the nodes' degrees."""
    spreads = []
    for seed in range(50):
        graph = _connect(
            seed, 10, latent_nodes=0, edge_offset=0.0, hub_scale=hub_scale, block_bonus=0.0
        )
        degree = graph.parents.sum(axis=0) + graph.parents.sum(axis=1)  # (n,)
        spreads.append(degree.std())
    return float(np.mean(spreads))


def test_a_larger_hub_scale_spreads_the_degrees() -> None:
    """hub_scale scales the Cauchy node terms, so a large one makes a few nodes hubs."""
    assert _degree_spread(2.0) > 1.2 * _degree_spread(0.0)
