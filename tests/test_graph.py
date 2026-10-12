"""The graph module's seams: what a Graph guarantees on its own,
and connect, the Cauchy generating graph over observed and latent nodes.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.graph import Graph, connect
from tabdgp.hyperprior import Hyperprior
from tests.fixtures.build import graphOf, point


@pytest.mark.parametrize(('child', 'parent'), [(0, 1), (1, 1)], ids=['above', 'on'])
def test_a_parent_on_or_above_the_diagonal_fails(child: int, parent: int) -> None:
    """parents[j, i] says i is a parent of j; only i < j keeps the names in topological order."""
    parents = [[], []]
    parents[child].append(parent)
    with pytest.raises(ValidationError):
        graphOf(parents)


# x0 -> x1 -> x2: a root, a linear node and an MLP node, each with exactly the fields it draws
CHAIN = {
    'names': ['x0', 'x1', 'x2'],
    'parents': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=bool),
    'latent': np.zeros(3, dtype=bool),
    'mechanism': [None, 'linear', 'mlp'],
    'activation': [None, None, 'tanh'],
    'noise_scale': [None, 0.1, 0.2],
    'init_scale': [None, None, 1.5],
}

MISMATCHED = [
    pytest.param({'names': ['x0', 'x1']}, id='names one short'),
    pytest.param({'latent': np.zeros(2, dtype=bool)}, id='latent one short'),
    pytest.param({'mechanism': [None, 'linear']}, id='mechanism one short'),
    pytest.param({'parents': np.zeros((3, 2), dtype=bool)}, id='parents not square'),
    pytest.param({'activation': [None, None]}, id='activation one short'),
    pytest.param({'noise_scale': [None, 0.1]}, id='noise scale one short'),
    pytest.param({'init_scale': [None, None]}, id='init scale one short'),
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


def test_a_graph_records_what_each_node_draws_for_itself() -> None:
    """Each node draws its own family, noise scale, activation and init scale;
    keeping them keeps the dataset white-box.
    """
    graph = Graph(**CHAIN)
    assert graph.mechanism == [None, 'linear', 'mlp']
    assert graph.activation == [None, None, 'tanh']
    assert graph.noise_scale == [None, 0.1, 0.2]
    assert graph.init_scale == [None, None, 1.5]


UNUSED_OR_MISSING = [
    pytest.param({'mechanism': ['linear', 'linear', 'mlp']}, id='a root with a family'),
    pytest.param({'mechanism': [None, None, 'mlp']}, id='a node with parents without a family'),
    pytest.param({'noise_scale': [0.1, 0.1, 0.2]}, id='a root with a noise scale'),
    pytest.param(
        {'noise_scale': [None, None, 0.2]},
        id='a node with parents without a noise scale',
    ),
    pytest.param({'activation': [None, 'tanh', 'tanh']}, id='a linear node with an activation'),
    pytest.param({'init_scale': [None, 1.0, 1.5]}, id='a linear node with an init scale'),
    pytest.param({'activation': [None, None, None]}, id='an mlp node without an activation'),
    pytest.param({'init_scale': [None, None, None]}, id='an mlp node without an init scale'),
]


@pytest.mark.parametrize('change', UNUSED_OR_MISSING)
def test_a_node_holds_a_value_exactly_for_the_fields_it_draws(change: dict) -> None:
    """A recorded value that played no part would claim it did; a missing one hides a draw."""
    with pytest.raises(ValidationError):
        Graph(**CHAIN | change)


def test_an_empty_graph_is_valid() -> None:
    """A budget of 0 with no latent nodes draws a dataset of rows and no nodes at all."""
    graphOf([])


# connect carries whatever names the plan gives it; letters show that it mints none of its own
OBSERVED = list('abcdefghij')


def _connect(seed: int, n_observed: int, **overrides: object) -> Graph:
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(**overrides).resolveDataset(rng)
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
    """A latent node can be a common cause of columns but never an effect of one."""
    graph = _connect(0, 5, latent_nodes=3)
    assert graph.latent.tolist() == [True] * 3 + [False] * 5
    assert graph.names[:3] == ['z0', 'z1', 'z2']
    assert sorted(graph.names[3:]) == ['a', 'b', 'c', 'd', 'e']


def _roots(graph: Graph) -> np.ndarray:
    """(n,) bool: the nodes without parents."""
    return ~graph.parents.any(axis=1)


@pytest.mark.parametrize('family', ['linear', 'mlp'])
def test_every_node_with_parents_draws_its_family_from_the_mechanism_shares(family: str) -> None:
    """The mechanism shares are honoured; a root is its noise, so it draws no family."""
    for seed in range(20):
        graph = _connect(seed, 8, mechanism_shares=point('mechanism_shares', family))
        expected = [None if root else family for root in _roots(graph)]
        assert graph.mechanism == expected


# every pair of nodes an edge, so every node after the first has parents
DENSE = {'edge_offset': 30.0, 'hub_scale': 0.0, 'latent_nodes': 0}


def test_each_node_with_parents_draws_its_own_noise_scale() -> None:
    """Real tables mix strong and weak dependence, so the nodes of one dataset differ in noise."""
    graph = _connect(0, 6, **DENSE)
    assert graph.noise_scale[0] is None
    assert len(set(graph.noise_scale[1:])) == 5


def test_each_mlp_node_draws_its_own_activation_and_init_scale_and_a_linear_node_neither() -> None:
    """The activation shares are honoured per MLP node; a linear node is its readout alone."""
    sin = point('activation_shares', 'sin')
    mlp = _connect(
        0, 6, mechanism_shares=point('mechanism_shares', 'mlp'), activation_shares=sin, **DENSE
    )
    assert mlp.activation == [None] + ['sin'] * 5
    assert mlp.init_scale[0] is None
    assert len(set(mlp.init_scale[1:])) == 5
    linear = _connect(0, 6, mechanism_shares=point('mechanism_shares', 'linear'), **DENSE)
    assert linear.activation == [None] * 6
    assert linear.init_scale == [None] * 6


def test_a_point_override_gives_every_node_its_value() -> None:
    """A caller who wants one noise scale for the whole dataset passes it as a point."""
    graph = _connect(
        0,
        6,
        noise_scale=0.3,
        init_scale=2.0,
        **DENSE,
        mechanism_shares=point('mechanism_shares', 'mlp'),
    )
    assert graph.noise_scale[1:] == [0.3] * 5
    assert graph.init_scale[1:] == [2.0] * 5


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
    assert (first.mechanism, first.activation) == (second.mechanism, second.activation)
    assert (first.noise_scale, first.init_scale) == (second.noise_scale, second.init_scale)


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
