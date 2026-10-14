"""The graph module's seams: what a Graph guarantees on its own,
and connect, the Cauchy generating graph over observed and latent nodes.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.column import Column, Grouping, assign
from tabdgp.graph import Graph, connect
from tabdgp.hyperprior import Hyperprior
from tests.fixtures.build import columnsOf, graphOf, groupingOf, point


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
    'unit': ['row'] * 3,
    'group_share': [None] * 3,
    'group_factor': [None] * 3,
}

MISMATCHED = [
    pytest.param({'names': ['x0', 'x1']}, id='names one short'),
    pytest.param({'latent': np.zeros(2, dtype=bool)}, id='latent one short'),
    pytest.param({'mechanism': [None, 'linear']}, id='mechanism one short'),
    pytest.param({'parents': np.zeros((3, 2), dtype=bool)}, id='parents not square'),
    pytest.param({'activation': [None, None]}, id='activation one short'),
    pytest.param({'noise_scale': [None, 0.1]}, id='noise scale one short'),
    pytest.param({'init_scale': [None, None]}, id='init scale one short'),
    pytest.param({'unit': ['row'] * 2}, id='unit one short'),
    pytest.param({'group_share': [None] * 2}, id='group share one short'),
    pytest.param({'group_factor': [None] * 2}, id='group factor one short'),
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
    """The mask and the x…/z… names both tell latent from observed nodes;
    they may not disagree.
    """
    with pytest.raises(ValidationError):
        Graph(**CHAIN | change)


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


# g0 is a grouping factor; x1 is a root at g0's unit, x2 a linear child of x1 at the row
GROUPED = {
    'names': ['g0', 'x1', 'x2'],
    'parents': np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]], dtype=bool),
    'latent': np.zeros(3, dtype=bool),
    'mechanism': [None, None, 'linear'],
    'activation': [None] * 3,
    'noise_scale': [None, None, 0.1],
    'init_scale': [None] * 3,
    'unit': ['g0', 'g0', 'row'],
    'group_share': [None, None, 0.3],
    'group_factor': [None, None, 'g0'],
}


def test_a_grouped_graph_that_meets_every_check_constructs() -> None:
    """The control for the checks below: each fails for its change to GROUPED alone."""
    Graph(**GROUPED)


UNKNOWN = [
    pytest.param({'unit': ['g0', 'g1', 'row']}, id='a factor not among the names'),
    pytest.param({'unit': ['g0', 'cell', 'row']}, id='neither the row nor a factor'),
    pytest.param({'group_factor': [None, None, 'g1']}, id='a group factor not among the names'),
]


@pytest.mark.parametrize('change', UNKNOWN)
def test_a_node_at_an_unknown_unit_fails(change: dict) -> None:
    """A unit is the row or a grouping factor of the dataset, and so is the factor of a group part;
    any other name has no groups to be drawn per.
    """
    with pytest.raises(ValidationError):
        Graph(**GROUPED | change)


def _parentsOf(*edges: tuple[int, int]) -> np.ndarray:
    """(3, 3) bool: the adjacency of three nodes with the given (child, parent) edges."""
    parents = np.zeros((3, 3), dtype=bool)
    for child, parent in edges:
        parents[child, parent] = True
    return parents


FACTOR_EDGES = [
    pytest.param(
        {
            'parents': _parentsOf((1, 0), (2, 1)),
            'mechanism': [None, 'linear', 'linear'],
            'noise_scale': [None, 0.1, 0.1],
        },
        id='a parent of a group-level node',
    ),
    pytest.param({'parents': _parentsOf((2, 0), (2, 1))}, id='a parent of a row-unit node'),
    pytest.param(
        {
            'names': ['g0', 'g1', 'x2'],
            'unit': ['g0', 'g1', 'row'],
            'parents': _parentsOf((1, 0)),
            'mechanism': [None, 'linear', None],
            'noise_scale': [None, 0.1, None],
        },
        id='a child of another factor',
    ),
]


@pytest.mark.parametrize('change', FACTOR_EDGES)
def test_a_g_node_with_an_edge_fails(change: dict) -> None:
    """A grouping factor's values are its codes, drawn by the size draw:
    it reads no parent, and a node reads it only through its unit.
    """
    with pytest.raises(ValidationError):
        Graph(**GROUPED | change)


FACTOR_UNITS = [
    pytest.param({'unit': ['row', 'g0', 'row']}, id='at the row'),
    pytest.param(
        {
            'names': ['g0', 'g1', 'x2'],
            'unit': ['g0', 'g0', 'row'],
            'parents': _parentsOf(),
            'mechanism': [None] * 3,
            'noise_scale': [None] * 3,
        },
        id='at another factor',
    ),
]


@pytest.mark.parametrize('change', FACTOR_UNITS)
def test_a_g_node_at_another_unit_fails(change: dict) -> None:
    """A grouping factor takes one value per group of its own, its group's code."""
    with pytest.raises(ValidationError):
        Graph(**GROUPED | change)


MISPLACED_PARTS = [
    pytest.param({'group_share': [None, 0.3, 0.3]}, id='a group share on a group-level node'),
    pytest.param({'group_factor': [None, 'g0', 'g0']}, id='a group factor on a group-level node'),
    pytest.param({'group_share': [0.3, None, 0.3]}, id='a group share on a grouping factor'),
    pytest.param(
        {'names': ['x0', 'x1', 'x2'], 'unit': ['row'] * 3, 'group_factor': [None] * 3},
        id='a group share in a flat graph',
    ),
]


@pytest.mark.parametrize('change', MISPLACED_PARTS)
def test_a_two_part_field_on_a_group_level_node_fails(change: dict) -> None:
    """Only a row-unit node mixes a group part and a row part;
    a group-level node is drawn at its unit, and a flat graph has no groups to mix.
    """
    with pytest.raises(ValidationError):
        Graph(**GROUPED | change)


MISSING_PARTS = [
    pytest.param({'group_share': [None] * 3}, id='a row-unit node without a group share'),
    pytest.param({'group_factor': [None] * 3}, id='a row-unit node without a group factor'),
    pytest.param(
        {
            'parents': _parentsOf(),
            'mechanism': [None] * 3,
            'noise_scale': [None] * 3,
            'group_share': [None] * 3,
            'group_factor': [None] * 3,
        },
        id='a row-unit root without either',
    ),
]


@pytest.mark.parametrize('change', MISSING_PARTS)
def test_a_missing_two_part_field_on_a_row_unit_node_fails(change: dict) -> None:
    """Every row-unit node of a grouped graph mixes a group part, roots included:
    a root's group part is pure group-level noise.
    """
    with pytest.raises(ValidationError):
        Graph(**GROUPED | change)


# g0 and g1 are grouping factors; x2 at g1's unit before x3 at g0's, both roots
LATER_UNIT_FIRST = {
    'names': ['g0', 'g1', 'x2', 'x3'],
    'parents': np.zeros((4, 4), dtype=bool),
    'latent': np.zeros(4, dtype=bool),
    'mechanism': [None] * 4,
    'activation': [None] * 4,
    'noise_scale': [None] * 4,
    'init_scale': [None] * 4,
    'unit': ['g0', 'g1', 'g1', 'g0'],
    'group_share': [None] * 4,
    'group_factor': [None] * 4,
}

OUT_OF_ORDER = [
    pytest.param(
        {
            'unit': ['g0', 'row', 'g0'],
            'parents': _parentsOf(),
            'mechanism': [None] * 3,
            'noise_scale': [None] * 3,
            'group_share': [None, 0.3, None],
            'group_factor': [None, 'g0', None],
        },
        id='the row before a factor',
    ),
    pytest.param(LATER_UNIT_FIRST, id='a later factor before an earlier one'),
    pytest.param(
        {
            'names': ['x0', 'g1', 'x2'],
            'unit': ['g1', 'g1', 'row'],
            'parents': _parentsOf((2, 0)),
            'group_factor': [None, None, 'g1'],
        },
        id='a factor after another node',
    ),
]


@pytest.mark.parametrize('change', OUT_OF_ORDER)
def test_units_out_of_order_fail(change: dict) -> None:
    """The factors, then their units in the factors' order, the row last:
    a parent may be at the child's unit or a coarser one, so a coarser unit comes first.
    """
    with pytest.raises(ValidationError):
        Graph(**GROUPED | change)


# connect carries whatever names the plan gives it; letters show that it mints none of its own,
# skipping g and z, the prefixes of grouping factors and latent nodes
OBSERVED = list('abcdefhijk')
FLAT = groupingOf([])


def _connect(
    seed: int, observed_nodes: int, grouping: Grouping = FLAT, **overrides: object
) -> Graph:
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(**overrides).resolveDataset(rng)
    return connect(resolved, grouping, columnsOf(OBSERVED[:observed_nodes]), rng)


def test_zero_observed_and_zero_latent_nodes_give_an_empty_graph() -> None:
    """A budget of 0 with no latent nodes is a dataset of rows alone; the graph must still exist."""
    graph = _connect(0, 0, latent_nodes=0)
    assert graph.names == []
    assert graph.parents.shape == (0, 0)


def test_connect_on_a_flat_grouping_is_item_1s_graph() -> None:
    """A flat table keeps item 1's graph: every node at the row, no g node, no two-part field."""
    graph = _connect(0, 6, latent_nodes=2)
    assert graph.unit == ['row'] * 8
    assert graph.group_share == [None] * 8
    assert graph.group_factor == [None] * 8
    assert not any(name.startswith('g') for name in graph.names)


# g1 nests in g0, g2 is crossed with both; connect reads the factors and their chains, not the codes
GROUPING = groupingOf(
    [[0, 0, 1, 1], [0, 1, 2, 3], [0, 1, 0, 1]],
    nests_in=[None, 'g0', None],
)
UNITS = ['g0', 'g1', 'g2', 'row']


def _ranks(graph: Graph) -> list[int]:
    """Each node's unit as its position among the factors and the row, the g nodes left out."""
    return [UNITS.index(unit) for unit in graph.unit[3:]]


def test_g_nodes_come_first_then_the_units_in_index_order_and_the_row_last() -> None:
    """A parent is at its child's unit or a coarser one, so the coarser units come first;
    crossed units have no order between them and take the factors' fixed one.
    """
    units_seen = set()
    for seed in range(20):
        graph = _connect(seed, 8, GROUPING, latent_nodes=3)
        assert graph.names[:3] == ['g0', 'g1', 'g2']
        assert graph.unit[:3] == ['g0', 'g1', 'g2']
        assert len(graph.names) == 3 + 3 + 8
        assert _ranks(graph) == sorted(_ranks(graph))
        units_seen |= set(graph.unit[3:])
    assert units_seen == set(UNITS)


def _latentAt(graph: Graph, unit: str) -> list[bool]:
    """The latent mask of the nodes at one unit, in the graph's order, the g nodes left out."""
    return [bool(graph.latent[j]) for j in range(3, len(graph.names)) if graph.unit[j] == unit]


def test_latent_nodes_precede_observed_ones_within_a_unit() -> None:
    """Within a unit a latent node can be a common cause of columns but never an effect of one;
    the latent names follow the order.
    """
    for seed in range(20):
        graph = _connect(seed, 8, GROUPING, latent_nodes=4)
        for unit in UNITS:
            assert _latentAt(graph, unit) == sorted(_latentAt(graph, unit), reverse=True)
        latent_names = [name for name in graph.names if name.startswith('z')]
        assert latent_names == ['z0', 'z1', 'z2', 'z3']


# (child's unit, parent's unit) for GROUPING: a child reads its unit and the coarser ones of its
# chain; g2 is crossed, so neither reads the other's, and the row reads every unit
READS = {
    ('g0', 'g0'),
    ('g1', 'g0'),
    ('g1', 'g1'),
    ('g2', 'g2'),
    ('row', 'g0'),
    ('row', 'g1'),
    ('row', 'g2'),
    ('row', 'row'),
}


def _unitPairs(graph: Graph) -> set[tuple[str, str]]:
    """The (child's unit, parent's unit) of every edge."""
    child, parent = np.nonzero(graph.parents)
    return {(graph.unit[j], graph.unit[i]) for j, i in zip(child, parent)}


def test_no_edge_crosses_out_of_a_chain() -> None:
    """None between crossed units, none into a group-level node from the row.

    Every edge is drawn, so the mask alone decides which remain, and every pair it allows appears.
    """
    pairs_seen = set()
    for seed in range(20):
        graph = _connect(
            seed,
            8,
            GROUPING,
            latent_nodes=4,
            group_level_share=0.5,
            edge_offset=30.0,
            hub_scale=0.0,
        )
        assert _unitPairs(graph) <= READS
        pairs_seen |= _unitPairs(graph)
    assert pairs_seen == READS


def test_group_level_share_1_puts_every_node_at_a_factor_and_0_at_the_row() -> None:
    """The share is the probability that a node is group-level, latent nodes included."""
    for seed in range(10):
        group_level = _connect(seed, 8, GROUPING, latent_nodes=3, group_level_share=1.0)
        assert 'row' not in group_level.unit
        within_group = _connect(seed, 8, GROUPING, latent_nodes=3, group_level_share=0.0)
        assert within_group.unit[3:] == ['row'] * 11


def _rowShares(graph: Graph) -> list[float]:
    """The group shares of the row-unit nodes, in the graph's order."""
    return [share for share, unit in zip(graph.group_share, graph.unit) if unit == 'row']


def test_every_row_unit_node_draws_its_own_group_share() -> None:
    """Real tables mix strongly and weakly confounded columns, so the nodes of one dataset differ."""
    graph = _connect(0, 8, GROUPING, latent_nodes=3, group_level_share=0.5)
    shares = _rowShares(graph)
    assert len(shares) >= 2
    assert len(set(shares)) == len(shares)
    assert all(0 <= share <= 1 for share in shares)


def test_a_point_group_share_gives_every_row_unit_node_its_value() -> None:
    """A study that wants one confounding strength for the whole dataset passes it as a point."""
    graph = _connect(0, 8, GROUPING, latent_nodes=3, group_share=0.25)
    assert graph.unit.count('row') >= 1
    assert _rowShares(graph) == [0.25] * graph.unit.count('row')


def test_group_factor_is_drawn_among_the_factors() -> None:
    """Each row-unit node is confounded with respect to one factor, any of the dataset's."""
    factors_seen = set()
    for seed in range(10):
        graph = _connect(seed, 8, GROUPING, group_level_share=0.0)
        factors_seen |= set(graph.group_factor[3:])
    assert factors_seen == {'g0', 'g1', 'g2'}


def test_the_adjacency_is_strictly_lower_triangular() -> None:
    """parents[j, i] with i < j only:
    the names are in topological order, so propagation is one pass.
    """
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
    """(d,) bool: the nodes without parents."""
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
        d = len(graph.names)
        densities.append(graph.parents.sum() / (d * (d - 1) / 2))
    return float(np.mean(densities))


def test_density_is_monotone_in_the_offset() -> None:
    """The offset is the one knob that moves the graph from near-empty to near-complete."""
    no_hubs_no_blocks = {'hub_scale': 0.0, 'block_bonus': 0.0}
    densities = [_density(offset, **no_hubs_no_blocks) for offset in [-8, -4, 0, 4, 8]]
    assert densities[0] < 0.01
    assert densities[-1] > 0.99
    assert densities == sorted(densities)


def _cliques(parents: np.ndarray) -> np.ndarray:
    """Whether the undirected skeleton is a disjoint union of cliques:
    its reachability is itself.
    """
    adjacent = parents | parents.T | np.eye(len(parents), dtype=bool)  # (d, d)
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


@pytest.mark.parametrize('grouping', [FLAT, GROUPING], ids=['flat', 'grouped'])
def test_connect_is_deterministic_under_a_fixed_generator(grouping: Grouping) -> None:
    """A key reproduces its dataset bitwise only if every step draws from the generator alone."""
    first, second = _connect(7, 9, grouping), _connect(7, 9, grouping)
    assert first.names == second.names
    assert np.array_equal(first.parents, second.parents)
    assert (first.mechanism, first.activation) == (second.mechanism, second.activation)
    assert (first.noise_scale, first.init_scale) == (second.noise_scale, second.init_scale)
    assert (first.unit, first.group_share) == (second.unit, second.group_share)
    assert first.group_factor == second.group_factor


def test_a_column_with_more_levels_than_its_factors_groups_stays_at_the_row() -> None:
    """A group-level column takes one value per group, so it shows at most G of its levels;
    one with more would leave a level no row has, an all-zero dummy column.
    """
    grouping = groupingOf([[0, 0, 1, 1, 2, 2]])  # one factor of 3 groups
    alpha = {'level_alpha': 1.0}
    columns = {
        'a': Column(type='binary', levels=2, parameters=alpha),
        'b': Column(type='categorical', levels=3, parameters=alpha),
        'c': Column(type='ordered', levels=4, parameters=alpha),
    }
    for seed in range(10):
        rng = np.random.default_rng(seed)
        resolved = Hyperprior(group_level_share=1.0).resolveDataset(rng)
        graph = connect(resolved, grouping, columns, rng)
        unit = dict(zip(graph.names, graph.unit))
        assert [unit['a'], unit['b'], unit['c']] == ['g0', 'g0', 'row']


def test_the_units_a_graph_accepts_are_the_groupings_factors_and_the_row() -> None:
    """graph tells a grouping factor by its g prefix and column names the factors:
    the two lists must agree, or a drawn grouping would make an invalid graph.
    """
    for seed in range(20):
        rng = np.random.default_rng(seed)
        resolved = Hyperprior.grouped(rows=200).resolveDataset(rng)
        grouping = assign(resolved, rng)
        graph = connect(resolved, grouping, columnsOf(OBSERVED[:5]), rng)
        factors = [name for name in graph.names if name.startswith('g')]
        assert [*factors, 'row'] == grouping.chain('row')
        assert set(graph.unit) <= set(grouping.chain('row'))


def _degreeSpread(hub_scale: float) -> float:
    """The mean over 50 graphs of 10 nodes of the standard deviation of the nodes' degrees."""
    spreads = []
    for seed in range(50):
        graph = _connect(
            seed, 10, latent_nodes=0, edge_offset=0.0, hub_scale=hub_scale, block_bonus=0.0
        )
        degree = graph.parents.sum(axis=0) + graph.parents.sum(axis=1)  # (d,)
        spreads.append(degree.std())
    return float(np.mean(spreads))


def test_a_larger_hub_scale_spreads_the_degrees() -> None:
    """hub_scale scales the Cauchy node terms, so a large one makes a few nodes hubs."""
    assert _degreeSpread(2.0) > 1.2 * _degreeSpread(0.0)
