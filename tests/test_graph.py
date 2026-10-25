"""The graph module's seams: what a Graph guarantees on its own,
and connect, the Cauchy generating graph over observed and latent nodes.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from tabdgp.column import Column, Grouping, assign
from tabdgp.graph import Graph, connect
from tabdgp.hyperprior import Hyperprior
from tests.fixtures.build import columnsOf, drawNothing, graphOf, groupingOf, point


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


def test_a_chain_that_meets_every_check_constructs() -> None:
    """The control for the checks below: each fails for its change to CHAIN alone.
    With its last node named y, it is a flat graph with the outcome.
    """
    Graph(**CHAIN)
    Graph(**CHAIN | {'names': ['x0', 'x1', 'y']})


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


MISLABELED = [
    pytest.param({'latent': np.array([True, False, False])}, id='an x node marked latent'),
    pytest.param({'names': ['z0', 'x1', 'x2']}, id='a z node not marked latent'),
]


@pytest.mark.parametrize('change', MISLABELED)
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
    """The control for the checks below: each fails for its change to GROUPED alone.
    With its last node named y, it is a graph with the outcome.
    """
    Graph(**GROUPED)
    Graph(**GROUPED | {'names': ['g0', 'x1', 'y']})


def test_a_y_not_last_fails() -> None:
    """The outcome is a sink: a node after it could read it, a covariate as its effect."""
    with pytest.raises(ValidationError):
        Graph(**CHAIN | {'names': ['x0', 'y', 'x2']})


def test_a_y_at_a_factors_unit_fails() -> None:
    """The outcome is drawn once per row, never once per group."""
    group_level_y = {
        'names': ['g0', 'x1', 'y'],
        'unit': ['g0', 'g0', 'g0'],
        'group_share': [None] * 3,
        'group_factor': [None] * 3,
    }
    with pytest.raises(ValidationError):
        Graph(**GROUPED | group_level_y)


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
    a parent may be at the child's unit or a superior one, so a superior unit comes first.
    """
    with pytest.raises(ValidationError):
        Graph(**GROUPED | change)


# connect carries whatever names the plan gives it; letters show that it mints none of its own,
# skipping g and z, the prefixes of grouping factors and latent nodes
OBSERVED = list('abcdefhijk')
FLAT = groupingOf([])


def _connect(
    seed: int,
    observed_nodes: int,
    grouping: Grouping = FLAT,
    with_y: bool = False,
    **overrides: object,
) -> Graph:
    """A graph over a plan of continuous columns, with y under the scm form,
    else without y unless the overrides name another form.
    """
    rng = np.random.default_rng(seed)
    form = {'outcome_form': 'scm' if with_y else 'none'}
    resolved = Hyperprior(**form | overrides).resolveDataset(rng)
    columns = columnsOf(OBSERVED[:observed_nodes])
    return connect(resolved, grouping, columns, rng)


def test_zero_observed_and_zero_latent_nodes_give_an_empty_graph() -> None:
    """A budget of 0 with no latent nodes is a dataset of rows alone; the graph must still exist."""
    graph = _connect(0, 0, latent_nodes=0)
    assert graph.names == []
    assert graph.parents.shape == (0, 0)


def test_connect_on_a_flat_grouping_is_spec_1s_graph() -> None:
    """A flat table keeps spec #1's graph: every node at the row, no g node, no two-part field."""
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
    """A parent is at its child's unit or a superior one, so the superior units come first;
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


# (child's unit, parent's unit) for GROUPING: a child reads its unit and the superior ones of its
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
    """(v,) bool: the nodes without parents."""
    return ~graph.parents.any(axis=1)


@pytest.mark.parametrize('family', ['linear', 'mlp'])
def test_every_node_with_parents_draws_its_family_from_the_mechanism_shares(family: str) -> None:
    """The mechanism shares are honored; a root is its noise, so it draws no family."""
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
    """The activation shares are honored per MLP node; a linear node is its readout alone."""
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
        v = len(graph.names)
        densities.append(graph.parents.sum() / (v * (v - 1) / 2))
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
    adjacent = parents | parents.T | np.eye(len(parents), dtype=bool)  # (v, v)
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
    """A key reproduces its dataset bitwise only if every step draws from the generator alone.
    Under a model, so that y, which no column of the plan names, is drawn too.
    """
    first = _connect(7, 9, grouping, outcome_form=drawNothing)
    second = _connect(7, 9, grouping, outcome_form=drawNothing)
    assert first.names == second.names
    assert np.array_equal(first.parents, second.parents)
    assert (first.mechanism, first.activation) == (second.mechanism, second.activation)
    assert (first.noise_scale, first.init_scale) == (second.noise_scale, second.init_scale)
    assert (first.unit, first.group_share) == (second.unit, second.group_share)
    assert first.group_factor == second.group_factor


def test_a_column_with_more_levels_than_its_factors_groups_stays_at_the_row() -> None:
    """A group-level column takes one value per group, so it shows at most m of its levels;
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
        assert [*factors, 'row'] == grouping.traceChain('row')
        assert set(graph.unit) <= set(grouping.traceChain('row'))


# one factor of two groups; seed 2 puts nodes at both units and draws both families
ONE_FACTOR = groupingOf([[0, 0, 1, 1]])


def test_connect_without_y_is_spec_22s_graph() -> None:
    """A plan without y, under outcome_form = 'none',
    keeps the dataset its key drew before the outcome.
    The literals are spec #22's graph for this key.
    """
    graph = _connect(2, 3, ONE_FACTOR, latent_nodes=1, group_level_share=0.5)
    child, parent = np.nonzero(graph.parents)
    assert graph.names == ['g0', 'c', 'z0', 'b', 'a']
    assert graph.unit == ['g0', 'g0', 'row', 'row', 'row']
    assert list(zip(child.tolist(), parent.tolist())) == [(3, 1), (3, 2), (4, 3)]
    assert graph.mechanism == [None, None, None, 'linear', 'mlp']
    # the scales pass through the platform's log and exp, whose last digit differs between platforms
    noise_scale = [None, None, None, 0.7493527189794704, 0.140907138595985]
    assert graph.noise_scale == pytest.approx(noise_scale, rel=1e-12)
    group_share = [None, None, 0.3932290382601226, 0.0019036411256082356, 0.00010978727102732418]
    assert graph.group_share == pytest.approx(group_share, rel=1e-12)


@given(seed=st.integers(0, 2**32 - 1), factors=st.integers(0, 3))
def test_y_is_last_and_at_the_row_with_no_children_under_every_grouping(
    seed: int, factors: int
) -> None:
    """No covariate is an effect of the outcome, and the outcome is drawn once per row,
    even when every other node is group-level and every pair of nodes an edge.
    """
    rng = np.random.default_rng(seed)
    hyperprior = Hyperprior.grouped(
        rows=100, factors=factors, group_level_share=1.0, edge_offset=30.0, hub_scale=0.0
    )
    resolved = hyperprior.resolveDataset(rng)
    grouping = assign(resolved, rng)
    columns = columnsOf(OBSERVED[:5])
    graph = connect(resolved, grouping, columns, rng)
    assert graph.names[-1] == 'y'
    assert graph.unit[-1] == 'row'
    assert not graph.parents[:, -1].any()


def test_y_is_in_the_graph_under_a_model_without_a_planned_column_and_absent_under_none() -> None:
    """A model writes y's column itself, so the plan holds none, but every form turns y's node
    into y: the graph draws it like the scm form's, last at the row, with the outcome's noise.
    """
    model = _connect(0, 4, GROUPING, outcome_form=drawNothing, outcome_noise_scale=2.0, **DENSE)
    assert (model.names[-1], model.unit[-1], model.noise_scale[-1]) == ('y', 'row', 2.0)
    assert model.group_factor[-1] in ['g0', 'g1', 'g2']
    none = _connect(0, 4, GROUPING, outcome_form='none')
    assert 'y' not in none.names


def test_y_draws_outcome_noise_scale_and_the_other_nodes_noise_scale() -> None:
    """The outcome's share of unexplained variance reaches far above a covariate's,
    so its noise has a field of its own; a point override of each shows which a node reads.
    """
    graph = _connect(0, 5, with_y=True, noise_scale=0.3, outcome_noise_scale=2.0, **DENSE)
    assert graph.noise_scale == [None] + [0.3] * 4 + [2.0]


def test_a_root_y_records_no_noise_scale() -> None:
    """A root is its noise alone, so y without parents draws no scale, as every root."""
    no_edges = {'edge_offset': -30.0, 'hub_scale': 0.0, 'block_bonus': 0.0}
    graph = _connect(0, 4, with_y=True, outcome_noise_scale=2.0, **no_edges)
    assert not graph.parents[-1].any()
    assert (graph.mechanism[-1], graph.noise_scale[-1]) == (None, None)


def test_y_may_have_zero_parents_and_may_have_a_latent_parent() -> None:
    """No edge into y is forced or forbidden beyond the mask:
    a null result is a valid dataset, and a latent parent makes the outcome endogenous.
    """
    parentless_seen = False
    latent_parent_seen = False
    for seed in range(50):
        graph = _connect(seed, 4, GROUPING, with_y=True, latent_nodes=2)
        parents_of_y = graph.parents[-1]  # (v,)
        parentless_seen |= not parents_of_y.any()
        latent_parent_seen |= bool((parents_of_y & graph.latent).any())
    assert parentless_seen
    assert latent_parent_seen


def test_y_draws_a_group_share_and_group_factor_only_in_a_grouped_graph() -> None:
    """The outcome mixes a group part and a row part like any row-unit node;
    a flat table has no groups to mix.
    """
    grouped = _connect(0, 4, GROUPING, with_y=True, group_share=0.25)
    assert grouped.group_share[-1] == 0.25
    assert grouped.group_factor[-1] in ['g0', 'g1', 'g2']
    flat = _connect(0, 4, with_y=True, group_share=0.25)
    assert (flat.group_share[-1], flat.group_factor[-1]) == (None, None)


def _degreeSpread(hub_scale: float) -> float:
    """The mean over 50 graphs of 10 nodes of the standard deviation of the nodes' degrees."""
    spreads = []
    for seed in range(50):
        graph = _connect(
            seed, 10, latent_nodes=0, edge_offset=0.0, hub_scale=hub_scale, block_bonus=0.0
        )
        degree = graph.parents.sum(axis=0) + graph.parents.sum(axis=1)  # (v,)
        spreads.append(degree.std())
    return float(np.mean(spreads))


def test_a_larger_hub_scale_spreads_the_degrees() -> None:
    """hub_scale scales the Cauchy node terms, so a large one makes a few nodes hubs."""
    assert _degreeSpread(2.0) > 1.2 * _degreeSpread(0.0)
