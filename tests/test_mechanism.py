"""The mechanism module's seam: propagate, every node's values from its parents and noise."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tabdgp.column import Grouping
from tabdgp.graph import Graph, connect
from tabdgp.hyperprior import Hyperprior
from tabdgp.mechanism import ACTIVATIONS, propagate
from tests.fixtures.build import columnsOf, graphOf, groupingOf, point

OBSERVED = list('abcdefhijklm')  # no g, a grouping factor's prefix
FLAT = groupingOf([])


def _balanced(n: int, groups: list[int], nests_in: list[str | None] | None = None) -> Grouping:
    """A grouping of n rows with balanced groups, each crossed factor's codes a random
    permutation and a nested factor's groups splitting each coarser group evenly.
    """
    if nests_in is None:
        nests_in = [None] * len(groups)
    rng = np.random.default_rng(len(groups))
    codes = []
    for groups_of_k, nests_in_of_k in zip(groups, nests_in):
        if nests_in_of_k is None:
            codes.append(rng.permutation(n) % groups_of_k)
            continue
        coarser = codes[int(nests_in_of_k[1:])]  # (n,)
        ratio = groups_of_k // (coarser.max() + 1)  # finer groups per coarser group
        finer = np.zeros(n, dtype=np.int64)  # (n,)
        for group in range(coarser.max() + 1):
            rows = np.flatnonzero(coarser == group)
            finer[rows] = group * ratio + np.arange(len(rows)) % ratio
        codes.append(finer)
    return groupingOf([codes_of_k.tolist() for codes_of_k in codes], nests_in)


def _size(grouping: Grouping, unit: str, n: int) -> int:
    return n if unit == 'row' else grouping.metadata[unit].levels


# g1 nests in g0 and g2 is crossed with both
GROUPED = _balanced(1000, [20, 60, 15], nests_in=[None, 'g0', None])
GROUPINGS = [pytest.param(FLAT, id='flat'), pytest.param(GROUPED, id='grouped')]


def _matrix(nodes: dict[str, np.ndarray]) -> np.ndarray:
    """(n, d): the nodes side by side in the graph's order."""
    return np.stack(list(nodes.values()), axis=1)


def _propagateOver(
    grouping: Grouping, seed: int, observed_nodes: int, n: int, **overrides: object
) -> dict[str, np.ndarray]:
    """Node values over a graph drawn by connect on the grouping, each at its unit."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(rows=n, **overrides).resolveDataset(rng)
    graph = connect(resolved, grouping, columnsOf(OBSERVED[:observed_nodes]), rng)
    return propagate(graph, resolved, grouping, rng)


def _propagate(seed: int, observed_nodes: int, n: int, **overrides: object) -> np.ndarray:
    """Node values over a flat graph drawn by connect, as one matrix."""
    return _matrix(_propagateOver(FLAT, seed, observed_nodes, n, **overrides))


def test_propagate_names_every_node_with_its_own_contiguous_array() -> None:
    """draw joins nodes to columns by name and hands the latent ones out as they are.

    A view into every node's values would keep them all alive and read strided.
    """
    rng = np.random.default_rng(0)
    resolved = Hyperprior(rows=100, latent_nodes=2).resolveDataset(rng)
    graph = connect(resolved, FLAT, columnsOf(OBSERVED[:4]), rng)
    nodes = propagate(graph, resolved, FLAT, rng)
    assert list(nodes) == graph.names
    for values in nodes.values():
        assert values.shape == (100,)
        assert values.flags.c_contiguous and values.base is None


def test_every_node_is_z_scored() -> None:
    """Relative noise and the marginals downstream assume every node has mean 0 and sd 1."""
    for seed in range(10):
        values = _propagate(seed, 8, 500, latent_nodes=2)
        assert values.shape == (500, 10)
        assert values.dtype == np.float32
        np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-5)
        np.testing.assert_allclose(values.std(axis=0), 1, atol=1e-5)


def _run(
    graph: Graph, n: int, seed: int = 0, grouping: Grouping = FLAT, **overrides: object
) -> np.ndarray:
    """Node values over a hand-built graph, as one matrix."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(rows=n, **overrides).resolveDataset(rng)
    return _matrix(propagate(graph, resolved, grouping, rng))


def test_a_root_is_noise() -> None:
    """A root's values are its own standard-normal noise."""
    graph = graphOf([[], [], [], []])
    values = _run(graph, 20000)
    correlations = np.corrcoef(values.T)  # (4, 4)
    off_diagonal = correlations[~np.eye(4, dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.05
    kurtosis = (values**4).mean(axis=0)  # (4,), 3 for a normal
    np.testing.assert_allclose(kurtosis, 3, atol=0.2)


def test_a_node_with_parents_correlates_with_them_at_low_noise() -> None:
    """The mechanism carries the parent's signal:
    with little noise, a linear map stays a linear map.
    """
    graph = graphOf([[], [0]], noise_scale=0.01)
    for seed in range(10):
        values = _run(graph, 2000, seed)
        correlation = np.corrcoef(values.T)[0, 1]
        assert abs(correlation) > 0.99


def test_each_node_takes_its_own_noise_scale() -> None:
    """Two children of one root differ only in their noise: the quiet one follows the root,
    the loud one barely (R² = 1 / (1 + 10²), a correlation of 0.1).
    """
    graph = replace(graphOf([[], [0], [0]]), noise_scale=[None, 0.01, 10.0])
    correlations = np.corrcoef(_run(graph, 2000).T)[0]  # (3,)
    assert abs(correlations[1]) > 0.99
    assert abs(correlations[2]) < 0.2


def test_a_constant_mechanism_output_z_scores_to_zeros() -> None:
    """A constant output must become zeros, never NaN nor rounding residue scaled to unit variance.

    With zero hidden weights an MLP outputs its readout of activation(bias), a nonzero constant;
    its linear child then sees a zero column and outputs zeros. Without noise both stay constant.
    At this width and row count the readout's BLAS sums would leave rows a last bit apart, so
    the constant must be caught before the readout.
    """
    graph = graphOf([[], [0], [1]], [None, 'mlp', 'linear'], init_scale=0.0, noise_scale=0.0)
    for seed in range(10):
        values = _run(graph, 1001, seed, hidden=64)
        assert (values[:, 1:] == 0).all()


def test_a_deep_mlp_at_a_small_init_scale_never_goes_constant() -> None:
    """Each layer shrinks the signal by its init scale while the N(0, 1) biases keep an offset
    near 1, so a deep MLP at 0.1 carries a small spread over a large offset. It is real signal:
    a guard by tolerance, 1000 eps of the largest value, zeroed three tanh layers of width 8
    in 101 of 500 seeds.
    """
    graph = graphOf([[], [0]], [None, 'mlp'], noise_scale=0.0, init_scale=0.1)
    for seed in range(20):
        values = _run(graph, 2000, seed, layers=3, hidden=8)
        assert values[:, 1].std() > 0


@pytest.mark.parametrize('grouping', GROUPINGS)
def test_values_are_finite_for_every_family(grouping: Grouping) -> None:
    """No family may overflow or divide by zero, even at the widest, deepest and steepest MLP,
    at a factor's unit or mixed from two parts.

    scamd zeroed its non-finite values and so hid them.
    """
    steep = {'init_scale': 3.0, 'layers': 3, 'hidden': 64, 'latent_nodes': 3}
    for activation in ACTIVATIONS:
        for family in ['linear', 'mlp']:
            shares = {
                'activation_shares': point('activation_shares', activation),
                'mechanism_shares': point('mechanism_shares', family),
                'group_level_share': 0.5,
            }
            for seed in range(5):
                nodes = _propagateOver(grouping, seed, 12, 1000, edge_offset=2.0, **steep, **shares)
                for values in nodes.values():
                    assert np.isfinite(values).all()


@pytest.mark.parametrize('grouping', GROUPINGS)
def test_propagate_is_deterministic_under_a_fixed_generator(grouping: Grouping) -> None:
    """A key reproduces its dataset bitwise only if every step draws from the generator alone."""
    first = _propagateOver(grouping, 7, 9, 1000, group_level_share=0.5)
    second = _propagateOver(grouping, 7, 9, 1000, group_level_share=0.5)
    assert list(first) == list(second)
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_every_node_is_z_scored_at_its_unit_and_has_its_units_length() -> None:
    """A node at a factor's unit is one value per group, z-scored there; a row-unit node is n
    values, z-scored over the rows. draw broadcasts the former to the rows.
    """
    # g1 nests in g0 and g2 is crossed with both
    grouping = _balanced(600, [20, 60, 15], nests_in=[None, 'g0', None])
    for seed in range(10):
        rng = np.random.default_rng(seed)
        resolved = Hyperprior(rows=600, latent_nodes=2, group_level_share=0.5).resolveDataset(rng)
        graph = connect(resolved, grouping, columnsOf(OBSERVED[:8]), rng)
        nodes = propagate(graph, resolved, grouping, rng)
        assert list(nodes) == graph.names[3:]
        for name, values in nodes.items():
            unit = graph.unit[graph.names.index(name)]
            assert values.shape == (_size(grouping, unit, 600),)
            assert values.dtype == np.float32
            np.testing.assert_allclose(values.mean(), 0, atol=1e-5)
            np.testing.assert_allclose(values.std(), 1, atol=1e-5)


def _nodes(
    graph: Graph, n: int, grouping: Grouping, seed: int = 0, **overrides: object
) -> dict[str, np.ndarray]:
    """Node values over a hand-built grouped graph, each at its unit."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(rows=n, **overrides).resolveDataset(rng)
    return propagate(graph, resolved, grouping, rng)


def test_a_group_level_nodes_parents_are_read_at_its_unit() -> None:
    """A node at the finer factor of a chain reads a parent at the coarser one as the coarser
    group's value of each finer group, so a linear child at low noise follows it there.
    """
    grouping = _balanced(600, [20, 60], nests_in=[None, 'g0'])
    # x2 is a root at g0's unit, x3 its linear child at g1's
    graph = graphOf([[], [], [], [2]], unit=['g0', 'g1', 'g0', 'g1'], noise_scale=0.01)
    for seed in range(10):
        nodes = _nodes(graph, 600, grouping, seed)
        parent = grouping.broadcast(nodes['x2'], 'g0', 'g1')  # (60,)
        assert nodes['x3'].shape == (60,)
        assert abs(np.corrcoef(parent, nodes['x3'])[0, 1]) > 0.99


def _betweenShare(values: np.ndarray, codes: np.ndarray) -> float:
    """The share of a node's variance over rows that lies between the groups of a factor."""
    means = np.bincount(codes, weights=values) / np.bincount(codes)  # (G,)
    between = ((means[codes] - values.mean()) ** 2).sum()
    return float(between / ((values - values.mean()) ** 2).sum())


def test_a_row_unit_node_at_group_share_1_is_constant_within_its_group_factors_groups() -> None:
    """At share 1 the row part has weight 0: a root is group-level noise, and a child of a row
    parent keeps nothing of it. Both are one value per group of the group factor.
    """
    grouping = _balanced(600, [20, 15])
    # x2 and x3 are roots at the row, x4 a linear child of x3; the group factor is g1
    graph = graphOf(
        [[], [], [], [], [3]],
        unit=['g0', 'g1', 'row', 'row', 'row'],
        group_share=1.0,
        group_factor='g1',
    )
    for seed in range(10):
        nodes = _nodes(graph, 600, grouping, seed)
        for name in ['x2', 'x4']:
            assert nodes[name].shape == (600,)
            assert _betweenShare(nodes[name], grouping.codes['g1']) == pytest.approx(1.0)
            assert len(np.unique(nodes[name])) == 15


def test_a_row_unit_node_at_group_share_0_has_a_between_group_share_at_chance_level() -> None:
    """At share 0 the group part has weight 0 and the node is its row part alone: the group
    means differ only by chance, a share near (G − 1) / (n − 1), here 0.005. The override that
    switches group confounding off rests on this.
    """
    grouping = _balanced(4000, [20])
    # x1 is a root at the row, x2 a linear child of it
    graph = graphOf([[], [], [1]], unit=['g0', 'row', 'row'], group_share=0.0)
    for seed in range(10):
        nodes = _nodes(graph, 4000, grouping, seed)
        for name in ['x1', 'x2']:
            assert _betweenShare(nodes[name], grouping.codes['g0']) < 0.02


@pytest.mark.parametrize('share', [0.2, 0.5, 0.8])
def test_the_between_group_share_over_rows_is_near_the_group_share_under_balanced_groups(
    share: float,
) -> None:
    """The group part and the row part are z-scored over the rows and the row part's group means
    differ only by chance, so sqrt(s) and sqrt(1 − s) make s the node's between-group share:
    the statistic the measurement pass records. A row-unit parent carries its own group variance
    into the row part, so its child's share lies above s.
    """
    grouping = _balanced(4000, [20])
    # x1 is a root at g0 and x2 its linear child at the row; x3 is a root at the row, x4 its child
    graph = graphOf(
        [[], [], [1], [], [3]], unit=['g0', 'g0', 'row', 'row', 'row'], group_share=share
    )
    for seed in range(10):
        nodes = _nodes(graph, 4000, grouping, seed)
        for name in ['x2', 'x3']:
            assert _betweenShare(nodes[name], grouping.codes['g0']) == pytest.approx(
                share, abs=0.03
            )
        assert _betweenShare(nodes['x4'], grouping.codes['g0']) > share + 0.03


def test_the_group_part_follows_a_group_level_parent_at_low_noise() -> None:
    """A parent at the group factor's unit goes into the group part, which carries its signal
    like item 1's mechanism: at share 1 and little noise, the child follows it over the rows.
    """
    grouping = _balanced(600, [20])
    # x1 is a root at g0, x2 its linear child at the row
    graph = graphOf([[], [], [1]], unit=['g0', 'g0', 'row'], group_share=1.0, noise_scale=0.01)
    for seed in range(10):
        nodes = _nodes(graph, 600, grouping, seed)
        parent = grouping.broadcast(nodes['x1'], 'g0', 'row')  # (600,)
        assert abs(np.corrcoef(parent, nodes['x2'])[0, 1]) > 0.99


def test_a_crossed_factors_parent_goes_into_the_row_part() -> None:
    """A parent at a unit outside the group factor's chain cannot be read at that factor; it
    goes into the row part, so the child follows it at share 0 and loses it at share 1,
    where it is constant within the group factor's groups whatever the parent does.
    """
    grouping = _balanced(600, [20, 15])

    # x2 is a root at g1, x3 its linear child at the row with the group factor g0
    def _graph(share: float) -> Graph:
        return graphOf(
            [[], [], [], [2]],
            unit=['g0', 'g1', 'g1', 'row'],
            group_share=share,
            group_factor='g0',
            noise_scale=0.01,
        )

    for seed in range(10):
        parent = grouping.broadcast(_nodes(_graph(0.0), 600, grouping, seed)['x2'], 'g1', 'row')
        row_only = _nodes(_graph(0.0), 600, grouping, seed)['x3']
        assert abs(np.corrcoef(parent, row_only)[0, 1]) > 0.99
        group_only = _nodes(_graph(1.0), 600, grouping, seed)['x3']
        assert _betweenShare(group_only, grouping.codes['g0']) == pytest.approx(1.0)
        assert abs(np.corrcoef(parent, group_only)[0, 1]) < 0.2


def test_a_parentless_group_part_is_group_level_noise() -> None:
    """A row-unit root's group part is one standard-normal noise per group of its group factor,
    independent across roots, like item 1's root at the row.
    """
    grouping = _balanced(40000, [20000])
    # x1 and x2 are roots at the row
    graph = graphOf([[], [], []], unit=['g0', 'row', 'row'], group_share=1.0)
    nodes = _nodes(graph, 40000, grouping)
    first_rows = np.unique(grouping.codes['g0'], return_index=True)[1]  # (20000,)
    groups = np.stack([nodes['x1'][first_rows], nodes['x2'][first_rows]], axis=1)  # (20000, 2)
    assert abs(np.corrcoef(groups.T)[0, 1]) < 0.05
    kurtosis = (groups**4).mean(axis=0)  # (2,), 3 for a normal
    np.testing.assert_allclose(kurtosis, 3, atol=0.2)


def test_propagate_skips_the_grouping_factors() -> None:
    """A grouping factor's values are its codes, which Grouping holds; it has no mechanism."""
    grouping = groupingOf([[0, 0, 1, 1], [0, 1, 0, 1]])
    graph = graphOf([[], [], [], [2]], unit=['g0', 'g1', 'g0', 'row'])
    rng = np.random.default_rng(0)
    resolved = Hyperprior(rows=4).resolveDataset(rng)
    assert list(propagate(graph, resolved, grouping, rng)) == ['x2', 'x3']


# item 1's propagation, recorded from the module at commit 6063e89 on two graphs connect draws
# at rows 300, latent_nodes 1 and the observed nodes a to d: the first row, the last row and
# the absolute sum, the rows as exact float32 values; seed 1 draws three MLP nodes and seed 7
# three linear ones
ITEM_1 = {
    1: (
        [
            -1.46487295627594,
            0.005196389276534319,
            -1.3860321044921875,
            0.6441646218299866,
            -1.9364700317382812,
        ],
        [
            1.3785829544067383,
            -0.573655903339386,
            1.2348989248275757,
            0.10027249902486801,
            1.0218738317489624,
        ],
        1193.7367689563544,
    ),
    7: (
        [
            1.4009796380996704,
            0.0024338674265891314,
            0.23955166339874268,
            -0.16038000583648682,
            -0.7962819933891296,
        ],
        [
            -1.7905281782150269,
            0.6712811589241028,
            0.3729669153690338,
            -0.38211488723754883,
            1.5618715286254883,
        ],
        1198.659372946393,
    ),
}


@pytest.mark.parametrize('seed', ITEM_1)
def test_the_flat_case_is_item_1s_propagation_bit_for_bit(seed: int) -> None:
    """A flat dataset is item 1's: the units are all the row, every broadcast the identity,
    and the generator is read in item 1's order, the noise before the weights.
    """
    first, last, total = ITEM_1[seed]
    values = _propagate(seed, 4, 300, latent_nodes=1)
    assert values.shape == (300, 5)
    assert np.array_equal(values[0], np.array(first, dtype=np.float32))
    assert np.array_equal(values[-1], np.array(last, dtype=np.float32))
    assert np.abs(values).sum(dtype=np.float64) == total


def test_an_empty_graph_propagates_to_no_nodes() -> None:
    """A budget of 0 with no latent nodes is a dataset of rows alone, as draw expects."""
    rng = np.random.default_rng(0)
    resolved = Hyperprior(rows=50, latent_nodes=0).resolveDataset(rng)
    graph = connect(resolved, FLAT, {}, rng)
    assert propagate(graph, resolved, FLAT, rng) == {}


def test_every_activation_is_non_linear() -> None:
    """An MLP of identities is a linear node under another name: its z-scored output is a
    uniformly random direction over its parents, like a linear readout's, so item 4's fit could
    not tell the two shares apart.
    """
    x = np.linspace(-3, 3, 13)
    for name, activate in ACTIVATIONS.items():
        slope, intercept = np.polyfit(x, activate(x), 1)
        assert not np.allclose(activate(x), slope * x + intercept), name


def test_an_mlp_node_is_non_linear_in_its_parent() -> None:
    """At low noise, a linear map keeps the parent's correlation near 1; a sin MLP must lose it."""
    graph = graphOf([[], [0]], [None, 'mlp'], activation='sin', noise_scale=0.01, init_scale=3.0)
    for seed in range(10):
        values = _run(graph, 2000, seed)
        assert abs(np.corrcoef(values.T)[0, 1]) < 0.9
