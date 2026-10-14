"""The mechanism module's seam: propagate, every node's values from its parents and noise."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from tabdgp.graph import Graph, connect
from tabdgp.hyperprior import Hyperprior
from tabdgp.mechanism import ACTIVATIONS, propagate
from tests.fixtures.build import graphOf, groupingOf, point

OBSERVED = list('abcdefghijkl')
FLAT = groupingOf([])


def _matrix(nodes: dict[str, np.ndarray]) -> np.ndarray:
    """(n, d): the nodes side by side in the graph's order."""
    return np.stack(list(nodes.values()), axis=1)


def _propagate(seed: int, observed_nodes: int, n: int, **overrides: object) -> np.ndarray:
    """Node values over a graph drawn by connect, as one matrix."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(rows=n, **overrides).resolveDataset(rng)
    graph = connect(resolved, FLAT, OBSERVED[:observed_nodes], rng)
    return _matrix(propagate(graph, resolved, rng))


def test_propagate_names_every_node_with_its_own_contiguous_array() -> None:
    """draw joins nodes to columns by name and hands the latent ones out as they are.

    A view into every node's values would keep them all alive and read strided.
    """
    rng = np.random.default_rng(0)
    resolved = Hyperprior(rows=100, latent_nodes=2).resolveDataset(rng)
    graph = connect(resolved, FLAT, OBSERVED[:4], rng)
    nodes = propagate(graph, resolved, rng)
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


def _run(graph: Graph, n: int, seed: int = 0, **overrides: object) -> np.ndarray:
    """Node values over a hand-built graph, as one matrix."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(rows=n, **overrides).resolveDataset(rng)
    return _matrix(propagate(graph, resolved, rng))


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


def test_values_are_finite_for_every_family() -> None:
    """No family may overflow or divide by zero, even at the widest, deepest and steepest MLP.

    scamd zeroed its non-finite values and so hid them.
    """
    steep = {'init_scale': 3.0, 'layers': 3, 'hidden': 64, 'latent_nodes': 3}
    for activation in ACTIVATIONS:
        for family in ['linear', 'mlp']:
            shares = {
                'activation_shares': point('activation_shares', activation),
                'mechanism_shares': point('mechanism_shares', family),
            }
            for seed in range(5):
                values = _propagate(seed, 12, 1000, edge_offset=2.0, **steep, **shares)
                assert np.isfinite(values).all()


def test_propagate_is_deterministic_under_a_fixed_generator() -> None:
    """A key reproduces its dataset bitwise only if every step draws from the generator alone."""
    first = _propagate(7, 9, 300)
    second = _propagate(7, 9, 300)
    assert np.array_equal(first, second)


def test_an_empty_graph_propagates_to_no_nodes() -> None:
    """A budget of 0 with no latent nodes is a dataset of rows alone, as draw expects."""
    rng = np.random.default_rng(0)
    resolved = Hyperprior(rows=50, latent_nodes=0).resolveDataset(rng)
    graph = connect(resolved, FLAT, [], rng)
    assert propagate(graph, resolved, rng) == {}


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
