"""The mechanism module's seam: propagate, every node's values from its parents and noise."""

from __future__ import annotations

import numpy as np
import pytest

from tabdgp.dataset import Graph
from tabdgp.graph import connect
from tabdgp.hyperprior import Hyperprior
from tabdgp.mechanism import propagate
from tests.fixtures.build import graphOf, point

OBSERVED = list('abcdefghijkl')


def _matrix(nodes: dict[str, np.ndarray]) -> np.ndarray:
    """(N, n): the nodes side by side in the graph's order."""
    return np.stack(list(nodes.values()), axis=1)


def _propagate(seed: int, n_observed: int, n_obs: int, **overrides: object) -> np.ndarray:
    """Node values over a graph drawn by connect, as one matrix."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(n_obs=n_obs, **overrides).resolve(rng)
    graph = connect(resolved, OBSERVED[:n_observed], rng)
    return _matrix(propagate(graph, resolved, rng))


def test_propagate_names_every_node_with_its_own_contiguous_array() -> None:
    """draw joins nodes to columns by name and hands the latent ones out as they are.

    A view into every node's values would keep them all alive and read strided.
    """
    rng = np.random.default_rng(0)
    resolved = Hyperprior(n_obs=100, latent_nodes=2).resolve(rng)
    graph = connect(resolved, OBSERVED[:4], rng)
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


def _run(graph: Graph, n_obs: int, seed: int = 0, **overrides: object) -> np.ndarray:
    """Node values over a hand-built graph, as one matrix."""
    rng = np.random.default_rng(seed)
    resolved = Hyperprior(n_obs=n_obs, **overrides).resolve(rng)
    return _matrix(propagate(graph, resolved, rng))


@pytest.mark.parametrize('family', ['linear', 'mlp'])
def test_a_root_is_noise(family: str) -> None:
    """A root's values are its own standard-normal noise, whatever family it drew."""
    graph = graphOf([[], [], [], []], [family] * 4)
    values = _run(graph, 20000)
    correlations = np.corrcoef(values.T)  # (4, 4)
    off_diagonal = correlations[~np.eye(4, dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.05
    kurtosis = (values**4).mean(axis=0)  # (4,), 3 for a normal
    np.testing.assert_allclose(kurtosis, 3, atol=0.2)


IDENTITY = {'tanh': 0.0, 'relu': 0.0, 'sin': 0.0, 'identity': 1.0}


def test_a_node_with_parents_correlates_with_them_at_low_noise() -> None:
    """The mechanism carries the parent's signal: with little noise, a linear map stays a linear map.

    An MLP with the identity activation is linear too, so both families must keep it.
    """
    for family in ['linear', 'mlp']:
        graph = graphOf([[], [0]], ['linear', family])
        for seed in range(10):
            values = _run(graph, 2000, seed, noise_scale=0.01, activation_shares=IDENTITY)
            correlation = np.corrcoef(values.T)[0, 1]
            assert abs(correlation) > 0.99


def test_a_constant_mechanism_output_z_scores_to_zeros() -> None:
    """A constant output must become zeros, never NaN nor rounding residue scaled to unit variance.

    With zero hidden weights an MLP outputs its readout of activation(bias), a nonzero constant;
    its linear child then sees a zero column and outputs zeros. Without noise both stay constant.
    At this width and row count the readout's BLAS sums leave rows a last bit apart, the residue
    a guard of std > 0 would scale up.
    """
    graph = graphOf([[], [0], [1]], ['linear', 'mlp', 'linear'])
    for seed in range(10):
        values = _run(graph, 1001, seed, init_scale=0.0, noise_scale=0.0, hidden=64)
        assert (values[:, 1:] == 0).all()


ACTIVATIONS = ['tanh', 'relu', 'sin', 'identity']


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


def test_an_empty_graph_gives_rows_without_columns() -> None:
    """A budget of 0 with no latent nodes is a dataset of rows alone, as draw expects."""
    rng = np.random.default_rng(0)
    resolved = Hyperprior(n_obs=50, latent_nodes=0).resolve(rng)
    graph = connect(resolved, [], rng)
    assert propagate(graph, resolved, rng) == {}
