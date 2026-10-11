"""core.py is a throwaway core draw of TabDGP, timed by the throughput lab note.

It has the shape of the eventual `draw`: one dataset from its key, vectorised over its own rows,
with the spec's provisional defaults. It lives beside the notebook because spawned pool workers
cannot import a function defined in a notebook.

pipeline:
---------
the key                                                      default_rng([seed, index])
-> the per-dataset hyperparameters                           _hyperparameters
-> the Cauchy DAG over latent and observed nodes             _graph
-> node values by MLP propagation with relative noise        _propagate
-> each observed node rank-matched or thresholded            _columns
"""

from __future__ import annotations

import numpy as np

N_OBSERVED = 20


# =====================================================================================
# mechanisms
# =====================================================================================
# the activations an MLP node draws from, one per node.


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _identity(x: np.ndarray) -> np.ndarray:
    return x


ACTIVATIONS = [np.tanh, _relu, np.sin, _identity]


def _zscore(x: np.ndarray) -> np.ndarray:
    # a constant output (a dead relu, or any layer after one) z-scores to zeros, never to NaN
    # nor to the mean's rounding residue scaled up to unit variance
    centred = x - x.mean()
    std = centred.std()
    if std <= 1000 * np.finfo(x.dtype).eps * np.abs(x).max():
        return np.zeros_like(x)
    return centred / std


def _logUniform(rng: np.random.Generator, low: float, high: float) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


# =====================================================================================
# the core draw
# =====================================================================================


def drawCore(
    seed: int, index: int, n_obs: int, mlp_share: float = 1.0, dtype: type = np.float64
) -> dict[str, np.ndarray]:
    # mlp_share 1 makes every node an MLP, the slow case; the spec's default mix is about 0.5
    rng = np.random.default_rng([seed, index])
    params = _hyperparameters(rng) | {'mlp_share': mlp_share, 'dtype': dtype}
    parents = _graph(params, rng)
    values = _propagate(parents, params, n_obs, rng)
    columns = _columns(values[params['latent'] :], params, rng)
    return {'columns': columns, 'latent': values[: params['latent']], 'parents': parents}


def _hyperparameters(rng: np.random.Generator) -> dict:
    return {
        'latent': int(rng.integers(0, 4)),
        'edge_offset': rng.uniform(-4, 2),
        'hub_scale': _logUniform(rng, 0.1, 2),
        'hidden': int(_logUniform(rng, 4, 65)),
        'layers': int(rng.integers(1, 4)),
        'activation_shares': rng.dirichlet(np.ones(len(ACTIVATIONS))),
        'init_scale': _logUniform(rng, 0.1, 3),
        'noise_scale': _logUniform(rng, 0.01, 1),
        'lognormal_sigma': _logUniform(rng, 0.25, 1.5),
    }


def _graph(params: dict, rng: np.random.Generator) -> np.ndarray:
    # TabICL's Cauchy DAG: edge i -> j (i < j) with probability sigmoid(offset + out_i + in_j),
    # heavy-tailed per-node terms make hubs; latent nodes come first in the order
    n = params['latent'] + N_OBSERVED
    out_term = params['hub_scale'] * rng.standard_cauchy(n)  # (n,)
    in_term = params['hub_scale'] * rng.standard_cauchy(n)  # (n,)
    logits = params['edge_offset'] + in_term[:, None] + out_term[None, :]  # (n, n), [j, i]
    edges = rng.uniform(size=(n, n)) < 0.5 * (
        1 + np.tanh(logits / 2)
    )  # a sigmoid that cannot overflow
    return np.tril(edges, k=-1)  # parents[j, i]: i is a parent of j


def _propagate(
    parents: np.ndarray, params: dict, n_obs: int, rng: np.random.Generator
) -> np.ndarray:
    n = parents.shape[0]
    activations = rng.choice(len(ACTIVATIONS), size=n, p=params['activation_shares'])
    layers = np.where(rng.uniform(size=n) < params['mlp_share'], params['layers'], 0)
    values = np.empty(
        (n, n_obs), params['dtype']
    )  # (n, N), node-major so each node's values are contiguous
    for j in range(n):
        inputs = values[parents[j]].T  # (N, n_parents)
        if inputs.shape[1] == 0:
            values[j] = _zscore(rng.standard_normal(n_obs, params['dtype']))  # a root is its noise
            continue
        output = _mlp(inputs, layers[j], params, ACTIVATIONS[activations[j]], rng)
        # relative noise: z-score, add noise at a scale relative to 1, z-score again
        noisy = _zscore(output) + params['noise_scale'] * rng.standard_normal(
            n_obs, params['dtype']
        )
        values[j] = _zscore(noisy)
    return values


def _mlp(
    inputs: np.ndarray, layers: int, params: dict, activation, rng: np.random.Generator
) -> np.ndarray:
    # fan-in scaled weights and a bias per unit keep hidden layers bounded on z-scored inputs;
    # zero layers is the linear mechanism
    hidden = inputs
    for _ in range(layers):
        fan_in = hidden.shape[1]
        scale = params['init_scale'] / fan_in**0.5
        weights = scale * rng.standard_normal((fan_in, params['hidden']), params['dtype'])
        bias = rng.standard_normal(params['hidden'], params['dtype'])
        hidden = activation(hidden @ weights + bias)  # (N, hidden)
    width = hidden.shape[1]
    readout = rng.standard_normal(width, params['dtype']) / width**0.5
    return hidden @ readout  # (N,)


def _columns(observed: np.ndarray, params: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    # half the columns rank-matched to a lognormal, half thresholded to binary
    n = observed.shape[0]
    is_binary = rng.uniform(size=n) < 0.5
    columns = {}
    for j in range(n):
        if is_binary[j]:
            columns[f'x{j}'] = _binary(observed[j], rng)
        else:
            columns[f'x{j}'] = _rankMatch(observed[j], params['lognormal_sigma'], rng)
    return columns


def _rankMatch(values: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    # Iman and Conover (1982): the sorted marginal sample placed by the node's ranks
    sample = np.sort(rng.lognormal(0, sigma, values.shape[0]).astype(values.dtype))
    matched = np.empty_like(sample)
    matched[np.argsort(values)] = sample
    return matched


def _binary(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # threshold at the quantile of a drawn level share
    share = rng.uniform(0.05, 0.95)
    return (values > np.quantile(values, share)).astype(np.int64)
