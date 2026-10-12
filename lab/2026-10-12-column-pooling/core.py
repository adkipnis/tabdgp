"""core.py is a throwaway draw of TabDGP whose ten pooled fields vary per column or node, for #16.

`tabdgp.draw` resolves every field once per dataset, so all lognormal columns of a dataset share
one sigma and all nodes one noise scale. Here the public steps (`resolve`, `plan`, `connect`,
`apply`) run in `draw`'s order, and only what reads the ten fields is redone: each column or node
gets its own value of a field, pooled across the dataset as `Pooling` names.
Under `today` the draw is `tabdgp.draw`'s, bit for bit.
The pooled values come from a generator of their own, so the plan, the graph and the node noise
are the same draws under every pooling.
It lives beside the notebook because spawned pool workers cannot import a notebook's functions.

pipeline:
---------
the key, the hyperprior and a pooling
-> the resolved hyperprior, the column plan, the graph       resolve, plan, connect
-> each column's or node's value of every pooled field        spread
-> node values with each node's noise and init scale          _propagate
-> the typed columns, each with its own parameters            apply
-> the per-table summaries and the values their readers drew  measure
"""

from __future__ import annotations

import math
from collections.abc import Callable
from statistics import NormalDist
from typing import Literal

import numpy as np

from tabdgp.column import apply, plan
from tabdgp.dataset import Column, Graph, Key
from tabdgp.graph import connect
from tabdgp.hyperprior import Distribution, Hyperprior, LogUniform, Uniform
from tabdgp.mechanism import ACTIVATIONS

# today: one value per dataset; independent: one per column or node; alike: a Gaussian copula
# around a per-dataset centre; share: the centre or an own draw; median: every unit at the
# field's median, the floor the graph and the family shares leave
Pooling = Literal['today', 'independent', 'alike', 'share', 'median']

# =====================================================================================
# the pooled fields
# =====================================================================================
# which family reads each marginal field; a column of another family draws it unread.

MARGINAL_FIELDS = {
    'student_df': ['student_t'],
    'lognormal_sigma': ['lognormal', 'zero_inflated_lognormal'],
    'zero_share': ['zero_inflated_lognormal'],
    'beta_a': ['beta'],
    'beta_b': ['beta'],
    'mixture_gap': ['normal_mixture'],
    'mixture_weight': ['normal_mixture'],
}
COLUMN_FIELDS = [*MARGINAL_FIELDS, 'level_alpha']
NODE_FIELDS = ['noise_scale', 'init_scale']
NORMAL = NormalDist()


def cdf(distribution: Uniform | LogUniform, x: np.ndarray) -> np.ndarray:
    """Where x lies in a field's distribution, in [0, 1]: closed form for the two interval laws."""
    low, high = distribution.support()
    if isinstance(distribution, LogUniform):
        return np.log(x / low) / math.log(high / low)
    if isinstance(distribution, Uniform):
        return (x - low) / (high - low)
    raise TypeError(f'no closed-form cdf for {distribution}')


def quantile(distribution: Uniform | LogUniform, u: np.ndarray) -> np.ndarray:
    low, high = distribution.support()
    if isinstance(distribution, LogUniform):
        return low * np.exp(u * math.log(high / low))
    if isinstance(distribution, Uniform):
        return low + u * (high - low)
    raise TypeError(f'no closed-form quantile for {distribution}')


# =====================================================================================
# the draw
# =====================================================================================


def drawPooled(seed: int, index: int, pooling: Pooling, hyperprior: Hyperprior) -> dict:
    """Dataset `index` of corpus `seed` with its ten pooled fields drawn per column or node."""
    rng = Key(seed=seed, index=index).generator()
    resolved = hyperprior.resolve(rng)
    metadata = plan(resolved, rng)
    graph = connect(resolved, list(metadata), rng)
    spread_rng = np.random.default_rng([seed, index, 1])
    alikeness = float(spread_rng.random())  # one per dataset, shared by the ten fields
    parameters = {}  # field -> (units,), per column in plan order or per node in graph order
    for name in COLUMN_FIELDS + NODE_FIELDS:
        size = len(metadata) if name in COLUMN_FIELDS else len(graph.names)
        prior = getattr(hyperprior, name)
        centre = getattr(resolved, name)
        parameters[name] = spread(prior, centre, alikeness, size, pooling, spread_rng)
    nodes = _propagate(graph, resolved, parameters['noise_scale'], parameters['init_scale'], rng)
    columns = {}
    for k, (name, column) in enumerate(metadata.items()):
        # the families and the codes read their parameters from a resolved hyperprior, so each
        # column gets a copy holding its own
        own = {field: float(parameters[field][k]) for field in COLUMN_FIELDS}
        columns[name] = apply(nodes[name], column, resolved.model_copy(update=own), rng)
    return {
        'columns': columns,
        'metadata': metadata,
        'graph': graph,
        'parameters': parameters,
        'alikeness': alikeness,
    }


def measure(seed: int, index: int, pooling: Pooling, hyperprior: Hyperprior) -> dict:
    """The table's summaries and the values its readers drew, small enough for a pool to return."""
    dataset = drawPooled(seed, index, pooling, hyperprior)
    read = {}
    for name, values in dataset['parameters'].items():
        read[name] = values[readers(name, dataset['metadata'], dataset['graph'])]
    summaries = summarise(dataset['columns'], dataset['metadata'])
    return {'summaries': summaries, 'parameters': read, 'alikeness': dataset['alikeness']}


def readers(field: str, metadata: dict[str, Column], graph: Graph) -> np.ndarray:
    """Which columns or nodes read a field, a bool per unit: the rest drew it unread."""
    if field in MARGINAL_FIELDS:
        families = MARGINAL_FIELDS[field]
        return np.array([column.marginal in families for column in metadata.values()])
    if field == 'level_alpha':
        return np.array([column.type != 'continuous' for column in metadata.values()])
    child = graph.parents.any(axis=1)  # (n,), a root reads neither scale
    if field == 'noise_scale':
        return child
    return child & (np.array(graph.mechanism) == 'mlp')


def spread(
    prior: object,
    centre: float,
    alikeness: float,
    size: int,
    pooling: Pooling,
    rng: np.random.Generator,
) -> np.ndarray:
    """Each unit's value of one field, its law the prior's under every pooling but median."""
    if not isinstance(prior, Distribution) or pooling == 'today':  # an overridden point stays
        return np.full(size, centre)
    if pooling == 'median':
        return np.full(size, quantile(prior, np.array(0.5)))
    if pooling == 'independent':
        return quantile(prior, rng.random(size))
    if pooling == 'share':
        own = quantile(prior, rng.random(size))  # (size,)
        return np.where(rng.random(size) < alikeness, centre, own)
    # alike: a Gaussian copula, z = ρ z_centre + √(1 − ρ²) ε, so ρ = 1 is today and ρ = 0 is
    # independent, and every unit's law is the prior's whatever ρ
    z_centre = NORMAL.inv_cdf(float(cdf(prior, np.array(centre))))
    noise = math.sqrt(1 - alikeness**2) * rng.standard_normal(size)  # (size,)
    z = alikeness * z_centre + noise  # (size,)
    return quantile(prior, _phi(z))


def _phi(z: np.ndarray) -> np.ndarray:
    return np.array([NORMAL.cdf(float(value)) for value in z])


# =====================================================================================
# propagation
# =====================================================================================
# mechanism.propagate and its _mlp with the node's own noise and init scale in place of the
# dataset's; the generator is read in the same order, so `today` reproduces it bit for bit.


def _propagate(
    graph: Graph,
    resolved: Hyperprior,
    noise_scale: np.ndarray,
    init_scale: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n = len(graph.names)
    n_obs = resolved.n_obs
    activations = resolved.activation_shares.chooseMany(rng, n)  # (n,)
    values = np.empty((n_obs, n), dtype=np.float32)  # (N, n)
    for j in range(n):
        noise = rng.standard_normal(n_obs, dtype=np.float32)  # (N,)
        if not graph.parents[j].any():
            values[:, j] = _zscore(noise)
            continue
        layers = resolved.layers if graph.mechanism[j] == 'mlp' else 0
        inputs = values[:, graph.parents[j]]  # (N, p)
        activation = ACTIVATIONS[activations[j]]
        output = _mlp(inputs, layers, activation, float(init_scale[j]), resolved.hidden, rng)
        # a python float keeps the sum float32, as the package's resolved scalar does
        values[:, j] = _zscore(_zscore(output) + float(noise_scale[j]) * noise)
    return {name: values[:, j].copy() for j, name in enumerate(graph.names)}  # (N,) each


def _mlp(
    inputs: np.ndarray,
    layers: int,
    activation: Callable[[np.ndarray], np.ndarray],
    init_scale: float,
    hidden_width: int,
    rng: np.random.Generator,
) -> np.ndarray:
    hidden = inputs  # (N, p)
    for _ in range(layers):
        fan_in = hidden.shape[1]
        scale = init_scale / fan_in**0.5
        weights = scale * rng.standard_normal((fan_in, hidden_width), dtype=np.float32)
        bias = rng.standard_normal(hidden_width, dtype=np.float32)  # (hidden,)
        hidden = activation(hidden @ weights + bias)  # (N, hidden)
    readout = rng.standard_normal(hidden.shape[1], dtype=np.float32)  # (hidden,) or (p,)
    return hidden @ readout


def _zscore(x: np.ndarray) -> np.ndarray:
    centred = x - x.mean()
    std = centred.std()
    if std <= 1000 * np.finfo(x.dtype).eps * np.abs(x).max():
        return np.zeros_like(x)
    return centred / std


# =====================================================================================
# the table summaries
# =====================================================================================
# what a classifier two-sample test on per-table summary vectors would see: per summary its
# mean and sd over the table's pairs or columns.

SUMMARIES = [
    'spearman mean',
    'spearman sd',
    'skew mean',
    'skew sd',
    'kurtosis mean',
    'kurtosis sd',
    'largest share mean',
    'largest share sd',
]


def summarise(columns: dict[str, np.ndarray], metadata: dict[str, Column]) -> dict[str, float]:
    """Each summary of SUMMARIES, NaN where the table has too few columns for it."""
    table = np.stack(list(columns.values()), axis=1).astype(np.float64)  # (N, p)
    continuous = np.array([column.type == 'continuous' for column in metadata.values()])  # (p,)
    skew, kurtosis = _moments(table[:, continuous])  # (c,) each
    per_unit = {
        'spearman': _spearman(table),
        'skew': skew,
        'kurtosis': kurtosis,
        'largest share': _largestShare(table[:, ~continuous]),
    }
    summaries = {}
    for name, values in per_unit.items():
        summaries[f'{name} mean'], summaries[f'{name} sd'] = _meanSd(values)
    return summaries


def _spearman(table: np.ndarray) -> np.ndarray:
    # the absolute pearson correlation of average ranks, over all pairs of columns; average
    # ranks give a coded column's tied rows one rank per level
    p = table.shape[1]
    ranks = np.empty_like(table)  # (N, p)
    for j in range(p):
        _, inverse, counts = np.unique(table[:, j], return_inverse=True, return_counts=True)
        ends = np.cumsum(counts)  # (levels,), the last rank of each value, 1-based
        ranks[:, j] = (ends - (counts - 1) / 2)[inverse]
    correlation = np.atleast_2d(np.corrcoef(ranks, rowvar=False))  # (p, p)
    upper = np.triu_indices(p, k=1)
    return np.abs(correlation[upper])  # (p (p − 1) / 2,)


def _moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # the sample skewness g1 and excess kurtosis g2 of each column
    centred = x - x.mean(axis=0)  # (N, c)
    variance = (centred**2).mean(axis=0)  # (c,)
    skew = (centred**3).mean(axis=0) / variance**1.5
    kurtosis = (centred**4).mean(axis=0) / variance**2 - 3
    return skew, kurtosis


def _largestShare(codes: np.ndarray) -> np.ndarray:
    n_obs, k = codes.shape
    largest = np.empty(k)  # (k,)
    for j in range(k):
        largest[j] = np.bincount(codes[:, j].astype(np.int64)).max() / n_obs
    return largest


def _meanSd(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean()) if len(values) >= 1 else math.nan
    sd = float(values.std(ddof=1)) if len(values) >= 2 else math.nan
    return mean, sd
