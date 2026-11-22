"""distance.py compares four distances for the precision's reach, on real tables and a toy (#97).

The precision ranks a real table's reach among the reaches of its draws,
where a table's reach is its distance to its nearest real tables of other sources.
A distance codes each axis, then sums the gaps over the axes. The four candidates:

- A: ranks pooled over the real tables and their draws, the mean absolute difference.
- B: ranks among the real tables of other sources, the mean absolute difference
  (`fidelity.precision`'s, which the run checks to the bit).
- C: B with each block of axes weighted alike.
- D: the median and interquartile range of the real tables of other sources, Euclidean,
  with each block of axes weighted alike (after Structural Coverage, arXiv 2609.06912).

Each candidate is read at 1 to 50 neighbors, with the mean, the median or the largest of their
distances. The checks were fixed before the run: the gap between the defaults and
`dependence='none'` in its standard errors over resampled sources, the same gap without the
relation axes, the top three axes' share of the draws' excess reach, how far a real table's pooled
rank moves between two arms, the neighbors two candidates share, and the toy's doses.
distance.ipynb reads the report.

pipeline:
---------
each half of the real tables inside the default sizes, and their draws per arm (flat.py)
-> per candidate, each table's and each draw's distances to its nearest
   real tables of other sources, sorted                               compareReal, _measureNearest
-> the precision per number of neighbors and reach                    _rankReaches
-> the checks                                                         _scoreCandidates, _measureGap
the toy's real tables and draws, per case and seed                    compareToy, _drawToy
-> the same precision                                                 _scoreToy
-> distance-report.csv

Run from the repository root, in about half an hour on a laptop:

    uv run python experiments/fidelity/distance.py
"""

from __future__ import annotations

import math
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from tabdgp import fidelity, summary

# =====================================================================================
# the files, the arms and the candidates
# =====================================================================================

HERE = Path(__file__).parent
REAL = HERE / 'flat-real.csv'  # flat.py's measured tables
# a row per statistic, data (a half or the toy), arm, candidate, neighbors, reach and seed
REPORT = HERE / 'distance-report.csv'
COLUMNS = ['statistic', 'data', 'arm', 'candidate', 'neighbors', 'reach', 'seed', 'value']
SEED = 2026
DRAWS = 39  # per real table, as in flat.py
AXES = list(summary.AXES)
ARMS = ['default', 'independent', 'tabicl']
PARTS = ['certifying', 'fitting']
NEIGHBORS = [1, 3, 5, 10, 20, 50]
REACHES = [
    'mean',
    'median',
    'largest',
]  # a table's reach as this summary of its neighbors' distances
CHOSEN = (5, 'mean')  # the reach of `fidelity.precision`
RESAMPLES = 2000  # of the sources, for a gap's standard error

Candidate = Literal['A', 'B', 'C', 'D']
# pooled ranks, ranks among the real tables of other sources, or their robust scale
Coding = Literal['pooled', 'among', 'robust']


@dataclass(frozen=True)
class Distance:
    """How a candidate measures the distance between two tables."""

    coding: Coding
    order: int  # 1 for the mean absolute difference, 2 for the Euclidean distance
    blocks_alike: bool  # each block of axes weighs alike, else each axis does


CANDIDATES: dict[Candidate, Distance] = {
    'A': Distance(coding='pooled', order=1, blocks_alike=False),
    'B': Distance(coding='among', order=1, blocks_alike=False),
    'C': Distance(coding='among', order=1, blocks_alike=True),
    'D': Distance(coding='robust', order=2, blocks_alike=True),
}


def _nameBlock(axis: str) -> str:
    """An axis's block: the sizes, a type's shape (categorical and ordered together),
    the relations or y.
    """
    if axis.startswith('y_'):
        return 'y'
    if '_relation_' in axis or axis == 'condition_index':
        return 'relations'
    if axis in ['rows', 'design_columns', 'columns'] or axis.startswith('levels_'):
        return 'sizes'
    # a type's share of the columns
    if axis.count('_') == 1 and axis.endswith('_share'):
        return 'sizes'
    type = axis.split('_')[0]
    if type in ['categorical', 'ordered']:
        return 'coded shape'
    return f'{type} shape'


# =====================================================================================
# the real tables
# =====================================================================================


def compareReal() -> list[dict]:
    """Every candidate on each half of the real tables inside the default sizes, under each arm.
    The neighbors are the real tables of the same half.
    """
    real = pd.read_csv(REAL)
    draws = {}
    for arm in ARMS:
        draws[arm] = pd.read_csv(HERE / f'flat-draws-{arm}-inside.csv.gz')
    blocks = []
    for axis in AXES:
        blocks.append(_nameBlock(axis))

    records = []
    for part in PARTS:
        tables = real[(real['part'] == part) & (real['stratum'] == 'inside')]
        real_vectors = tables[AXES].to_numpy()  # (t, p)
        sources = tables['source'].to_numpy()  # (t,)
        draw_vectors = {}
        for arm in ARMS:
            draw_vectors[arm] = _pairDraws(draws[arm], tables)
        records += _scoreCandidates(part, real_vectors, draw_vectors, sources, blocks)
    return records


def _pairDraws(draws: pd.DataFrame, tables: pd.DataFrame) -> np.ndarray:
    """The draws of the tables, DRAWS per table in the tables' order."""
    own = draws[draws['position'].isin(tables['position'])]
    expected = np.repeat(tables['position'].to_numpy(), DRAWS)
    if not np.array_equal(own['position'].to_numpy(), expected):
        raise ValueError('the draws are not those of the real tables: run flat.py again')
    return own[AXES].to_numpy().reshape(len(tables), DRAWS, len(AXES))  # (t, s, p)


def _scoreCandidates(
    part: str, real: np.ndarray, draws: dict[str, np.ndarray], sources: np.ndarray, blocks: list
) -> list[dict]:
    """The precision of every candidate and arm on one half, and the checks."""
    rng = np.random.default_rng([SEED, PARTS.index(part)])
    resamples = _resampleSources(sources, rng)
    records = []
    neighbors = {}
    for candidate in CANDIDATES:
        precision = {}
        for arm in ARMS:
            measured = _measureNearest(candidate, real, draws[arm], sources, blocks)
            precision[arm] = _rankReaches(measured['nearest'])
            cell = {'data': part, 'arm': arm, 'candidate': candidate}
            records += _recordReaches(cell, precision[arm])
            if arm == 'default':
                neighbors[candidate] = measured['neighbors']
                excess = np.sort(measured['excess'])[::-1]  # (p,), the largest first
                top_three = {
                    'statistic': 'top_three_share',
                    **cell,
                    'value': excess[:3].sum() / excess.sum(),
                }
                records.append(top_three)

        # B is fidelity's precision
        if candidate == 'B':
            for arm in ARMS:
                measured = fidelity.precision(real, draws[arm], sources).per_table
                if not np.array_equal(precision[arm][CHOSEN], measured):
                    raise ValueError(f'B differs from fidelity.precision under {arm}')

        # the gap between the defaults and independent covariates, at every reach
        cell = {
            'statistic': 'gap',
            'data': part,
            'arm': 'default - independent',
            'candidate': candidate,
        }
        for (count, reach), default in precision['default'].items():
            gap = _measureGap(default, precision['independent'][count, reach], resamples)
            records.append({**cell, 'neighbors': count, 'reach': reach, 'value': gap})

        # the same gap without the relation axes, at the chosen reach
        bare = {}
        for arm in ['default', 'independent']:
            measured = _measureNearest(candidate, real, draws[arm], sources, blocks, 'relations')
            bare[arm] = _rankReaches(measured['nearest'])[CHOSEN]
        gap = _measureGap(bare['default'], bare['independent'], resamples)
        count, reach = CHOSEN
        cell = {**cell, 'statistic': 'gap_without_relations', 'neighbors': count, 'reach': reach}
        records.append({**cell, 'value': gap})

    # the real tables' nearest real tables under two candidates, at the defaults
    for first, second in combinations(CANDIDATES, 2):
        shared = _countShared(neighbors[first], neighbors[second])
        cell = {'data': part, 'arm': 'default', 'candidate': f'{first}, {second}'}
        records.append({'statistic': 'shared_neighbors', **cell, 'value': shared})

    # a real table's pooled rank between the defaults' pool and TabICL's
    default_codes, _ = _rankPooled(real, draws['default'])
    tabicl_codes, _ = _rankPooled(real, draws['tabicl'])
    moved = float(np.nanmean(np.abs(default_codes - tabicl_codes)))
    cell = {'data': part, 'arm': 'default, tabicl', 'candidate': 'A'}
    records.append({'statistic': 'pooled_rank_moved', **cell, 'value': moved})
    return records


# -------------------------------------------------------------------------------------
# a candidate's distances -> the precision
# -------------------------------------------------------------------------------------


def _measureNearest(
    candidate: Candidate,
    real: np.ndarray,
    draws: np.ndarray,
    sources: np.ndarray,
    blocks: list[str],
    without: str | None = None,
) -> dict[str, np.ndarray]:
    """Per real table, its and its draws' sorted distances to their nearest real tables of other
    sources, its own 5 nearest by their index, and each axis's mean excess of the draws' parts
    of their 5 nearest distances over the real table's. A block `without` weighs nothing.
    """
    distance = CANDIDATES[candidate]
    weights = _weighAxes(blocks, distance.blocks_alike, without)  # (p,)
    if distance.coding == 'pooled':
        real, draws = _rankPooled(real, draws)
    t, s, p = draws.shape
    nearest = np.empty((t, 1 + s, max(NEIGHBORS)))
    neighbors = np.empty((t, CHOSEN[0]), dtype=int)
    excess = np.zeros(p)
    for table in range(t):
        elsewhere = np.flatnonzero(sources != sources[table])  # (o,)
        pair = np.concatenate([real[[table]], draws[table]])  # (1 + s, p)
        parts = _measureParts(distance, pair, real[elsewhere], weights)  # (1 + s, o, p)
        apart = (parts.sum(axis=2) / weights.sum()) ** (1 / distance.order)  # (1 + s, o)
        order = np.argsort(apart, axis=1)
        nearest[table] = np.take_along_axis(apart, order[:, : max(NEIGHBORS)], axis=1)
        neighbors[table] = elsewhere[order[0, : CHOSEN[0]]]
        closest = order[:, : CHOSEN[0], None]
        parts_closest = np.take_along_axis(parts, closest, axis=1).mean(axis=1)  # (1 + s, p)
        excess += parts_closest[1:].mean(axis=0) - parts_closest[0]
    return {'nearest': nearest, 'neighbors': neighbors, 'excess': excess / t}


def _weighAxes(blocks: list[str], alike: bool, without: str | None) -> np.ndarray:
    """Each axis's weight: 1, or 1 over its block's size if blocks weigh alike."""
    weights = np.ones(len(blocks))
    if alike:
        for block in set(blocks):
            members = np.array([name == block for name in blocks])
            weights[members] = 1 / members.sum()
    if without is not None:
        weights[np.array([name == without for name in blocks])] = 0.0
    return weights  # (p,)


def _measureParts(
    distance: Distance, pair: np.ndarray, others: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Each axis's weighted part of the distance between each table of the pair and each other."""
    match distance.coding:
        case 'pooled':
            # the values are pooled ranks already
            pair_codes, other_codes, costs = pair, others, np.ones(len(weights))
        case 'among':
            pair_codes = _rankAmong(pair, others)
            other_codes = _rankAmong(others, others)
            costs = np.ones(len(weights))
        case 'robust':
            pair_codes, other_codes, costs = _scaleRobust(pair, others)
    gaps = np.abs(pair_codes[:, None] - other_codes[None])  # (1 + s, o, p)
    # an axis only one of two tables shows costs the axis's cost, and one neither shows 0
    shown = ~np.isnan(pair)[:, None]
    other_shown = ~np.isnan(others)[None]
    gaps = np.where(shown != other_shown, costs, gaps)
    gaps = np.where(~shown & ~other_shown, 0.0, gaps)
    return weights * gaps**distance.order


def _rankPooled(real: np.ndarray, draws: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The real tables (t, p) and their draws (t, s, p), ranked among all of them per axis."""
    t, s, p = draws.shape
    pooled = np.concatenate([real, draws.reshape(t * s, p)])
    codes = _rankAmong(pooled, pooled)
    return codes[:t], codes[t:].reshape(t, s, p)


def _rankAmong(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Each value's share of the reference's values below it, per axis (a tie counts half).
    It is NaN where the value is missing or the reference shows no value on its axis.
    """
    ranks = np.full(values.shape, np.nan)  # (n, p)
    for axis in range(values.shape[1]):
        ordered = np.sort(reference[~np.isnan(reference[:, axis]), axis])
        if len(ordered) == 0:
            continue
        shown = ~np.isnan(values[:, axis])
        below = np.searchsorted(ordered, values[shown, axis], side='left')
        at_or_below = np.searchsorted(ordered, values[shown, axis], side='right')
        ranks[shown, axis] = (below + at_or_below) / 2 / len(ordered)
    return ranks  # (n, p)


def _scaleRobust(pair: np.ndarray, others: np.ndarray) -> tuple[np.ndarray, ...]:
    """The pair and the others as (x - median) / IQR of the others per axis,
    and the others' range on that scale, the cost of a mismatched axis.
    """
    # an infinite value counts as the largest finite one,
    # an IQR of 0 falls back to the sd, and an sd of 0 to 1
    with warnings.catch_warnings():
        # an axis no other table shows leaves every statistic of it NaN
        warnings.simplefilter('ignore', RuntimeWarning)
        largest = np.nanmax(np.where(np.isfinite(others), others, np.nan), axis=0)  # (p,)
        capped_others = np.where(np.isinf(others), largest, others)
        capped_pair = np.where(np.isinf(pair), largest, pair)
        median = np.nanmedian(capped_others, axis=0)
        upper, lower = np.nanpercentile(capped_others, [75, 25], axis=0)
        spread = np.nanstd(capped_others, axis=0)
        scale = np.where(upper - lower > 0, upper - lower, spread)
        scale = np.where(scale > 0, scale, 1.0)
        median = np.where(np.isnan(median), 0.0, median)
        other_codes = (capped_others - median) / scale
        extent = np.nanmax(other_codes, axis=0) - np.nanmin(other_codes, axis=0)
    costs = np.where(np.isnan(extent), 0.0, extent)
    return (capped_pair - median) / scale, other_codes, costs


def _rankReaches(nearest: np.ndarray) -> dict[tuple[int, str], np.ndarray]:
    """The precision per table, for each number of neighbors and reach."""
    precision = {}
    for count in NEIGHBORS:
        closest = nearest[..., :count]  # (t, 1 + s, count)
        reaches = {
            'mean': closest.mean(axis=-1),
            'median': np.median(closest, axis=-1),
            'largest': closest[..., -1],
        }
        for reach, values in reaches.items():
            # a draw within reach lies below its real table,
            # so the real table's rank among its draws is the precision
            ranks = fidelity.recall(values[:, :1], values[:, 1:, None]).ranks  # (t, 1)
            precision[count, reach] = ranks[:, 0]
    return precision


def _recordReaches(cell: dict, precision: dict[tuple[int, str], np.ndarray]) -> list[dict]:
    records = []
    for (count, reach), per_table in precision.items():
        value = float(per_table.mean())
        records.append(
            {'statistic': 'precision', **cell, 'neighbors': count, 'reach': reach, 'value': value}
        )
    return records


# -------------------------------------------------------------------------------------
# the checks
# -------------------------------------------------------------------------------------


def _resampleSources(sources: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """RESAMPLES sets of the tables, each the tables of as many sources drawn with replacement."""
    names = np.unique(sources)
    members = {}
    for name in names:
        members[name] = np.flatnonzero(sources == name)
    resamples = []
    for _ in range(RESAMPLES):
        drawn = rng.choice(names, len(names))
        resamples.append(np.concatenate([members[name] for name in drawn]))
    return resamples


def _measureGap(first: np.ndarray, second: np.ndarray, resamples: list[np.ndarray]) -> float:
    """The difference of two arms' mean precision, in its standard errors over the resamples."""
    differences = []
    for rows in resamples:
        differences.append(first[rows].mean() - second[rows].mean())
    return float((first.mean() - second.mean()) / np.std(differences))


def _countShared(first: np.ndarray, second: np.ndarray) -> float:
    """How many of a real table's nearest real tables two candidates share, on average."""
    shared = []
    for row_first, row_second in zip(first.tolist(), second.tolist()):
        shared.append(len(set(row_first) & set(row_second)))
    return float(np.mean(shared))


# =====================================================================================
# the toy
# =====================================================================================
# real tables like the measured ones, on which a fault is dosed:
# copied schema axes, blocks of content axes a table shows or lacks, one heavy-tailed block,
# and one axis where the real tables spread over [0, 1] with 14 % tied at 1

TOY_TABLES = 375  # each its own source
TOY_SCHEMA = 16  # the axes a draw copies from its real table
TOY_BLOCKS = 5  # of content axes, each shown or lacked whole
TOY_WIDTH = 12  # the axes of a block
TOY_FACTORS = 5  # behind the content axes
TOY_SHOWN = [0.9, 0.6, 0.4, 0.2, 0.5]  # the share of the tables that show each block
TOY_SEEDS = 10
TOY_CASES = {
    'calibrated': {},
    'spread 0.8': {'spread': 0.8},
    'spread 1.1': {'spread': 1.1},
    'spread 1.25': {'spread': 1.25},
    'spread 1.5': {'spread': 1.5},
    'independent 0.1': {'broken': 0.1},
    'independent 0.3': {'broken': 0.3},
    'piled': {'piled': True},
}


def compareToy() -> list[dict]:
    """Every candidate on each case of the toy and each seed."""
    tasks = []
    for case in TOY_CASES:
        for seed in range(TOY_SEEDS):
            tasks.append((case, seed))
    records = []
    with ProcessPoolExecutor() as pool:
        for result in pool.map(_scoreToy, tasks):
            records += result
    return records


def _scoreToy(task: tuple[str, int]) -> list[dict]:
    case, seed = task
    real, draws = _drawToy(np.random.default_rng(seed), **TOY_CASES[case])
    sources = np.arange(TOY_TABLES)
    blocks = _nameToyBlocks()
    records = []
    for candidate in CANDIDATES:
        measured = _measureNearest(candidate, real, draws, sources, blocks)
        cell = {'data': 'toy', 'arm': case, 'candidate': candidate, 'seed': seed}
        records += _recordReaches(cell, _rankReaches(measured['nearest']))
    return records


def _drawToy(
    rng: np.random.Generator, spread: float = 1.0, broken: float = 0.0, piled: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The real tables (t, p) and their draws (t, s, p).
    The draws' content is `spread` times as wide as the real tables', a share `broken` of each
    axis's factor variance independent, and the piled axis at its edge if `piled`.
    """
    content_axes = TOY_BLOCKS * TOY_WIDTH
    schema = rng.standard_normal((TOY_TABLES, TOY_SCHEMA))
    loading = rng.standard_normal((TOY_FACTORS, content_axes)) / math.sqrt(TOY_FACTORS)
    tilt = rng.standard_normal((TOY_SCHEMA, TOY_FACTORS)) / math.sqrt(TOY_SCHEMA)
    shown = rng.random((TOY_TABLES, TOY_BLOCKS)) < np.array(TOY_SHOWN)
    mask = np.repeat(shown, TOY_WIDTH, axis=1)  # (t, content axes)
    copies = np.repeat(schema[:, None], DRAWS, axis=1)  # (t, s, schema axes)

    real_content = _drawToyContent(rng, schema, loading, tilt, spread=1.0, broken=0.0)
    draw_content = _drawToyContent(rng, copies, loading, tilt, spread, broken)
    # the piled axis belongs to the second block
    piled_shown = mask[:, TOY_WIDTH]  # (t,)
    real_pile = _drawToyPile(rng, (TOY_TABLES,), piled=False)
    draw_pile = _drawToyPile(rng, (TOY_TABLES, DRAWS), piled)

    real = [
        schema,
        np.where(mask, real_content, np.nan),
        np.where(piled_shown, real_pile, np.nan)[:, None],
    ]
    draws = [
        copies,
        np.where(mask[:, None], draw_content, np.nan),
        np.where(piled_shown[:, None], draw_pile, np.nan)[:, :, None],
    ]
    return np.concatenate(real, axis=1), np.concatenate(draws, axis=2)


def _drawToyContent(
    rng: np.random.Generator,
    schema: np.ndarray,
    loading: np.ndarray,
    tilt: np.ndarray,
    spread: float,
    broken: float,
) -> np.ndarray:
    """The content axes: factors that the schema tilts, loaded onto the axes, plus noise."""
    shape = schema.shape[:-1]
    content_axes = loading.shape[1]
    factors = (schema @ tilt + rng.standard_normal((*shape, TOY_FACTORS))) / math.sqrt(2)
    related = factors @ loading
    # an independent part with each axis's factor variance
    loose = rng.standard_normal((*shape, content_axes)) * np.sqrt((loading**2).sum(axis=0))
    signal = math.sqrt(1 - broken) * related + math.sqrt(broken) * loose
    values = spread * (signal + 0.5 * rng.standard_normal((*shape, content_axes)))
    # the first block is heavy-tailed
    heavy = np.exp(2 * values[..., :TOY_WIDTH])
    return np.concatenate([heavy, values[..., TOY_WIDTH:]], axis=-1)


def _drawToyPile(rng: np.random.Generator, shape: tuple[int, ...], piled: bool) -> np.ndarray:
    """Uniform on [0, 1] with 14 % at 1, or if `piled` 84 % at 1 and the rest above 0.9."""
    if piled:
        at_edge = rng.random(shape) < 0.84
        return np.where(at_edge, 1.0, rng.uniform(0.9, 1.0, shape))
    at_edge = rng.random(shape) < 0.14
    return np.where(at_edge, 1.0, rng.random(shape))


def _nameToyBlocks() -> list[str]:
    """The toy's blocks of axes, in the order `_drawToy` concatenates them."""
    blocks = ['schema'] * TOY_SCHEMA
    for block in range(TOY_BLOCKS):
        blocks += [f'content {block}'] * TOY_WIDTH
    blocks.append('content 1')  # the piled axis
    return blocks


# =====================================================================================
# main
# =====================================================================================

if __name__ == '__main__':
    records = compareReal()
    records += compareToy()
    report = pd.DataFrame(records, columns=COLUMNS)
    report.to_csv(REPORT, index=False, lineterminator='\n')
