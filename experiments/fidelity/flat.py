"""flat.py compares what `Scm.flat()` draws with the real flat tables of mixedbench 1.0.15 (#93).

Each real table is prepared (preparation.py) and measured into its summary vector.
Each arm then draws 39 datasets at the real table's schema, measured the same way,
so the real table can be ranked among its own draws on every axis.
The arms are the defaults, the defaults with independent covariates (`dependence='none'`,
the negative control) and TabICL v2's prior, which states no column types.
flat.ipynb reads the report.

pipeline:
---------
mixedbench's flat sample
-> each table from the cache, prepared                              _prepareTables
-> its schema (the overrides that restate it) and its stratum       _stateSchema, _findStratum
-> its source's part, by a fixed split                              _splitSources
-> its summary vector                                               measure
-> flat-real.csv, and flat-left-out.csv with a reason per table
per arm and stratum:
-> 39 draws at each real table's schema, measured                   run, _drawScm, _drawTabicl
-> flat-draws-<arm>-<stratum>.csv.gz
per arm, part and stratum:
-> recall per axis against its gate, the precision at 5, 3 and 10 neighbors,
   the two-sample test and the distance per axis                    report, _reportArm
-> the same test and distances between the two parts                _reportParts
-> the tables outside each size field, the distance of the
   type mix and the levels to the fields' own draws                 _reportSchema
-> flat-report.csv

Run from the repository root. The tables inside the default sizes take minutes on a laptop;
the large tables' measure and the draws of the tables outside run on the cluster (flat.sbatch).
The TabICL arm needs its own dependency group:

    uv run --group registry python experiments/fidelity/flat.py measure --cache <dir>
    uv run --group registry python experiments/fidelity/flat.py run default inside
    uv run --group registry python experiments/fidelity/flat.py run independent inside
    uv run --group registry --group tabicl python experiments/fidelity/flat.py run tabicl inside
    uv run --group registry python experiments/fidelity/flat.py report --strata inside
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import pandas as pd
from mixedbench import registry
from mixedbench.datasets import Loader
from mixedbench.registry.row import readCsv
from preparation import Prepared, prepare
from scipy.stats import binom

from tabdgp import Scm, drawDataset, fidelity, summary
from tabdgp.column import Column

# =====================================================================================
# the files, the arms and the sizes
# =====================================================================================

HERE = Path(__file__).parent
REAL = HERE / 'flat-real.csv'  # a row per measured table
LEFT_OUT = HERE / 'flat-left-out.csv'  # a row per table the preparation empties
REPORT = HERE / 'flat-report.csv'  # a row per statistic, arm, part, stratum and axis
SEED = 2026
# draws per real table and arm, so that 2 / (DRAWS + 1) = 5 % of the real tables
# fall outside their draws' range on an axis if they are draws of the generator
DRAWS = 39
PERMUTATIONS = 100  # of the two-sample test's and the distances' null
SCHEMA_DRAWS = 1000  # of the defaults, for the type mix and the levels

Arm = Literal['default', 'independent', 'tabicl']
Part = Literal[
    'fitting', 'certifying'
]  # the hyperprior is fitted on one and certified on the other
# inside the sizes the defaults draw, outside them, or too large to pair with draws
Stratum = Literal['inside', 'outside', 'large']
ARMS: tuple[Arm, ...] = get_args(Arm)
PARTS: tuple[Part, ...] = get_args(Part)
PAIRED: tuple[Stratum, ...] = ('inside', 'outside')
IDENTITY = ['position', 'source', 'object', 'part', 'stratum', 'response_type', 'schema']
AXES = list(summary.AXES)
# a table past one of these is measured once on the cluster and never drawn:
# the relations loop over the pairs of columns, the design holds rows × design columns
MOST_COLUMNS = 300
MOST_DESIGN_COLUMNS = 3000
MOST_CELLS = 10**8

# =====================================================================================
# the real tables
# =====================================================================================


def measure(cache: Path, large: bool) -> None:
    """Writes the measured tables and the left-out ones.
    The large tables are measured by a run of their own, each run keeping the other's rows:
    after a change to the preparation or to mixedbench's version, both runs are owed.
    """
    measured = {}
    if REAL.exists():
        for record in pd.read_csv(REAL, dtype=str, keep_default_na=False).to_dict('records'):
            if (record['stratum'] == 'large') != large:
                measured[int(record['position'])] = record
    parts = _splitSources()
    left_out = []
    for position, row, prepared in _prepareTables(cache):
        identity = {'source': row['source'], 'object': row['object']}
        if prepared.reason is not None:
            left_out.append(identity | {'reason': prepared.reason})
            continue
        schema = _stateSchema(prepared)
        stratum = _findStratum(prepared)
        if (stratum == 'large') != large:
            continue
        vector = summary.measure(prepared.covariates, prepared.outcome)  # (p,)
        described = {
            'position': str(position),
            'part': parts[row['source']],
            'stratum': stratum,
            'response_type': row['response_type'],
            'schema': json.dumps(schema),
        }
        measured[position] = identity | described | _formatVector(vector)
    ordered = [measured[position] for position in sorted(measured)]
    _writeCsv(pd.DataFrame(ordered, columns=IDENTITY + AXES), REAL)
    if not large:
        _writeCsv(pd.DataFrame(left_out, columns=['source', 'object', 'reason']), LEFT_OUT)


# -------------------------------------------------------------------------------------
# the flat sample -> each table prepared, with its schema, stratum and part
# -------------------------------------------------------------------------------------


def _prepareTables(cache: Path) -> Iterator[tuple[int, dict, Prepared]]:
    """Each table of the flat sample: its position, its row with the response's type, prepared."""
    loader = Loader(cache)
    sources = registry.sources()
    flat = readCsv(registry.path('flat.csv'))
    responses = readCsv(registry.path('flat_response_statistics.csv'))
    response_types = {}
    for response in responses.to_dict('records'):
        response_types[response['source'], response['object']] = response['response_type']
    for position, row in enumerate(flat.to_dict('records')):
        table = loader.load(sources[row['source']], row['object'])
        # a table without a documented model names no response
        if row['response'] == 'none':
            row['response_type'] = ''
            prepared = prepare(table, None, None)
        else:
            row['response_type'] = response_types[row['source'], row['object']]
            prepared = prepare(table, row['response'], row['response_type'])
        yield position, row, prepared


def _stateSchema(prepared: Prepared) -> dict:
    """The overrides of `Scm.flat` that restate the table: its rows, each column's type, y's."""
    stated = []
    for column in prepared.covariates.metadata.values():
        stated.append(_stateColumn(column))
    schema = {'rows': prepared.covariates.rows, 'columns': stated}
    if prepared.outcome is None:
        schema['outcome_source'] = 'none'
    else:
        schema['outcome_column'] = _stateColumn(prepared.outcome.metadata['y'])
    return schema


def _stateColumn(column: Column) -> str | list:
    """A column as `Scm` states one: its type, with k where k is free."""
    if column.type in ['categorical', 'ordered']:
        return [column.type, column.levels]
    return column.type


def _findStratum(prepared: Prepared) -> Stratum:
    """Whether the defaults draw the table's sizes, an override does, or it is too large to pair."""
    rows = prepared.covariates.rows
    metadata = prepared.covariates.metadata
    design_columns = sum(column.design_columns for column in metadata.values())
    cells = rows * design_columns
    if len(metadata) > MOST_COLUMNS or design_columns > MOST_DESIGN_COLUMNS or cells > MOST_CELLS:
        return 'large'
    scm = Scm.flat()
    drawn_rows = scm.rows.low <= rows <= scm.rows.high
    # the drawn budget is cut at half the rows
    most_design_columns = min(scm.design_columns.high, rows // 2)
    drawn_columns = scm.design_columns.low <= design_columns <= most_design_columns
    return 'inside' if drawn_rows and drawn_columns else 'outside'


def _splitSources() -> dict[str, Part]:
    """Half the sources to each part, so that no source's tables are in both."""
    flat = readCsv(registry.path('flat.csv'))
    sources = sorted(set(flat['source']))
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(sources)
    half = len(sources) // 2
    parts: dict[str, Part] = {}
    for rank, source in enumerate(shuffled.tolist()):
        parts[source] = 'fitting' if rank < half else 'certifying'
    return parts


# -------------------------------------------------------------------------------------
# the files' format
# -------------------------------------------------------------------------------------


def _formatVector(vector: np.ndarray) -> dict[str, str]:
    """A summary vector by axis at 8 significant digits, which a rerun on another machine
    reproduces, blank where the table lacks the axis.
    """
    formatted = {}
    for axis, value in zip(AXES, vector.tolist()):
        formatted[axis] = '' if math.isnan(value) else f'{value:.8g}'
    return formatted


def _writeCsv(table: pd.DataFrame, path: Path) -> None:
    # the same bytes on every system and run: one line end, no timestamp in a gzip header
    compression = {'method': 'gzip', 'mtime': 0} if path.suffix == '.gz' else None
    table.to_csv(path, index=False, lineterminator='\n', compression=compression)


def _pathDraws(arm: Arm, stratum: Stratum) -> Path:
    return HERE / f'flat-draws-{arm}-{stratum}.csv.gz'


# =====================================================================================
# the paired draws
# =====================================================================================


def run(arm: Arm, stratum: Stratum) -> None:
    """Writes the arm's draws for one stratum: DRAWS summary vectors per real table."""
    real = pd.read_csv(REAL)
    paired = real[real['stratum'] == stratum]
    records = []
    for table in paired.to_dict('records'):
        position = table['position']
        schema = json.loads(table['schema'])
        match arm:
            case 'default':
                vectors = _drawScm(position, schema)
            case 'independent':
                vectors = _drawScm(position, schema | {'dependence': 'none'})
            case 'tabicl':
                vectors = _drawTabicl(position, schema)
        for draw, vector in enumerate(vectors):
            records.append({'position': position, 'draw': draw} | _formatVector(vector))
    draws = pd.DataFrame(records, columns=['position', 'draw', *AXES])
    _writeCsv(draws, _pathDraws(arm, stratum))


def _drawScm(position: int, overrides: dict) -> list[np.ndarray]:
    """DRAWS datasets of `Scm.flat` under the overrides, measured."""
    scm = Scm.flat(**overrides)
    vectors = []
    for draw in range(DRAWS):
        # a table's keys follow its position in the flat sample, whichever tables are left out
        dataset = drawDataset(SEED, position * DRAWS + draw, scm)
        vectors.append(summary.measure(dataset.covariates, dataset.outcome))  # (p,)
    return vectors


def _drawTabicl(position: int, schema: dict) -> list[np.ndarray]:
    """DRAWS datasets of TabICL v2's prior at the table's rows and columns, measured.
    The prior states no types, so each column is read as a real table's numbers are,
    and y by the task: a regression's is continuous, a classification's categorical.
    """
    # torch and tabicl are the dependency group `tabicl`, which only this arm needs
    import torch
    from tabicl.prior import PriorDataset

    stated = schema.get('outcome_column')
    regression = stated in [None, 'continuous', 'count']
    if stated == 'binary':
        classes = 2
    elif regression:
        classes = 0
    else:
        classes = stated[1]
    # the prior reads the global generators, so each table seeds them
    seed = SEED + position
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    columns = len(schema['columns'])
    prior = PriorDataset(
        regression=regression,
        batch_size=DRAWS,
        batch_size_per_gp=1,  # each draw with hyperparameters of its own
        min_features=columns,
        max_features=columns,
        max_classes=classes,
        max_seq_len=schema['rows'],
        prior_type='graph_scm',  # the prior TabICL v2 is pretrained on
        n_jobs=1,
    )
    features, targets, kept, _, _ = prior.get_batch()  # (s, n, c), (s, n), (s,)
    vectors = []
    for draw in range(DRAWS):
        # the prior drops its constant columns and pads the block with zeros
        table = pd.DataFrame(features[draw, :, : kept[draw]].numpy())
        if stated is None:
            prepared = prepare(table, None, None)
        else:
            table['response'] = targets[draw].numpy()
            response_type = 'continuous' if regression else 'categorical'
            prepared = prepare(table, 'response', response_type)
        vectors.append(summary.measure(prepared.covariates, prepared.outcome))  # (p,)
    return vectors


# =====================================================================================
# the report
# =====================================================================================


def report(arms: list[Arm], strata: list[Stratum]) -> None:
    """Writes every statistic of the report, a row per statistic, arm, part, stratum and axis."""
    real = pd.read_csv(REAL)
    records = _reportSchema(real)
    for stratum in strata:
        records += _reportParts(real, stratum)
        for arm in arms:
            draws = pd.read_csv(_pathDraws(arm, stratum))
            records += _reportArm(arm, stratum, real, draws)
    columns = ['statistic', 'arm', 'part', 'stratum', 'axis', 'value']
    _writeCsv(pd.DataFrame(records, columns=columns), REPORT)


# -------------------------------------------------------------------------------------
# an arm's draws against the real tables
# -------------------------------------------------------------------------------------


def _reportArm(arm: Arm, stratum: Stratum, real: pd.DataFrame, draws: pd.DataFrame) -> list[dict]:
    """Recall, precision, the two-sample test and the distances of one arm in one stratum,
    per part. The precision's neighbors are the real tables of the same part and stratum.
    """
    records = []
    for part in PARTS:
        cell = {'arm': arm, 'part': part, 'stratum': stratum}
        tables = real[(real['part'] == part) & (real['stratum'] == stratum)]
        real_vectors = tables[AXES].to_numpy()  # (t, p)
        own = draws[draws['position'].isin(tables['position'])]
        # check that the draws are these tables': DRAWS per table, in the tables' order
        expected = np.repeat(tables['position'].to_numpy(), DRAWS)
        if not np.array_equal(own['position'].to_numpy(), expected):
            raise ValueError(
                f'the draws of {arm} are not those of the {stratum} tables: run it again'
            )
        draw_vectors = own[AXES].to_numpy().reshape(len(tables), DRAWS, len(AXES))

        recall = fidelity.recall(real_vectors, draw_vectors)
        per_axis = {
            'tables': recall.tables,
            'outside': recall.outside,
            'below': np.sum(recall.ranks == 0, axis=0),  # the tables under every draw
            'gate': _gateRecall(recall.tables),
            'mean_rank': recall.mean_rank,
        }
        for statistic, values in per_axis.items():
            records += _recordAxes(statistic, cell, AXES, values)
        # the gate reads the precision at 5 neighbors, and 3 and 10 show how far it moves
        statistics = {'precision': 5, 'precision_3': 3, 'precision_10': 10}
        sources = tables['source'].to_numpy()
        for statistic, neighbors in statistics.items():
            precision = fidelity.precision(real_vectors, draw_vectors, sources, neighbors)
            records.append({'statistic': statistic, **cell, 'value': precision.share})
        # two samples of t tables each: the real ones and the first draw of each
        seeds = [SEED, ARMS.index(arm), PARTS.index(part), PAIRED.index(stratum)]
        records += _reportTwoSamples(real_vectors, draw_vectors[:, 0], AXES, seeds, cell)
    return records


def _gateRecall(tables: np.ndarray) -> np.ndarray:
    """The most real tables that may lie outside their draws' range before an axis fails:
    5 % lie outside under a calibrated generator, and the tolerance is the quantile
    1 − 0.01 / p of that binomial count, over the p axes.
    """
    return binom.ppf(1 - 0.01 / len(AXES), tables, 2 / (DRAWS + 1))  # (p,)


def _reportTwoSamples(
    a: np.ndarray, b: np.ndarray, axes: list[str], seeds: list[int], cell: dict
) -> list[dict]:
    """The two-sample test of two samples of summary vectors, and their distance per axis."""
    rng = np.random.default_rng(seeds)
    test = fidelity.c2st(a, b, rng, PERMUTATIONS)
    records = [
        {'statistic': 'c2st', **cell, 'value': test.statistic},
        {'statistic': 'c2st_p', **cell, 'value': test.p_value},
    ]
    records += _recordDistances('distance', a, b, axes, rng, cell)
    return records


def _recordDistances(
    statistic: str,
    a: np.ndarray,
    b: np.ndarray,
    axes: list[str],
    rng: np.random.Generator,
    cell: dict,
) -> list[dict]:
    """The distance per axis between two samples, and its p-value."""
    comparisons = fidelity.distances(a, b, rng, PERMUTATIONS)
    statistics = [comparison.statistic for comparison in comparisons]
    p_values = [comparison.p_value for comparison in comparisons]
    records = _recordAxes(statistic, cell, axes, statistics)
    records += _recordAxes(f'{statistic}_p', cell, axes, p_values)
    return records


def _recordAxes(
    statistic: str, cell: dict, axes: list[str], values: list | np.ndarray
) -> list[dict]:
    records = []
    for axis, value in zip(axes, values):
        records.append({'statistic': statistic, **cell, 'axis': axis, 'value': float(value)})
    return records


# -------------------------------------------------------------------------------------
# the real tables alone: their two parts, their sizes and their schemas
# -------------------------------------------------------------------------------------


def _reportParts(real: pd.DataFrame, stratum: Stratum) -> list[dict]:
    """The two-sample test and the distances between the two parts of a stratum's real tables,
    the reference for an arm's: real tables of other sources differ this much.
    Beside them, the share of each part's tables that lack an axis.
    """
    records = []
    vectors = {}
    for part in PARTS:
        tables = real[(real['part'] == part) & (real['stratum'] == stratum)]
        vectors[part] = tables[AXES].to_numpy()  # (t, p)
        cell = {'arm': 'real', 'part': part, 'stratum': stratum}
        missing_share = np.isnan(vectors[part]).mean(axis=0)  # (p,)
        records += _recordAxes('missing_share', cell, AXES, missing_share)
    cell = {'arm': 'real', 'part': 'fitting vs certifying', 'stratum': stratum}
    seeds = [SEED, len(ARMS), PAIRED.index(stratum)]
    records += _reportTwoSamples(vectors['fitting'], vectors['certifying'], AXES, seeds, cell)
    return records


def _reportSchema(real: pd.DataFrame) -> list[dict]:
    """Per part: the share of the real tables outside each size field's support, and the
    distance of the type mix and the levels between the tables inside and the defaults' draws,
    beside each side's mean and the share of its tables that show the axis.
    """
    scm = Scm.flat()
    drawn = []
    for index in range(SCHEMA_DRAWS):
        dataset = drawDataset(SEED + 1, index, scm)  # a seed apart from the paired draws'
        drawn.append(summary.measure(dataset.covariates, dataset.outcome))
    drawn = pd.DataFrame(drawn, columns=AXES)
    mix = [f'{type}_share' for type in summary.TYPES] + ['levels_median', 'levels_max']

    records = _recordMix({'arm': 'default', 'part': 'every', 'stratum': 'inside'}, drawn[mix])
    for part in PARTS:
        tables = real[real['part'] == part]
        cell = {'arm': 'real', 'part': part, 'stratum': 'every'}
        rows = tables['rows']
        design_columns = tables['design_columns']
        outside = {
            f'rows < {scm.rows.low}': rows < scm.rows.low,
            f'rows > {scm.rows.high}': rows > scm.rows.high,
            f'design_columns > {scm.design_columns.high}': design_columns > scm.design_columns.high,
            'design_columns > rows / 2': design_columns > rows // 2,
        }
        shares = [float(beyond.mean()) for beyond in outside.values()]
        records += _recordAxes('outside_share', cell, list(outside), shares)

        inside = tables[tables['stratum'] == 'inside']
        cell = {'arm': 'real', 'part': part, 'stratum': 'inside'}
        records += _recordMix(cell, inside[mix])
        seeds = [SEED, len(ARMS) + 1, PARTS.index(part)]
        rng = np.random.default_rng(seeds)
        real_mix = inside[mix].to_numpy()  # (t, 7)
        drawn_mix = drawn[mix].to_numpy()  # (SCHEMA_DRAWS, 7)
        records += _recordDistances('schema_distance', real_mix, drawn_mix, mix, rng, cell)
    return records


def _recordMix(cell: dict, mix: pd.DataFrame) -> list[dict]:
    """Each axis of the type mix and the levels: its mean over the tables that show it,
    and their share.
    """
    axes = list(mix.columns)
    records = _recordAxes('schema_mean', cell, axes, mix.mean())
    records += _recordAxes('schema_shown', cell, axes, mix.notna().mean())
    return records


# =====================================================================================
# main
# =====================================================================================

if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    steps = parser.add_subparsers(dest='step', required=True)
    measuring = steps.add_parser('measure')
    measuring.add_argument('--cache', type=Path, required=True, help="mixedbench's cache of its sources")
    measuring.add_argument('--large', action='store_true', help='the large tables alone, on the cluster')
    running = steps.add_parser('run')
    running.add_argument('arm', choices=ARMS)
    running.add_argument('stratum', choices=PAIRED)
    reporting = steps.add_parser('report')
    reporting.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    reporting.add_argument('--strata', nargs='+', choices=PAIRED, default=list(PAIRED))
    args = parser.parse_args()
    # fmt: on

    match args.step:
        case 'measure':
            measure(args.cache, args.large)
        case 'run':
            run(args.arm, args.stratum)
        case 'report':
            report(args.arms, args.strata)
