"""registry.py compares the scale draws of a GLMM's prior on the registry's designs (#77, #78).

simulated.py reads each measure off the ground truth on TabDGP's own covariates.
Here each registry fit lends its design as lme4 builds it from the documented formula
(X, Z, the rows it keeps and the offset), each candidate draws a y on it,
and lme4 refits the documented formula with y as its left side,
so each refit is its registry fit with y swapped, and both are lme4 estimates.
The registry holds only fits with estimates, so each candidate draws until REFITS refits have them.
The ground truth and what the prior drew are logged beside each refit,
and the real y's mean is kept beside the design's sizes, which fits.csv lacks.
registry.ipynb reads the records against the registry's fits.

pipeline:
---------
the registry's fits
-> a task per kept fit, a reason per dropped one                      select, _screenFit
-> registry-fits.csv, the array job's tasks, with the real y's mean
per task and range of draws:
-> each arm's refits with estimates before the range                  run
-> lme4's design of the documented formula, X standardized            _readDesign
-> the blocks TabDGP's draw builds on it                              _measureBlocks
-> per draw and arm still short, the joint prior under the arm's
   overrides, each factor correlated as lme4 fits it                   _drawJointPrior
-> the prior at the arm's scales, the parameters, y
   and the ground truth's measures                                     measureTruth
-> lme4's refit of y and its measures                                  _refit
-> what the prior drew                                                 _describePrior
-> a record per refit, in the task's run directory
-> registry.csv.gz, each arm's draws until REFITS refits have estimates,
   the arms not collected as registry.csv.gz held them                 collect

Run from the repository root: select once, the tasks as cluster array jobs
(registry.sbatch), then collect:

    uv run --group registry python experiments/scale-draw/registry.py select --cache <dir>
    uv run --group registry python experiments/scale-draw/registry.py run <task> \
        --cache <dir> --survey <dir> [--draws <first> <stop>] [--arms <arm> ...]
    uv run --group registry python experiments/scale-draw/registry.py collect [--arms <arm> ...]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from candidates import (
    ARMS,
    OVERRIDES,
    Arm,
    Candidate,
    Factors,
    buildCandidate,
    buildEta,
    measureTruth,
)
from mixedbench import registry
from mixedbench.frequentist import lme4
from mixedbench.registry.row import Row, readCsv

from tabdgp import Glmm, likelihood
from tabdgp.field import Distribution
from tabdgp.prior import Block, Prior

# =====================================================================================
# the fits and the draws
# =====================================================================================

HERE = Path(__file__).parent
FITS = HERE / 'registry-fits.csv'  # the array job's tasks
SEED = 2026
# refits with estimates per fit and arm, so that 2 / (REFITS + 1) = 5 % of the fits
# fall outside their refits' range if real tables are draws of the prior
REFITS = 39
DRAWS = 2 * REFITS  # per fit and arm at most
TIMEOUT = 2 * 60 * 60  # seconds per refit
ESTIMATED = ['ok', 'singular', 'warning']  # the statuses of a fit with estimates
# the drawn y's column, beside the documented response, which a covariate may read
# (rabe_hesketh's pups counts it per dam), and named apart from a covariate y (SASmixed's Demand)
RESPONSE = 'y_drawn'
NO_COLUMNS = ['rethinking/data/PrussianHorses.csv/1']  # `kicks ~ 0 + (1 | corps) + (1 | year)`

# =====================================================================================
# the tasks
# =====================================================================================


def select(tables: registry.Tables) -> pd.DataFrame:
    """Every model the registry fitted, a task, the design's sizes and the real y's mean
    per kept fit, a reason per dropped one."""
    fits = readCsv(registry.path('fits.csv'))
    # fits.csv has a row per factor
    models = fits.drop_duplicates(['source', 'object', 'model_index'])
    rows = {row.id: row for row in registry.rows()}
    records = []
    task = 0
    for fit in models.to_dict('records'):
        row = rows[f'{fit["source"]}/{fit["object"]}/{fit["model_index"]}']
        screened = _screenFit(row, fit, tables)
        record = {'task': None, 'id': row.id, 'family': fit['family'], 'status': fit['status']}
        if screened['reason'] is None:
            record['task'] = task
            task += 1
        records.append(record | screened)
    # integers with blanks, where a dropped fit has no task or sizes
    sizes = {
        'task': 'Int64',
        'rows': 'Int64',
        'rank': 'Int64',
        'factors': 'Int64',
        'random_effects': 'Int64',
    }
    return pd.DataFrame(records).astype(sizes)


def _screenFit(row: Row, fit: dict, tables: registry.Tables) -> dict:
    """Why a fit is dropped (None if kept), and a kept fit's sizes and real y's mean."""
    # a fit without estimates (an error, a model in words) leaves nothing to compare
    if fit['status'] not in ESTIMATED:
        return {'reason': 'no estimates'}
    # mixedbench 1.0.13's design fails on an X without columns, which has no intercept
    if row.id in NO_COLUMNS:
        return {'reason': 'no intercept'}
    design = _readDesign(tables.loadFitted(row), row.formula, fit['family'])
    # the design holds the rows the registry's fit read (a subset or weighted copies included)
    if len(design['rows']) != int(fit['n_obs']):
        raise ValueError(
            f'{row.id}: the design has {len(design["rows"])} rows, the fit {fit["n_obs"]}'
        )
    # the prior's fixed block opens with the intercept
    if design['columns'][:1] != ['(Intercept)']:
        return {'reason': 'no intercept'}
    random_effects = 0
    for z, _ in design['factors'].values():
        random_effects += z.shape[1]
    return {
        'reason': None,
        'rows': len(design['rows']),
        'rank': len(design['columns']),  # d, the intercept included
        'factors': len(design['factors']),
        'random_effects': random_effects,
        'y_mean': design['y_mean'],  # over the rows, as lme4 reads y
    }


def _readDesign(table: pd.DataFrame, formula: str, family: str) -> dict:
    """lme4's design of the formula on the table: the rows it keeps, y's mean over them,
    X's column names and X standardized, the offset, and per factor in lme4's order
    its random columns and codes, its groups, and whether lme4 correlates its random effects."""
    built = lme4.buildDesign(table, formula, family)

    # a continuous column, a transform or a product z-scored over the rows, as the contract
    # scales a continuous covariate, and a 0/1 column (the intercept, a dummy) as is,
    # so that the refit's intercept and slopes absorb the change
    continuous = ~np.isin(built.X, [0, 1]).all(axis=0)  # (d,)
    matrix = built.X.copy()  # (n, d)
    values = built.X[:, continuous]
    matrix[:, continuous] = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)

    factors: Factors = {}
    groups = {}
    correlated = []
    for factor in built.factors:
        factors[factor.name] = (factor.Z, factor.codes)
        groups[factor.name] = len(factor.levels)
        # `(1 + x || g)` is two terms of g, whose random effects lme4 leaves uncorrelated
        if len(factor.terms) == 1:
            correlated.append(factor.name)
    return {
        'rows': built.rows,
        'y_mean': float(built.y.mean()),  # as lme4 reads y
        'columns': list(built.columns),
        'matrix': matrix,
        'offset': built.offset,
        'factors': factors,
        'groups': groups,
        'correlated': correlated,
    }


# =====================================================================================
# a task
# =====================================================================================


def run(
    task: int, draws: range, tables: registry.Tables, candidates: dict[Arm, Candidate], runs: Path
) -> None:
    """Draws each arm's y on the task's fit at `draws` and refits each by lme4,
    a record per refit in the task's run directory, which replaces an earlier run's.
    Past the first REFITS draws, an arm draws only while short of REFITS refits with estimates."""
    fits = pd.read_csv(FITS)
    kept = fits[fits['task'].notna()]
    [fit] = kept[kept['task'] == task].to_dict('records')
    dir = runs / str(task)

    # each arm's refits with estimates before the range, where the range reaches past
    # the first REFITS draws, so that its earlier draws must all have run
    estimated = {}
    for arm in candidates:
        estimated[arm] = 0
        if draws.stop > REFITS:
            for draw in range(draws.start):
                record = _readRecord(dir, arm, draw)
                estimated[arm] += record['fit_status'] in ESTIMATED
    if draws.start >= REFITS and min(estimated.values()) >= REFITS:
        return  # every arm has its refits

    row = {row.id: row for row in registry.rows()}[fit['id']]
    table = tables.loadFitted(row)
    family = fit['family']
    factor_columns = int(kept['factors'].max())  # a column per factor of the widest fit
    dir.mkdir(parents=True, exist_ok=True)
    design = _readDesign(table, row.formula, family)
    blocks = _measureBlocks(design)
    formula = f'{RESPONSE} ~' + row.formula.split('~', 1)[1]  # the documented right side

    for draw in draws:
        streams = [SEED, task, draw]
        for arm, candidate in candidates.items():
            if draw >= REFITS and estimated[arm] >= REFITS:
                continue  # the arm has its refits

            # every arm draws from the same streams, so arms differ by their overrides and scales
            joint = _drawJointPrior(
                family,
                blocks,
                design['correlated'],
                OVERRIDES[arm],
                np.random.default_rng([*streams, 0]),
            )
            prior = candidate.rescale(joint, family, np.random.default_rng([*streams, 1]))
            parameters = prior.drawParameters(np.random.default_rng([*streams, 2]))
            eta = buildEta(
                design['matrix'], design['factors'], parameters, design['offset']
            )  # (n,)
            scale = None
            if parameters.sigma_eps is not None:
                scale = np.full(len(eta), parameters.sigma_eps)  # (n,)
            # y's three families have no shape
            y = likelihood.draw(family, eta, scale, None, np.random.default_rng([*streams, 3]))
            truth = measureTruth(
                family,
                design['matrix'],
                design['factors'],
                parameters,
                factor_columns,
                design['offset'],
            )

            # lme4's refit on the raw table, y missing on the rows lme4 dropped
            response = np.full(len(table), math.nan)  # (N,) the loaded rows
            response[design['rows']] = y
            refit_table = table.assign(**{RESPONSE: response})
            measured = _refit(refit_table, formula, family, factor_columns)
            estimated[arm] += measured['status'] in ESTIMATED

            # what the prior drew, so that a margin splits by any of its draws,
            # and y's mean, which flags a y of a few ones or counts that lme4 cannot refit
            record = {'task': task, 'draw': draw, 'arm': arm}
            record |= _describePrior(prior, factor_columns)
            record['intercept'] = float(parameters.beta[0])
            record['y_mean'] = float(y.mean())
            for name, value in truth.items():
                record[f'truth_{name}'] = value
            for name, value in measured.items():
                record[f'fit_{name}'] = value
            (dir / f'{arm}-{draw}.json').write_text(json.dumps(record, indent=2))


def _readRecord(dir: Path, arm: str, draw: int) -> dict:
    """An arm's record at a draw in the task's run directory; raises where it never ran."""
    return json.loads((dir / f'{arm}-{draw}.json').read_text())


def _measureBlocks(design: dict) -> list[Block]:
    """The blocks TabDGP's draw builds on the design, from what each coefficient adds
    to var(eta) at scale 1."""
    # a fixed effect adds its column's variance (the intercept nothing)
    blocks = [Block(name='fixed', variances=design['matrix'].var(axis=0).tolist())]
    for factor, (z, _) in design['factors'].items():
        # a random effect adds E[z²] (an intercept 1)
        variances = (z**2).mean(axis=0)  # (q,)
        groups = design['groups'][factor]
        blocks.append(Block(name=factor, variances=variances.tolist(), groups=groups))
    return blocks


def _drawJointPrior(
    family: str,
    blocks: list[Block],
    correlated: list[str],
    overrides: dict[str, Distribution],
    rng: np.random.Generator,
) -> Prior:
    """The joint draw's prior over the blocks under an arm's overrides of Glmm's defaults,
    each factor's random effects correlated as lme4 fits them:
    an LKJ for a factor in one term, independent for `(x || g)`."""
    hyperprior = Glmm(family=family, correlated_share=1.0, **overrides)
    prior = hyperprior.resolveDataset(rng).drawPrior(blocks, rng)
    lkj_eta = {}
    for factor, shape in prior.lkj_eta.items():
        lkj_eta[factor] = shape if factor in correlated else None
    return dataclasses.replace(prior, lkj_eta=lkj_eta)


def _refit(table: pd.DataFrame, formula: str, family: str, factor_columns: int) -> dict:
    """How lme4's fit of the formula ended, its sd and max |eta|,
    and its variance partition as fits.csv reads a fit's,
    an ICC per factor in `factor_columns` columns, NaN where the fit has no estimates."""
    # the fit's files (the table among them) go once it is read
    with tempfile.TemporaryDirectory() as temporary:
        fitted = lme4.fit(table, formula, family, Path(temporary), TIMEOUT)
    measures = {
        'status': fitted.status,
        'seconds': fitted.seconds,
        'message': fitted.message,
        'sd_eta': math.nan,
        'max_abs_eta': math.nan,
        'r2_marginal': math.nan,
        'r2_conditional': math.nan,
        'fixed_share': math.nan,
    }
    for k in range(factor_columns):
        measures[f'icc_{k + 1}'] = math.nan
    if fitted.status not in ESTIMATED:
        return measures

    measures['sd_eta'] = fitted.sd_eta
    measures['max_abs_eta'] = fitted.max_abs_eta
    measures['r2_marginal'] = fitted.r2_marginal
    measures['r2_conditional'] = fitted.r2_conditional
    # a singular intercept-only fit explains nothing, so it has no fixed share
    explained = fitted.sigma2_fixed + fitted.sigma2_random
    if explained > 0:
        measures['fixed_share'] = fitted.sigma2_fixed / explained
    # the factors in lme4's order, the design's and that of the fit's rows in fits.csv
    for k, icc in enumerate(fitted.icc.values()):
        measures[f'icc_{k + 1}'] = icc
    return measures


def _describePrior(prior: Prior, factor_columns: int) -> dict:
    """What the prior drew: β's and σ_α's families, τ, the intercept's location and scale,
    Xβ's share of τ², and per factor in lme4's order its share, the spread of its σ_α scales
    and its LKJ shape, in `factor_columns` columns, NaN past the fit's factors."""
    fixed, *factors = prior.blocks
    described = {
        'beta_family': prior.beta_family,
        'sigma_alpha_family': prior.sigma_alpha_family,
        'predictor_sd': prior.predictor_sd,
        'intercept_location': prior.beta_locations[0],
        'intercept_scale': prior.beta_scales[0],
        'split_fixed': prior.split[fixed.name],
    }
    for k in range(factor_columns):
        described[f'split_{k + 1}'] = math.nan
        described[f'sigma_alpha_spread_{k + 1}'] = math.nan
        described[f'lkj_eta_{k + 1}'] = math.nan
    for k, block in enumerate(factors):
        # the level of a factor's scales is τ² times its share, so their spread is what they add:
        # the sd of their logs, 0 for one random effect or a shared scale
        log_scales = np.log(prior.sigma_alpha_scales[block.name])  # (q,)
        described[f'split_{k + 1}'] = prior.split[block.name]
        described[f'sigma_alpha_spread_{k + 1}'] = float(log_scales.std())
        described[f'lkj_eta_{k + 1}'] = prior.lkj_eta[block.name]  # None for independent effects
    return described


# =====================================================================================
# the records
# =====================================================================================


def collect(runs: Path, arms: list[Arm]) -> pd.DataFrame:
    """The arms' records on each task until REFITS refits have estimates (all DRAWS if fewer do),
    in task order.
    It skips a task an arm never ran on, and raises where a draw it keeps never ran."""
    fits = pd.read_csv(FITS)
    records = []
    for task in fits['task'].dropna().astype(int):
        dir = runs / str(task)
        for arm in arms:
            if not (dir / f'{arm}-0.json').exists():
                continue  # a task the arm never ran on
            estimated = 0
            for draw in range(DRAWS):
                record = _readRecord(dir, arm, draw)
                records.append(record)
                estimated += record['fit_status'] in ESTIMATED
                if estimated == REFITS:
                    break
    return pd.DataFrame(records)


# =====================================================================================
# main
# =====================================================================================

if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description="Compares the scale draws on the registry's designs.")
    parser.add_argument('command', choices=['select', 'run', 'collect'],
                        help='list the tasks, run one, or collect every task into registry.csv.gz')
    parser.add_argument('task', type=int, nargs='?', help='the task to run, in registry-fits.csv')
    parser.add_argument('--cache', type=Path, help="mixedbench's cache of the registry's tables")
    parser.add_argument('--draws', type=int, nargs=2, default=[0, DRAWS], metavar=('FIRST', 'STOP'),
                        help='the draws to run, FIRST to STOP - 1')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS,
                        help='the arms to run, or to collect beside the others in registry.csv.gz')
    parser.add_argument('--survey', type=Path,
                        help="the practice survey's data directory, read by the survey arm")
    parser.add_argument('--runs', type=Path, default=HERE / 'registry-runs',
                        help='the directory of the run directories, one per task')
    # fmt: on
    args = parser.parse_args()
    if args.command in ['select', 'run'] and args.cache is None:
        parser.error(f'{args.command} reads --cache')

    match args.command:
        case 'select':
            fits = select(registry.Tables(args.cache))
            fits.round(4).to_csv(FITS, index=False)  # y's mean at the records' 4 decimals
        case 'run':
            if args.task is None:
                parser.error('run reads a task')
            if 'survey' in args.arms and args.survey is None:
                parser.error('the survey arm reads --survey')
            candidates = {}
            for arm in args.arms:
                candidates[arm] = buildCandidate(arm, args.survey)
            run(args.task, range(*args.draws), registry.Tables(args.cache), candidates, args.runs)
        case 'collect':
            records = collect(args.runs, args.arms)
            # the arms not collected as registry.csv.gz held them,
            # each task's arms in ARMS' order and each arm's draws in order
            if set(args.arms) != set(ARMS):
                earlier = pd.read_csv(HERE / 'registry.csv.gz')
                kept = earlier[~earlier['arm'].isin(args.arms)]
                records = pd.concat([kept, records])
                arm_order = records['arm'].map(ARMS.index)
                records = records.iloc[np.lexsort([arm_order, records['task']])]
            # gzip's header at a fixed time, so that a rerun writes the same bytes
            compression = {'method': 'gzip', 'mtime': 0}
            records.round(4).to_csv(HERE / 'registry.csv.gz', index=False, compression=compression)
