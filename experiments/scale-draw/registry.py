"""registry.py compares the scale draws of a GLMM's prior on the registry's designs (#77, #78).

simulated.py reads each measure off the ground truth on TabDGP's own covariates.
Here each registry fit lends its design as lme4 builds it from the documented formula
(X, Z, the rows it keeps and the offset), each candidate draws a y on it,
and lme4 refits the documented formula with y as its left side,
so each refit is its registry fit with y swapped, and both are lme4 estimates.
The registry holds only fits with estimates, so each candidate draws until REFITS refits have them.
The ground truth and what the prior drew are logged beside each refit,
and each fit's real y is refitted once, for the max |eta| that fits.csv lacks.
registry.ipynb reads the records against the registry's fits.

pipeline:
---------
the registry's fits
-> a task per kept fit, a reason per dropped one                      select, _screenFit
-> registry-fits.csv, the array job's tasks
per task and range of draws:
-> each arm's refits with estimates before the range                  run
-> the real y's refit, in the range from draw 0                       _refit
-> lme4's design of the documented formula (registry.R),
   X standardized                                                      _readDesign
-> the blocks TabDGP's draw builds on it                              _measureBlocks
-> per draw, the joint prior, each factor correlated as lme4 fits it  _drawJointPrior
-> per arm still short, the prior at its scales, the parameters, y
   and the ground truth's measures                                     measureTruth
-> lme4's refit of y, its measures and max |eta| (registry.R)         _refit, _measureFit
-> what the prior drew                                                 _describePrior
-> a record per refit, in its run directory
-> registry.csv.gz, each arm's draws until REFITS refits have estimates,
   and registry-real.csv, the real y's refits                          collect

Run from the repository root: select once, the tasks as cluster array jobs
(registry.sbatch), then collect:

    uv run --group registry python experiments/scale-draw/registry.py select --cache <dir>
    uv run --group registry python experiments/scale-draw/registry.py run <task> \
        --cache <dir> --survey <dir> [--draws <first> <stop>] [--arms <arm> ...]
    uv run --group registry python experiments/scale-draw/registry.py collect
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from candidates import ARMS, Candidate, Factors, buildCandidate, buildEta, measureTruth
from mixedbench import registry
from mixedbench.frequentist import lme4
from mixedbench.registry.row import Row, readCsv

from tabdgp import Glmm, likelihood
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
# mixedbench's script of a fit, whose definitions registry.R loads
LME4_R = Path(lme4.__file__).with_name('lme4.R')
RECORD = 'record.json'  # a refit's record, in its run directory

# =====================================================================================
# the tasks
# =====================================================================================


def select(tables: registry.Tables) -> pd.DataFrame:
    """Every model the registry fitted, a task and the design's sizes per kept fit,
    a reason per dropped one."""
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
    """Why a fit is dropped (None if kept), and a kept fit's sizes."""
    # a fit without estimates (an error, a model in words) leaves nothing to compare
    if fit['status'] not in ESTIMATED:
        return {'reason': 'no estimates'}
    design = _readDesign(tables.load(row), row.formula, fit['family'])
    # a fit of a subset or of weighted copies of the rows, which the loaded table lacks
    if len(design['rows']) != int(fit['n_obs']):
        return {'reason': "rows unlike the fit's"}
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
    }


# =====================================================================================
# a task
# =====================================================================================


def run(
    task: int, draws: range, tables: registry.Tables, candidates: dict[str, Candidate], runs: Path
) -> None:
    """Draws each arm's y on the task's fit at `draws` and refits each by lme4,
    a record per refit in its own run directory, which replaces an earlier run's.
    Past the first REFITS draws, an arm draws only while short of REFITS refits with estimates.
    The range from draw 0 also refits the real y."""
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
    table = tables.load(row)
    family = fit['family']
    factor_columns = int(kept['factors'].max())  # a column per factor of the widest fit
    if draws.start == 0:
        real = _refit(table, row.formula, family, dir / 'real', factor_columns)
        (dir / 'real' / RECORD).write_text(json.dumps({'task': task} | real, indent=2))
    design = _readDesign(table, row.formula, family)
    blocks = _measureBlocks(design)
    formula = f'{RESPONSE} ~' + row.formula.split('~', 1)[1]  # the documented right side

    for draw in draws:
        streams = [SEED, task, draw]
        joint = _drawJointPrior(
            family, blocks, design['correlated'], np.random.default_rng([*streams, 0])
        )
        for arm, candidate in candidates.items():
            if draw >= REFITS and estimated[arm] >= REFITS:
                continue  # the arm has its refits

            # every arm draws from the same streams, so arms differ by their scales
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
            refit_dir = dir / f'{arm}-{draw}'
            response = np.full(len(table), math.nan)  # (N,) the loaded rows
            response[design['rows']] = y
            refit_table = table.assign(**{RESPONSE: response})
            measured = _refit(refit_table, formula, family, refit_dir, factor_columns)
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
            (refit_dir / RECORD).write_text(json.dumps(record, indent=2))


def _readRecord(dir: Path, arm: str, draw: int) -> dict:
    """An arm's record at a draw in the task's run directory; raises where it never ran."""
    return json.loads((dir / f'{arm}-{draw}' / RECORD).read_text())


def _refit(table: pd.DataFrame, formula: str, family: str, dir: Path, factor_columns: int) -> dict:
    """lme4's fit of the formula as mixedbench fits it, its files in `dir` but the table,
    which replace an earlier run's, and its measures."""
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir(parents=True)
    _runR('refit', table, formula, family, dir)
    return _measureFit(dir, factor_columns)


def _runR(
    mode: Literal['design', 'refit'], table: pd.DataFrame, formula: str, family: str, dir: Path
) -> None:
    """registry.R in `mode` on the table and the formula, in the files mixedbench writes for R,
    the table deleted once R ran."""
    lme4._write(table, formula, family, dir, TIMEOUT)
    with open(dir / 'R.log', 'w') as log:
        subprocess.run(
            ['Rscript', HERE / 'registry.R', mode, dir, LME4_R],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    (dir / 'table.csv').unlink()


def _measureFit(dir: Path, factor_columns: int) -> dict:
    """How the refit in `dir` ended, its sd and max |eta|,
    and its variance partition as fits.csv reads a fit's,
    an ICC per factor in `factor_columns` columns, NaN where the fit has no estimates."""
    fitted = lme4._read(dir)
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

    # σ̂² of a Gaussian fit, the link's variance of a Bernoulli or Poisson one
    residual = 0.0
    if fitted.sigma is not None:
        residual = fitted.sigma**2
    explained = fitted.sigma2_fixed + sum(fitted.variances.values())
    total = explained + residual + fitted.sigma2_distribution
    measures['sd_eta'] = fitted.sd_eta
    measures['max_abs_eta'] = json.loads((dir / 'eta.json').read_text())['max_abs_eta']
    measures['r2_marginal'] = fitted.sigma2_fixed / total
    measures['r2_conditional'] = explained / total
    # a singular intercept-only fit explains nothing, so it has no fixed share
    if explained > 0:
        measures['fixed_share'] = fitted.sigma2_fixed / explained
    # the factors in lme4's order, the design's and that of the fit's rows in fits.csv
    for k, variance in enumerate(fitted.variances.values()):
        measures[f'icc_{k + 1}'] = variance / total
    return measures


def _readDesign(table: pd.DataFrame, formula: str, family: str) -> dict:
    """lme4's design of the formula on the table, read by registry.R:
    the rows it keeps, X's column names and X standardized, the offset,
    and per factor in lme4's order its random columns and codes, its groups,
    and whether lme4 correlates its random effects."""
    with tempfile.TemporaryDirectory() as temporary:
        dir = Path(temporary)
        _runR('design', table, formula, family, dir)
        built = json.loads((dir / 'design.json').read_text())

    # a continuous column, a transform or a product z-scored over the rows, as the contract
    # scales a continuous covariate, and a 0/1 column (the intercept, a dummy) as is,
    # so that the refit's intercept and slopes absorb the change
    raw = np.array(built['X'], dtype=np.float64)  # (n, d)
    continuous = ~np.isin(raw, [0, 1]).all(axis=0)  # (d,)
    matrix = raw.copy()
    values = raw[:, continuous]
    matrix[:, continuous] = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)

    factors: Factors = {}
    groups = {}
    correlated = []
    for factor in built['factors']:
        name = factor['name']
        z = np.array(factor['z'], dtype=np.float64)  # (n, q) as lme4 builds it
        codes = np.array(factor['codes']) - 1  # (n,) R counts levels from 1
        factors[name] = (z, codes)
        groups[name] = factor['levels']
        # `(1 + x || g)` is two terms of g, whose random effects lme4 leaves uncorrelated
        if factor['terms'] == 1:
            correlated.append(name)
    return {
        'rows': np.array(built['rows']) - 1,  # (n,) positions in the table
        'columns': built['columns'],
        'matrix': matrix,
        'offset': np.array(built['offset'], dtype=np.float64),  # (n,)
        'factors': factors,
        'groups': groups,
        'correlated': correlated,
    }


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
    family: str, blocks: list[Block], correlated: list[str], rng: np.random.Generator
) -> Prior:
    """The joint draw's prior over the blocks, each factor's random effects correlated
    as lme4 fits them: an LKJ for a factor in one term, independent for `(x || g)`."""
    hyperprior = Glmm(family=family, correlated_share=1.0)
    prior = hyperprior.resolveDataset(rng).drawPrior(blocks, rng)
    lkj_eta = {}
    for factor, shape in prior.lkj_eta.items():
        lkj_eta[factor] = shape if factor in correlated else None
    return dataclasses.replace(prior, lkj_eta=lkj_eta)


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


def collect(runs: Path) -> dict[str, pd.DataFrame]:
    """Each arm's records on each task until REFITS refits have estimates (all DRAWS if fewer do),
    in task order, and each task's real refit; raises where a draw it keeps never ran."""
    fits = pd.read_csv(FITS)
    records = []
    real = []
    for task in fits['task'].dropna().astype(int):
        dir = runs / str(task)
        real.append(json.loads((dir / 'real' / RECORD).read_text()))
        for arm in ARMS:
            estimated = 0
            for draw in range(DRAWS):
                record = _readRecord(dir, arm, draw)
                records.append(record)
                estimated += record['fit_status'] in ESTIMATED
                if estimated == REFITS:
                    break
    return {'records': pd.DataFrame(records), 'real': pd.DataFrame(real)}


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
                        help='the candidates to compare')
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
            fits.to_csv(FITS, index=False)
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
            collected = collect(args.runs)
            # gzip's header at a fixed time, so that a rerun writes the same bytes
            compression = {'method': 'gzip', 'mtime': 0}
            collected['records'].round(4).to_csv(
                HERE / 'registry.csv.gz', index=False, compression=compression
            )
            collected['real'].round(4).to_csv(HERE / 'registry-real.csv', index=False)
