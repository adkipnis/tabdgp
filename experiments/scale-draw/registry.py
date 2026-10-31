"""registry.py compares the scale draws of a GLMM's prior on the registry's designs (#77).

simulated.py reads each measure off the ground truth on TabDGP's own covariates.
Here each registry fit lends its design (X and Z, as mixedbench's contract codes them),
each candidate draws a y on it, and lme4 refits that y as the registry fitted the real one,
so the refits and the registry's fits are both lme4 estimates.
The ground truth is logged beside each refit.
registry.ipynb reads the records against the registry's fits.

pipeline:
---------
the registry's fits
-> a task per kept fit, a reason per dropped one                      select, _screenFit
-> registry-fits.csv, the array job's tasks
per task:
-> its problem's standardized design, blocks, table and formula       run, _readDesign
-> per draw, the joint prior, each factor correlated as fitted        _drawJointPrior
-> per arm, the prior at its scales, the parameters, y
   and the ground truth's measures                                     measureTruth
-> lme4's refit of y and its measures                                  _measureFit
-> records.json in the task's run directory
-> registry.csv.gz, one row per task, draw and arm                    collect

Run from the repository root: select once, the tasks as a cluster array job
(registry.sbatch), then collect:

    uv run --group registry python experiments/scale-draw/registry.py select --cache <dir>
    uv run --group registry python experiments/scale-draw/registry.py run <task> \
        --cache <dir> --survey <dir>
    uv run --group registry python experiments/scale-draw/registry.py collect
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from candidates import ARMS, Candidate, Factors, buildCandidate, buildEta, measureTruth
from mixedbench import frequentist, registry
from mixedbench.contract import Problem, Refused
from mixedbench.frequentist import Fit
from mixedbench.references.handoff import Handoff
from mixedbench.registry.row import Row, readCsv

from tabdgp import Glmm, likelihood
from tabdgp.prior import Block, Prior

# =====================================================================================
# the fits and the draws
# =====================================================================================

HERE = Path(__file__).parent
FITS = HERE / 'registry-fits.csv'  # the array job's tasks
SEED = 2026
DRAWS = 10  # per fit and arm
TIMEOUT = 30 * 60  # seconds per refit
ESTIMATED = ['ok', 'singular', 'warning']  # the statuses of a fit with estimates
CALL = re.compile(r'[A-Za-z_.][A-Za-z0-9_.]*\s*\(')  # a function call in a formula

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
    # the contract codes the raw columns, not log(y), I(x^2) or factor(year)
    if CALL.search(row.formula):
        return {'reason': 'a transform in the formula'}
    try:
        problem = Problem.build(row, tables.load(row))
    except Refused as refused:
        return {'reason': f'refused: {refused.reason}'}
    # a fit of a subset or of weighted copies of the rows, which the contract leaves out
    if len(problem.table) != int(fit['n_obs']):
        return {'reason': "rows unlike the fit's"}
    random_effects = 0
    for factor in problem.factors:
        random_effects += 1 + len(problem.slopes(factor))  # the intercept and the slopes
    return {
        'reason': None,
        'rows': len(problem.table),
        'rank': len(problem.design),  # d, the intercept included
        'factors': len(problem.factors),
        'random_effects': random_effects,
    }


# =====================================================================================
# a task
# =====================================================================================


def run(task: int, tables: registry.Tables, candidates: dict[str, Candidate], runs: Path) -> None:
    """Draws each arm's y on the task's fit DRAWS times and refits each by lme4,
    the records in the task's run directory, which replace an earlier run's."""
    fits = pd.read_csv(FITS)
    kept = fits[fits['task'].notna()]
    [id] = kept.loc[kept['task'] == task, 'id']
    rows = {row.id: row for row in registry.rows()}
    problem = Problem.build(rows[id], tables.load(rows[id]))
    family = problem.family
    design = _readDesign(problem)
    icc_columns = int(kept['factors'].max())  # an ICC per factor of the widest fit
    dir = runs / str(task)
    if dir.exists():  # no file of an earlier run stays beside this run's records
        shutil.rmtree(dir)

    records = []
    for draw in range(DRAWS):
        streams = [SEED, task, draw]
        joint = _drawJointPrior(problem, design['blocks'], np.random.default_rng([*streams, 0]))
        for arm, candidate in candidates.items():
            # every arm draws from the same streams, so arms differ by their scales
            prior = candidate.rescale(joint, family, np.random.default_rng([*streams, 1]))
            parameters = prior.drawParameters(np.random.default_rng([*streams, 2]))
            eta = buildEta(design['matrix'], design['factors'], parameters)  # (n,)
            scale = None
            if parameters.sigma_eps is not None:
                scale = np.full(len(eta), parameters.sigma_eps)  # (n,)
            # y's three families have no shape
            y = likelihood.draw(family, eta, scale, None, np.random.default_rng([*streams, 3]))
            truth = measureTruth(
                family, design['matrix'], design['factors'], parameters, icc_columns
            )

            # lme4's refit, its files kept but the table, which the streams draw again
            refit_dir = dir / f'{arm}-{draw}'
            table = design['table'].assign(y=y)
            fitted = frequentist.fit(table, design['formula'], family, refit_dir, TIMEOUT)
            (refit_dir / 'table.csv').unlink()
            measured = _measureFit(fitted, icc_columns)

            # what the prior drew, so that a margin splits by tail, τ or the intercept,
            # and y's mean, which flags a y of a few ones or counts that lme4 cannot refit
            record = {
                'task': task,
                'draw': draw,
                'arm': arm,
                'beta_family': prior.beta_family,
                'sigma_alpha_family': prior.sigma_alpha_family,
                'predictor_sd': prior.predictor_sd,
                'intercept': float(parameters.beta[0]),
                'y_mean': float(y.mean()),
            }
            for name, value in truth.items():
                record[f'truth_{name}'] = value
            for name, value in measured.items():
                record[f'fit_{name}'] = value
            records.append(record)
    (dir / 'records.json').write_text(json.dumps(records, indent=2))


def _readDesign(problem: Problem) -> dict:
    """The standardized design matrix, each factor's random columns and codes, the blocks
    TabDGP's draw builds from them, and the table and formula lme4 refits."""
    standardized = problem.standardized('full')  # continuous columns z-scored over the rows
    matrix = standardized[list(problem.design)].to_numpy(dtype=np.float64)  # (n, d)
    # a fixed effect adds its column's variance to var(eta) at scale 1 (the intercept nothing)
    blocks = [Block(name='fixed', variances=matrix.var(axis=0).tolist())]
    factors: Factors = {}
    for factor in problem.factors:
        # the intercept, then the slopes the reference formula gives the factor
        positions = [0]
        for slope in problem.slopes(factor):
            positions.append(problem.design.index(slope))
        z = matrix[:, positions]  # (n, q)
        factors[factor] = (z, standardized[factor].to_numpy())
        # a random effect adds E[z²] to var(eta) at scale 1 (the intercept 1)
        variances = (z**2).mean(axis=0)  # (q,)
        groups = len(problem.levels[factor])
        blocks.append(Block(name=factor, variances=variances.tolist(), groups=groups))

    # the reference formula on aliases R reads as plain names (y, x1, g1, ...)
    handoff = Handoff(problem)
    table = standardized[list(handoff.aliases.values())].set_axis(list(handoff.aliases), axis=1)
    return {
        'matrix': matrix,
        'factors': factors,
        'blocks': blocks,
        'table': table,
        'formula': handoff._formula(),
    }


def _drawJointPrior(problem: Problem, blocks: list[Block], rng: np.random.Generator) -> Prior:
    """The joint draw's prior over the blocks, each factor's random effects correlated
    as the reference formula fits them: an LKJ for `(x | g)`, independent for `(x || g)`."""
    hyperprior = Glmm(family=problem.family, correlated_share=1.0)
    prior = hyperprior.resolveDataset(rng).drawPrior(blocks, rng)
    lkj_eta = {}
    for factor, shape in prior.lkj_eta.items():
        lkj_eta[factor] = shape if factor in problem.correlations else None
    return dataclasses.replace(prior, lkj_eta=lkj_eta)


def _measureFit(fitted: Fit, icc_columns: int) -> dict:
    """How the refit ended, its sd of eta and its variance partition as fits.csv reads a fit's,
    an ICC per factor in `icc_columns` columns, NaN where the fit has no estimates."""
    measures = {
        'status': fitted.status,
        'seconds': fitted.seconds,
        'message': fitted.message,
        'sd_eta': math.nan,
        'r2_marginal': math.nan,
        'r2_conditional': math.nan,
        'fixed_share': math.nan,
    }
    for k in range(icc_columns):
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
    measures['r2_marginal'] = fitted.sigma2_fixed / total
    measures['r2_conditional'] = explained / total
    # a singular intercept-only fit explains nothing, so it has no fixed share
    if explained > 0:
        measures['fixed_share'] = fitted.sigma2_fixed / explained
    # lme4 orders the factors by their groups, so each is read by its alias g1, g2, ...
    for k in range(len(fitted.variances)):
        measures[f'icc_{k + 1}'] = fitted.variances[f'g{k + 1}'] / total
    return measures


# =====================================================================================
# the records
# =====================================================================================


def collect(runs: Path) -> pd.DataFrame:
    """Every task's records, in task order; raises where a task never ran."""
    fits = pd.read_csv(FITS)
    records = []
    for task in fits['task'].dropna().astype(int):
        records += json.loads((runs / str(task) / 'records.json').read_text())
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
            run(args.task, registry.Tables(args.cache), candidates, args.runs)
        case 'collect':
            records = collect(args.runs)
            # gzip's header at a fixed time, so that a rerun writes the same bytes
            compression = {'method': 'gzip', 'mtime': 0}
            records.round(4).to_csv(HERE / 'registry.csv.gz', index=False, compression=compression)
