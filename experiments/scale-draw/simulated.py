"""simulated.py compares the scale draws of a GLMM's prior on simulated datasets (#76).

A prior's scales decide how much of y's variance eta explains, how it splits among the blocks,
and how often eta saturates the logit and the log link.
The registry fixes each table's design, so only simulated datasets show how a scale draw behaves
as the design grows: each cell sets one size of the grouped preset to a point and draws the rest.
Every measure is read off the ground truth, so no fit stands between a prior and its curve.
simulated.ipynb reads the records and draws the curves.

pipeline:
---------
the arms, the survey's data directory and the datasets per cell
-> each arm's candidate                                                 buildCandidate
-> each cell's SCM hyperprior and GLMM overrides, one size at a point    _buildHyperpriors
-> per dataset, the covariates                                           drawDataset
-> per family, the joint prior over the covariates' blocks               modelOutcome
-> per arm, the prior at its scales and the parameters drawn from it     _measureArms
-> eta's saturation and sd, and the variance partition                   measureTruth
-> simulated.csv.gz, one row per cell, dataset, family and arm, with what the prior drew
   (simulated-light.csv.gz under light tails)

Run from the repository root, once with and once without --light-tails:

    uv run python experiments/scale-draw/simulated.py --survey <the survey's data directory>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from candidates import ARMS, FAMILIES, Candidate, Factors, buildCandidate, measureTruth

from tabdgp import Covariates, Glmm, LogUniformInt, Scm, UniformInt, drawDataset, modelOutcome

# =====================================================================================
# the cells
# =====================================================================================

SEED = 2026
SWEEP = {
    'rank': [1, 2, 4, 8, 16, 32],  # d, the design matrix's, the intercept included
    'factors': [1, 2, 3],
    'random_effects': [1, 3, 6],  # Glmm's, the most design columns a factor's terms take
}
GROUPED = Scm.grouped()  # the preset's sizes, whose tops a raised floor keeps
ICC_COLUMNS = max(SWEEP['factors'])  # an ICC per factor of the widest cell
# the GLMM overrides of a light-tailed run, where each budget holds for Normal effects
LIGHT_TAILS = {
    'beta_family_shares': {'normal': 1.0, 'student_t': 0.0, 'cauchy': 0.0},
    'sigma_alpha_family_shares': {
        'half_normal': 1.0,
        'half_student_t': 0.0,
        'half_cauchy': 0.0,
        'exponential': 0.0,
    },
}

# =====================================================================================
# the sweep
# =====================================================================================


def sweep(candidates: dict[str, Candidate], datasets: int, tails: dict) -> pd.DataFrame:
    """Every arm's measures, one row per cell, dataset, family and arm,
    with the GLMM overrides `tails` in every cell.
    """
    records = []
    for field, values in SWEEP.items():
        for value in values:
            hyperpriors = _buildHyperpriors(field, value)
            overrides = {**hyperpriors['glmm'], **tails}
            cell = {'field': field, 'value': value}
            for index in range(datasets):
                covariates = drawDataset(SEED, index, hyperpriors['scm']).covariates
                dataset_records = _measureArms(cell, index, covariates, overrides, candidates)
                records.extend(dataset_records)
    return pd.DataFrame(records)


def _buildHyperpriors(field: str, value: int) -> dict:
    """The cell's SCM hyperprior and GLMM overrides, one size set to a point."""
    match field:
        case 'rank':
            # n >= 2d at the widest point, the same rows at every point
            rows = LogUniformInt(2 * max(SWEEP['rank']), GROUPED.rows.high)
            scm = {'design_columns': value - 1, 'rows': rows}  # besides the intercept
            glmm = {}
        case 'factors':
            scm = {'factors': value}
            glmm = {}
        case 'random_effects':
            # a point q above the design's width fails, so every design fits the widest point
            narrowest = max(SWEEP['random_effects']) - 1  # without the intercept
            scm = {'design_columns': UniformInt(narrowest, GROUPED.design_columns.high)}
            glmm = {'random_effects': value}
    # the covariates alone, the GLMM draws y
    return {'scm': Scm.grouped(outcome_source='none', **scm), 'glmm': glmm}


def _measureArms(
    cell: dict,
    index: int,
    covariates: Covariates,
    overrides: dict,
    candidates: dict[str, Candidate],
) -> list[dict]:
    """Each family's joint prior rescaled by each arm and measured, one row per family and arm."""
    design = covariates.toDesign()
    records = []
    for family in FAMILIES:
        joint = modelOutcome(SEED, index, covariates, Glmm(family=family, **overrides))
        factors: Factors = {}
        for factor, terms in joint.random_terms.items():
            positions = [design.names.index(term) for term in terms]
            factors[factor] = (design.matrix[:, positions], covariates.grouping.codes[factor])
        for arm, candidate in candidates.items():
            # every arm draws its parameters from the same stream, so arms differ by their scales
            prior = candidate.rescale(joint.prior, family, np.random.default_rng([SEED, index, 1]))
            parameters = prior.drawParameters(np.random.default_rng([SEED, index, 2]))
            measures = measureTruth(family, design.matrix, factors, parameters, ICC_COLUMNS)
            # what the prior drew, so that a curve splits by tail, τ or the intercept
            drawn = {
                'beta_family': prior.beta_family,
                'sigma_alpha_family': prior.sigma_alpha_family,
                'predictor_sd': prior.predictor_sd,
                'intercept': float(parameters.beta[0]),
            }
            records.append(
                {**cell, 'index': index, 'family': family, 'arm': arm, **drawn, **measures}
            )
    return records


# =====================================================================================
# main
# =====================================================================================

if __name__ == '__main__':
    # fmt: off
    parser = argparse.ArgumentParser(description='Compares the scale draws on simulated datasets.')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS,
                        help='the candidates to compare')
    parser.add_argument('--survey', type=Path,
                        help="the practice survey's data directory, read by the survey arm")
    parser.add_argument('--datasets', type=int, default=1000, help='per cell')
    parser.add_argument('--light-tails', action='store_true',
                        help='Normal β and half-normal σ_α only, written to simulated-light.csv.gz')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent,
                        help='the directory of the records')
    # fmt: on
    args = parser.parse_args()
    if 'survey' in args.arms and args.survey is None:
        parser.error('the survey arm reads --survey')

    candidates = {}
    for arm in args.arms:
        candidates[arm] = buildCandidate(arm, args.survey)
    tails = {}
    name = 'simulated.csv.gz'
    if args.light_tails:
        tails = LIGHT_TAILS
        name = 'simulated-light.csv.gz'
    records = sweep(candidates, args.datasets, tails)
    # gzip's header at a fixed time, so that a rerun writes the same bytes
    compression = {'method': 'gzip', 'mtime': 0}
    records.round(4).to_csv(args.out / name, index=False, compression=compression)
