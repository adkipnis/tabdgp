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
-> eta's saturation and the variance partition                           _measure
-> simulated.csv.gz, one row per cell, dataset, family and arm, with what the prior drew

Run from the repository root:

    uv run python experiments/scale-draw/simulated.py --survey <the survey's data directory>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from candidates import ARMS, FAMILIES, Candidate, Family, buildCandidate

from tabdgp import Covariates, Glmm, LogUniformInt, Scm, UniformInt, drawDataset, modelOutcome
from tabdgp.column import Grouping
from tabdgp.covariates import Design
from tabdgp.prior import Parameters

# =====================================================================================
# the cells and the bounds
# =====================================================================================

SEED = 2026
SWEEP = {
    'design_columns': [1, 2, 4, 8, 16, 32],
    'factors': [1, 2, 3],
    'random_effects': [1, 3, 6],  # Glmm's, the most design columns a factor's terms take
}
GROUPED = Scm.grouped()  # the preset's sizes, whose tops a raised floor keeps
LOGIT_BOUND = 10  # some |eta| above it saturates the logit
RATE_BOUND = 1e6  # some Poisson rate above it saturates the log link

# =====================================================================================
# the sweep
# =====================================================================================


def sweep(candidates: dict[str, Candidate], datasets: int) -> pd.DataFrame:
    """Every arm's measures, one row per cell, dataset, family and arm."""
    records = []
    for field, values in SWEEP.items():
        for value in values:
            hyperpriors = _buildHyperpriors(field, value)
            cell = {'field': field, 'value': value}
            for index in range(datasets):
                covariates = drawDataset(SEED, index, hyperpriors['scm']).covariates
                dataset_records = _measureArms(
                    cell, index, covariates, hyperpriors['glmm'], candidates
                )
                records.extend(dataset_records)
    return pd.DataFrame(records)


def _buildHyperpriors(field: str, value: int) -> dict:
    """The cell's SCM hyperprior and GLMM overrides, one size set to a point."""
    match field:
        case 'design_columns':
            # n >= 2d at the widest point, the same rows at every point
            rows = LogUniformInt(2 * max(SWEEP['design_columns']), GROUPED.rows.high)
            scm = {'design_columns': value, 'rows': rows}
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
        for arm, candidate in candidates.items():
            # every arm draws its parameters from the same stream, so arms differ by their scales
            prior = candidate.rescale(joint.prior, family, np.random.default_rng([SEED, index, 1]))
            parameters = prior.drawParameters(np.random.default_rng([SEED, index, 2]))
            measures = _measure(family, design, covariates.grouping, joint.random_terms, parameters)
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


def _measure(
    family: Family,
    design: Design,
    grouping: Grouping,
    random_terms: dict[str, list[str]],
    parameters: Parameters,
) -> dict[str, float]:
    """The saturation of eta and the variance partition on the link scale
    (Nakagawa, Johnson and Schielzeth 2017, as the registry reads it off a fit).
    """
    # the fixed part's variance, by n − 1 as the registry takes it
    eta = design.matrix @ parameters.beta  # (n,)
    sigma2_fixed = float(eta.var(ddof=1))

    # each factor's variance, the mean over rows of z'Σz (Johnson 2014), and its part of eta
    sigma2_factors = []
    for factor, terms in random_terms.items():
        positions = [design.names.index(term) for term in terms]
        z = design.matrix[:, positions]  # (n, q)
        sd = parameters.sigma_alpha[factor]
        covariance = parameters.corr_alpha[factor] * np.outer(sd, sd)  # (q, q)
        row_variances = ((z @ covariance) * z).sum(axis=1)  # (n,) z'Σz
        sigma2_factors.append(float(row_variances.mean()))
        alpha = grouping.broadcast(parameters.alpha[factor], factor, 'row')  # (n, q)
        eta = eta + (z * alpha).sum(axis=1)

    # y's variance around eta: the residual's, or the link's distribution-specific one
    match family:
        case 'gaussian':
            sigma2_noise = parameters.sigma_eps**2
            saturated = math.nan  # the identity link never saturates
        case 'bernoulli':
            sigma2_noise = math.pi**2 / 3
            saturated = float(np.abs(eta).max() > LOGIT_BOUND)
        case 'poisson':
            # ln(1 + 1/λ), λ the mean rate over the rows,
            # which the registry's null model estimates as exp(β_0 + σ²_0 / 2)
            # (summed in logs, so that a rate past float64 is no NaN)
            log_rate = float(np.logaddexp.reduce(eta)) - math.log(len(eta))
            sigma2_noise = float(np.logaddexp(0, -log_rate))
            saturated = float(eta.max() > math.log(RATE_BOUND))

    sigma2_random = sum(sigma2_factors)
    total = sigma2_fixed + sigma2_random + sigma2_noise
    measures = {
        'saturated': saturated,
        'r2_marginal': sigma2_fixed / total,
        'r2_conditional': (sigma2_fixed + sigma2_random) / total,
        'fixed_share': sigma2_fixed / (sigma2_fixed + sigma2_random),
    }
    for k in range(max(SWEEP['factors'])):
        measures[f'icc_{k + 1}'] = math.nan  # a factor the dataset lacks
        if k < len(sigma2_factors):
            measures[f'icc_{k + 1}'] = sigma2_factors[k] / total
    return measures


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
    parser.add_argument('--out', type=Path, default=Path(__file__).parent,
                        help='the directory of the records')
    # fmt: on
    args = parser.parse_args()
    if 'survey' in args.arms and args.survey is None:
        parser.error('the survey arm reads --survey')

    candidates = {}
    for arm in args.arms:
        candidates[arm] = buildCandidate(arm, args.survey)
    records = sweep(candidates, args.datasets)
    # gzip's header at a fixed time, so that a rerun writes the same bytes
    compression = {'method': 'gzip', 'mtime': 0}
    records.round(4).to_csv(args.out / 'simulated.csv.gz', index=False, compression=compression)
