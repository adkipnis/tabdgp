"""simulated.py compares the scale draws of a GLMM's prior on simulated datasets (#76).

A prior's scales decide how much of y's variance eta explains, how it splits among the blocks,
and how often eta saturates the logit and the log link.
The registry fixes each table's design, so only simulated datasets show how a scale draw behaves
as the design grows: each cell sets one size of the grouped preset to a point and draws the rest.
Every measure is read off the ground truth, so no fit stands between a prior and its curve.

pipeline:
---------
the arms, the survey's data directory and the datasets per cell
-> each cell's SCM hyperprior and GLMM overrides, one size at a point     _buildHyperpriors
-> per dataset, the covariates                                           drawDataset
-> per family, the joint prior over the covariates' blocks               modelOutcome
-> per arm, the prior at its scales and the parameters drawn from it     _measureArms
-> eta's saturation and the variance partition                           _measure
-> per family, each measure's curves over each swept size as a PDF       _plotCurves
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from candidates import ARMS, Candidate, buildCandidate

from tabdgp import Covariates, Glmm, LogUniformInt, Scm, UniformInt, drawDataset, modelOutcome
from tabdgp.column import Grouping
from tabdgp.covariates import Design
from tabdgp.prior import Parameters

SEED = 2026
FAMILIES = ['gaussian', 'bernoulli', 'poisson']
SWEEP = {
    'design_columns': [1, 2, 4, 8, 16, 32],
    'factors': [1, 2, 3],
    'random_effects': [1, 3, 6],  # Glmm's, the most design columns a factor's terms take
}
GROUPED = Scm.grouped(outcome_source='none')  # the covariates alone, the GLMM draws y
LOGIT_BOUND = 10  # some |eta| above it saturates the logit
RATE_BOUND = 1e6  # some Poisson rate above it saturates the log link
MEASURES = {
    'saturated': 'saturated share',
    'r2_marginal': '$R^2_m$',
    'r2_conditional': '$R^2_c$',
    'fixed_share': '$R^2_m / R^2_c$',
    'icc_1': 'ICC of factor 1',
    'icc_2': 'ICC of factor 2',
    'icc_3': 'ICC of factor 3',
}
# three hues (joint, R2D2M2, budget-free), a dashed line for each one's second variant
ARM_STYLES = {
    'joint': {'color': '#2a78d6', 'linestyle': '-'},
    'r2d2m2-0.5': {'color': '#eb6834', 'linestyle': '-'},
    'r2d2m2-1': {'color': '#eb6834', 'linestyle': '--'},
    'survey': {'color': '#1baf7a', 'linestyle': '-'},
    'unit': {'color': '#1baf7a', 'linestyle': '--'},
}

# =====================================================================================
# the sweep
# =====================================================================================


def sweep(candidates: dict[str, Candidate], datasets: int) -> pd.DataFrame:
    """Every arm's measures, one row per cell, dataset and family."""
    records = []
    for field, values in SWEEP.items():
        for value in values:
            hyperpriors = _buildHyperpriors(field, value)
            for index in range(datasets):
                covariates = drawDataset(SEED, index, hyperpriors['scm']).covariates
                arms = _measureArms(covariates, index, hyperpriors['glmm'], candidates)
                for record in arms:
                    records.append({'field': field, 'value': value, 'index': index, **record})
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
    return {'scm': Scm.grouped(outcome_source='none', **scm), 'glmm': glmm}


def _measureArms(
    covariates: Covariates, index: int, overrides: dict, candidates: dict[str, Candidate]
) -> list[dict]:
    """Each family's joint prior on the covariates, rescaled by each arm, and its measures."""
    design = covariates.toDesign()
    records = []
    for family in FAMILIES:
        joint = modelOutcome(SEED, index, covariates, Glmm(family=family, **overrides))
        for arm, candidate in candidates.items():
            # every arm draws its parameters from the same stream, so arms differ by their scales
            prior = candidate.rescale(joint.prior, family, np.random.default_rng([SEED, index, 1]))
            parameters = prior.drawParameters(np.random.default_rng([SEED, index, 2]))
            measures = _measure(family, design, covariates.grouping, joint.random_terms, parameters)
            records.append({'family': family, 'arm': arm, **measures})
    return records


def _measure(
    family: str,
    design: Design,
    grouping: Grouping,
    random_terms: dict[str, list[str]],
    parameters: Parameters,
) -> dict[str, float]:
    """eta's saturation and the variance partition on the link scale
    (Nakagawa, Johnson and Schielzeth 2017), as the registry reads it off a fit.
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
        sigma2_factors.append(float(((z @ covariance) * z).sum(axis=1).mean()))
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
            # ln(1 + 1/λ), λ = exp(β_0 + σ²_0 / 2) with σ²_0 the random intercepts' variances
            intercept_variance = 0.0
            for sd in parameters.sigma_alpha.values():
                intercept_variance += float(sd[0]) ** 2
            log_rate = float(parameters.beta[0]) + intercept_variance / 2
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
# the curves
# =====================================================================================


def _plotCurves(records: pd.DataFrame, family: str, path: Path) -> None:
    """One family's measures over each swept size: per arm the median and its 10-90 % band
    (the saturated share alone, a share of datasets)."""
    table = records[records['family'] == family]
    measures = []
    for measure in MEASURES:
        if table[measure].notna().any():  # a Gaussian y never saturates
            measures.append(measure)
    figure, axes = plt.subplots(
        len(measures),
        len(SWEEP),
        figsize=(11, 2.1 * len(measures)),
        sharey='row',
        squeeze=False,
        layout='constrained',
    )
    for row, measure in enumerate(measures):
        for column, field in enumerate(SWEEP):
            ax = axes[row, column]
            cells = table[table['field'] == field].groupby(['arm', 'value'])[measure]
            for arm in table['arm'].unique():
                # a marker per cell, so a measure with one cell (a third factor's ICC) shows
                style = {
                    **ARM_STYLES[arm],
                    'label': arm,
                    'linewidth': 2,
                    'marker': 'o',
                    'markersize': 4,
                }
                if measure == 'saturated':
                    ax.plot(cells.mean()[arm], **style)
                else:
                    ax.plot(cells.median()[arm], **style)
                    low = cells.quantile(0.1)[arm]
                    high = cells.quantile(0.9)[arm]
                    ax.fill_between(low.index, low, high, color=style['color'], alpha=0.12)
            ax.set_xticks(SWEEP[field])
            if field == 'design_columns':
                ax.set_xscale('log', base=2)
                ax.set_xticks(SWEEP[field], [str(value) for value in SWEEP[field]])
            ax.grid(color='#e4e3df', linewidth=0.6)
            ax.spines[['top', 'right']].set_visible(False)
            if row == len(measures) - 1:
                ax.set_xlabel(field)
            if column == 0:
                ax.set_ylabel(MEASURES[measure])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=len(labels), frameon=False)
    figure.suptitle(f'{family} y')
    figure.savefig(path, metadata={'CreationDate': None})
    plt.close(figure)


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
                        help='the directory of the PDFs')
    # fmt: on
    args = parser.parse_args()
    if 'survey' in args.arms and args.survey is None:
        parser.error('the survey arm reads --survey')

    candidates = {}
    for arm in args.arms:
        candidates[arm] = buildCandidate(arm, args.survey)
    records = sweep(candidates, args.datasets)
    for family in FAMILIES:
        _plotCurves(records, family, args.out / f'simulated-{family}.pdf')
