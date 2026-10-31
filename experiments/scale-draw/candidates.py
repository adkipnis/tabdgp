"""candidates.py defines the scale draws a GLMM's prior is compared under (#76, #77).

TabDGP draws a GLMM's prior scales jointly from one budget, the variance of eta.
Its rivals are R2D2M2, which spreads an R²-induced budget over every coefficient by one Dirichlet,
and the budget-free scales a user states per block.
Each candidate takes the prior `Glmm.drawPrior` drew and replaces its scales
(the slopes', σ_α's and σ_ε's) before the parameters are drawn from it.
The families, the intercept, the slopes' offsets δ_j, the LKJ shapes and the random terms
stay the joint draw's, so two candidates differ by their scales alone.
A slope's location follows its new scale (δ_j times it),
and τ, the split and r are recomputed from what the new scales spend.

pipeline:
---------
an arm
-> its candidate                                                    buildCandidate
-> per dataset, the joint prior, y's family and a generator
-> the scales, by candidate                                         Joint, R2d2m2, Stated
   -> one survey model's stated scales, a blank block filled        Survey.drawModel
   -> what each coefficient adds to var(eta) at scale 1, by block   _measureVariances
-> the prior at those scales, with the budget fields they realize   _replaceScales
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import ClassVar, Literal, get_args

import numpy as np
import pandas as pd
from pydantic import ConfigDict, PositiveFloat
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.field import Distribution
from tabdgp.prior import Glmm, Prior

# =====================================================================================
# the arms and y's families
# =====================================================================================

Arm = Literal['joint', 'r2d2m2-0.5', 'r2d2m2-1', 'survey', 'unit']
ARMS: list[Arm] = list(get_args(Arm))
Family = Literal['gaussian', 'bernoulli', 'poisson']  # the families whose y's variance is known
FAMILIES: list[Family] = list(get_args(Family))

# =====================================================================================
# the practice survey
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Survey:
    """The scales the practice survey's models state, one row per model that states one."""

    BLOCKS: ClassVar[list[str]] = ['beta', 'sd', 'dispersion']  # the blocks a candidate reads
    FLAT_SCALE: ClassVar[int] = 100  # a stated scale this large is flat, read as blank

    table: pd.DataFrame  # a column per block in BLOCKS, NaN where the model states none

    @classmethod
    def fromDir(cls, data_dir: Path) -> Survey:
        """The stated scales below FLAT_SCALE, σ_ε's only from a Normal model's dispersion."""
        priors = pd.read_csv(data_dir / 'priors.csv')
        models = pd.read_csv(data_dir / 'papers.csv')
        # sd_normalized holds the scale as an sd where it was computable, else words
        priors['scale'] = pd.to_numeric(priors['sd_normalized'], errors='coerce')
        table = priors.pivot(index=['paper_id', 'model_index'], columns='block', values='scale')
        table = table[cls.BLOCKS]
        # another family's dispersion is a shape, not σ_ε's scale
        family = models.set_index(['paper_id', 'model_index'])['family']
        normal = family.reindex(table.index) == 'Normal'
        table['dispersion'] = table['dispersion'].where(normal)
        table = table.where(table < cls.FLAT_SCALE)
        stating = table.notna().any(axis=1)
        return cls(table=table[stating].reset_index(drop=True))

    def drawModel(self, rng: np.random.Generator) -> dict[str, float]:
        """One model's stated scale per block, a blank one drawn from the block's stated scales."""
        row = self.table.iloc[rng.integers(len(self.table))]
        scales = {}
        for block in self.BLOCKS:
            scale = row[block]
            if math.isnan(scale):
                stated = self.table[block].dropna()
                scale = stated.iloc[rng.integers(len(stated))]
            scales[block] = float(scale)
        return scales


# =====================================================================================
# the candidates
# =====================================================================================


@pydantic_dataclass(frozen=True)
class Joint:
    """The joint draw, the prior as `Glmm.drawPrior` drew it."""

    def rescale(self, prior: Prior, family: Family, rng: np.random.Generator) -> Prior:
        return prior


@pydantic_dataclass(frozen=True)
class R2d2m2:
    """The R2D2M2 prior (Aguilar and Bürkner 2023, eq. 7 and 10-11):
    R² drawn afresh from the joint draw's prior on r, τ² = R² / (1 − R²) and σ² by y's family,
    φ ~ Dirichlet(a) over every slope and random term,
    and coefficient j's scale √(σ² τ² φ_j / v_j), so that j adds σ² τ² φ_j to var(eta).
    The joint draw's split and gate are dropped, so a block's share follows its count of terms.
    """

    SIGNAL_SHARE: ClassVar[Distribution] = Glmm.model_fields['signal_share'].default  # R²'s prior

    concentration: PositiveFloat  # Dirichlet

    def rescale(self, prior: Prior, family: Family, rng: np.random.Generator) -> Prior:
        signal_share = self.SIGNAL_SHARE.draw(rng)  # R²

        # σ², y's variance around eta, linearized at the intercept's location on the logit
        # and the log link (Yanchenko, Bondell and Reich 2024, eq. 9-10)
        intercept_location = prior.beta_locations[0]
        sigma_eps_scale = None
        match family:
            case 'gaussian':
                noise = 1 - signal_share  # y at unit variance, as under the joint draw
                sigma_eps_scale = math.sqrt(noise)
            case 'bernoulli':
                p0 = 1 / (1 + math.exp(-intercept_location))
                noise = 1 / (p0 * (1 - p0))
            case 'poisson':
                noise = math.exp(-intercept_location)
        budget = noise * signal_share / (1 - signal_share)  # σ² τ², var(eta) without the intercept

        # one Dirichlet over every coefficient's share of the budget, the intercept outside
        variances = _measureVariances(prior)
        coefficients = sum(len(block_variances) for block_variances in variances.values())
        shares = rng.dirichlet([self.concentration] * coefficients)  # (d − 1 + Σ_k q_k,)
        scales = {}
        start = 0
        for name, block_variances in variances.items():
            block_shares = shares[start : start + len(block_variances)]
            scales[name] = np.sqrt(budget * block_shares / block_variances)
            start += len(block_variances)

        # the joint draw's families stay, where the paper has a Normal β and σ_α at its scale,
        # so heavy tails overshoot this budget as they overshoot the joint's
        return _replaceScales(prior, scales, sigma_eps_scale)


@pydantic_dataclass(frozen=True)
class Stated:
    """The budget-free draw: each block at the scale a user states
    (one survey model's per dataset, 1 everywhere without a survey),
    and σ_ε's scale at the stated dispersion.
    A block adds what it would if every coefficient had the stated scale, whatever d and q.
    """

    # a stand-in for independent scales per coefficient, since var(eta) grows alike with each term
    # (more spread across datasets, as one scale moves a whole block, and the same on every link)
    UNIT_SCALES: ClassVar[dict[str, float]] = {'beta': 1.0, 'sd': 1.0, 'dispersion': 1.0}

    survey: Survey | None = None

    def rescale(self, prior: Prior, family: Family, rng: np.random.Generator) -> Prior:
        stated = self.UNIT_SCALES
        if self.survey is not None:
            stated = self.survey.drawModel(rng)

        # the slopes at β's scale, each factor's terms at the sd's, the intercept the joint's
        variances = _measureVariances(prior)
        fixed, *factors = prior.blocks
        scales = {
            fixed.name: self._spreadScale(
                prior.beta_scales[1:], variances[fixed.name], stated['beta']
            )
        }
        for block in factors:
            scales[block.name] = self._spreadScale(
                prior.sigma_alpha_scales[block.name], variances[block.name], stated['sd']
            )
        sigma_eps_scale = None
        if prior.sigma_eps_scale is not None:
            sigma_eps_scale = stated['dispersion']
        return _replaceScales(prior, scales, sigma_eps_scale)

    @staticmethod
    def _spreadScale(joint_scales: list[float], variances: np.ndarray, scale: float) -> np.ndarray:
        """A block's scales at one stated scale, spread by the joint draw's weights (its gate),
        so the block adds to var(eta) what the stated scale would add if every coefficient shared it
        (each coefficient exactly at the stated scale when the gate's weights are constant).
        """
        joint = np.array(joint_scales)  # (d − 1,) or (q,)
        if joint.size == 0:
            return joint  # an intercept-only fixed block has no slope to scale
        spent = float((joint**2 * variances).sum())
        stated = scale**2 * float(variances.sum())
        return joint * math.sqrt(stated / spent)


Candidate = Joint | R2d2m2 | Stated


def buildCandidate(arm: Arm, survey_dir: Path | None) -> Candidate:
    """The candidate an arm names (the survey arm reads the practice survey's data directory)."""
    match arm:
        case 'joint':
            return Joint()
        case 'r2d2m2-0.5':
            return R2d2m2(concentration=0.5)
        case 'r2d2m2-1':
            return R2d2m2(concentration=1.0)
        case 'survey':
            return Stated(survey=Survey.fromDir(survey_dir))
        case 'unit':
            return Stated()


# =====================================================================================
# the scales a candidate replaces
# =====================================================================================


def _measureVariances(prior: Prior) -> dict[str, np.ndarray]:
    """What each coefficient adds to var(eta) at scale 1, by block (the intercept left out)."""
    fixed, *factors = prior.blocks
    offsets = _recoverOffsets(prior)
    # E[β_j²] = τ_j² (1 + δ_j²) at a Normal β, as the joint draw spends it
    variances = {fixed.name: (1 + offsets**2) * np.array(fixed.variances[1:])}  # (d − 1,)
    for block in factors:
        variances[block.name] = np.array(block.variances)  # (q,) E[z²], the intercept's 1 first
    return variances


def _recoverOffsets(prior: Prior) -> np.ndarray:
    """Each slope's offset δ_j = ν_j / τ_j (0 for a slope without a location)."""
    locations = np.array(prior.beta_locations[1:])
    scales = np.array(prior.beta_scales[1:])
    return locations / scales  # (d − 1,)


def _replaceScales(
    prior: Prior, scales: dict[str, np.ndarray], sigma_eps_scale: float | None
) -> Prior:
    """The prior at a candidate's scales by block, with the budget fields they realize (τ, φ, r)."""
    fixed, *factors = prior.blocks

    # each block's part of var(eta) at a Normal draw, and y's share at the scales
    variances = _measureVariances(prior)
    spent = {}
    for name, block_scales in scales.items():
        spent[name] = float((block_scales**2 * variances[name]).sum())
    predictor_variance = sum(spent.values())
    split = {}
    for name, part in spent.items():
        split[name] = part / predictor_variance
    signal_share = None
    if sigma_eps_scale is not None:
        signal_share = predictor_variance / (predictor_variance + sigma_eps_scale**2)

    # a slope's location stays its offset times its scale
    slope_scales = scales[fixed.name]
    slope_locations = _recoverOffsets(prior) * slope_scales
    sigma_alpha_scales = {}
    for block in factors:
        sigma_alpha_scales[block.name] = scales[block.name].tolist()
    return dataclasses.replace(
        prior,
        beta_locations=[prior.beta_locations[0], *slope_locations.tolist()],
        beta_scales=[prior.beta_scales[0], *slope_scales.tolist()],
        sigma_alpha_scales=sigma_alpha_scales,
        sigma_eps_scale=sigma_eps_scale,
        signal_share=signal_share,
        predictor_sd=math.sqrt(predictor_variance),
        split=split,
    )
