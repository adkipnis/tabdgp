"""prior.py defines how we draw the parameters of a GLM or GLMM outcome.

A generalized linear model (GLM) links an outcome y to the linear predictor eta = Xβ
of a design matrix X by the canonical link of a likelihood family (e.g. Normal, Bernoulli).
A generalized linear mixed model (GLMM) adds random effects per grouping factor k,
eta = Xβ + Σ_k Z_k α_k, so each group gets its own additional intercept (and optionally slopes).
This module samples and returns the ground truth (parameters, hyperparameters).

Optionally, the linear predictor can be misspecified in a controlled way:
We add the structural causal model's (SCM) value for y
(standardized and scaled by λ > 0 times the sd of eta).
This way, y's value can be nonlinear in its causes and depend on unobserved latents.
β and α stay the truth, so a fitted GLMM is wrong by exactly the added term.

Parameters are drawn from priors, whose scales are drawn from fixed hyperpriors.
Independent prior scales would saturate many datasets (e.g. R² near 1),
but capping them would degrade the prior. Instead, the scales are drawn jointly per dataset:
- **Predictor sd τ:** the sd of eta without the intercept.
- **Split φ:** each term's signal, its share of var(eta):
  φ_fixed for Xβ, the rest split among Z_k α_k.
- **Prior scales:** set so that each term adds its share of var(eta).

This adapts the R2D2M2 prior (Aguilar and Bürkner, 2023),
without its Dirichlet over all effects (which rarely lets a random intercept
carry the signal in eta, which is observed in real datasets with strong grouping).

pipeline:
---------
the (optional) overrides
-> validation at construction                                 Glm, Glmm
-> a point for each field drawn once per dataset              Glm.resolveDataset
   (a factor resolves its budget and its LKJ shape later)
-> given y's family and the blocks, the families chosen,
   r, τ, φ, each block's scale and each factor's LKJ shape    _drawContext
-> β and σ_ε from their families at those scales              Glm.drawParameters
-> each factor's σ_α, Ω and α per group                       _drawRandomEffects
"""

from __future__ import annotations

import math
from typing import Annotated, ClassVar

import numpy as np
from pydantic import ConfigDict, Field, NonNegativeFloat, PositiveInt
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.field import (
    Beta,
    Config,
    LogUniform,
    NonNegative,
    Positive,
    PositiveCount,
    Real,
    Share,
    Simplex,
    Uniform,
    UniformInt,
)

# =====================================================================================
# what the draw reads and returns
# =====================================================================================


@pydantic_dataclass(frozen=True)
class Block:
    """A block of eta's coefficients drawn under one budget:
    fixed effects or a factor's random effects (one set per group).
    """

    name: str  # 'fixed', or the factor's
    # what each coefficient adds to var(eta) at scale 1, the intercept first:
    # a fixed effect its column's variance (the intercept's 0), a random effect its E[z²]
    variances: Annotated[list[NonNegativeFloat], Field(min_length=1)]
    groups: PositiveInt = 1  # m, the fixed effects' one set for all rows

    @property
    def width(self) -> int:
        return len(self.variances)

    @property
    def variance(self) -> float:
        """What the block's term adds to var(eta) at scale 1."""
        return sum(self.variances)


@pydantic_dataclass(frozen=True)
class PriorContext:
    """The resolved hyperparameters a GLM's or GLMM's parameters were drawn with.

    - Families: the family of each factor's random-effect sds and the residual sd's.
    - Signal: the signal share r and the predictor sd τ.
    - Split: each block's share φ of var(eta).
    - Scales: each block's prior scale, the intercept's and the residual's.
    - Correlation: each factor's LKJ shape, if its effects are correlated.
    """

    sd_alpha_family: str | None  # None on a GLM
    sigma_eps_family: str | None  # None unless the family has a residual
    signal_share: float | None  # r, None unless the family is Normal
    predictor_sd: float  # τ
    split: dict[str, float]  # φ by block name
    scales: dict[str, float]  # by block name
    intercept_scale: float  # σ_0
    residual_scale: float | None  # s_ε, None unless the family is Normal
    lkj_eta: dict[str, int | None]  # the LKJ shape by factor, None if its effects are independent


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Parameters:
    """A GLM's or GLMM's parameters on the standardized scale. All arrays are float32."""

    beta: np.ndarray  # (d,) intercept first
    sigma_eps: float | None  # None unless the likelihood family is Normal
    sd_alpha: dict[str, np.ndarray]  # (q,) by factor
    corr_alpha: dict[str, np.ndarray]  # (q, q) by factor
    alpha: dict[str, np.ndarray]  # (m, q) by factor


# =====================================================================================
# the config
# =====================================================================================

# the sd families, uniform: what a random-effect sd and the residual sd are drawn from
SD_FAMILY_SHARES = {
    'half_normal': 0.25,
    'half_student_t': 0.25,  # ν = 3
    'half_cauchy': 0.25,
    'exponential': 0.25,
}


class Glm(Config):
    """The prior of a generalized linear model (GLM) over the design matrix.

    - Families: y's response family, with its canonical link,
      the family of the fixed effects β and the family the residual sd σ_ε is drawn from.
    - Locations: the intercept's, on the link scale, and each slope's,
      zero for most slopes and a multiple of the slope's scale for the rest.
    - Scales: the predictor sd τ is the sd of eta without the intercept.
      Under the Normal family τ² = r, the signal share r being the share of y's variance
      that eta explains, so that y has unit variance at the scales.
      Each slope's scale is the block's times a log-normal weight, at spread 0 the block's.
      A scale is its family's scale parameter, so a draw's variance is the family's at it:
      E[σ_ε²] = s_ε² E[X²], X the standard law of σ_ε's family.
    - Deviation: the score added to eta at λ · sd(Xβ + Σ_k Z_k α_k),
      the population sd over the rows.
    """

    SHARES: ClassVar[tuple[str, ...]] = (
        'family_shares',
        'beta_family_shares',
        'sigma_eps_family_shares',
    )
    PER_FACTOR: ClassVar[tuple[str, ...]] = ()  # the fields each factor resolves on its own

    # ruff cannot trace Config to BaseModel, so it reads a shares default as a class attribute
    family_shares: Simplex = {'normal': 1 / 3, 'bernoulli': 1 / 3, 'poisson': 1 / 3}  # noqa: RUF012
    beta_family_shares: Simplex = {'normal': 0.68, 'student_t': 0.30, 'cauchy': 0.02}  # noqa: RUF012
    sigma_eps_family_shares: Simplex = SD_FAMILY_SHARES  # the family σ_ε is drawn from
    intercept_location: Real = Uniform(-3, 3)  # μ_0, on the link scale
    intercept_scale: NonNegative = LogUniform(0.1, 10)  # σ_0
    location_share: Share = 0.1  # the probability that a slope has a location
    location_scale: NonNegative = 2.0  # a located slope's location, in its own scale, is N(0, this)
    scale_spread: NonNegative = 0.0  # the sd of each slope's log scale around the block's
    signal_share: Share = Beta(2, 2)  # r
    predictor_sd: NonNegative = LogUniform(0.25, 4)  # τ, unless the family is Normal
    deviation_scale: NonNegative = 0.0  # λ

    def resolveDataset(self, rng: np.random.Generator) -> Glm:
        """Resolves each field drawn once per dataset, in declaration order."""
        points = {}
        for name in type(self).model_fields:
            if name in self.PER_FACTOR:
                points[name] = getattr(self, name)  # defer resolution to each factor
            else:
                points[name] = self.resolveField(name, rng)
        return type(self)(**points)

    def drawParameters(
        self, family: str, blocks: list[Block], rng: np.random.Generator
    ) -> tuple[PriorContext, Parameters]:
        """Draws the scales jointly, then each parameter from its family at its scale exactly.
        On a resolved Glm, for the response family chosen.
        """
        context = self._drawContext(family, blocks, rng)
        fixed, *factors = blocks
        intercept = rng.normal(0, context.intercept_scale, size=1)  # (1,)
        slopes = rng.normal(0, context.scales[fixed.name], size=fixed.width - 1)  # (d − 1,)
        beta = np.concatenate([intercept, slopes])  # (d,)
        sigma_eps = None
        if context.residual_scale is not None:
            sigma_eps = self._drawSd(context.sigma_eps_family, context.residual_scale, rng)
        sd_alpha, corr_alpha, alpha = self._drawRandomEffects(context, factors, rng)
        # the parameters are float32, the precision of the columns they are recorded with
        parameters = Parameters(
            beta=beta.astype(np.float32),
            sigma_eps=sigma_eps,
            sd_alpha=sd_alpha,
            corr_alpha=corr_alpha,
            alpha=alpha,
        )
        return context, parameters

    def _drawContext(
        self, family: str, blocks: list[Block], rng: np.random.Generator
    ) -> PriorContext:
        sigma_eps_family = None
        signal_share = None
        predictor_sd = self.predictor_sd
        residual_scale = None
        if family == 'normal':
            sigma_eps_family = self.sigma_eps_family_shares.choose(rng)
            signal_share = self.signal_share
            # τ² = r and s_ε² = 1 − r, so that y has unit variance at the scales
            predictor_sd = math.sqrt(self.signal_share)
            residual_scale = math.sqrt(1 - self.signal_share)
        sd_alpha_family = self._drawSdAlphaFamily(rng)
        split = self._drawSplit(blocks, rng)
        lkj_eta = self._drawCorrelations(blocks, rng)
        scales = {}
        for block in blocks:
            # a block without covariate columns adds nothing to var(eta), so no budget reaches it
            budget = split[block.name] / block.variance if block.variance > 0 else 0.0
            scales[block.name] = predictor_sd * math.sqrt(budget)
        return PriorContext(
            sd_alpha_family=sd_alpha_family,
            sigma_eps_family=sigma_eps_family,
            signal_share=signal_share,
            predictor_sd=predictor_sd,
            split=split,
            scales=scales,
            intercept_scale=self.intercept_scale,
            residual_scale=residual_scale,
            lkj_eta=lkj_eta,
        )

    def _drawSdAlphaFamily(self, rng: np.random.Generator) -> str | None:
        return None  # a GLM has no random effects

    def _drawSplit(self, blocks: list[Block], rng: np.random.Generator) -> dict[str, float]:
        return {blocks[0].name: 1.0}  # a GLM's one block takes all of var(eta)

    def _drawCorrelations(
        self, blocks: list[Block], rng: np.random.Generator
    ) -> dict[str, int | None]:
        return {}  # a GLM has no random effects

    @staticmethod
    def _drawSd(
        family: str, scale: float, rng: np.random.Generator, size: int | None = None
    ) -> float | np.ndarray:
        """Draws from the sd family F at its scale, one value or size of them, as numpy does:
        the scale times |X|, X F's standard law.
        """
        match family:
            case 'half_normal':
                standard = rng.standard_normal(size)
            case 'half_student_t':
                standard = rng.standard_t(3, size)
            case 'half_cauchy':
                standard = rng.standard_cauchy(size)
            case 'exponential':
                standard = rng.standard_exponential(size)
        return scale * abs(standard)

    def _drawRandomEffects(
        self, context: PriorContext, factors: list[Block], rng: np.random.Generator
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        sd_alpha = {}
        corr_alpha = {}
        alpha = {}
        for block in factors:
            scale = context.scales[block.name]
            sd = self._drawSd(context.sd_alpha_family, scale, rng, size=block.width)  # (q,)
            corr = np.eye(block.width)  # (q, q)
            lkj_eta = context.lkj_eta[block.name]
            if lkj_eta is not None:
                corr = self._drawLkj(block.width, lkj_eta, rng)  # (q, q)
            standard = rng.standard_normal((block.groups, block.width))  # (m, q)
            # α ~ N(0, diag(σ) Ω diag(σ)) per group: Ω's Cholesky factor correlates, σ scales
            effects = standard @ np.linalg.cholesky(corr).T * sd  # (m, q)
            alpha[block.name] = effects.astype(np.float32)
            sd_alpha[block.name] = sd.astype(np.float32)
            corr_alpha[block.name] = corr.astype(np.float32)
        return sd_alpha, corr_alpha, alpha

    @staticmethod
    def _drawLkj(q: int, lkj_eta: int, rng: np.random.Generator) -> np.ndarray:
        """Draws a correlation matrix from the LKJ prior at shape lkj_eta by the C-vine
        of Lewandowski, Kurowicka and Joe (2009, section 3.2).
        Each partial correlation is a Beta on (−1, 1),
        turned into a correlation by the recursion over the variables before it.
        """
        partial = np.zeros((q, q))  # (q, q)
        corr = np.eye(q)  # (q, q)
        concentration = lkj_eta + (q - 1) / 2  # the paper's β
        for k in range(q - 1):
            concentration -= 1 / 2
            for i in range(k + 1, q):
                share = rng.beta(concentration, concentration)  # on (0, 1)
                partial[k, i] = 2 * share - 1  # on (−1, 1)
                p = partial[k, i]
                for j in reversed(range(k)):
                    shrink = math.sqrt((1 - partial[j, i] ** 2) * (1 - partial[j, k] ** 2))
                    p = p * shrink + partial[j, i] * partial[j, k]
                corr[k, i] = p
                corr[i, k] = p
        return corr


class Glmm(Glm):
    """The prior of a generalized linear mixed model (GLMM): a GLM with random effects.

    - Split: the fixed block takes the share φ_fixed of var(eta),
      and the factors' blocks split the rest by a Dirichlet.
    - Random effects: each factor has an intercept and slopes on covariate columns,
      up to its own budget q_k, in design columns.
    - Correlation: a factor's effects are correlated with probability correlated_share,
      by an LKJ prior, and independent otherwise.
    - Families: the random-effect sds σ_α have a family of their own, apart from σ_ε's.
    """

    SHARES: ClassVar[tuple[str, ...]] = (*Glm.SHARES, 'sd_alpha_family_shares')
    PER_FACTOR: ClassVar[tuple[str, ...]] = ('random_effects', 'lkj_eta')

    sd_alpha_family_shares: Simplex = (
        SD_FAMILY_SHARES  # the family each factor's σ_α are drawn from
    )
    fixed_share: Share = Beta(1, 1)  # φ_fixed
    split_alpha: Positive = 1.0  # the Dirichlet's concentration over the factors' blocks
    random_effects: PositiveCount = UniformInt(1, 6)  # q_k per factor, truncated at d
    correlated_share: Share = 0.55
    lkj_eta: PositiveCount = UniformInt(1, 4)  # the LKJ shape per correlated factor

    def _drawSdAlphaFamily(self, rng: np.random.Generator) -> str | None:
        return self.sd_alpha_family_shares.choose(rng)

    def _drawSplit(self, blocks: list[Block], rng: np.random.Generator) -> dict[str, float]:
        fixed, *factors = blocks
        # a fixed block without covariate columns adds nothing to var(eta), so the factors take all
        fixed_share = self.fixed_share if fixed.variance > 0 else 0.0
        split = {fixed.name: fixed_share}
        shares = rng.dirichlet([self.split_alpha] * len(factors))  # (f,)
        for block, share in zip(factors, shares):
            split[block.name] = (1 - fixed_share) * float(share)
        return split

    def _drawCorrelations(
        self, blocks: list[Block], rng: np.random.Generator
    ) -> dict[str, int | None]:
        _, *factors = blocks
        lkj_eta = {}
        for block in factors:
            lkj_eta[block.name] = None  # independent effects, Ω the identity
            # a lone intercept has nothing to correlate with
            if block.width > 1 and rng.random() < self.correlated_share:
                lkj_eta[block.name] = self.resolveField('lkj_eta', rng)
        return lkj_eta
