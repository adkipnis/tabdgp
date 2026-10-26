"""prior.py defines how we draw the parameters of a GLM or GLMM outcome.

A generalized linear model (GLM) links an outcome y to the linear predictor eta = Xβ
with design matrix X. The link function differs per likelihood family (e.g. Normal, Bernoulli).
A generalized linear mixed model (GLMM) adds random effects for each grouping factor k,
eta = Xβ + Σ_k Z_k α_k, so each group gets its own additional intercept (and optionally slopes).
This module samples priors and parameters for these models.

The priors are specified by their family (e.g. Normal, Student-t)
and hyperparameters (e.g. location, scale). Each model parameter is assigned its own prior;
the family is shared per parameter type (e.g. fixed effects),
but each dimension gets its own hyperparameter.

Independent prior scales would saturate many datasets (e.g. R² near 1),
but capping them would degrade the prior. Instead, the scales are drawn jointly per dataset:
- **Predictor sd τ:** the sd of eta without the intercept.
- **Split φ:** each term's signal, its share of var(eta):
  φ_fixed for Xβ, the rest split among Z_k α_k.
- **Prior scales:** set so that each term adds its share of var(eta).
This adapts the R2D2M2 prior (Aguilar and Bürkner, 2023),
without its Dirichlet over all effects (which rarely lets a random intercept
carry the signal in eta, which is observed in real datasets with strong grouping).

Optionally, the linear predictor can be misspecified in a controlled way:
We add the structural causal model's (SCM) value for y
(standardized and scaled by λ > 0 times the sd of eta).

pipeline:
---------
the (optional) overrides
-> validation at construction                                 Glm, Glmm
-> a point per field drawn once per dataset                   Glm.resolveDataset
   (later: each factor's q_k and LKJ shape, the family's shape)
-> the prior for y's family over the blocks                   Glm.drawContext
-> β and σ_ε                                                  PriorContext.drawParameters
-> each factor's σ_α, Ω and α per group                       PriorContext._drawRandomEffects
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
    """A block of eta's coefficients: fixed effects or a factor's random effects (per factor)."""

    name: str  # 'fixed', or the factor's name
    variances: Annotated[list[NonNegativeFloat], Field(min_length=1)]
    groups: PositiveInt = 1  # m

    @property
    def width(self) -> int:
        return len(self.variances)

    @property
    def variance(self) -> float:
        """What the block's term adds to var(eta) at scale 1."""
        return sum(self.variances)


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Parameters:
    """A model's parameters (on the standardized scale). All arrays are float32."""

    beta: np.ndarray  # (d,) intercept first
    sigma_eps: (
        float | None
    )  # None unless the family is Normal or Student-t # TODO: either rename to sd_eps or better yet, join with shape
    shape: float | None  # the family's second parameter, None for Normal, Bernoulli and Poisson
    sigma_alpha: dict[str, np.ndarray]  # (q,) by factor
    corr_alpha: dict[str, np.ndarray]  # (q, q) by factor
    alpha: dict[str, np.ndarray]  # (m, q) by factor


@pydantic_dataclass(frozen=True)
class PriorContext:
    """The prior from which model parameters are drawn:

    - Fixed effects: family, locations, scales
    - Random effects: sd family, scales (per random effect), LKJ shape
    - Residual: family, scale (for Normal and Student-t outcomes)
    - Budget: signal share r, predictor sd τ, each block's share φ of var(eta)
    """

    beta_family: str
    beta_locations: list[float]  # μ, (d,) intercept first
    beta_scales: list[float]  # σ, (d,) intercept first
    sigma_alpha_family: str | None  # None on a GLM
    sigma_alpha_scales: dict[str, list[float]]  # s_α, (q,) by factor
    lkj_eta: dict[str, int | None]  # by factor, None if its effects are independent
    sigma_eps_family: str | None  # None unless the family has a residual
    sigma_eps_scale: float | None  # s_ε, None unless the family has a residual
    # ν, φ, σ_c, Gamma shape or beta precision (None for Normal, Bernoulli and Poisson)
    shape: float | None
    signal_share: float | None  # r, None unless the family is Normal or Student-t
    predictor_sd: float  # τ
    split: dict[str, float]  # φ by block name
    blocks: list[Block]  # fixed, then the factors'

    def drawParameters(self, rng: np.random.Generator) -> Parameters:
        """Draws each parameter from its prior, with no caps or rescaling."""
        _, *factors = self.blocks
        locations = np.array(self.beta_locations)  # (d,)
        scales = np.array(self.beta_scales)  # (d,)
        standard = self._drawStandard(self.beta_family, rng, len(scales))  # (d,)
        beta = locations + scales * standard  # (d,)
        sigma_eps = None
        if self.sigma_eps_scale is not None:
            sigma_eps = self._drawSd(self.sigma_eps_family, self.sigma_eps_scale, rng)
        sigma_alpha, corr_alpha, alpha = self._drawRandomEffects(factors, rng)
        # float32, like the columns
        return Parameters(
            beta=beta.astype(np.float32),
            sigma_eps=sigma_eps,
            shape=self.shape,
            sigma_alpha=sigma_alpha,
            corr_alpha=corr_alpha,
            alpha=alpha,
        )

    @staticmethod
    def _drawStandard(
        family: str, rng: np.random.Generator, size: int | None
    ) -> float | np.ndarray:
        """Draws from a family at location 0 and scale 1, a half family as its full one."""
        match family.removeprefix('half_'):
            case 'normal':
                return rng.standard_normal(size)
            case 'student_t':
                return rng.standard_t(3, size)
            case 'cauchy':
                return rng.standard_cauchy(size)
            case 'exponential':
                return rng.standard_exponential(size)

    @classmethod
    def _drawSd(
        cls,
        family: str,
        scale: float | np.ndarray,
        rng: np.random.Generator,
        size: int | None = None,
    ) -> float | np.ndarray:
        """Draws from an sd family at its scale: scale · |X|, X the family's standard law."""
        standard = cls._drawStandard(family, rng, size)
        return scale * abs(standard)

    def _drawRandomEffects(
        self, factors: list[Block], rng: np.random.Generator
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        sigma_alpha = {}
        corr_alpha = {}
        alpha = {}
        for block in factors:
            scale = np.array(self.sigma_alpha_scales[block.name])  # (q,)
            sd = self._drawSd(self.sigma_alpha_family, scale, rng, size=block.width)  # (q,)
            corr = np.eye(block.width)  # (q, q)
            lkj_eta = self.lkj_eta[block.name]
            if lkj_eta is not None:
                corr = self._drawLkj(block.width, lkj_eta, rng)  # (q, q)
            standard = rng.standard_normal((block.groups, block.width))  # (m, q)
            # α ~ N(0, diag(σ) Ω diag(σ)) per group: Ω's Cholesky factor correlates, σ scales
            effects = standard @ np.linalg.cholesky(corr).T * sd  # (m, q)
            alpha[block.name] = effects.astype(np.float32)
            sigma_alpha[block.name] = sd.astype(np.float32)
            corr_alpha[block.name] = corr.astype(np.float32)
        return sigma_alpha, corr_alpha, alpha

    @staticmethod
    def _drawLkj(q: int, lkj_eta: int, rng: np.random.Generator) -> np.ndarray:
        """Draws a correlation matrix from LKJ(lkj_eta)
        by the C-vine of Lewandowski, Kurowicka and Joe (2009, section 3.2).
        Each partial correlation is a Beta on (−1, 1), turned into a correlation by recursion.
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


# =====================================================================================
# the config
# =====================================================================================

# the families of σ_α and σ_ε, uniform
SD_FAMILY_SHARES = {
    'half_normal': 0.25,
    'half_student_t': 0.25,  # ν = 3
    'half_cauchy': 0.25,
    'exponential': 0.25,
}


class Glm(Config):
    """The prior of a generalized linear model (GLM).

    - Families: of y (with its canonical link), of β and of σ_ε.
    - Locations: the intercept's (on the link scale) and each slope's (mostly zero).
    - Scales: τ is the sd of eta without the intercept.
      For Normal and Student-t outcomes τ² = r, the share of y's variance eta explains,
      so y has unit variance (ν / (ν − 2) for Student-t).
      Each slope's scale is the block's times a log-normal weight (the block's at spread 0).
      Each scale is its family's scale parameter, e.g. E[σ_ε²] = s_ε² E[X²],
      X the standard law of σ_ε's family.
    - Shape: the second parameter of a family at share zero, resolved for the chosen one only.
    - Deviation: the score added to eta, scaled by λ · sd(Xβ + Σ_k Z_k α_k) over the rows.
    """

    SHARES: ClassVar[tuple[str, ...]] = (
        'family_shares',
        'beta_family_shares',
        'sigma_eps_family_shares',
    )
    PER_FACTOR: ClassVar[tuple[str, ...]] = ()  # the fields each factor resolves on its own
    # resolved for the chosen family only, so a family at share zero costs no draw
    PER_FAMILY: ClassVar[dict[str, str]] = {
        'student_t': 'residual_df',
        'negative_binomial': 'dispersion',
        'latent_bernoulli': 'latent_scale',
        'gamma': 'gamma_shape',
        'beta': 'beta_precision',
    }

    # ruff cannot trace Config to BaseModel, so it reads a shares default as a class attribute
    family_shares: Simplex = {  # noqa: RUF012
        'normal': 1 / 3,
        'bernoulli': 1 / 3,
        'poisson': 1 / 3,
        # the contaminated families (each with a shape), drawn only by override
        'student_t': 0.0,
        'negative_binomial': 0.0,
        'latent_bernoulli': 0.0,
        'gamma': 0.0,
        'beta': 0.0,
    }
    beta_family_shares: Simplex = {'normal': 0.68, 'student_t': 0.30, 'cauchy': 0.02}  # noqa: RUF012
    sigma_eps_family_shares: Simplex = SD_FAMILY_SHARES  # σ_ε's family
    intercept_location: Real = Uniform(-3, 3)  # μ_0, on the link scale
    intercept_scale: NonNegative = LogUniform(0.1, 10)  # σ_0
    location_share: Share = 0.1  # the probability that a slope has a location
    location_scale: NonNegative = 2.0  # a located slope's location, in its own scale, is N(0, this)
    scale_spread: NonNegative = 0.0  # the sd of each slope's log scale around the block's
    signal_share: Share = Beta(2, 2)  # r
    predictor_sd: NonNegative = LogUniform(0.25, 4)  # τ, unless the family is Normal or Student-t
    deviation_scale: NonNegative = 0.0  # λ
    residual_df: Positive = LogUniform(3, 10)  # ν of the Student-t family
    dispersion: Positive = LogUniform(1, 10)  # φ of the negative binomial, variance μ + μ² / φ
    latent_scale: NonNegative = LogUniform(0.5, 2)  # σ_c of the latent-noise Bernoulli
    gamma_shape: Positive = LogUniform(1, 10)
    beta_precision: Positive = LogUniform(2, 50)  # of the beta family, not of β

    def resolveDataset(self, rng: np.random.Generator) -> Glm:
        """Resolves each field drawn once per dataset, in declaration order."""
        points = {}
        for name in type(self).model_fields:
            if name in self.PER_FACTOR or name in self.PER_FAMILY.values():
                points[name] = getattr(self, name)  # defer resolution to each factor or family
            else:
                points[name] = self.resolveField(name, rng)
        return type(self)(**points)

    def drawContext(
        self, family: str, blocks: list[Block], rng: np.random.Generator
    ) -> PriorContext:
        """Draws the prior for y's family over the blocks, on a resolved Glm."""
        beta_family = self.beta_family_shares.choose(rng)
        sigma_eps_family = None
        signal_share = None
        predictor_sd = self.predictor_sd
        sigma_eps_scale = None
        shape = None
        if family in self.PER_FAMILY:
            shape = self.resolveField(self.PER_FAMILY[family], rng)
        if family in ['normal', 'student_t']:
            sigma_eps_family = self.sigma_eps_family_shares.choose(rng)
            signal_share = self.signal_share
            # τ² = r and s_ε² = 1 − r, so that y has unit variance at the scales
            predictor_sd = math.sqrt(self.signal_share)
            sigma_eps_scale = math.sqrt(1 - self.signal_share)
        sigma_alpha_family = self._drawSigmaAlphaFamily(rng)
        split = self._drawSplit(blocks, rng)
        lkj_eta = self._drawCorrelations(blocks, rng)
        fixed, *factors = blocks
        budget = predictor_sd**2 * split[fixed.name]  # τ² φ_fixed, what Xβ adds to var(eta)
        slope_locations, slope_scales = self._drawSlopes(fixed, budget, rng)
        sigma_alpha_scales = {}
        for block in factors:
            # s_α² tr E[z z'] = τ² φ_k, tr ≥ 1 by the intercept; one scale for all its effects
            scale = predictor_sd * math.sqrt(split[block.name] / block.variance)
            sigma_alpha_scales[block.name] = [scale] * block.width
        return PriorContext(
            beta_family=beta_family,
            beta_locations=[self.intercept_location, *slope_locations],
            beta_scales=[self.intercept_scale, *slope_scales],
            sigma_alpha_family=sigma_alpha_family,
            sigma_alpha_scales=sigma_alpha_scales,
            lkj_eta=lkj_eta,
            sigma_eps_family=sigma_eps_family,
            sigma_eps_scale=sigma_eps_scale,
            shape=shape,
            signal_share=signal_share,
            predictor_sd=predictor_sd,
            split=split,
            blocks=blocks,
        )

    def _drawSigmaAlphaFamily(self, rng: np.random.Generator) -> str | None:
        return None  # a GLM has no random effects

    def _drawSplit(self, blocks: list[Block], rng: np.random.Generator) -> dict[str, float]:
        return {blocks[0].name: 1.0}  # a GLM's one block takes all of var(eta)

    def _drawCorrelations(
        self, blocks: list[Block], rng: np.random.Generator
    ) -> dict[str, int | None]:
        return {}  # a GLM has no random effects

    def _drawSlopes(
        self, fixed: Block, budget: float, rng: np.random.Generator
    ) -> tuple[list[float], list[float]]:
        """Draws each slope's location μ_j = δ_j σ_j and scale σ_j = σ_β w_j,
        with σ_β solved from the budget σ_β² Σ_j w_j² (1 + δ_j²) var_j,
        what Xβ adds to var(eta) for a Normal β (a heavier-tailed β adds more).
        """
        slopes = fixed.width - 1
        located = rng.random(slopes) < self.location_share  # (d − 1,)
        offsets = rng.normal(0, self.location_scale, slopes)  # (d − 1,) δ_j of a located slope
        delta = np.where(located, offsets, 0.0)  # (d − 1,)
        log_weights = rng.normal(0, self.scale_spread, slopes)  # (d − 1,)
        weights = np.exp(log_weights)  # (d − 1,) w_j
        variances = np.array(fixed.variances[1:])  # (d − 1,)
        spent = float((weights**2 * (1 + delta**2) * variances).sum())  # at σ_β = 1
        # without a varying column, nothing spends the budget
        sigma_beta = math.sqrt(budget / spent) if spent > 0 else 0.0
        slope_scales = sigma_beta * weights  # (d − 1,)
        slope_locations = delta * slope_scales  # (d − 1,)
        return slope_locations.tolist(), slope_scales.tolist()


class Glmm(Glm):
    """The prior of a generalized linear mixed model (GLMM): a GLM with random effects.

    - Split: φ_fixed of var(eta) for Xβ, the rest split among the factors by a Dirichlet.
    - Random effects: per factor, an intercept and slopes, up to q_k design columns.
    - Correlation: LKJ with probability correlated_share, independent otherwise.
    - Families: σ_α has its own, apart from σ_ε's.
    """

    SHARES: ClassVar[tuple[str, ...]] = (*Glm.SHARES, 'sigma_alpha_family_shares')
    PER_FACTOR: ClassVar[tuple[str, ...]] = ('random_effects', 'lkj_eta')

    sigma_alpha_family_shares: Simplex = SD_FAMILY_SHARES  # σ_α's family
    fixed_share: Share = Beta(1, 1)  # φ_fixed
    split_alpha: Positive = 1.0  # the Dirichlet's concentration over the factors' blocks
    random_effects: PositiveCount = UniformInt(1, 6)  # q_k per factor, truncated at d
    correlated_share: Share = 0.55
    lkj_eta: PositiveCount = UniformInt(1, 4)  # the LKJ shape per correlated factor

    def _drawSigmaAlphaFamily(self, rng: np.random.Generator) -> str | None:
        return self.sigma_alpha_family_shares.choose(rng)

    def _drawSplit(self, blocks: list[Block], rng: np.random.Generator) -> dict[str, float]:
        fixed, *factors = blocks
        # without covariate columns, Xβ adds nothing to var(eta), so the factors take all
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
