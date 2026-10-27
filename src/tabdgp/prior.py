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
-> the prior for y's family over the blocks                   Glm.drawPrior
-> σ_α's family, each factor's LKJ shape and σ_α scales       Glmm.drawPrior
-> β and σ_ε                                                  Prior.drawParameters
-> each factor's σ_α, Ω and α per group                       Prior._drawRandomEffects
"""

from __future__ import annotations

import dataclasses
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


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Parameters:
    """A model's parameters (on the standardized scale). All arrays are float32."""

    beta: np.ndarray  # (d,) intercept first
    sigma_eps: float | None  # None unless the family is Normal or Student-t
    y_shape: float | None  # y's second parameter, None for Normal, Bernoulli and Poisson
    sigma_alpha: dict[str, np.ndarray]  # (q,) by factor
    corr_alpha: dict[str, np.ndarray]  # (q, q) by factor
    alpha: dict[str, np.ndarray]  # (m, q) by factor


@pydantic_dataclass(frozen=True)
class Prior:
    """The prior from which model parameters are drawn:

    - Fixed effects: family, locations, scales
    - Random effects: sd family, scales (per random effect), LKJ shape
    - Residual: family, scale (for Normal and Student-t outcomes)
    - Budget: signal share r, predictor sd τ, each block's share φ of var(eta)
    """

    beta_family: str
    beta_locations: list[float]  # ν_β, (d,) intercept first
    beta_scales: list[float]  # τ_β, (d,) intercept first
    sigma_alpha_family: str | None  # None on a GLM
    sigma_alpha_scales: dict[str, list[float]]  # τ_α, σ_α's prior scale, (q,) by factor
    lkj_eta: dict[str, int | None]  # by factor, None if its effects are independent
    sigma_eps_family: str | None  # None unless the family has a residual
    sigma_eps_scale: float | None  # τ_ε, σ_ε's prior scale, None unless the family has a residual
    y_shape: float | None
    signal_share: float | None  # r, the share of var(y) eta explains (Normal and Student-t only)
    predictor_sd: float  # τ, the sd of eta (without the intercept)
    split: dict[str, float]  # φ, each block's share of τ²
    blocks: list[Block]

    def drawParameters(self, rng: np.random.Generator) -> Parameters:
        """Draws each parameter from its prior, with no caps or rescaling."""
        locations = np.array(self.beta_locations)  # (d,)
        scales = np.array(self.beta_scales)  # (d,)
        standard = self._drawStandard(self.beta_family, rng, len(scales))  # (d,)
        beta = locations + scales * standard  # (d,)
        sigma_eps = None
        if self.sigma_eps_scale is not None:
            sigma_eps = self._drawSd(self.sigma_eps_family, self.sigma_eps_scale, rng)
        random_effects = self._drawRandomEffects(rng)
        return Parameters(
            beta=beta.astype(np.float32),
            sigma_eps=sigma_eps,
            y_shape=self.y_shape,
            sigma_alpha=random_effects['sigma_alpha'],
            corr_alpha=random_effects['corr_alpha'],
            alpha=random_effects['alpha'],
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
        """Draws from an sd family at its scale: scale · |X|, X the family's standard distribution."""
        standard = cls._drawStandard(family, rng, size)
        return scale * abs(standard)  # abs() for half-normal, half-student and half-cauchy

    def _drawRandomEffects(self, rng: np.random.Generator) -> dict[str, dict[str, np.ndarray]]:
        # α ~ N(0, diag(σ) Ω diag(σ)) per group
        random_effects = {'sigma_alpha': {}, 'corr_alpha': {}, 'alpha': {}}  # Parameters' fields
        for block in self.blocks[1:]:  # the factors, after the fixed block
            scale = np.array(self.sigma_alpha_scales[block.name])  # (q,)
            sd = self._drawSd(self.sigma_alpha_family, scale, rng, size=block.width)  # (q,)
            corr = np.eye(block.width)  # (q, q)
            lkj_eta = self.lkj_eta[block.name]
            if lkj_eta is not None:
                corr = self._drawLkj(block.width, lkj_eta, rng)  # (q, q)
            standard = rng.standard_normal((block.groups, block.width))  # (m, q)
            effects = standard @ np.linalg.cholesky(corr).T * sd  # (m, q)
            random_effects['sigma_alpha'][block.name] = sd.astype(np.float32)
            random_effects['corr_alpha'][block.name] = corr.astype(np.float32)
            random_effects['alpha'][block.name] = effects.astype(np.float32)
        return random_effects

    @staticmethod
    def _drawLkj(q: int, lkj_eta: int, rng: np.random.Generator) -> np.ndarray:
        """Draws a correlation matrix from LKJ(lkj_eta)
        using the C-vine according to Lewandowski, Kurowicka and Joe (2009).
        """
        partial = np.zeros((q, q))  # (q, q)
        corr = np.eye(q)  # (q, q)
        concentration = lkj_eta + (q - 1) / 2
        for k in range(q - 1):
            concentration -= 1 / 2
            for i in range(k + 1, q):
                share = rng.beta(concentration, concentration)
                partial[k, i] = 2 * share - 1  # expand support to (-1, 1)
                p = partial[k, i]
                # to make the correlation matrix positive definite
                # p shrinks by what each previous variable leaves "unexplained"
                for j in reversed(range(k)):
                    prod = (1 - partial[j, i] ** 2) * (1 - partial[j, k] ** 2)
                    shrink = math.sqrt(prod)
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
    'half_student_t': 0.25,  # with df = 3
    'half_cauchy': 0.25,
    'exponential': 0.25,
}


class Glm(Config):
    """The hyperprior of a generalized linear model (GLM),
    from which the prior and in turn the model parameters are drawn:

    - Families: of y, of β and of σ_ε
    - Locations: of intercept and slopes
    - Scales: shared or per coefficient within a block
    - Shape: y's second parameter (e.g. Student-t's df)
    - Budget: τ², the variance of eta without the intercept
    - Deviation: the score added to eta, scaled by λ · sd(Xβ + Σ_k Z_k α_k)
    """

    SHARES: ClassVar[tuple[str, ...]] = (
        'y_family_shares',
        'beta_family_shares',
        'sigma_eps_family_shares',
    )
    PER_FACTOR: ClassVar[tuple[str, ...]] = ()
    PER_Y_FAMILY: ClassVar[dict[str, str]] = {
        'student_t': 'residual_df',
        'negative_binomial': 'dispersion',
        'latent_bernoulli': 'latent_scale',
        'gamma': 'gamma_shape',
        'beta': 'beta_precision',
    }

    y_family_shares: Simplex = {  # noqa: RUF012
        'normal': 1 / 3,
        'bernoulli': 1 / 3,
        'poisson': 1 / 3,
        # useful for misspecification tests
        'student_t': 0.0,
        'negative_binomial': 0.0,
        'latent_bernoulli': 0.0,
        'gamma': 0.0,
        'beta': 0.0,
    }
    beta_family_shares: Simplex = {'normal': 0.68, 'student_t': 0.30, 'cauchy': 0.02}  # noqa: RUF012
    sigma_eps_family_shares: Simplex = SD_FAMILY_SHARES  # σ_ε's family
    intercept_location: Real = Uniform(-3, 3)  # ν_0 on the link scale
    intercept_scale: NonNegative = LogUniform(0.1, 10)  # τ_0
    location_share: Share = 0.1  # the probability that a slope has a location
    location_scale: NonNegative = 2.0  # the sd of a located slope's δ_j = ν_j / τ_j
    varying_scale_share: Share = 0.5  # the probability that a block's scales vary per coefficient
    scale_spread: NonNegative = Uniform(0, 2)  # the sd of a log weight in a varying block
    signal_share: Share = Beta(2, 2)  # r
    predictor_sd: NonNegative = LogUniform(0.25, 4)  # τ (unless the family is Normal or Student-t)
    # for misspecification
    deviation_scale: NonNegative = 0.0  # λ
    residual_df: Positive = LogUniform(3, 10)  # of the Student-t family
    dispersion: Positive = LogUniform(1, 10)  # φ of the negative binomial, variance μ + μ² / φ
    latent_scale: NonNegative = LogUniform(0.5, 2)  # σ_c of the latent-noise Bernoulli
    gamma_shape: Positive = LogUniform(1, 10)
    beta_precision: Positive = LogUniform(2, 50)

    def resolveDataset(self, rng: np.random.Generator) -> Glm:
        """Resolves each field that is drawn once per dataset."""
        deferred = (*self.PER_FACTOR, *self.PER_Y_FAMILY.values())
        points = {}
        for name in type(self).model_fields:
            if name in deferred:
                points[name] = getattr(self, name)
            else:
                points[name] = self.resolveField(name, rng)
        return type(self)(**points)

    def drawPrior(self, y_family: str, blocks: list[Block], rng: np.random.Generator) -> Prior:
        """Draws the prior for y's family over the blocks after the GLM is resolved."""
        # β's family, and optionally y's shape
        beta_family = self.beta_family_shares.choose(rng)
        y_shape = None
        if y_family in self.PER_Y_FAMILY:
            y_shape = self.resolveField(self.PER_Y_FAMILY[y_family], rng)
        # σ_ε's family and scale, if y's family has a residual
        sigma_eps_family = None
        signal_share = None
        predictor_sd = self.predictor_sd
        sigma_eps_scale = None
        if y_family in ['normal', 'student_t']:
            sigma_eps_family = self.sigma_eps_family_shares.choose(rng)
            signal_share = self.signal_share
            # τ² = r and τ_ε² = 1 − r, so that y has unit variance at the scales
            predictor_sd = math.sqrt(self.signal_share)
            sigma_eps_scale = math.sqrt(1 - self.signal_share)
        # the split of τ² between the blocks, and the slopes spending Xβ's share
        split = self._drawSplit(blocks, rng)
        fixed = blocks[0]
        budget = predictor_sd**2 * split[fixed.name]  # τ² φ_fixed (what Xβ adds to var(eta))
        slope_locations, slope_scales = self._drawSlopes(fixed, budget, rng)
        return Prior(
            beta_family=beta_family,
            beta_locations=[self.intercept_location, *slope_locations],
            beta_scales=[self.intercept_scale, *slope_scales],
            sigma_alpha_family=None,  # a GLM has no random effects
            sigma_alpha_scales={},
            lkj_eta={},
            sigma_eps_family=sigma_eps_family,
            sigma_eps_scale=sigma_eps_scale,
            y_shape=y_shape,
            signal_share=signal_share,
            predictor_sd=predictor_sd,
            split=split,
            blocks=blocks,
        )

    def _drawSplit(self, blocks: list[Block], rng: np.random.Generator) -> dict[str, float]:
        return {blocks[0].name: 1.0}  # a GLM's one block takes all of var(eta)

    # TODO: confusing name, we don't draw the slopes, we draw the their prior parameters
    # also what's the reason to split the intercept priors from the slope? even if the intercept prior is not affected by the budget, it would be cleaner to move its lines here and rename the method, no?
    def _drawSlopes(
        self, fixed: Block, budget: float, rng: np.random.Generator
    ) -> tuple[list[float], list[float]]:
        """Draws each slope's location and scale:

        - Location: ν_j = δ_j τ_j, δ_j nonzero for a located slope only
        - Scale: τ_j, spending the budget Σ_j τ_j² (1 + δ_j²) var_j
        - Budget: what Xβ adds to var(eta) at a Normal β, more at a heavier-tailed one
        """
        slopes = fixed.width - 1
        located = rng.random(slopes) < self.location_share  # (d − 1,)
        offsets = rng.normal(0, self.location_scale, slopes)  # (d − 1,) δ_j of a located slope
        delta = np.where(located, offsets, 0.0)  # (d − 1,)
        column_variances = np.array(fixed.variances[1:])  # (d − 1,) var_j
        variances = (1 + delta**2) * column_variances  # (d − 1,) E[β_j²] var_j at τ_j = 1
        slope_scales = self._drawScales(variances, budget, rng)  # (d − 1,)
        slope_locations = delta * slope_scales  # (d − 1,)
        return slope_locations.tolist(), slope_scales.tolist()

    # TODO: rewrite docstring more intuitively, this is confusing
    def _drawScales(
        self, variances: np.ndarray, budget: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Draws a block's scales s w_j from what each coefficient adds to var(eta) at scale 1:

        - Weights: w_j log-normal at scale_spread with probability varying_scale_share, else 1
        - Scale: s solves s² Σ_j w_j² variances_j = budget
        """
        weights = np.ones(len(variances))  # (d − 1,) or (q,)
        if rng.random() < self.varying_scale_share:
            log_weights = rng.normal(0, self.scale_spread, len(variances))  # (d − 1,) or (q,)
            weights = np.exp(log_weights)  # (d − 1,) or (q,)
        spent = float((weights**2 * variances).sum())  # at s = 1
        # without a column that adds to var(eta), nothing spends the budget
        scale = math.sqrt(budget / spent) if spent > 0 else 0.0
        return scale * weights  # (d − 1,) or (q,)


class Glmm(Glm):
    """The hyperprior of a generalized linear mixed model (GLMM), a GLM with random effects:

    - Split: φ_fixed of τ² for Xβ, the rest split among the factors by a Dirichlet
    - Random effects: per factor, an intercept and slopes, up to q_k design columns
    - Correlation: LKJ with probability correlated_share, independent otherwise
    - Families: σ_α has its own, apart from σ_ε's
    """

    SHARES: ClassVar[tuple[str, ...]] = (*Glm.SHARES, 'sigma_alpha_family_shares')
    PER_FACTOR: ClassVar[tuple[str, ...]] = ('random_effects', 'lkj_eta')

    sigma_alpha_family_shares: Simplex = SD_FAMILY_SHARES  # σ_α's family
    fixed_share: Share = Beta(1, 1)  # φ_fixed
    split_alpha: Positive = 1.0  # the Dirichlet's concentration over the factors' blocks
    random_effects: PositiveCount = UniformInt(1, 6)  # q_k per factor, truncated at d
    correlated_share: Share = 0.55
    lkj_eta: PositiveCount = UniformInt(1, 4)  # the LKJ shape per correlated factor

    def drawPrior(self, y_family: str, blocks: list[Block], rng: np.random.Generator) -> Prior:
        """Draws a GLM's prior, then σ_α's family, each factor's LKJ shape and its σ_α scales."""
        prior = super().drawPrior(y_family, blocks, rng)
        _, *factors = blocks
        sigma_alpha_family = self.sigma_alpha_family_shares.choose(rng)
        lkj_eta = self._drawCorrelations(factors, rng)
        sigma_alpha_scales = self._drawSigmaAlphaScales(
            factors, prior.predictor_sd, prior.split, rng
        )
        return dataclasses.replace(
            prior,
            sigma_alpha_family=sigma_alpha_family,
            sigma_alpha_scales=sigma_alpha_scales,
            lkj_eta=lkj_eta,
        )

    def _drawSplit(self, blocks: list[Block], rng: np.random.Generator) -> dict[str, float]:
        fixed, *factors = blocks
        # without covariate columns, Xβ adds nothing to var(eta), so the factors take all
        fixed_share = self.fixed_share if sum(fixed.variances) > 0 else 0.0
        split = {fixed.name: fixed_share}
        shares = rng.dirichlet([self.split_alpha] * len(factors))  # (f,)
        for block, share in zip(factors, shares):
            split[block.name] = (1 - fixed_share) * float(share)
        return split

    def _drawCorrelations(
        self, factors: list[Block], rng: np.random.Generator
    ) -> dict[str, int | None]:
        lkj_eta = {}
        for block in factors:
            lkj_eta[block.name] = None  # independent effects, Ω the identity
            # a lone intercept has nothing to correlate with
            if block.width > 1 and rng.random() < self.correlated_share:
                lkj_eta[block.name] = self.resolveField('lkj_eta', rng)
        return lkj_eta

    def _drawSigmaAlphaScales(
        self,
        factors: list[Block],
        predictor_sd: float,
        split: dict[str, float],
        rng: np.random.Generator,
    ) -> dict[str, list[float]]:
        sigma_alpha_scales = {}
        for block in factors:
            budget = predictor_sd**2 * split[block.name]  # τ² φ_k, what Z_k α_k adds to var(eta)
            variances = np.array(block.variances)  # (q,) E[z_j²], the intercept's 1 first
            scales = self._drawScales(variances, budget, rng)  # (q,)
            sigma_alpha_scales[block.name] = scales.tolist()
        return sigma_alpha_scales
