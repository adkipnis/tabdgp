"""prior.py defines the prior of a GLM or GLMM: its config, its scales' joint draw, its parameters.

The scales are drawn jointly per dataset.
The predictor sd τ sets the sd of η without the intercept, and the split φ shares var(η) among
the blocks of coefficients: the fixed block takes φ_fixed, and a GLMM's factors split the rest.
Each block's scale divides its share by the variance its term adds to η at scale 1,
so that a block of dummies realizes the share a block of z-scored columns does.
Given the scales, each parameter is drawn from its stated family exactly:
nothing is clipped, rerolled or rescaled.
A factor's random effects are drawn per group, correlated by an LKJ prior or independent.
The draw reads a block's size, never its columns.

pipeline:
---------
the (optional) overrides
-> validation at construction                                 Glm, Glmm
-> a point for each field drawn once per dataset              Glm.resolveDataset
   (a factor resolves its budget and its η later)
-> given y's family and the blocks, the families chosen,
   r, τ, φ, each block's scale and each factor's η             _drawContext
-> β and σ_ε from their families at those scales,
   then each factor's σ_α, Ω and α per group                   Glm.drawParameters
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, NonNegativeFloat, PositiveInt
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.field import (
    Beta,
    Config,
    LogUniform,
    NonNegative,
    Positive,
    PositiveCount,
    Share,
    Simplex,
    UniformInt,
)

# =====================================================================================
# what the draw reads and returns
# =====================================================================================


@pydantic_dataclass(frozen=True)
class Block:
    """A block of η's coefficients drawn at one scale:
    the fixed effects, or a factor's random effects, one set per group.
    The draw reads its size, never its columns.
    """

    name: str  # 'fixed', or the factor's
    width: PositiveInt  # its coefficients per group, the intercept included
    variance: NonNegativeFloat  # what its term adds to var(η) at scale 1: tr cov X if fixed
    groups: PositiveInt = 1  # m, the fixed effects' one set for all rows


@pydantic_dataclass(frozen=True)
class PriorContext:
    """The resolved hyperparameters a GLM's or GLMM's parameters were drawn with.

    - Families: F, the sd family chosen, and the Normal family's residual law.
    - Signal: the signal share r and the predictor sd τ, the sd of η without the intercept.
    - Split: each block's share φ of var(η), and its scale, σ_β for the fixed block.
    - Residual: σ_ε is drawn from F at the residual scale s_ε, under the Normal family only.
    - Correlation: a factor's random effects are correlated by an LKJ prior of shape η,
      or independent.
    """

    sd_family: str
    residual_law: str | None  # None unless the family is Normal
    signal_share: float | None  # r, None unless the family is Normal
    predictor_sd: float  # τ
    split: dict[str, float]  # φ by block name
    scales: dict[str, float]  # by block name
    intercept_scale: float  # σ_0
    residual_scale: float | None  # s_ε, None unless the family is Normal
    lkj_eta: dict[str, int | None]  # η by factor, None if its effects are independent


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Parameters:
    """A GLM's or GLMM's parameters on the standardized scale, as drawn.
    Each factor's random effects have their sds, their correlation and one row per group.
    """

    beta: np.ndarray  # (d,) float32, the intercept first
    sigma_eps: float | None  # None unless the family is Normal
    sd_alpha: dict[str, np.ndarray]  # (q,) float32 by factor, none for a GLM
    corr_alpha: dict[str, np.ndarray]  # (q, q) float32 by factor
    alpha: dict[str, np.ndarray]  # (m, q) float32 by factor


# =====================================================================================
# the config
# =====================================================================================


class Glm(Config):
    """The prior of a generalized linear model (GLM) over the design matrix.

    - Families: y's response family, with its canonical link,
      and the family F the residual scale is drawn from.
    - Scales: the predictor sd τ is the sd of η without the intercept.
      Under the Normal family τ² = r, the signal share r being the share of y's variance
      that η explains, so that y has unit variance at the scales.
      σ_ε is drawn from F at s_ε, so E[σ_ε²] = s_ε² E[X²], X F's standard law.
    - Deviation: the score added to η at λ · sd(Xβ + Σ_k Z_k α_k),
      the population sd over the rows.
    """

    SHARES: ClassVar[tuple[str, ...]] = ('family_shares', 'sd_family_shares', 'residual_shares')
    PER_FACTOR: ClassVar[tuple[str, ...]] = ()  # the fields each factor resolves on its own

    # ruff cannot trace Config to BaseModel, so it reads a shares default as a class attribute
    family_shares: Simplex = {'normal': 1 / 3, 'bernoulli': 1 / 3, 'poisson': 1 / 3}  # noqa: RUF012
    sd_family_shares: Simplex = {  # noqa: RUF012
        'half_normal': 0.25,
        'half_student_t': 0.25,  # ν = 3
        'half_cauchy': 0.25,
        'exponential': 0.25,
    }  # F, the family of the residual scale σ_ε
    residual_shares: Simplex = {'normal': 1.0}  # the Normal family's residual law  # noqa: RUF012
    intercept_scale: NonNegative = LogUniform(0.1, 10)  # σ_0
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
        slopes = rng.normal(0, context.scales['fixed'], size=fixed.width - 1)  # (d − 1,)
        beta = np.concatenate([intercept, slopes])  # (d,)
        sigma_eps = None
        if context.residual_scale is not None:
            sigma_eps = self._drawSd(context.sd_family, context.residual_scale, rng)
        sd_alpha = {}
        corr_alpha = {}
        alpha = {}
        for block in factors:
            scale = context.scales[block.name]
            sd = self._drawSd(context.sd_family, scale, rng, size=block.width)  # (q,)
            corr = np.eye(block.width)  # (q, q)
            eta = context.lkj_eta[block.name]
            if eta is not None:
                corr = self._drawLkj(block.width, eta, rng)  # (q, q)
            standard = rng.standard_normal((block.groups, block.width))  # (m, q)
            # α ~ N(0, diag(σ) Ω diag(σ)) per group: Ω's Cholesky factor correlates, σ scales
            effects = standard @ np.linalg.cholesky(corr).T * sd  # (m, q)
            # the parameters are float32, the precision of the columns they are recorded with
            alpha[block.name] = effects.astype(np.float32)
            sd_alpha[block.name] = sd.astype(np.float32)
            corr_alpha[block.name] = corr.astype(np.float32)
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
        sd_family = self.sd_family_shares.choose(rng)
        residual_law = None
        signal_share = None
        predictor_sd = self.predictor_sd
        residual_scale = None
        if family == 'normal':
            residual_law = self.residual_shares.choose(rng)
            signal_share = self.signal_share
            # τ² = r and s_ε² = 1 − r, so that y has unit variance at the scales
            predictor_sd = math.sqrt(self.signal_share)
            residual_scale = math.sqrt(1 - self.signal_share)
        split = self._drawSplit(blocks, rng)
        lkj_eta = self._drawCorrelations(blocks, rng)
        scales = {}
        for block in blocks:
            # a block without covariate columns adds nothing to var(η), so no budget reaches it
            budget = split[block.name] / block.variance if block.variance > 0 else 0.0
            scales[block.name] = predictor_sd * math.sqrt(budget)
        return PriorContext(
            sd_family=sd_family,
            residual_law=residual_law,
            signal_share=signal_share,
            predictor_sd=predictor_sd,
            split=split,
            scales=scales,
            intercept_scale=self.intercept_scale,
            residual_scale=residual_scale,
            lkj_eta=lkj_eta,
        )

    def _drawSplit(self, blocks: list[Block], rng: np.random.Generator) -> dict[str, float]:
        return {'fixed': 1.0}  # a GLM's one block takes all of var(η)

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

    @staticmethod
    def _drawLkj(q: int, eta: int, rng: np.random.Generator) -> np.ndarray:
        """Draws a correlation matrix from the LKJ prior at shape η by the C-vine
        of Lewandowski, Kurowicka and Joe (2009, section 3.2).
        Each partial correlation is a Beta on (−1, 1),
        turned into a correlation by the recursion over the variables before it.
        """
        partial = np.zeros((q, q))  # (q, q)
        corr = np.eye(q)  # (q, q)
        concentration = eta + (q - 1) / 2  # the paper's β
        for k in range(q - 1):
            concentration -= 1 / 2
            for i in range(k + 1, q):
                partial[k, i] = 2 * rng.beta(concentration, concentration) - 1
                p = partial[k, i]
                for j in reversed(range(k)):
                    shrink = math.sqrt((1 - partial[j, i] ** 2) * (1 - partial[j, k] ** 2))
                    p = p * shrink + partial[j, i] * partial[j, k]
                corr[k, i] = p
                corr[i, k] = p
        return corr


class Glmm(Glm):
    """The prior of a generalized linear mixed model (GLMM): a GLM with random effects.

    - Split: the fixed block takes the share φ_fixed of var(η),
      and the factors' blocks split the rest by a Dirichlet.
    - Random effects: each factor has an intercept and slopes on covariate columns,
      up to its own budget q_k, in design columns.
    - Correlation: a factor's effects are correlated with probability correlated_share,
      by an LKJ prior of shape η, and independent otherwise.
    """

    PER_FACTOR: ClassVar[tuple[str, ...]] = ('random_effects', 'lkj_eta')

    fixed_share: Share = Beta(1, 1)  # φ_fixed
    split_alpha: Positive = 1.0  # the Dirichlet's concentration over the factors' blocks
    random_effects: PositiveCount = UniformInt(1, 6)  # q_k per factor, truncated at d
    correlated_share: Share = 0.55
    lkj_eta: PositiveCount = UniformInt(1, 4)  # η per correlated factor

    def _drawSplit(self, blocks: list[Block], rng: np.random.Generator) -> dict[str, float]:
        fixed, *factors = blocks
        # a fixed block without covariate columns adds nothing to var(η), so the factors take all
        fixed_share = self.fixed_share if fixed.variance > 0 else 0.0
        split = {'fixed': fixed_share}
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
