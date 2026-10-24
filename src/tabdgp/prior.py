"""prior.py defines the GLM's prior: its config, the joint draw of its scales and its parameters.

The scales are drawn jointly per dataset.
The predictor sd τ sets the sd of η without the intercept, and the split φ shares var(η) among
the blocks of coefficients.
Each block's scale divides its share by the variance its term adds to η at scale 1,
so that a block of dummies realizes the share a block of z-scored columns does.
Given the scales, each parameter is drawn from its stated family exactly:
nothing is clipped, rerolled or rescaled.
The draw reads a block's size, never its columns.

pipeline:
---------
the (optional) overrides
-> validation at construction                                 Glm
-> a point for each field                                     Glm.resolveDataset
-> given y's family and the blocks, the families chosen,
   r, τ, φ and each block's scale                              _drawContext
-> β and σ_ε from their families at those scales               Glm.drawParameters
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np
from pydantic import ConfigDict, NonNegativeFloat, PositiveInt
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.field import Beta, Fields, LogUniform, NonNegative, Share, Simplex

# =====================================================================================
# what the draw reads and returns
# =====================================================================================


@pydantic_dataclass(frozen=True)
class Block:
    """A block of η's coefficients drawn at one scale, the fixed effects for a GLM.
    The draw reads its size, never its columns.
    """

    name: str  # 'fixed'
    width: PositiveInt  # its coefficients, the intercept included
    variance: NonNegativeFloat  # what its term adds to var(η) at scale 1: tr cov X if fixed


@pydantic_dataclass(frozen=True)
class PriorContext:
    """The resolved hyperparameters a GLM's parameters were drawn with.

    - Families: F, the sd family chosen, and the Normal family's residual law.
    - Signal: the signal share r and the predictor sd τ, the sd of η without the intercept.
    - Split: each block's share φ of var(η), and its scale, σ_β for the fixed block.
    - Residual: σ_ε is drawn from F at the residual scale s_ε, under the Normal family only.
    """

    sd_family: str
    residual_law: str | None  # None unless the family is Normal
    signal_share: float  # r
    predictor_sd: float  # τ
    split: dict[str, float]  # φ by block name
    scales: dict[str, float]  # by block name
    intercept_scale: float  # σ_0
    residual_scale: float | None  # s_ε, None unless the family is Normal


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Parameters:
    """A GLM's parameters on the standardized scale, as drawn."""

    beta: np.ndarray  # (d,) float64, the intercept first
    sigma_eps: float | None  # None unless the family is Normal


# =====================================================================================
# the config
# =====================================================================================


class Glm(Fields):
    """The prior of a generalized linear model (GLM) over the design matrix.

    - Families: y's response family, with its canonical link,
      and the family F the residual scale is drawn from.
    - Scales: the predictor sd τ is the sd of η without the intercept.
      Under the Normal family τ² = r, the signal share r being the share of y's variance
      that η explains, so that y has unit variance.
    - Deviation: the score added to η at deviation_scale · sd(η).
    """

    SHARES: ClassVar[tuple[str, ...]] = ('family_shares', 'sd_family_shares', 'residual_shares')

    family_shares: Simplex = {'normal': 1 / 3, 'bernoulli': 1 / 3, 'poisson': 1 / 3}
    sd_family_shares: Simplex = {
        'half_normal': 0.25,
        'half_student_t': 0.25,  # ν = 3
        'half_cauchy': 0.25,
        'exponential': 0.25,
    }  # F, the family of the residual scale σ_ε
    residual_shares: Simplex = {'normal': 1.0}  # the Normal family's residual law
    intercept_scale: NonNegative = LogUniform(0.1, 10)  # σ_0
    signal_share: Share = Beta(2, 2)  # r
    predictor_sd: NonNegative = LogUniform(0.25, 4)  # τ, unless the family is Normal
    deviation_scale: NonNegative = 0.0  # λ

    def resolveDataset(self, rng: np.random.Generator) -> Glm:
        """Resolves every field, in declaration order."""
        points = {}
        for name in type(self).model_fields:
            points[name] = self.resolveField(name, rng)
        return type(self)(**points)

    def drawParameters(
        self, family: str, blocks: list[Block], rng: np.random.Generator
    ) -> tuple[PriorContext, Parameters]:
        """Draws the scales jointly, then each parameter from its family at its scale exactly.
        On a resolved Glm, for the response family chosen.
        """
        context = self._drawContext(family, blocks, rng)
        [fixed] = blocks
        intercept = rng.normal(0, context.intercept_scale, size=1)  # (1,)
        slopes = rng.normal(0, context.scales['fixed'], size=fixed.width - 1)  # (d − 1,)
        beta = np.concatenate([intercept, slopes])  # (d,)
        sigma_eps = None
        if context.residual_scale is not None:
            sigma_eps = self._drawSd(context.sd_family, context.residual_scale, rng)
        return context, Parameters(beta=beta, sigma_eps=sigma_eps)

    def _drawContext(
        self, family: str, blocks: list[Block], rng: np.random.Generator
    ) -> PriorContext:
        normal = family == 'normal'
        sd_family = self.sd_family_shares.choose(rng)
        residual_law = self.residual_shares.choose(rng) if normal else None
        # under the Normal family τ² = r, so that y has unit variance
        predictor_sd = math.sqrt(self.signal_share) if normal else self.predictor_sd
        split = {'fixed': 1.0}  # a GLM's one block takes all of var(η)
        scales = {}
        for block in blocks:
            # a block without covariate columns adds nothing to var(η), so no budget reaches it
            budget = split[block.name] / block.variance if block.variance > 0 else 0.0
            scales[block.name] = predictor_sd * math.sqrt(budget)
        residual_scale = math.sqrt(1 - self.signal_share) if normal else None
        return PriorContext(
            sd_family=sd_family,
            residual_law=residual_law,
            signal_share=self.signal_share,
            predictor_sd=predictor_sd,
            split=split,
            scales=scales,
            intercept_scale=self.intercept_scale,
            residual_scale=residual_scale,
        )

    @staticmethod
    def _drawSd(family: str, scale: float, rng: np.random.Generator) -> float:
        """A draw from the sd family F at its scale: the scale times |X|, X F's standard law."""
        match family:
            case 'half_normal':
                standard = rng.standard_normal()
            case 'half_student_t':
                standard = rng.standard_t(3)
            case 'half_cauchy':
                standard = rng.standard_cauchy()
            case 'exponential':
                standard = rng.standard_exponential()
        return scale * abs(float(standard))
