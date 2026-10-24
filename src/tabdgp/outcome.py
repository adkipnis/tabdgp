"""outcome.py defines what a form reads and what it returns, and TabDGP's own model, the GLM.

A form reads the covariates: the observed columns at the rows, their metadata,
the grouping and the score (the outcome node's z-scored value in the SCM, None without one).
A GLM codes the covariates as the design matrix and draws y from its family through the link.

pipeline:
---------
the covariates at the rows, their metadata, the grouping and the score  Covariates
-> the columns a form writes, their metadata and its truth              Outcome
-> a model's outcome                                                    run
   a GLM's:
   -> the design matrix, the intercept first                            codeDesign
   -> the prior's scales and parameters over the block fixed            Glm.drawParameters
   -> η = Xβ, then y from the family through its canonical link         _drawGlm
   a study's:
   -> its call, its result checked at the covariates' rows              _callModel
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping
from tabdgp.hyperprior import Model
from tabdgp.prior import Block, Glm, PriorContext

# =====================================================================================
# the checks
# =====================================================================================
# the covariates, an outcome and a model's result each hold columns at one set of rows.


def _checkOneLength(arrays: list[np.ndarray | None], of: str) -> None:
    lengths = {len(array) for array in arrays if array is not None}
    if len(lengths) > 1:
        raise ValueError(f'the arrays of {of} have the lengths {sorted(lengths)}, not one')


def _checkNames(columns: dict[str, np.ndarray], metadata: dict[str, Column]) -> None:
    if columns.keys() != metadata.keys():
        raise ValueError(f'the columns {list(columns)} have metadata {list(metadata)}')


# =====================================================================================
# what a form reads
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Covariates:
    """The observed columns a form reads: the grouping factors' codes and the covariates."""

    columns: dict[str, np.ndarray]  # (n,) each
    metadata: dict[str, Column]
    grouping: Grouping
    score: np.ndarray | None  # (n,) float32, None for a table without an SCM

    def __post_init__(self) -> None:
        # check that every array has one length, each column its metadata, and none is y
        _checkOneLength([*self.columns.values(), self.score], 'the covariates')
        _checkNames(self.columns, self.metadata)
        if 'y' in self.columns:
            raise ValueError("a covariate cannot be named 'y', the outcome's name")

    @property
    def rows(self) -> int:
        """n, the length of every array, the score's in a table without a column."""
        for array in [*self.columns.values(), self.score]:
            if array is not None:
                return len(array)
        raise ValueError('covariates without a column or a score have no rows')


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Design:
    """The design matrix: the covariates coded for a linear predictor, the intercept first."""

    matrix: np.ndarray  # (n, d) float64
    names: list[str]  # the terms: 'intercept', then each covariate's
    means: dict[str, float]  # each continuous column's, before z-scoring
    sds: dict[str, float]  # each continuous column's population sd over the n rows


# =====================================================================================
# what a form returns
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Outcome:
    """The columns a form writes (y today) and their metadata.
    A form's own subclass adds its truth.
    """

    columns: dict[str, np.ndarray]  # (n,) each, in its type's dtype
    metadata: dict[str, Column]

    def __post_init__(self) -> None:
        # check that each column has its metadata, the role outcome, its type's dtype and one length
        _checkNames(self.columns, self.metadata)
        for name, column in self.metadata.items():
            if column.role != 'outcome':
                raise ValueError(f'{name} has the role {column.role}, not outcome')
            dtype = self.columns[name].dtype
            if dtype != column.dtype:
                raise ValueError(
                    f'the {column.type} column {name} is {dtype}, not {column.dtype.__name__}'
                )
        _checkOneLength(list(self.columns.values()), 'the outcome')


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ScmOutcome(Outcome):
    """The scm form's y and its truth, the score it rank-matched onto y's marginal."""

    score: np.ndarray  # (n,) float32

    def __post_init__(self) -> None:
        super().__post_init__()
        _checkOneLength([*self.columns.values(), self.score], 'the outcome')


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class GlmOutcome(Outcome):
    """A GLM's y and its truth, on the standardized scale as drawn and on the raw columns.

    - Terms: the design's columns, the intercept first, each with one coefficient in beta.
    - Deviation: the score's part of η at the deviation scale λ, zero at λ = 0.
    - Coding: each continuous column's mean and sd, which map beta onto the raw columns.
    """

    family: str  # 'normal', 'bernoulli' or 'poisson'
    link: str  # the family's canonical link
    fixed_terms: list[str]  # the design's names
    context: PriorContext
    beta: np.ndarray  # (d,) float64
    sigma_eps: float | None  # None unless the family is Normal
    deviation: np.ndarray  # (n,) float64
    deviation_scale: float  # λ
    means: dict[str, float]  # the design's
    sds: dict[str, float]  # the design's

    def __post_init__(self) -> None:
        super().__post_init__()
        # check that beta has one coefficient per term and the deviation one value per row
        if self.beta.shape != (len(self.fixed_terms),):
            raise ValueError(
                f'beta has the shape {self.beta.shape}, not one per {self.fixed_terms}'
            )
        _checkOneLength([*self.columns.values(), self.deviation], 'the outcome')

    @property
    def beta_raw(self) -> np.ndarray:
        """beta on the raw columns, which gives the same η as beta on the design matrix."""
        intercept = float(self.beta[0])
        slopes = []
        for term, coefficient in zip(self.fixed_terms[1:], self.beta[1:]):
            if term in self.sds:
                # β (x − mean) / sd = (β / sd) x − β mean / sd
                slope = float(coefficient) / self.sds[term]
                intercept -= slope * self.means[term]
            else:
                slope = float(coefficient)  # a 0/1 column is coded as it is
            slopes.append(slope)
        return np.array([intercept, *slopes])  # (d,)


# =====================================================================================
# the families
# =====================================================================================
# each draws y from its linear predictor η through its canonical link,
# at a residual scale per row that only the Normal family has (None for the others).


class Normal:
    """y ~ N(η, σ_ε), the identity link."""

    LINK = 'identity'
    COLUMN = Column(type='continuous', role='outcome')

    @staticmethod
    def draw(rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None) -> np.ndarray:
        noise = rng.standard_normal(len(eta))  # (n,)
        y = eta + scale * noise  # (n,)
        return y.astype(np.float32)


class Bernoulli:
    """y ~ Bernoulli(p), η the logit of p."""

    LINK = 'logit'
    COLUMN = Column(type='binary', levels=2, role='outcome')

    @staticmethod
    def draw(rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None) -> np.ndarray:
        p = 0.5 * (1 + np.tanh(eta / 2))  # (n,) the inverse logit, without exp's overflow
        uniforms = rng.random(len(eta))  # (n,)
        return (uniforms < p).astype(np.int64)


class Poisson:
    """y ~ Poisson(λ), η the log of λ."""

    LINK = 'log'
    COLUMN = Column(type='count', role='outcome')
    RATE_LIMIT = 9.2e18  # numpy refuses a rate above about 9.22e18, int64's largest count

    @staticmethod
    def draw(rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None) -> np.ndarray:
        # above the limit no int64 holds the count, so y saturates at int64's largest
        drawn = eta < math.log(Poisson.RATE_LIMIT)  # (n,)
        rate = np.exp(np.where(drawn, eta, 0.0))  # (n,) float64, exp(η) wherever it is drawn
        counts = rng.poisson(rate)  # (n,) int64
        return np.where(drawn, counts, np.iinfo(np.int64).max)


FAMILIES = {'normal': Normal, 'bernoulli': Bernoulli, 'poisson': Poisson}


# =====================================================================================
# the model's call
# =====================================================================================


def run(model: Model, covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """Draws the outcome of TabDGP's own model, or of a study's, from the covariates."""
    match model:
        case Glm():
            return _drawGlm(model, covariates, rng)
        case _:
            return _callModel(model, covariates, rng)


# -------------------------------------------------------------------------------------
# a GLM: the covariates -> the design matrix -> η -> y
# -------------------------------------------------------------------------------------


def _drawGlm(glm: Glm, covariates: Covariates, rng: np.random.Generator) -> GlmOutcome:
    resolved = glm.resolveDataset(rng)
    rows = covariates.rows
    design = codeDesign(covariates.columns, covariates.metadata, rows)
    name = resolved.family_shares.choose(rng)
    family = FAMILIES[name]
    # the fixed term adds tr cov X to var(η) at scale 1, the intercept nothing
    variance = float(design.matrix.var(axis=0).sum())
    fixed = Block(name='fixed', width=len(design.names), variance=variance)
    context, parameters = resolved.drawParameters(name, [fixed], rng)
    deviation = np.zeros(rows)  # (n,)
    eta = design.matrix @ parameters.beta + deviation  # (n,)
    scale = None
    if parameters.sigma_eps is not None:
        scale = np.full(rows, parameters.sigma_eps)  # (n,)
    y = family.draw(rng, eta, scale)  # (n,)
    return GlmOutcome(
        columns={'y': y},
        metadata={'y': family.COLUMN},
        family=name,
        link=family.LINK,
        fixed_terms=design.names,
        context=context,
        beta=parameters.beta,
        sigma_eps=parameters.sigma_eps,
        deviation=deviation,
        deviation_scale=resolved.deviation_scale,
        means=design.means,
        sds=design.sds,
    )


def codeDesign(columns: dict[str, np.ndarray], metadata: dict[str, Column], rows: int) -> Design:
    """The design matrix of the covariates, in their order after the intercept.
    A grouping factor and the outcome are left out by role.
    rows is n, which a table without a covariate cannot tell.
    """
    names = ['intercept']
    coded = [np.ones(rows)]  # (n,) each
    means = {}
    sds = {}
    for name, column in metadata.items():
        if column.role != 'covariate':
            continue
        values = columns[name].astype(np.float64)  # (n,)
        if column.type in ['continuous', 'count']:
            means[name] = float(values.mean())
            sds[name] = float(values.std())  # the population sd, over the n rows
            names.append(name)
            coded.append((values - means[name]) / sds[name])
        elif column.type == 'binary':
            names.append(name)
            coded.append(values)
        else:
            # k − 1 dummies, the first level in the codes' order dropped
            for level in range(1, column.levels):
                names.append(f'{name}={level}')
                coded.append((values == level).astype(np.float64))
    matrix = np.stack(coded, axis=1)  # (n, d)
    return Design(matrix=matrix, names=names, means=means, sds=sds)


# -------------------------------------------------------------------------------------
# a study's model: its call -> its result checked
# -------------------------------------------------------------------------------------


def _callModel(model: Model, covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """Calls a study's model on the covariates and the generator.
    What it returns must be an Outcome at the covariates' rows; a failure names the model.
    """
    outcome = model(covariates, rng)
    if not isinstance(outcome, Outcome):
        raise TypeError(f'{model!r} returned a {type(outcome).__name__}, not an Outcome')
    arrays = [*covariates.columns.values(), covariates.score, *outcome.columns.values()]
    _checkOneLength(arrays, f'{model!r} and its covariates')
    # the dataset holds both, so a shared name would replace the covariate
    shared = covariates.columns.keys() & outcome.columns.keys()
    if shared:
        raise ValueError(f'{model!r} wrote the covariates {sorted(shared)}')
    return outcome
