"""outcome.py defines what a form reads and returns, and TabDGP's own models, the GLM and GLMM.

A form reads the covariates: the observed columns at the rows, their metadata,
the grouping and the score (the outcome node's z-scored value in the SCM, None without one).
A GLM codes the covariates as the design matrix and draws y from its family through the link.
A GLMM adds random effects per grouping factor on some of the design's columns.

pipeline:
---------
the covariates at the rows, their metadata, the grouping and the score  Covariates
-> the columns a form writes, their metadata and its truth              Outcome
-> a model's outcome                                                    run
   a GLMM's, a GLM's without factors:
   -> the design matrix, the intercept first                            codeDesign
   -> each factor's random terms within its budget                      _chooseRandomTerms
   -> the prior's scales and parameters over the blocks                 Glm.drawParameters
   -> η = Xβ + Σ_k Z_k α_k, plus the deviation λ · sd(Xβ + Σ_k Z_k α_k) · score,
      then y from the family through its link                           _drawGlmm
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
from tabdgp.prior import Block, Glm, Glmm, PriorContext

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
    positions: dict[str, list[int]]  # each covariate's columns of the matrix
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
    - Deviation: the score's part of η at the deviation scale λ,
      zero at λ = 0 or without a score.
    - Coding: each continuous column's mean and sd, which map beta onto the raw columns.
    """

    family: str  # 'normal', 'bernoulli' or 'poisson'
    link: str  # the family's canonical link
    fixed_terms: list[str]  # the design's names
    context: PriorContext
    beta: np.ndarray  # (d,) float32
    sigma_eps: float | None  # None unless the family is Normal
    deviation: np.ndarray  # (n,) float32
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
        return self._buildRawMap(self.fixed_terms) @ self.beta  # (d,)

    def _buildRawMap(self, terms: list[str]) -> np.ndarray:
        """The matrix A that takes a block's coefficients, the intercept first,
        from the design matrix to the raw columns.
        A 0/1 column is coded as it is, so its row is the identity's.
        """
        raw_map = np.eye(len(terms), dtype=np.float32)  # (q, q)
        for j, term in enumerate(terms):
            if term in self.sds:
                # β (x − mean) / sd = (β / sd) x − β mean / sd
                raw_map[j, j] = 1 / self.sds[term]
                raw_map[0, j] = -self.means[term] / self.sds[term]
        return raw_map


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class GlmmOutcome(GlmOutcome):
    """A GLMM's y and its truth: a GLM's, and each grouping factor's random effects.

    - Random terms: a factor's intercept and slopes, terms of the design,
      each with one random effect per group.
    - Covariance: a group's random effects are N(0, Σ), with Σ = diag(sd) corr diag(sd).
    - Raw scale: the random intercept absorbs each slope's α · mean / sd, like the fixed one,
      so Σ's intercept row and column change on the raw columns, not only its diagonal.
    """

    random_terms: dict[str, list[str]]  # by factor, the intercept first
    sd_alpha: dict[str, np.ndarray]  # (q,) float32 by factor
    corr_alpha: dict[str, np.ndarray]  # (q, q) float32 by factor
    alpha: dict[str, np.ndarray]  # (m, q) float32 by factor, a row per group

    def __post_init__(self) -> None:
        super().__post_init__()
        # check that every field has each factor, sized by the factor's q terms
        factors = list(self.random_terms)
        for field in ['sd_alpha', 'corr_alpha', 'alpha']:
            named = list(getattr(self, field))
            if named != factors:
                raise ValueError(f'{field} has the factors {named}, not {factors}')
        for factor, terms in self.random_terms.items():
            q = len(terms)
            sd_shape = self.sd_alpha[factor].shape
            if sd_shape != (q,):
                raise ValueError(f'sd_alpha of {factor} has the shape {sd_shape}, not ({q},)')
            corr_shape = self.corr_alpha[factor].shape
            if corr_shape != (q, q):
                raise ValueError(
                    f'corr_alpha of {factor} has the shape {corr_shape}, not ({q}, {q})'
                )
            alpha_shape = self.alpha[factor].shape
            if len(alpha_shape) != 2 or alpha_shape[1] != q:
                raise ValueError(f'alpha of {factor} has the shape {alpha_shape}, not (m, {q})')

    @property
    def alpha_raw(self) -> dict[str, np.ndarray]:
        """alpha on the raw columns, which gives the same η as alpha on the design matrix."""
        raw = {}
        for factor, terms in self.random_terms.items():
            raw[factor] = self.alpha[factor] @ self._buildRawMap(terms).T  # (m, q)
        return raw

    @property
    def sd_alpha_raw(self) -> dict[str, np.ndarray]:
        sds = {}
        for factor, covariance in self._transformCovariances().items():
            sds[factor] = np.sqrt(np.diag(covariance))  # (q,)
        return sds

    @property
    def corr_alpha_raw(self) -> dict[str, np.ndarray]:
        sds = self.sd_alpha_raw
        corrs = {}
        for factor, covariance in self._transformCovariances().items():
            corrs[factor] = covariance / np.outer(sds[factor], sds[factor])  # (q, q)
        return corrs

    def _transformCovariances(self) -> dict[str, np.ndarray]:
        """Each factor's Σ on the raw columns, A Σ A' for the raw map A of its terms."""
        covariances = {}
        for factor, terms in self.random_terms.items():
            sd = self.sd_alpha[factor]
            covariance = self.corr_alpha[factor] * np.outer(sd, sd)  # (q, q)
            raw_map = self._buildRawMap(terms)  # (q, q)
            covariances[factor] = raw_map @ covariance @ raw_map.T  # (q, q)
        return covariances


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
        case Glmm():
            # random effects for every grouping factor
            return _drawGlmm(model, covariates, list(covariates.grouping.metadata), rng)
        case Glm():
            return _drawGlmm(model, covariates, [], rng)
        case _:
            return _callModel(model, covariates, rng)


# -------------------------------------------------------------------------------------
# a GLMM, a GLM without factors: the covariates -> the design matrix -> the blocks -> η -> y
# -------------------------------------------------------------------------------------


def _drawGlmm(
    glm: Glm, covariates: Covariates, factors: list[str], rng: np.random.Generator
) -> GlmOutcome:
    resolved = glm.resolveDataset(rng)
    rows = covariates.rows
    design = codeDesign(covariates.columns, covariates.metadata, rows)
    name = resolved.family_shares.choose(rng)
    family = FAMILIES[name]
    # the fixed term adds tr cov X to var(η) at scale 1, the intercept nothing
    variance = float(design.matrix.var(axis=0).sum())
    blocks = [Block(name='fixed', width=len(design.names), variance=variance)]
    random_terms = {}
    random_columns = {}
    for factor in factors:
        budget = resolved.resolveField('random_effects', rng, high=len(design.names))
        positions = _chooseRandomTerms(design, budget, rng)
        z = design.matrix[:, positions]  # (n, q)
        # a random term adds tr E[z z'] to var(η) at scale 1, its intercept 1
        variance = float((z**2).mean(axis=0).sum())
        groups = covariates.grouping.metadata[factor].levels
        blocks.append(Block(name=factor, width=len(positions), variance=variance, groups=groups))
        random_terms[factor] = [design.names[i] for i in positions]
        random_columns[factor] = z
    context, parameters = resolved.drawParameters(name, blocks, rng)
    # η in float64 from the float32 parameters, which are so its exact truth
    eta = design.matrix @ parameters.beta  # (n,)
    for factor, z in random_columns.items():
        codes = covariates.grouping.codes[factor]  # (n,)
        eta = eta + (z * parameters.alpha[factor][codes]).sum(axis=1)  # (n,)
    deviation = np.zeros(rows, dtype=np.float32)  # (n,), a table's, which has no score
    if covariates.score is not None:
        # the score at λ times η's sd so far, the population sd over the n rows
        linear_sd = float(eta.std())
        deviation = resolved.deviation_scale * linear_sd * covariates.score  # (n,) float32
    eta = eta + deviation  # (n,)
    scale = None
    if parameters.sigma_eps is not None:
        scale = np.full(rows, parameters.sigma_eps)  # (n,)
    y = family.draw(rng, eta, scale)  # (n,)
    truth = {
        'columns': {'y': y},
        'metadata': {'y': family.COLUMN},
        'family': name,
        'link': family.LINK,
        'fixed_terms': design.names,
        'context': context,
        'beta': parameters.beta,
        'sigma_eps': parameters.sigma_eps,
        'deviation': deviation,
        'deviation_scale': resolved.deviation_scale,
        'means': design.means,
        'sds': design.sds,
    }
    if not factors:
        return GlmOutcome(**truth)
    return GlmmOutcome(
        **truth,
        random_terms=random_terms,
        sd_alpha=parameters.sd_alpha,
        corr_alpha=parameters.corr_alpha,
        alpha=parameters.alpha,
    )


def codeDesign(columns: dict[str, np.ndarray], metadata: dict[str, Column], rows: int) -> Design:
    """The design matrix of the covariates, in their order after the intercept.
    A grouping factor and the outcome are left out by role.
    rows is n, which a table without a covariate cannot tell.
    """
    names = ['intercept']
    coded = [np.ones(rows)]  # (n,) each
    positions = {}
    means = {}
    sds = {}
    for name, column in metadata.items():
        if column.role != 'covariate':
            continue
        start = len(names)
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
        positions[name] = list(range(start, len(names)))
    matrix = np.stack(coded, axis=1)  # (n, d)
    return Design(matrix=matrix, names=names, positions=positions, means=means, sds=sds)


def _chooseRandomTerms(design: Design, budget: int, rng: np.random.Generator) -> list[int]:
    """A factor's random terms, as positions in the design:
    the intercept, then whole covariates in a random order, each while it fits the budget.
    """
    positions = [0]
    covariates = list(design.positions)
    for i in rng.permutation(len(covariates)):
        columns = design.positions[covariates[i]]
        if len(positions) + len(columns) <= budget:
            positions.extend(columns)
    return sorted(positions)


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
