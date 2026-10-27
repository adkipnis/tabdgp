"""outcome.py defines the outcome y, and the GLM and GLMM models thereof.

A GLM codes the covariates as the design matrix and draws y from its likelihood.
A GLMM adds random effects per grouping factor on some of the design's columns.

pipeline:
---------
the covariates (from an SCM draw or a provided table)
-> a model's outcome: its columns, their metadata and its truth:        run
   -> the design matrix (intercept first)                               codeDesign
   -> each factor's block, its random terms within its budget           _buildRandomBlocks
   -> y's family, the stated prior over the blocks, then the parameters Glm.drawPrior,
                                                                        Prior.drawParameters
   -> eta = Xβ + Σ_k Z_k α_k, plus the deviation λ * sd(Xβ + Σ_k Z_k α_k) * score
                                                                        _drawGlmm
   -> y from its likelihood                                             likelihood.draw
   (a study's own model is called instead, and its outcome checked against the covariates)
"""

from __future__ import annotations

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping
from tabdgp.covariates import Covariates, checkNames, checkSameLength
from tabdgp.hyperprior import Model
from tabdgp.likelihood import COLUMNS, LINKS, draw # TODO: from tabdgp import likelihood, then use likelihood.draw down there
from tabdgp.prior import Block, Glm, Glmm, Parameters, Prior

# =====================================================================================
# the design matrix
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Design:
    """The design matrix: the covariates coded for a linear predictor, the intercept first."""

    matrix: np.ndarray  # (n, d) float64
    names: list[str]  # the terms: 'intercept', then each covariate's
    positions: dict[str, list[int]]  # each covariate's columns of the matrix
    means: dict[str, float]  # per column (continuous or count)
    sds: dict[str, float]  # per column population sd over the n rows


def codeDesign(columns: dict[str, np.ndarray], metadata: dict[str, Column], rows: int) -> Design:
    """The design matrix of the covariates (in their order, after the intercept)."""
    names = ['intercept']
    # rows is n, which a table without a covariate cannot tell
    coded = [np.ones(rows)]  # (n,) each
    positions: dict[str, list[int]] = {}
    means: dict[str, float] = {}
    sds: dict[str, float] = {}
    for name, column in metadata.items():
        # a grouping factor and the outcome are left out by role
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


# =====================================================================================
# the outcome and its ground truth
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Outcome:
    """The columns a model writes and their metadata (a model's subclass adds its truth)."""

    columns: dict[str, np.ndarray]  # (n,) each, in its type's dtype
    metadata: dict[str, Column]

    def __post_init__(self) -> None:
        # check that each column has its metadata, the role outcome, the correct dtype,
        # and the same length
        checkNames(self.columns, self.metadata)
        for name, column in self.metadata.items():
            if column.role != 'outcome':
                raise ValueError(f'{name} has the role {column.role}, not outcome')
            dtype = self.columns[name].dtype
            if dtype != column.dtype:
                raise ValueError(
                    f'the {column.type} column {name} is {dtype}, not {column.dtype.__name__}'
                )
        checkSameLength(list(self.columns.values()), 'the outcome')


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ScmOutcome(Outcome):
    """The SCM's y and its truth, the score it rank-matched onto y's marginal."""

    score: np.ndarray  # (n,) float32

    def __post_init__(self) -> None:
        super().__post_init__()
        checkSameLength([*self.columns.values(), self.score], 'the outcome')


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class GlmOutcome(Outcome):
    """A GLM's y and its ground truth, both on the standardized scale and on the raw columns.

    - Ground Truth: the stated prior and the parameters drawn from it (on standardized scale).
    - Terms: the design's columns (intercept first), each with a beta-coefficient.
    - Deviation: the score's part of eta at the deviation scale λ (0 at λ = 0 or without a score).
    - Standardization: each z-scored column's mean and sd, which map beta onto the raw columns.
    """

    link: str  # the canonical link of y's family
    fixed_terms: list[str]  # the design's names
    prior: Prior
    parameters: Parameters
    deviation: np.ndarray  # (n,) float32
    deviation_scale: float  # λ
    means: dict[str, float]  # of the design
    sds: dict[str, float]  # of the design

    def __post_init__(self) -> None:
        super().__post_init__()
        # check that beta has one coefficient per term and the deviation one value per row
        beta_shape = self.parameters.beta.shape
        if beta_shape != (len(self.fixed_terms),):
            raise ValueError(f'beta has the shape {beta_shape}, not one per {self.fixed_terms}')
        checkSameLength([*self.columns.values(), self.deviation], 'the outcome')

    @property
    def beta_raw(self) -> np.ndarray:
        """beta on the raw columns, which gives the same eta as beta on the design matrix."""
        return self._buildRawMap(self.fixed_terms) @ self.parameters.beta  # (d,)

    def _buildRawMap(self, terms: list[str]) -> np.ndarray:
        """The matrix A that maps a block's coefficients (intercept first) onto the raw columns."""
        # a column coded as it is (binary, a dummy) keeps the identity's row
        raw_map = np.eye(len(terms), dtype=np.float32)  # (q, q)
        for j, term in enumerate(terms):
            if term in self.sds:
                # β * (x − mean) / sd = (β / sd) * x − β * mean / sd
                raw_map[j, j] = 1 / self.sds[term]
                raw_map[0, j] = -self.means[term] / self.sds[term]
        return raw_map


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class GlmmOutcome(GlmOutcome):
    """A GLMM's y and its truth: a GLM's, and each grouping factor's random effects.

    - Random terms: a factor's intercept and slopes, terms of the design,
      each with one random effect per group.
    - Covariance: a group's random effects are N(0, Σ), with Σ = diag(sd) corr diag(sd).
    - Raw scale: the random intercept absorbs each slope's α * mean / sd, like the fixed one,
      so Σ's intercept row and column change on the raw columns, not only its diagonal.
    """

    random_terms: dict[str, list[str]]  # by factor, the intercept first

    def __post_init__(self) -> None:
        super().__post_init__()
        # check that every random-effects parameter has each factor, sized by its q terms
        factors = list(self.random_terms)
        parameters = self.parameters
        for field in ['sigma_alpha', 'corr_alpha', 'alpha']:
            named = list(getattr(parameters, field))
            if named != factors:
                raise ValueError(f'{field} has the factors {named}, not {factors}')
        for factor, terms in self.random_terms.items():
            q = len(terms)
            sd_shape = parameters.sigma_alpha[factor].shape
            if sd_shape != (q,):
                raise ValueError(f'sigma_alpha of {factor} has the shape {sd_shape}, not ({q},)')
            corr_shape = parameters.corr_alpha[factor].shape
            if corr_shape != (q, q):
                raise ValueError(
                    f'corr_alpha of {factor} has the shape {corr_shape}, not ({q}, {q})'
                )
            alpha_shape = parameters.alpha[factor].shape
            if len(alpha_shape) != 2 or alpha_shape[1] != q:
                raise ValueError(f'alpha of {factor} has the shape {alpha_shape}, not (m, {q})')

    @property
    def alpha_raw(self) -> dict[str, np.ndarray]:
        """alpha on the raw columns, which gives the same eta as alpha on the design matrix."""
        raw: dict[str, np.ndarray] = {}
        for factor, terms in self.random_terms.items():
            raw[factor] = self.parameters.alpha[factor] @ self._buildRawMap(terms).T  # (m, q)
        return raw

    @property
    def sigma_alpha_raw(self) -> dict[str, np.ndarray]:
        sds: dict[str, np.ndarray] = {}
        for factor, covariance in self._transformCovariances().items():
            sds[factor] = np.sqrt(np.diag(covariance))  # (q,)
        return sds

    def _transformCovariances(self) -> dict[str, np.ndarray]:
        """Each factor's Σ on the raw columns, A Σ A' for the raw map A of its terms."""
        covariances: dict[str, np.ndarray] = {}
        for factor, terms in self.random_terms.items():
            sd = self.parameters.sigma_alpha[factor]
            covariance = self.parameters.corr_alpha[factor] * np.outer(sd, sd)  # (q, q)
            raw_map = self._buildRawMap(terms)  # (q, q)
            covariances[factor] = raw_map @ covariance @ raw_map.T  # (q, q)
        return covariances

    @property
    def corr_alpha_raw(self) -> dict[str, np.ndarray]:
        corrs: dict[str, np.ndarray] = {}
        for factor, covariance in self._transformCovariances().items():
            sd = np.sqrt(np.diag(covariance))  # (q,)
            corrs[factor] = covariance / np.outer(sd, sd)  # (q, q)
        return corrs


# =====================================================================================
# the model's call
# =====================================================================================


def run(model: Model, covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """Draws the outcome of TabDGP's own model, or calls a study's and checks what it returns."""
    match model:
        case Glmm():
            # random effects for every grouping factor (a GLMM needs one)
            factors = list(covariates.grouping.metadata)
            if not factors:
                raise ValueError(
                    f'a {type(model).__name__} draws random effects per grouping factor, '
                    'but the covariates have none: use a Glm'
                )
            return _drawGlmm(model, covariates, factors, rng)
        case Glm():
            return _drawGlmm(model, covariates, [], rng)
        case _:
            # a study's model is checked on what it returns, each failure naming the model
            outcome = model(covariates, rng)
            if not isinstance(outcome, Outcome):
                raise TypeError(f'{model!r} returned a {type(outcome).__name__}, not an Outcome')
            arrays = [*covariates.columns.values(), covariates.score, *outcome.columns.values()]
            checkSameLength(arrays, f'{model!r} and its covariates')
            # the dataset holds both, so a shared name would replace the covariate
            shared = covariates.columns.keys() & outcome.columns.keys()
            if shared:
                raise ValueError(f'{model!r} wrote the covariates {sorted(shared)}')
            return outcome


# -------------------------------------------------------------------------------------
# a GLMM, a GLM without factors: the covariates -> the design matrix -> the blocks -> eta -> y
# -------------------------------------------------------------------------------------


def _drawGlmm(
    glm: Glm, covariates: Covariates, factors: list[str], rng: np.random.Generator
) -> GlmOutcome:
    # TODO: rename glm: Glm to hyperprior: Glm|Glmm
    resolved = glm.resolveDataset(rng)
    rows = covariates.rows

    # the blocks: the fixed effects', then each factor's random effects'
    design = codeDesign(covariates.columns, covariates.metadata, rows)
    # a fixed effect adds its column's variance to var(eta) at scale 1 (the intercept nothing)
    fixed = Block(name='fixed', variances=design.matrix.var(axis=0).tolist()) # TODO: this line is hard to read
    random_blocks, random_terms, random_columns = _buildRandomBlocks(
        resolved, design, covariates.grouping, factors, rng
    )

    # the stated prior over the blocks, then the parameters drawn from it
    prior = resolved.drawPrior([fixed, *random_blocks], rng)
    parameters = prior.drawParameters(rng)

    # eta = Xβ + Σ_k Z_k α_k in float64, so the float32 parameters are its exact truth # TODO: explain the float64 - float32 dynamic once in the header docstring, don't mention them otherwise
    eta = design.matrix @ parameters.beta  # (n,)
    for factor, z in random_columns.items():  # this iterator is empty for GLMs
        alpha = covariates.grouping.broadcast(parameters.alpha[factor], factor, 'row')  # (n, q)
        eta = eta + (z * alpha).sum(axis=1)  # (n,)

    # optionally add the deviation
    deviation = np.zeros(rows, dtype=np.float32)  # (n,)
    if covariates.score is not None:
        linear_sd = float(eta.std())  # the population sd over the n rows
        deviation = resolved.deviation_scale * linear_sd * covariates.score  # (n,) float32
    eta = eta + deviation  # (n,)

    # y from its likelihood, with σ_ε at every row (if y's family has a residual)
    scale = None
    if parameters.sigma_eps is not None:
        scale = np.full(rows, parameters.sigma_eps)  # (n,)
    y = draw(prior.y_family, eta, scale, prior.y_shape, rng)  # (n,)

    # the outcome, a GLMM's with its random terms
    fields = {
        'columns': {'y': y},
        'metadata': {'y': COLUMNS[prior.y_family]},
        'link': LINKS[prior.y_family],
        'fixed_terms': design.names,
        'prior': prior,
        'parameters': parameters,
        'deviation': deviation,
        'deviation_scale': resolved.deviation_scale,
        'means': design.means,
        'sds': design.sds,
    }
    if not factors:
        return GlmOutcome(**fields)
    return GlmmOutcome(**fields, random_terms=random_terms)



def _buildRandomBlocks(
    resolved: Glm,
    design: Design,
    grouping: Grouping,
    factors: list[str],
    rng: np.random.Generator,
) -> tuple[list[Block], dict[str, list[str]], dict[str, np.ndarray]]: # TODO: use dict of dicts instead of tuple
    """Each factor's block of random effects, its random terms and their columns of the design."""
    blocks: list[Block] = []
    random_terms: dict[str, list[str]] = {}
    random_columns: dict[str, np.ndarray] = {}
    for factor in factors:
        budget = resolved.resolveField('random_effects', rng, high=len(design.names))
        positions = _chooseRandomTerms(design, budget, rng)
        z = design.matrix[:, positions]  # (n, q)
        groups = grouping.metadata[factor].levels
        # a random effect adds E[z²] to var(eta) at scale 1 (the intercept 1)
        blocks.append(Block(name=factor, variances=(z**2).mean(axis=0).tolist(), groups=groups))
        random_terms[factor] = [design.names[i] for i in positions]
        random_columns[factor] = z
    return blocks, random_terms, random_columns


def _chooseRandomTerms(design: Design, budget: int, rng: np.random.Generator) -> list[int]:
    """A factor's random terms as sorted positions in the design."""
    positions = [0]  # the intercept
    names = list(design.positions)
    # whole covariates in a random order (each if its columns still fit the budget)
    for i in rng.permutation(len(names)):
        columns = design.positions[names[i]]
        if len(positions) + len(columns) <= budget:
            positions.extend(columns)
    return sorted(positions)
