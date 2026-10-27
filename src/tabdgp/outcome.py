"""outcome.py defines what a form returns, and TabDGP's own models, the GLM and GLMM.

A form reads the covariates and writes its columns, with its truth.
A GLM codes the covariates as the design matrix and draws y from its family through the link.
A GLMM adds random effects per grouping factor on some of the design's columns.

pipeline:
---------
the covariates, from a draw or a table
-> the columns a form writes, their metadata and its truth              Outcome
-> a model's outcome                                                    run
   a GLMM's, a GLM's without factors:
   -> the design matrix, the intercept first                            codeDesign
   -> each factor's block, its random terms within its budget           _buildRandomBlocks
   -> y's family, the stated prior over the blocks, then the parameters Glm.drawPrior,
                                                                        Prior.drawParameters
   -> η = Xβ + Σ_k Z_k α_k, plus the deviation λ · sd(Xβ + Σ_k Z_k α_k) · score,
      then y from the family through its link                           _drawGlmm
   a study's:
   -> its call, its result checked at the covariates' rows              _callModel
"""

from __future__ import annotations

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column, Grouping
from tabdgp.covariates import Covariates, checkNames, checkOneLength
from tabdgp.hyperprior import Model
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
        checkNames(self.columns, self.metadata)
        for name, column in self.metadata.items():
            if column.role != 'outcome':
                raise ValueError(f'{name} has the role {column.role}, not outcome')
            dtype = self.columns[name].dtype
            if dtype != column.dtype:
                raise ValueError(
                    f'the {column.type} column {name} is {dtype}, not {column.dtype.__name__}'
                )
        checkOneLength(list(self.columns.values()), 'the outcome')


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class ScmOutcome(Outcome):
    """The scm form's y and its truth, the score it rank-matched onto y's marginal."""

    score: np.ndarray  # (n,) float32

    def __post_init__(self) -> None:
        super().__post_init__()
        checkOneLength([*self.columns.values(), self.score], 'the outcome')


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class GlmOutcome(Outcome):
    """A GLM's y and its truth, on the standardized scale as drawn and on the raw columns.

    - Truth: the stated prior and the parameters drawn from it, on the standardized scale.
    - Terms: the design's columns, the intercept first, each with one coefficient in beta.
    - Deviation: the score's part of η at the deviation scale λ,
      zero at λ = 0 or without a score.
    - Coding: each continuous column's mean and sd, which map beta onto the raw columns.
    """

    link: str  # the family's canonical link
    fixed_terms: list[str]  # the design's names
    prior: Prior
    parameters: Parameters
    deviation: np.ndarray  # (n,) float32
    deviation_scale: float  # λ
    means: dict[str, float]  # the design's
    sds: dict[str, float]  # the design's

    def __post_init__(self) -> None:
        super().__post_init__()
        # check that beta has one coefficient per term and the deviation one value per row
        beta_shape = self.parameters.beta.shape
        if beta_shape != (len(self.fixed_terms),):
            raise ValueError(f'beta has the shape {beta_shape}, not one per {self.fixed_terms}')
        checkOneLength([*self.columns.values(), self.deviation], 'the outcome')

    @property
    def beta_raw(self) -> np.ndarray:
        """beta on the raw columns, which gives the same η as beta on the design matrix."""
        return self._buildRawMap(self.fixed_terms) @ self.parameters.beta  # (d,)

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
        """alpha on the raw columns, which gives the same η as alpha on the design matrix."""
        raw = {}
        for factor, terms in self.random_terms.items():
            raw[factor] = self.parameters.alpha[factor] @ self._buildRawMap(terms).T  # (m, q)
        return raw

    @property
    def sigma_alpha_raw(self) -> dict[str, np.ndarray]:
        sds = {}
        for factor, covariance in self._transformCovariances().items():
            sds[factor] = np.sqrt(np.diag(covariance))  # (q,)
        return sds

    @property
    def corr_alpha_raw(self) -> dict[str, np.ndarray]:
        corrs = {}
        for factor, covariance in self._transformCovariances().items():
            sd = np.sqrt(np.diag(covariance))  # (q,)
            corrs[factor] = covariance / np.outer(sd, sd)  # (q, q)
        return corrs

    def _transformCovariances(self) -> dict[str, np.ndarray]:
        """Each factor's Σ on the raw columns, A Σ A' for the raw map A of its terms."""
        covariances = {}
        for factor, terms in self.random_terms.items():
            sd = self.parameters.sigma_alpha[factor]
            covariance = self.parameters.corr_alpha[factor] * np.outer(sd, sd)  # (q, q)
            raw_map = self._buildRawMap(terms)  # (q, q)
            covariances[factor] = raw_map @ covariance @ raw_map.T  # (q, q)
        return covariances


# =====================================================================================
# the families
# =====================================================================================
# each draws y from its linear predictor η through its canonical link,
# at a residual scale per row that only the Normal and Student-t families have (None otherwise)
# and a shape, the family's second parameter, that only the contaminated families have.


class Normal:
    """y ~ N(η, σ_ε), the identity link."""

    LINK = 'identity'
    COLUMN = Column(type='continuous', role='outcome')

    @staticmethod
    def draw(
        rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None, shape: float | None
    ) -> np.ndarray:
        noise = rng.standard_normal(len(eta))  # (n,)
        y = eta + scale * noise  # (n,)
        return y.astype(Normal.COLUMN.dtype)


class Bernoulli:
    """y ~ Bernoulli(p), η the logit of p."""

    LINK = 'logit'
    COLUMN = Column(type='binary', levels=2, role='outcome')

    @staticmethod
    def draw(
        rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None, shape: float | None
    ) -> np.ndarray:
        p = Bernoulli.invertLogit(eta)  # (n,)
        uniforms = rng.random(len(eta))  # (n,)
        return (uniforms < p).astype(Bernoulli.COLUMN.dtype)

    @staticmethod
    def invertLogit(eta: np.ndarray) -> np.ndarray:
        """The inverse logit, without exp's overflow."""
        return 0.5 * (1 + np.tanh(eta / 2))


class Poisson:
    """y ~ Poisson(λ), η the log of λ."""

    LINK = 'log'
    COLUMN = Column(type='count', role='outcome')
    RATE_LIMIT = 9.2e18  # numpy refuses a rate above about 9.22e18, int64's largest count

    @staticmethod
    def draw(
        rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None, shape: float | None
    ) -> np.ndarray:
        rate = np.exp(np.minimum(eta, 700.0))  # (n,) float64, exp's overflow cut at 700
        return Poisson.drawCounts(rng, rate)

    @staticmethod
    def drawCounts(rng: np.random.Generator, rate: np.ndarray) -> np.ndarray:
        """Poisson counts at a rate per row, saturated at int64's largest above the limit,
        where no int64 holds the count.
        """
        drawn = rate < Poisson.RATE_LIMIT  # (n,)
        counts = rng.poisson(np.where(drawn, rate, 1.0))  # (n,) int64
        return np.where(drawn, counts, np.iinfo(np.int64).max)


class StudentT:
    """y = η + σ_ε t_df, the identity link, the shape df."""

    LINK = 'identity'
    COLUMN = Column(type='continuous', role='outcome')

    @staticmethod
    def draw(
        rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None, shape: float | None
    ) -> np.ndarray:
        noise = rng.standard_t(shape, len(eta))  # (n,)
        y = eta + scale * noise  # (n,)
        return y.astype(StudentT.COLUMN.dtype)


class NegativeBinomial:
    """y ~ Poisson(λ) with λ ~ Gamma(φ, mean e^η), the log link, the shape φ:
    the mean is e^η and the variance μ (1 + μ / φ), the Poisson's at φ → ∞.
    """

    LINK = 'log'
    COLUMN = Column(type='count', role='outcome')

    @staticmethod
    def draw(
        rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None, shape: float | None
    ) -> np.ndarray:
        mean = np.exp(np.minimum(eta, 700.0))  # (n,)
        # numpy's negative_binomial raises on a gamma rate above the Poisson's limit, which
        # this draws in two steps to saturate
        rate = rng.gamma(shape, mean / shape)  # (n,)
        return Poisson.drawCounts(rng, rate)


class LatentBernoulli:
    """y ~ Bernoulli(expit(η + σ_c ε)) with ε standard normal, the logit link, the shape σ_c."""

    LINK = 'logit'
    COLUMN = Column(type='binary', levels=2, role='outcome')

    @staticmethod
    def draw(
        rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None, shape: float | None
    ) -> np.ndarray:
        latent = eta + shape * rng.standard_normal(len(eta))  # (n,)
        return Bernoulli.draw(rng, latent, None, None)


class Gamma:
    """y ~ Gamma(k, mean e^η), the log link, the shape k: the variance is μ² / k.
    y is held inside float32's positive range, which a fitter's Gamma likelihood needs.
    """

    LINK = 'log'
    COLUMN = Column(type='continuous', role='outcome')

    @staticmethod
    def draw(
        rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None, shape: float | None
    ) -> np.ndarray:
        mean = np.exp(np.clip(eta, -700.0, 700.0))  # (n,)
        y = rng.gamma(shape, mean / shape)  # (n,)
        limits = np.finfo(np.float32)
        return np.clip(y, limits.tiny, limits.max).astype(Gamma.COLUMN.dtype)


class Beta:
    """y ~ Beta(μ φ, (1 − μ) φ), the logit link, the shape φ, the precision: the mean is
    μ = expit(η) and the variance μ (1 − μ) / (φ + 1).
    y is held inside float32's open interval (0, 1), which a fitter's beta likelihood needs.
    """

    LINK = 'logit'
    COLUMN = Column(type='continuous', role='outcome')
    MEAN_EDGE = 2.0**-53  # keeps both parameters positive, which numpy needs, at any η

    @staticmethod
    def draw(
        rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None, shape: float | None
    ) -> np.ndarray:
        mean = np.clip(Bernoulli.invertLogit(eta), Beta.MEAN_EDGE, 1 - Beta.MEAN_EDGE)  # (n,)
        y = rng.beta(mean * shape, (1 - mean) * shape)  # (n,)
        below_one = 1 - 2.0**-24  # the largest float32 below 1
        return np.clip(y, np.finfo(np.float32).tiny, below_one).astype(Beta.COLUMN.dtype)


Y_FAMILIES = {
    'normal': Normal,
    'bernoulli': Bernoulli,
    'poisson': Poisson,
    'student_t': StudentT,
    'negative_binomial': NegativeBinomial,
    'latent_bernoulli': LatentBernoulli,
    'gamma': Gamma,
    'beta': Beta,
}


# =====================================================================================
# the model's call
# =====================================================================================


def run(model: Model, covariates: Covariates, rng: np.random.Generator) -> Outcome:
    """Draws the outcome of TabDGP's own model, or of a study's, from the covariates."""
    match model:
        case Glmm():
            # random effects for every grouping factor, of which a GLMM needs one
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
    # a fixed effect adds its column's variance to var(η) at scale 1, the intercept nothing
    fixed = Block(name='fixed', variances=design.matrix.var(axis=0).tolist())
    random_blocks, random_terms, random_columns = _buildRandomBlocks(
        resolved, design, covariates.grouping, factors, rng
    )
    prior = resolved.drawPrior([fixed, *random_blocks], rng)
    family = Y_FAMILIES[prior.y_family]
    parameters = prior.drawParameters(rng)
    # η in float64 from the float32 parameters, which are so its exact truth
    eta = design.matrix @ parameters.beta  # (n,)
    for factor, z in random_columns.items():
        alpha = covariates.grouping.broadcast(parameters.alpha[factor], factor, 'row')  # (n, q)
        eta = eta + (z * alpha).sum(axis=1)  # (n,)
    deviation = np.zeros(rows, dtype=np.float32)  # (n,), a table's, which has no score
    if covariates.score is not None:
        # the score at λ times η's sd so far, the population sd over the n rows
        linear_sd = float(eta.std())
        deviation = resolved.deviation_scale * linear_sd * covariates.score  # (n,) float32
    eta = eta + deviation  # (n,)
    scale = None
    if parameters.sigma_eps is not None:
        scale = np.full(rows, parameters.sigma_eps)  # (n,)
    y = family.draw(rng, eta, scale, prior.y_shape)  # (n,)
    truth = {
        'columns': {'y': y},
        'metadata': {'y': family.COLUMN},
        'link': family.LINK,
        'fixed_terms': design.names,
        'prior': prior,
        'parameters': parameters,
        'deviation': deviation,
        'deviation_scale': resolved.deviation_scale,
        'means': design.means,
        'sds': design.sds,
    }
    if not factors:
        return GlmOutcome(**truth)
    return GlmmOutcome(**truth, random_terms=random_terms)


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


def _buildRandomBlocks(
    resolved: Glm,
    design: Design,
    grouping: Grouping,
    factors: list[str],
    rng: np.random.Generator,
) -> tuple[list[Block], dict[str, list[str]], dict[str, np.ndarray]]:
    """Each factor's block of random effects, its random terms and their columns of the design."""
    blocks = []
    random_terms = {}
    random_columns = {}
    for factor in factors:
        budget = resolved.resolveField('random_effects', rng, high=len(design.names))
        positions = _chooseRandomTerms(design, budget, rng)
        z = design.matrix[:, positions]  # (n, q)
        # a random effect adds E[z²] to var(η) at scale 1, the intercept 1
        groups = grouping.metadata[factor].levels
        blocks.append(Block(name=factor, variances=(z**2).mean(axis=0).tolist(), groups=groups))
        random_terms[factor] = [design.names[i] for i in positions]
        random_columns[factor] = z
    return blocks, random_terms, random_columns


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
    checkOneLength(arrays, f'{model!r} and its covariates')
    # the dataset holds both, so a shared name would replace the covariate
    shared = covariates.columns.keys() & outcome.columns.keys()
    if shared:
        raise ValueError(f'{model!r} wrote the covariates {sorted(shared)}')
    return outcome
