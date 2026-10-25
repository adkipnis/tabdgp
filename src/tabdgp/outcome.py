"""outcome.py defines what a form reads and returns, and TabDGP's own models, the GLM and GLMM.

A form reads the covariates: the observed columns at the rows, their metadata,
the grouping and the score (the outcome node's z-scored value in the SCM, None without one).
A GLM codes the covariates as the design matrix and draws y from its family through the link.
A GLMM adds random effects per grouping factor on some of the design's columns.

pipeline:
---------
the covariates at the rows, their metadata, the grouping and the score  Covariates
   or a table's, without a score, each column read by its metadata,
   declared or inferred from its dtype, and checked against its values  Covariates.fromPandas
-> the columns a form writes, their metadata and its truth              Outcome
-> a model's outcome                                                    run
   a GLMM's, a GLM's without factors:
   -> the design matrix, the intercept first                            codeDesign
   -> each factor's block, its random terms within its budget           _buildRandomBlocks
   -> the prior's scales and parameters over the blocks                 Glm.drawParameters
   -> η = Xβ + Σ_k Z_k α_k, plus the deviation λ · sd(Xβ + Σ_k Z_k α_k) · score,
      then y from the family through its link                           _drawGlmm
   a study's:
   -> its call, its result checked at the covariates' rows              _callModel
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from pydantic import ConfigDict, TypeAdapter
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
    # a table's declared metadata: Columns, or their fields as Dataset.toDict writes them
    METADATA: ClassVar[TypeAdapter] = TypeAdapter(dict[str, Column])
    # the names a form or a model writes beside the covariates, each with what it names
    RESERVED: ClassVar[dict[str, str]] = {
        'y': "the outcome's name",
        'intercept': "the design matrix's first term",
        'fixed': "the fixed effects' block",
    }

    def __post_init__(self) -> None:
        # check that every array has one length, each column its metadata, and none a reserved name
        _checkOneLength([*self.columns.values(), self.score], 'the covariates')
        _checkNames(self.columns, self.metadata)
        for name, reason in self.RESERVED.items():
            if name in self.columns:
                raise ValueError(f'a column cannot be named {name!r}, {reason}: rename it')

    @property
    def rows(self) -> int:
        """n, the length of every array, the score's in a table without a column."""
        for array in [*self.columns.values(), self.score]:
            if array is not None:
                return len(array)
        raise ValueError('covariates without a column or a score have no rows')

    @classmethod
    def fromPandas(
        cls,
        table: pd.DataFrame,
        factors: list[str] | None = None,
        metadata: dict[str, Column | dict[str, Any]] | None = None,
    ) -> Covariates:
        """The covariates of a table, each column read by its metadata and checked against it.
        The metadata is declared, a Column per column, or inferred from the dtypes,
        with the grouping factors named in factors.
        The outcome, which only a declaration names, is left out; a table has no SCM, so no score.
        """
        if (factors is None) == (metadata is None):
            raise ValueError('name the grouping factors to infer the metadata, or declare it')
        observed = list(table.columns)
        if metadata is not None:
            metadata = cls._validateMetadata(table, metadata)
            observed = [name for name, column in metadata.items() if column.role != 'outcome']
        cls._checkValues(table[observed])
        if metadata is None:
            metadata = cls._inferMetadata(table, factors)
        grouping = cls._readGrouping(table, metadata)
        columns = dict(grouping.codes)
        for name, column in metadata.items():
            if column.role == 'covariate':
                columns[name] = cls._readValues(table[name], column)
        observed_metadata = {name: metadata[name] for name in columns}
        return cls(columns=columns, metadata=observed_metadata, grouping=grouping, score=None)

    @classmethod
    def _validateMetadata(
        cls, table: pd.DataFrame, metadata: dict[str, Column | dict[str, Any]]
    ) -> dict[str, Column]:
        """The declared metadata as Columns, one for each of the table's columns."""
        declared = cls.METADATA.validate_python(metadata)
        if declared.keys() != set(table.columns):
            raise ValueError(
                f'the table has the columns {list(table.columns)}, the metadata {list(declared)}'
            )
        return declared

    @staticmethod
    def _checkValues(table: pd.DataFrame) -> None:
        """Checks that every column has a value in every row and at least two distinct ones."""
        for name in table.columns:
            values = table[name]
            if values.isna().any():
                raise ValueError(f'the column {name} has a missing value')
            if values.nunique() < 2:
                raise ValueError(
                    f'the column {name} takes one value only, '
                    "so no fit tells its effect from the intercept's: drop it"
                )

    @classmethod
    def _inferMetadata(cls, table: pd.DataFrame, factors: list[str]) -> dict[str, Column]:
        """The factors' metadata from their codes, in the order named, then each covariate's
        from its dtype, in the table's order.
        """
        codes = {}
        levels = {}
        for name in factors:
            codes[name], levels[name] = cls._codeLevels(table[name])
        metadata = {}
        for name in factors:
            nests_in = cls._findSuperior(name, codes, levels)
            metadata[name] = Column(
                type='categorical', levels=levels[name], role='grouping_factor', nests_in=nests_in
            )
        for name in table.columns:
            if name not in factors:
                metadata[name] = cls._inferCovariate(table[name])
        return metadata

    @staticmethod
    def _codeLevels(values: pd.Series) -> tuple[np.ndarray, int]:
        """A column's codes and k, its levels in a categorical's order or sorted,
        each present in some row.
        """
        if not isinstance(values.dtype, pd.CategoricalDtype):
            values = values.astype('category')  # the categories are the sorted values
        present = values.cat.remove_unused_categories()
        codes = present.cat.codes.to_numpy(dtype=np.int64)  # (n,)
        return codes, len(present.cat.categories)

    @classmethod
    def _findSuperior(
        cls, factor: str, codes: dict[str, np.ndarray], levels: dict[str, int]
    ) -> str | None:
        """The factor's immediate superior, wherever it is named:
        of the factors with fewer groups that each hold whole groups of it, the one with the most.
        """
        superior = None
        for other, other_codes in codes.items():
            fewer = levels[other] < levels[factor]
            if not fewer or not cls._nests(codes[factor], other_codes):
                continue
            if superior is None or levels[other] > levels[superior]:
                superior = other
        return superior

    @staticmethod
    def _nests(subordinate: np.ndarray, superior: np.ndarray) -> bool:
        """Whether each group of the subordinate factor lies within one group of the superior."""
        # the distinct pairs of a group and its superior group, one per group if it nests
        pairs = np.unique(np.stack([subordinate, superior], axis=1), axis=0)  # (pairs, 2)
        return len(pairs) == np.unique(subordinate).size

    @staticmethod
    def _inferCovariate(values: pd.Series) -> Column:
        """A covariate's metadata as its dtype states it,
        or binary at two values whatever its dtype, as mixedbench reads a column.
        """
        dtype = values.dtype
        categorical = isinstance(dtype, pd.CategoricalDtype)
        if values.nunique() == 2:
            return Column(type='binary', levels=2)
        if categorical or pd.api.types.is_string_dtype(values):
            if categorical and dtype.ordered:
                return Column(type='ordered', levels=values.nunique())
            return Column(type='categorical', levels=values.nunique())
        if pd.api.types.is_integer_dtype(dtype):
            if (values < 0).any():
                raise ValueError(
                    f'the integer column {values.name} has a negative value, so it is no count: '
                    'cast it to float if it is continuous, '
                    'or drop the rows where a negative value marks a missing one'
                )
            return Column(type='count')
        if pd.api.types.is_float_dtype(dtype):
            return Column(type='continuous')
        raise ValueError(
            f'the column {values.name} has the dtype {dtype}, which states no column type: '
            'cast it to float, int, bool, str or category'
        )

    @classmethod
    def _readGrouping(cls, table: pd.DataFrame, metadata: dict[str, Column]) -> Grouping:
        """The factors' codes, checked against their metadata's levels and nesting,
        and no two factors grouping the rows alike.
        """
        factors = {}
        codes = {}
        for name, column in metadata.items():
            if column.role == 'grouping_factor':
                factors[name] = column
                codes[name] = cls._readCodes(table[name], column)
        names = list(factors)
        for i, name in enumerate(names):
            superior = factors[name].nests_in
            if superior is not None and superior not in factors:
                raise ValueError(
                    f'the factor {name} nests in {superior}, which is no grouping factor'
                )
            if superior is not None and not cls._nests(codes[name], codes[superior]):
                raise ValueError(f'the factor {name} has a group across two groups of {superior}')
            for other in names[i + 1 :]:
                alike = cls._nests(codes[name], codes[other]) and cls._nests(
                    codes[other], codes[name]
                )
                if alike:
                    raise ValueError(
                        f'the factors {name} and {other} group the rows alike, one factor named twice'
                    )
        return Grouping(metadata=factors, codes=codes)

    @classmethod
    def _readCodes(cls, values: pd.Series, column: Column) -> np.ndarray:
        """A coded column's codes, checked against its k levels."""
        codes, k = cls._codeLevels(values)
        if k != column.levels:
            raise ValueError(
                f'the column {values.name} has {k} distinct values, its metadata {column.levels} levels'
            )
        return codes

    @classmethod
    def _readValues(cls, values: pd.Series, column: Column) -> np.ndarray:
        """A covariate's values in its type's dtype, checked against its type:
        one of its k levels if coded, a number if continuous, a non-negative integer if a count.
        """
        if column.levels is not None:
            return cls._readCodes(values, column)
        if not pd.api.types.is_numeric_dtype(values.dtype):
            raise ValueError(
                f'the {column.type} column {values.name} holds {values.dtype}, not numbers'
            )
        if column.type == 'continuous':
            return values.to_numpy(dtype=column.dtype)
        counts = values.to_numpy(dtype=column.dtype)  # (n,)
        if (counts < 0).any() or (counts != values.to_numpy()).any():
            raise ValueError(f'the count column {values.name} holds a value that is no count')
        return counts


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
        corrs = {}
        for factor, covariance in self._transformCovariances().items():
            sd = np.sqrt(np.diag(covariance))  # (q,)
            corrs[factor] = covariance / np.outer(sd, sd)  # (q, q)
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
        return y.astype(Normal.COLUMN.dtype)


class Bernoulli:
    """y ~ Bernoulli(p), η the logit of p."""

    LINK = 'logit'
    COLUMN = Column(type='binary', levels=2, role='outcome')

    @staticmethod
    def draw(rng: np.random.Generator, eta: np.ndarray, scale: np.ndarray | None) -> np.ndarray:
        p = 0.5 * (1 + np.tanh(eta / 2))  # (n,) the inverse logit, without exp's overflow
        uniforms = rng.random(len(eta))  # (n,)
        return (uniforms < p).astype(Bernoulli.COLUMN.dtype)


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
    name = resolved.family_shares.choose(rng)
    family = FAMILIES[name]
    # the fixed term adds tr cov X to var(η) at scale 1, the intercept nothing
    variance = float(design.matrix.var(axis=0).sum())
    fixed = Block(name='fixed', width=len(design.names), variance=variance)
    random_blocks, random_terms, random_columns = _buildRandomBlocks(
        resolved, design, covariates.grouping, factors, rng
    )
    context, parameters = resolved.drawParameters(name, [fixed, *random_blocks], rng)
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
        # a random term adds tr E[z z'] to var(η) at scale 1, its intercept 1
        variance = float((z**2).mean(axis=0).sum())
        groups = grouping.metadata[factor].levels
        blocks.append(Block(name=factor, width=len(positions), variance=variance, groups=groups))
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
    _checkOneLength(arrays, f'{model!r} and its covariates')
    # the dataset holds both, so a shared name would replace the covariate
    shared = covariates.columns.keys() & outcome.columns.keys()
    if shared:
        raise ValueError(f'{model!r} wrote the covariates {sorted(shared)}')
    return outcome
