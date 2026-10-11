"""hyperprior.py defines the hyperprior: every quantity of a dataset as a point or a distribution.

Each field defaults to a distribution; an override replaces it by a point value or another
distribution of the field's kind, e.g. `Hyperprior(n_obs=500)` or
`Hyperprior(n_obs=LogUniformInt(20, 5000))`.

pipeline:
---------
the overrides
-> validation at construction                            Hyperprior
-> one draw per field still a distribution, in order     resolve
"""

from __future__ import annotations

import math
from typing import Annotated

import numpy as np
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationInfo, field_validator
from pydantic.dataclasses import dataclass

# =====================================================================================
# distributions
# =====================================================================================
# the distributions a field can take; dataclasses so that `Uniform(-4, 2)` reads positionally.


class Distribution:
    """The base of every distribution a field of the hyperprior can take."""


class _Interval(Distribution):
    """A distribution on [low, high], validated when it is built."""

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f'low {self.low} is not below high {self.high}')

    def support(self) -> tuple[float, float]:
        return self.low, self.high


@dataclass(frozen=True)
class Uniform(_Interval):
    low: float
    high: float

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


class _LogInterval(_Interval):
    """An interval whose logarithm is drawn, so it must lie above 0."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.low <= 0:
            raise ValueError(f'a log-uniform needs low > 0, got {self.low}')


@dataclass(frozen=True)
class LogUniform(_LogInterval):
    low: float
    high: float

    def draw(self, rng: np.random.Generator) -> float:
        return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))


@dataclass(frozen=True)
class UniformInt(_Interval):
    low: int
    high: int

    def draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.low, self.high, endpoint=True))


@dataclass(frozen=True)
class LogUniformInt(_LogInterval):
    low: int
    high: int

    def draw(self, rng: np.random.Generator) -> int:
        # the floor of a log-uniform on [low, high + 1) gives each integer k the mass log(1 + 1/k);
        # min catches exp rounding up onto high + 1
        value = math.exp(rng.uniform(math.log(self.low), math.log(self.high + 1)))
        return min(math.floor(value), self.high)


@dataclass(frozen=True)
class Beta(Distribution):
    a: float
    b: float

    def __post_init__(self) -> None:
        if min(self.a, self.b) <= 0:
            raise ValueError(f'a beta needs shapes > 0, got {self.a} and {self.b}')

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.a, self.b))

    def support(self) -> tuple[float, float]:
        return 0.0, 1.0


@dataclass(frozen=True)
class Dirichlet(Distribution):
    alpha: dict[str, float]  # concentration per category

    def __post_init__(self) -> None:
        if min(self.alpha.values()) <= 0:
            raise ValueError(f'a dirichlet needs concentrations > 0, got {self.alpha}')

    def draw(self, rng: np.random.Generator) -> dict[str, float]:
        shares = rng.dirichlet(list(self.alpha.values()))
        return dict(zip(self.alpha, shares.tolist()))


class Shares(dict[str, float]):
    """A point on the simplex: each category's share, from which a category is chosen."""

    def choose(self, rng: np.random.Generator) -> str:
        return str(rng.choice(list(self), p=list(self.values())))

    def chooseMany(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.choice(list(self), size=size, p=list(self.values()))  # (size,)


Integer = int | UniformInt | LogUniformInt
Real = float | Uniform | LogUniform | Beta
Simplex = dict[str, float] | Dirichlet


def _within(
    ge: float = -math.inf, gt: float = -math.inf, le: float = math.inf, lt: float = math.inf
) -> AfterValidator:
    """A field's bounds, checked on a point or on the closure of a distribution's support."""

    def check(value: float | Distribution) -> float | Distribution:
        # an open bound admits a support that touches it: the endpoint is drawn with probability 0
        if isinstance(value, Distribution):
            low, high = value.support()
            inside = max(ge, gt) <= low and high <= min(le, lt)
        else:
            inside = ge <= value <= le and gt < value < lt
        if not inside:
            raise ValueError(f'{value} lies outside ge={ge}, gt={gt}, le={le}, lt={lt}')
        return value

    return AfterValidator(check)


Count = Annotated[Integer, _within(ge=0)]
PositiveCount = Annotated[Integer, _within(ge=1)]
NonNegative = Annotated[Real, _within(ge=0)]
Positive = Annotated[Real, _within(gt=0)]


# =====================================================================================
# the hyperprior
# =====================================================================================


class Hyperprior(BaseModel):
    """Every quantity of a dataset, each a point or a distribution, in the order they are resolved.

    The defaults are provisional until the hyperprior is fitted to mixedbench's registry.
    """

    model_config = ConfigDict(frozen=True, extra='forbid', validate_default=True)

    # sizes and columns
    n_obs: PositiveCount = LogUniformInt(20, 15000)  # rows
    design_columns: Count = UniformInt(0, 19)  # metabeta's d − 1
    type_shares: Simplex = Dirichlet(
        {'continuous': 0.88, 'binary': 0.66, 'categorical': 0.38, 'ordered': 0.08}
    )  # twice the envelope witnesses' shares, so that corners get mass
    levels_p: Annotated[Real, _within(gt=0, le=1)] = Beta(2, 3)  # K is 2 + Geometric(levels_p)
    max_levels: Annotated[int, Field(ge=3)] | None = None  # caps K; none keeps the registry's tail
    level_alpha: Positive = LogUniform(0.5, 5)  # Dirichlet concentration of a column's level shares

    # graph
    latent_nodes: Count = UniformInt(0, 3)
    edge_offset: Real = Uniform(-4, 2)  # the logit offset shared by every edge
    hub_scale: NonNegative = LogUniform(0.1, 2)  # Cauchy scale of the per-node out- and in-terms
    blocks: PositiveCount = UniformInt(1, 4)
    block_bonus: NonNegative = Uniform(0, 3)  # added to the edge logit within a block

    # mechanisms
    mechanism_shares: Simplex = Dirichlet({'linear': 1, 'mlp': 1})
    hidden: PositiveCount = LogUniformInt(4, 64)  # MLP width
    layers: PositiveCount = UniformInt(1, 3)  # MLP depth
    activation_shares: Simplex = Dirichlet({'tanh': 1, 'relu': 1, 'sin': 1, 'identity': 1})
    init_scale: NonNegative = LogUniform(0.1, 3)  # MLP weights are N(0, init_scale² / fan_in)
    noise_scale: NonNegative = LogUniform(0.01, 1)  # relative to the z-scored node

    # marginals
    marginal_shares: Simplex = Dirichlet(
        {
            'normal': 1,
            'lognormal': 1,
            'student_t': 1,
            'beta': 1,
            'uniform': 1,
            'zero_inflated_lognormal': 1,
            'normal_mixture': 1,
        }
    )
    student_df: Positive = LogUniform(2, 30)
    lognormal_sigma: Positive = LogUniform(0.25, 1.5)  # lognormal and zero-inflated lognormal
    zero_share: Annotated[Real, _within(ge=0, lt=1)] = Uniform(0.05, 0.6)  # zero-inflated lognormal
    beta_a: Positive = LogUniform(0.5, 5)
    beta_b: Positive = LogUniform(0.5, 5)
    mixture_gap: NonNegative = Uniform(1, 4)  # distance between the two normal means, in sds
    mixture_weight: Annotated[Real, _within(ge=0, le=1)] = Uniform(0.1, 0.5)

    @field_validator('type_shares', 'mechanism_shares', 'activation_shares', 'marginal_shares')
    @classmethod
    def _shares_name_the_default_categories(cls, value: Simplex, info: ValidationInfo) -> Simplex:
        categories = cls.categories(info.field_name)
        shares = value.alpha if isinstance(value, Dirichlet) else value
        if shares.keys() != set(categories):
            raise ValueError(f'shares name {list(shares)}, not {categories}')
        if not isinstance(value, Dirichlet) and (
            min(shares.values()) < 0 or not math.isclose(sum(shares.values()), 1)
        ):
            raise ValueError(f'shares {shares} do not lie on the simplex')
        # a draw takes the categories by place, so they follow the default's order, not the override's
        ordered = {name: shares[name] for name in categories}
        return Dirichlet(ordered) if isinstance(value, Dirichlet) else Shares(ordered)

    @classmethod
    def categories(cls, field: str) -> list[str]:
        """The categories a shares field names, in its default's order."""
        return list(cls.model_fields[field].default.alpha)

    def resolve(self, rng: np.random.Generator) -> Hyperprior:
        """The same fields with every distribution replaced by one draw from it."""
        points = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            points[name] = value.draw(rng) if isinstance(value, Distribution) else value
        return Hyperprior(**points)
