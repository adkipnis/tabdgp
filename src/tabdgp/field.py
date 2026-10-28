"""field.py defines how we resolve config fields.

A field's kind (integer, real, shares over named categories) declares which points
and distributions it takes.
Bounds are checked on a point or a distribution's support.

pipeline:
---------
a subclass's fields (each a point or a distribution)
-> validation at construction, shares in default order    Hyperprior
-> a point for one field (within the bounds it is given)  resolveField
"""

from __future__ import annotations

import math
from typing import Annotated, Any, ClassVar

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
    ValidationInfo,
    field_validator,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

# =====================================================================================
# distributions
# =====================================================================================
# numpy has no distribution objects: these classes are the config's values.
# They check their parameters when a config is built, not at a draw deep inside a run.


class _Interval:
    """A distribution on [low, high], validated when it is built."""

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError(f'low {self.low} is not below high {self.high}')

    def support(self) -> tuple[float, float]:
        return self.low, self.high


@pydantic_dataclass(frozen=True)
class Uniform(_Interval):
    low: float
    high: float

    def draw(self, rng: np.random.Generator) -> float:
        out = rng.uniform(self.low, self.high)
        return float(out)


@pydantic_dataclass(frozen=True)
class LogUniform(_Interval):
    low: PositiveFloat  # its logarithm is drawn
    high: float

    def draw(self, rng: np.random.Generator) -> float:
        log_low, log_high = math.log(self.low), math.log(self.high)
        log_out = rng.uniform(log_low, log_high)
        return math.exp(log_out)


@pydantic_dataclass(frozen=True)
class UniformInt(_Interval):
    low: int
    high: int

    def draw(
        self, rng: np.random.Generator, low: int | None = None, high: int | None = None
    ) -> int:
        bottom, top = _truncate(self.support(), low, high)
        out = rng.integers(bottom, top, endpoint=True)
        return int(out)


def _truncate(ends: tuple[float, float], low: int | None, high: int | None) -> tuple[float, float]:
    """The ends of a support truncated to [low, high]."""
    bottom, top = ends
    if low is not None:
        bottom = max(low, bottom)
    if high is not None:
        top = min(high, top)
    return bottom, top


@pydantic_dataclass(frozen=True)
class LogUniformInt(_Interval):
    low: PositiveInt  # its logarithm is drawn
    high: int

    def draw(
        self, rng: np.random.Generator, low: int | None = None, high: int | None = None
    ) -> int:
        # the floor of a log-uniform on [bottom, top + 1) gives each integer k a mass ∝ log(1 + 1/k)
        bottom, top = _truncate(self.support(), low, high)
        log_bottom, log_top_p1 = math.log(bottom), math.log(top + 1)
        log_out = rng.uniform(log_bottom, log_top_p1)
        out = math.exp(log_out)
        return min(math.floor(out), top)  # catch exp rounding


@pydantic_dataclass(frozen=True)
class Beta:
    a: PositiveFloat
    b: PositiveFloat

    def draw(self, rng: np.random.Generator) -> float:
        out = rng.beta(self.a, self.b)
        return float(out)

    @staticmethod
    def support() -> tuple[float, float]:
        return 0.0, 1.0


class Shares(dict[str, float]):
    """A point on the simplex: each category's share, from which a category is chosen."""

    def choose(self, rng: np.random.Generator) -> str:
        p = list(self.values())
        choice = rng.choice(list(self), p=p)
        return str(choice)


@pydantic_dataclass(frozen=True)
class Dirichlet:
    alpha: dict[str, PositiveFloat]  # concentration per category

    @property
    def values(self) -> list[float]:
        return list(self.alpha.values())

    def draw(self, rng: np.random.Generator) -> Shares:
        out = rng.dirichlet(self.values)
        return Shares(zip(self.alpha, out.tolist()))


# every distribution a field can take
Distribution = Uniform | LogUniform | UniformInt | LogUniformInt | Beta | Dirichlet

# the kinds of field: a point, or a distribution of points of that kind
Integer = int | UniformInt | LogUniformInt
Real = float | Uniform | LogUniform | Beta
Simplex = dict[str, NonNegativeFloat] | Dirichlet


def within(
    ge: float = -math.inf,  # >=
    gt: float = -math.inf,  # >
    le: float = math.inf,  # <=
    lt: float = math.inf,  # <
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


Count = Annotated[Integer, within(ge=0)]
PositiveCount = Annotated[Integer, within(ge=1)]
NonNegative = Annotated[Real, within(ge=0)]
Positive = Annotated[Real, within(gt=0)]
Share = Annotated[Real, within(ge=0, le=1)]


# =====================================================================================
# the fields
# =====================================================================================


class Hyperprior(BaseModel):
    """A config whose every field is a point or a distribution of points.
    A `shares` field holds shares or a Dirichlet over named categories.
    """

    model_config = ConfigDict(frozen=True, extra='forbid', validate_default=True)
    SHARES: ClassVar[tuple[str, ...]] = ()  # the shares fields; each subclass names its own

    @classmethod
    def categories(cls, field: str) -> list[str]:
        """Lists the categories of a field in default order."""
        default = cls.model_fields[field].default
        shares = default.alpha if isinstance(default, Dirichlet) else default
        return list(shares)

    @field_validator('*')
    @classmethod
    def _orderShares(cls, value: Any, info: ValidationInfo) -> Any:
        """Checks the categories named by an override and puts them in default order."""
        if info.field_name not in cls.SHARES:
            return value
        categories = cls.categories(info.field_name)
        if isinstance(value, Dirichlet):
            return Dirichlet(cls._inOrder(value.alpha, categories))
        if not math.isclose(sum(value.values()), 1):
            raise ValueError(f'shares {value} do not sum to 1')
        return Shares(cls._inOrder(value, categories))

    @staticmethod
    def _inOrder(shares: dict[str, float], categories: list[str]) -> dict[str, float]:
        if shares.keys() != set(categories):
            raise ValueError(f'shares name {list(shares)}, not {categories}')
        return {name: shares[name] for name in categories}

    def resolveField(
        self, field: str, rng: np.random.Generator, low: int | None = None, high: int | None = None
    ) -> Any:
        value = getattr(self, field)
        bounded = low is not None or high is not None
        if bounded:
            self._checkBounds(field, value, low, high)
        if isinstance(value, Distribution):
            value = value.draw(rng, low, high) if bounded else value.draw(rng)
        return value

    def _checkBounds(self, field: str, value: Integer, low: int | None, high: int | None) -> None:
        bottom, top = _truncate(self._support(value), low, high)
        if bottom > top:
            raise ValueError(f'{field} = {value} has no value from {bottom} to {top}')

    @staticmethod
    def _support(value: Integer | Real) -> tuple[float, float]:
        """The smallest and largest value a field can take.
        A field holds a distribution or a point, and a point is its own smallest and largest value.
        """
        if isinstance(value, Distribution):
            return value.support()
        return value, value
