"""hyperprior.py defines how every hyperparameter of the dataset is sampled.

We save hyperparameters as fields of the Hyperprior class.
- Each field (a size, the column types' shares, a noise scale) defaults to a distribution.
- An override replaces it by a point or by another distribution, checked against the field's bounds.
- A draw resolves every field to a point;
  the fields that shape a single column or node are resolved in a second step.

pipeline:
---------
the (optional) overrides
-> validation at construction                            Hyperprior
-> a point for each field drawn once per dataset         resolveDataset
-> a point per column or node                            resolveField
"""

from __future__ import annotations

import math
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
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
# They check their parameters when a hyperprior is built, not at a draw deep inside a run.


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
        return float(rng.uniform(self.low, self.high))


@pydantic_dataclass(frozen=True)
class LogUniform(_Interval):
    low: PositiveFloat  # its logarithm is drawn
    high: float

    def draw(self, rng: np.random.Generator) -> float:
        return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))


@pydantic_dataclass(frozen=True)
class UniformInt(_Interval):
    low: int
    high: int

    def draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.low, self.high, endpoint=True))


@pydantic_dataclass(frozen=True)
class LogUniformInt(_Interval):
    low: PositiveInt  # its logarithm is drawn
    high: int

    def draw(self, rng: np.random.Generator) -> int:
        # the floor of a log-uniform on [low, high + 1) gives each integer k the mass log(1 + 1/k);
        # min catches exp rounding up onto high + 1
        value = math.exp(rng.uniform(math.log(self.low), math.log(self.high + 1)))
        return min(math.floor(value), self.high)


@pydantic_dataclass(frozen=True)
class Beta:
    a: PositiveFloat
    b: PositiveFloat

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.a, self.b))

    def support(self) -> tuple[float, float]:
        return 0.0, 1.0


@pydantic_dataclass(frozen=True)
class Dirichlet:
    alpha: dict[str, PositiveFloat]  # concentration per category

    @property
    def values(self) -> list[float]:
        return list(self.alpha.values())

    def draw(self, rng: np.random.Generator) -> Shares:
        shares = rng.dirichlet(self.values).tolist()
        return Shares(zip(self.alpha, shares))


class Shares(dict[str, float]):
    """A point on the simplex: each category's share, from which a category is chosen."""

    def choose(self, rng: np.random.Generator) -> str:
        p = list(self.values())
        choice = rng.choice(list(self), p=p)
        return str(choice)


# every distribution a field can take
Distribution = Uniform | LogUniform | UniformInt | LogUniformInt | Beta | Dirichlet

# the kinds of field: a point, or a distribution of points of that kind
Integer = int | UniformInt | LogUniformInt
Real = float | Uniform | LogUniform | Beta
Simplex = dict[str, NonNegativeFloat] | Dirichlet


def _within(
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


Count = Annotated[Integer, _within(ge=0)]
PositiveCount = Annotated[Integer, _within(ge=1)]
NonNegative = Annotated[Real, _within(ge=0)]
Positive = Annotated[Real, _within(gt=0)]


# =====================================================================================
# the hyperprior
# =====================================================================================

MARGINAL_FIELDS = {
    'normal': [],
    'lognormal': ['lognormal_sigma'],
    'student_t': ['student_df'],
    'beta': ['beta_a', 'beta_b'],
    'uniform': [],
    'zero_inflated_lognormal': ['lognormal_sigma', 'zero_share'],
    'normal_mixture': ['mixture_gap', 'mixture_weight'],
}  # based on scamd's marginals
Family = Literal[tuple(MARGINAL_FIELDS)]  # for pydantic  # pyrefly: ignore[invalid-literal]


class Hyperprior(BaseModel):
    """The hyperparameters of a dataset (grouped by when they're used).

    - Columns: a table has rows and typed columns.
      A column is continuous or has levels (binary, categorical or ordered).
      A column with levels takes one of K levels in each row.
      Due to dummy-encoding, it takes K − 1 columns in the design matrix.
    - Graph: each column is an observed node of a directed acyclic graph (DAG).
      Latent nodes may affect observed nodes but have no columns assigned to them.
      Each possible edge has its own probability. It rises with the graph's overall density,
      when its parent or child is a hub or when both nodes are in the same block.
    - Mechanisms: for each node, its value is a function of its parents' values + noise.
      The function is linear or an MLP.
    - Marginals: the graph and its mechanisms decide only the dependence between columns
      through the ranks of their values. Each column's marginal shape is drawn separately,
      allowing easy finetuning using real columns (skew, zeros, heavy tails).
    - Shares: a Dirichlet distribution draws the dataset's shares of the column types,
      marginal families, mechanisms and activations.
      Each column or node chooses its own by those shares.
    """

    model_config = ConfigDict(frozen=True, extra='forbid', validate_default=True)
    PER_NODE: ClassVar[tuple[str, ...]] = (
        'init_scale',
        'noise_scale',
    )
    PER_COLUMN: ClassVar[tuple[str, ...]] = (
        'level_alpha',
        'student_df',
        'lognormal_sigma',
        'zero_share',
        'beta_a',
        'beta_b',
        'mixture_gap',
        'mixture_weight',
    )  # columns are manifested (i.e. non-latent) nodes
    SHARES: ClassVar[tuple[str, ...]] = (
        'type_shares',
        'mechanism_shares',
        'activation_shares',
        'marginal_shares',
    )

    # sizes and columns
    rows: PositiveCount = LogUniformInt(20, 15000)  # N, the table's length
    design_columns: Count = UniformInt(1, 19)  # design-matrix width
    type_shares: Simplex = Dirichlet(
        {'continuous': 0.9, 'binary': 0.7, 'categorical': 0.4, 'ordered': 0.1}
    )  # column types
    levels_p: Annotated[Real, _within(gt=0, le=1)] = Beta(2, 3)  # K is 2 + Geometric(levels_p)
    max_levels: Annotated[int, Field(ge=3)] | None = None  # optional upper bound for K
    level_alpha: Positive = LogUniform(0.5, 5)  # controls each column's level shares

    # graph
    latent_nodes: Count = UniformInt(0, 9)
    edge_offset: Real = Uniform(-4, 2)  # the logit offset shared by every edge
    hub_scale: NonNegative = LogUniform(0.1, 2)  # Cauchy scale of the per-node out- and in-terms
    blocks: PositiveCount = UniformInt(1, 4)  # groups of nodes that connect more often
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
    )  # a continuous column's family
    student_df: Positive = LogUniform(2, 30)
    lognormal_sigma: Positive = LogUniform(0.25, 1.5)
    zero_share: Annotated[Real, _within(ge=0, lt=1)] = Uniform(0.05, 0.6)  # zero-inflated lognormal
    beta_a: Positive = LogUniform(0.5, 5)
    beta_b: Positive = LogUniform(0.5, 5)
    mixture_gap: NonNegative = Uniform(1, 4)  # distance between two normal means (in SDs)
    mixture_weight: Annotated[Real, _within(ge=0, le=0.5)] = Uniform(0.1, 0.5)  # 2nd normal's share

    @field_validator(*SHARES)
    @classmethod
    def _orderShares(cls, value: Simplex, info: ValidationInfo) -> Simplex:
        """Puts the categories in the default order."""
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

    @classmethod
    def categories(cls, field: str) -> list[str]:
        return list(cls.model_fields[field].default.alpha)

    def resolveDataset(self, rng: np.random.Generator) -> Hyperprior:
        """Resolves each non-node field."""
        points = {}
        for name in type(self).model_fields:
            if name in self.PER_COLUMN + self.PER_NODE:
                points[name] = getattr(self, name)  # defer resolution to a later step
            else:
                points[name] = self.resolveField(name, rng)
        return Hyperprior(**points)

    def resolveField(self, field: str, rng: np.random.Generator) -> Any:
        value = getattr(self, field)
        if isinstance(value, Distribution):
            value = value.draw(rng)
        return value
