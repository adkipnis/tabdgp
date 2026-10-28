"""hyperprior.py defines how every hyperparameter of the dataset is sampled.

We save hyperparameters as fields of the Hyperprior class (a Config in field.py).
- Each field (a size, the column types' shares, a noise scale) defaults to a distribution,
  or to a point where one value serves every dataset (the outcome's form, its type shares).
- An override replaces it by a point or, where the field's type allows, by another distribution,
  checked against the field's bounds.
- A preset is the defaults for one kind of table (e.g. flat, grouped); overrides apply on top.
- A draw resolves every field to a point;
  the fields that shape a single column, node or grouping factor are resolved in a second step.
- Some counts are limited by another field's value (design_columns and groups by rows):
  they are drawn only from the values within that limit.

pipeline:
---------
the (optional) overrides
-> a preset with the overrides                           grouped, flat
-> validation at construction                            Hyperprior
-> a point for one field, in the bounds it is given      resolveField
-> a point for each field drawn once per dataset         resolveDataset
   (a column, node or factor resolves its own fields later, by resolveField)
-> a factor's groups (point or a draw)                   resolveGroups
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from pydantic import Field, model_validator

from tabdgp.field import (
    Beta,
    Config,
    Count,
    Dirichlet,
    Integer,
    LogUniform,
    LogUniformInt,
    NonNegative,
    Positive,
    PositiveCount,
    Real,
    Share,
    Simplex,
    Uniform,
    UniformInt,
    within,
)

MARGINAL_FIELDS = {
    'normal': [],
    'lognormal': ['lognormal_sigma'],
    'student_t': ['student_df'],
    'beta': ['beta_a', 'beta_b'],
    'uniform': [],
    'zero_inflated_lognormal': ['lognormal_sigma', 'zero_share'],
    'normal_mixture': ['mixture_gap', 'mixture_weight'],
    'poisson': ['poisson_log_rate'],
}  # based on scamd's marginals
Marginal = Literal[tuple(MARGINAL_FIELDS)]  # for pydantic  # pyrefly: ignore[invalid-literal]
OutcomeForm = Literal['none', 'scm']
MaxLevels = Annotated[int, Field(ge=3)]  # a cap on k that leaves a categorical its 3


# =====================================================================================
# the hyperprior
# =====================================================================================


class Hyperprior(Config):
    """The hyperparameters of a dataset (grouped by when they're used).

    - Columns: a table has rows and typed columns.
      A column is continuous, a count or has levels (binary, categorical or ordered).
      A column with levels takes one of k levels in each row.
      Due to dummy-encoding, it takes k − 1 columns in the design matrix.
    - Grouping: a grouping factor assigns every row to one of its m groups.
      A table has none (flat), one or multiple grouping factors.
      Each factor after the first either nests in the one before it (classes in schools)
      or is crossed with all before it (patients and doctors).
      Factors nested in one another form a chain (schools -> classes).
      A factor's depth is the number of factors above it in its chain,
      and each factor in a chain draws more groups than the one above it.
      A factor's groups are equally large or have Dirichlet-distributed sizes.
      A node is constant within the groups of one factor (group-level) or varies within them.
      A within-group node adds a group part (one value per group) to a row part (one per row).
      Its group share is how much of its variance comes from the group part.
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
      Each column or node chooses its own, proportional to those shares.
    - Outcome: a single node y. No other node reads it.
      Its form turns y's node (the score) into y under the SCM; under none, there is no y.
      Its type shares and noise scale are the SCM's.
    """

    PER_NODE: ClassVar[tuple[str, ...]] = (
        'init_scale',
        'noise_scale',
        'group_share',
        'outcome_noise_scale',
    )
    PER_FACTOR: ClassVar[tuple[str, ...]] = (
        'groups',
        'size_alpha',
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
        'poisson_log_rate',
    )  # columns are observed (i.e. non-latent) nodes
    SHARES: ClassVar[tuple[str, ...]] = (
        'type_shares',
        'mechanism_shares',
        'activation_shares',
        'marginal_shares',
        'outcome_type_shares',
    )

    # sizes and columns
    rows: PositiveCount = LogUniformInt(20, 15000)  # n, the table's length
    design_columns: Count = UniformInt(1, 19)  # design-matrix width
    type_shares: Simplex = Dirichlet(
        {'continuous': 0.9, 'binary': 0.7, 'categorical': 0.4, 'ordered': 0.1}
    )  # column types
    levels_p: Annotated[Real, within(gt=0, le=1)] = Beta(2, 3)  # k is 2 + Geometric(levels_p)
    max_levels: MaxLevels | None = None  # optional upper bound for k
    level_alpha: Positive = LogUniform(0.5, 5)  # controls each column's level shares

    # grouping
    factors: Count = UniformInt(1, 3)  # grouping factors; 0 is a flat table
    groups: Annotated[Integer, within(ge=2)] | tuple[Annotated[int, within(ge=2)], ...] = (
        LogUniformInt(2, 400)
    )  # m per factor (at most n) or one point per factor
    balanced_share: Share = 1 / 3  # probability that a factor's groups are of equal size
    size_alpha: Positive = LogUniform(0.5, 500)  # Dirichlet concentration of unequal sizes
    nested_share: Share = Uniform(0.2, 0.6)  # probability that a factor nests in the previous one
    group_level_share: Share = Beta(2, 4)  # probability that a node is constant within groups
    group_share: Share = Beta(0.5, 2)  # share of a within-group node's variance due to its group

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
    activation_shares: Simplex = Dirichlet({'tanh': 1, 'relu': 1, 'sin': 1})
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
    zero_share: Annotated[Real, within(ge=0, lt=1)] = Uniform(0.05, 0.6)  # zero-inflated lognormal
    beta_a: Positive = LogUniform(0.5, 5)
    beta_b: Positive = LogUniform(0.5, 5)
    mixture_gap: NonNegative = Uniform(1, 4)  # distance between two normal means (in SDs)
    mixture_weight: Annotated[Real, within(ge=0, le=0.5)] = Uniform(0.1, 0.5)  # 2nd normal's share
    poisson_log_rate: Annotated[Real, within(le=40)] = Uniform(-1, 6)

    # outcome
    outcome_form: OutcomeForm = 'scm'
    outcome_type_shares: Simplex = {  # noqa: RUF012, ruff cannot trace Config to BaseModel
        'continuous': 0.6,
        'binary': 0.2,
        'categorical': 0.05,
        'ordered': 0.05,
        'count': 0.1,
    }  # the outcome's column type
    outcome_max_levels: MaxLevels = 10
    outcome_noise_scale: NonNegative = LogUniform(0.1, 3)

    @classmethod
    def grouped(cls, **overrides: Any) -> Hyperprior:
        return cls(**overrides)

    @classmethod
    def flat(cls, **overrides: Any) -> Hyperprior:
        return cls(factors=0, **overrides)

    @model_validator(mode='after')
    def _checkOnePointPerFactor(self) -> Hyperprior:
        """Checks that a tuple of groups gives every factor one point."""
        if isinstance(self.groups, tuple) and self.factors != len(self.groups):
            raise ValueError(
                f'groups = {self.groups} needs factors = {len(self.groups)}, not {self.factors}'
            )
        return self

    @model_validator(mode='after')
    def _checkGroupsCanNest(self) -> Hyperprior:
        """Checks that groups has enough values for the longest chain the fields allow:
        its bottom factor must reach its floor, and a tuple must rise from each point to the next.
        """
        _, most_nested = self._support(self.nested_share)
        if isinstance(self.groups, tuple):
            # any factor may nest in the one before it, which then needs fewer groups
            pairs = zip(self.groups, self.groups[1:])
            rising = all(superior < subordinate for superior, subordinate in pairs)
            if most_nested > 0 and not rising:
                raise ValueError(
                    f'groups = {self.groups} cannot nest each factor in the one before it: '
                    'set nested_share = 0 or give each factor more groups than the one before it'
                )
            return self
        _, most_factors = self._support(self.factors)
        longest = most_factors if most_nested > 0 else 1
        _, highest = self._support(self.groups)
        if self._floorGroups(longest - 1) > highest:
            raise ValueError(
                f'groups = {self.groups} cannot nest {longest} factors: set nested_share = 0, '
                'fewer factors, or give groups a wider range'
            )
        return self  # pydantic's after-validator returns the model

    def _floorGroups(self, depth: int) -> int:
        """The smallest number of groups a factor can have at this depth.
        Each factor above it (in its chain) has fewer groups.
        """
        lowest, _ = self._support(self.groups)
        return lowest + depth

    def resolveDataset(self, rng: np.random.Generator) -> Hyperprior:
        """Resolves each field drawn once per dataset."""
        points = {}
        for name in type(self).model_fields:
            if name in self.PER_COLUMN + self.PER_NODE + self.PER_FACTOR:
                points[name] = getattr(self, name)  # defer resolution to a later step
            elif name == 'design_columns':
                points[name] = self.resolveField(name, rng, high=points['rows'] // 2)  # n >= 2d
            else:
                points[name] = self.resolveField(name, rng)
        return Hyperprior(**points)

    def resolveGroups(self, factor: int, depth: int, rng: np.random.Generator, high: int) -> int:
        """A factor's m: its own point of a tuple, else a draw from its depth's floor to high."""
        if isinstance(self.groups, tuple):
            groups = self.groups[factor]
            self._checkBounds('groups', groups, None, high)
            return groups
        return self.resolveField('groups', rng, low=self._floorGroups(depth), high=high)
