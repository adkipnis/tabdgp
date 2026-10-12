"""dataset.py defines the output contract: what one draw returns, and the key that identifies it."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from tabdgp.hyperprior import Hyperprior
from tabdgp.marginal import PARAMETERS, Family

Mechanism = Literal['linear', 'mlp']
Activation = Literal['tanh', 'relu', 'sin', 'identity']
ColumnType = Literal['continuous', 'binary', 'categorical', 'ordered']

# =====================================================================================
# the key
# =====================================================================================


@dataclass(frozen=True)
class Key:
    """The identity of one dataset: its generator is default_rng([seed, index])."""

    seed: Annotated[int, Field(ge=0)]
    index: Annotated[int, Field(ge=0)]

    def generator(self) -> np.random.Generator:
        return np.random.default_rng([self.seed, self.index])


# =====================================================================================
# columns
# =====================================================================================


@dataclass(frozen=True)
class Column:
    """The metadata of one observed column."""

    type: ColumnType
    levels: int | None = None  # K
    role: Literal['covariate'] = 'covariate'  # the only role until grouping factors arrive
    marginal: Family | None = None  # the family a continuous column was matched to
    parameters: dict[str, float] = Field(default_factory=dict)  # its own values of the fields

    def __post_init__(self) -> None:
        # binary is exactly K = 2 and categorical or ordered K ≥ 3, so no two types overlap
        if self.type == 'continuous':
            agree = self.levels is None
        elif self.type == 'binary':
            agree = self.levels == 2
        else:
            agree = self.levels is not None and self.levels >= 3
        if not agree:
            raise ValueError(f'a {self.type} column cannot have {self.levels} levels')
        if (self.type == 'continuous') != (self.marginal is not None):
            raise ValueError(f'a {self.type} column cannot have the marginal {self.marginal}')
        # exactly the values the column was drawn with: its family's, or its level concentration
        own = PARAMETERS[self.marginal] if self.type == 'continuous' else ['level_alpha']
        if self.parameters.keys() != set(own):
            raise ValueError(f'a {self.type} column draws {own}, not {list(self.parameters)}')

    @property
    def width(self) -> int:
        """Its design columns: 1 if continuous, K − 1 if not."""
        return 1 if self.levels is None else self.levels - 1


# =====================================================================================
# the generating graph
# =====================================================================================


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Graph:
    """Every node of a dataset in topological order, observed ones `x…`, latent ones `z…`."""

    names: list[str]
    parents: np.ndarray  # (n, n) bool, parents[j, i] if i is a parent of j
    latent: np.ndarray  # (n,) bool
    # each node's own choices, None or NaN where it uses none: a root is its noise, a linear
    # node has no activation and no init scale
    mechanism: list[Mechanism | None]
    activation: list[Activation | None]
    noise_scale: np.ndarray  # (n,) float
    init_scale: np.ndarray  # (n,) float

    def __post_init__(self) -> None:
        n = len(self.names)
        shapes = (
            self.parents.shape,
            self.latent.shape,
            len(self.mechanism),
            len(self.activation),
            self.noise_scale.shape,
            self.init_scale.shape,
        )
        if shapes != ((n, n), (n,), n, n, (n,), (n,)):
            raise ValueError(f'{n} names, but the per-node fields have shapes {shapes}')
        if np.triu(self.parents).any():
            raise ValueError('a parent on or above the diagonal breaks the topological order')
        root = ~self.parents.any(axis=1)  # (n,)
        not_mlp = np.array([family != 'mlp' for family in self.mechanism])  # (n,)
        unused = {
            'mechanism': ([family is None for family in self.mechanism], root),
            'noise_scale': (np.isnan(self.noise_scale), root),
            'activation': ([name is None for name in self.activation], not_mlp),
            'init_scale': (np.isnan(self.init_scale), not_mlp),
        }
        for field, (missing, unused_by) in unused.items():
            if not np.array_equal(missing, unused_by):
                raise ValueError(f'{field} is missing for nodes {missing}, unused by {unused_by}')


# =====================================================================================
# the dataset
# =====================================================================================


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Dataset:
    """What one draw returns; the arrays are held as given, never copied."""

    columns: dict[str, np.ndarray]  # (N,) each, float32 if continuous, int64 codes 0..K−1 if not
    metadata: dict[str, Column]
    graph: Graph
    latent: dict[str, np.ndarray]  # (N,) float32 each, z-scored
    hyperparameters: Hyperprior  # resolved, but for the fields each column or node draws
    key: Key
    version: str  # of TabDGP; a key reproduces a dataset only within one version
