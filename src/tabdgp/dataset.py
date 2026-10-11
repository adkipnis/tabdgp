"""dataset.py defines the output contract: what one draw returns, and the key that identifies it."""

from __future__ import annotations

from typing import Annotated, Literal

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from tabdgp.hyperprior import Hyperprior

# =====================================================================================
# the key
# =====================================================================================


@dataclass(frozen=True)
class Key:
    """The identity of one dataset: its generator is default_rng([seed, index])."""

    seed: Annotated[int, Field(ge=0)]
    index: Annotated[int, Field(ge=0)]


# =====================================================================================
# columns
# =====================================================================================


@dataclass(frozen=True)
class Column:
    """The metadata of one observed column."""

    type: Literal['continuous', 'binary', 'categorical', 'ordered']
    levels: int | None = None  # K
    role: Literal['covariate'] = 'covariate'
    marginal: str | None = None  # the family a continuous column was matched to

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
        if self.type != 'continuous' and self.marginal is not None:
            raise ValueError(f'a {self.type} column has no marginal, got {self.marginal}')


# =====================================================================================
# the generating graph
# =====================================================================================


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Graph:
    """Every node of a dataset in topological order, observed ones `x…`, latent ones `z…`."""

    names: list[str]
    parents: np.ndarray  # (n, n) bool, parents[j, i] if i is a parent of j
    latent: np.ndarray  # (n,) bool
    mechanism: list[Literal['linear', 'mlp', 'noise']]

    def __post_init__(self) -> None:
        n = len(self.names)
        shapes = (self.parents.shape, self.latent.shape, len(self.mechanism))
        if shapes != ((n, n), (n,), n):
            raise ValueError(f'{n} names, but parents, latent and mechanism have shapes {shapes}')
        if np.triu(self.parents).any():
            raise ValueError('a parent on or above the diagonal breaks the topological order')
        roots = ~self.parents.any(axis=1)  # (n,)
        noise = np.array(self.mechanism) == 'noise'  # (n,)
        if (roots != noise).any():
            raise ValueError(f'noise must be the mechanism of exactly the roots {roots.tolist()}')


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
    hyperparameters: Hyperprior  # resolved: points only
    key: Key
    version: str  # of TabDGP; a key reproduces a dataset only within one version
