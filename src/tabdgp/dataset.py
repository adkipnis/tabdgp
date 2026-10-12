"""dataset.py defines the output contract."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.hyperprior import MARGINAL_FIELDS, Family, Hyperprior

Mechanism = Literal['linear', 'mlp']
Activation = Literal['tanh', 'relu', 'sin', 'identity']
ColumnType = Literal['continuous', 'binary', 'categorical', 'ordered']

# =====================================================================================
# columns
# =====================================================================================


@pydantic_dataclass(frozen=True)
class Column:
    """Metadata of one observed column."""

    type: ColumnType
    levels: int | None = None  # K
    role: Literal['covariate'] = 'covariate'  # extended to grouping factors or outcome later
    marginal: Family | None = None  # for continuous columns
    parameters: dict[str, float] = Field(default_factory=dict)

    def __post_init__(self) -> None:
        # binary is exactly K = 2 and categorical or ordered K ≥ 3, so no two types overlap
        if self.type == 'continuous':
            agree = self.levels is None
        elif self.type == 'binary':
            agree = self.levels == 2
        else:  # categorical | ordered
            agree = self.levels is not None and self.levels >= 3
        if not agree:
            raise ValueError(f'a {self.type} column cannot have {self.levels} levels')
        if (self.type == 'continuous') != (self.marginal is not None):
            raise ValueError(f'a {self.type} column cannot have the marginal {self.marginal}')
        # check that parameters hold exactly the fields this column drew
        if self.type == 'continuous':
            own = MARGINAL_FIELDS[self.marginal]
        else:
            own = ['level_alpha']
        if self.parameters.keys() != set(own):
            raise ValueError(f'a {self.type} column draws {own}, not {list(self.parameters)}')

    @property
    def design_columns(self) -> int:
        return 1 if self.levels is None else self.levels - 1


# =====================================================================================
# the generating graph
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Graph:
    """Every node of a dataset in topological order (observed `x`, latent `z`).

    - A root (a node without parents) is just its noise and draws nothing.
    - A linear node draws a noise scale.
    - An MLP node draws a noise scale, an activation and an init scale.
    - None at position i for any field <=> node i does not draw that field.
    """

    names: list[str]
    parents: np.ndarray  # (n, n) bool, parents[i, j] is True if j is a parent of i
    latent: np.ndarray  # (n,) bool
    mechanism: list[Mechanism | None]
    activation: list[Activation | None]
    noise_scale: list[float | None]
    init_scale: list[float | None]

    def __post_init__(self) -> None:
        n = len(self.names)
        shapes = (
            self.parents.shape,
            self.latent.shape,
            len(self.mechanism),
            len(self.activation),
            len(self.noise_scale),
            len(self.init_scale),
        )
        if shapes != ((n, n), (n,), n, n, n, n):
            raise ValueError(f'{n} names, but the per-node fields have shapes {shapes}')
        named_latent = [name.startswith('z') for name in self.names]  # (n,)
        if not np.array_equal(self.latent, named_latent):
            raise ValueError(f'the latent mask {self.latent.tolist()} disagrees with {self.names}')
        if np.triu(self.parents).any():
            raise ValueError('a parent on or above the diagonal breaks topological order')
        # check that each node holds a value for the fields it draws
        root = ~self.parents.any(axis=1)  # (n,)
        not_mlp = np.array([family != 'mlp' for family in self.mechanism])  # (n,)
        unused = {
            'mechanism': (self.mechanism, root),
            'noise_scale': (self.noise_scale, root),
            'activation': (self.activation, not_mlp),
            'init_scale': (self.init_scale, not_mlp),
        }
        for field, (values, unused_by) in unused.items():
            missing = [value is None for value in values]  # (n,)
            if not np.array_equal(missing, unused_by):
                raise ValueError(f'{field} is missing for nodes {missing}, unused by {unused_by}')


# =====================================================================================
# the dataset
# =====================================================================================


@pydantic_dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class Dataset:
    """A dataset draw in its white-box format.
    Each array is stored as drawn (not copied), so a consumer can share its memory.
    """

    columns: dict[str, np.ndarray]  # (N,) each, float32 if continuous, int64 codes 0..K−1 otherwise
    metadata: dict[str, Column]
    graph: Graph
    latent: dict[str, np.ndarray]  # (N,) float32 each, z-scored
    hyperparameters: Hyperprior  # resolved, except the fields each column or node draws
    seed: int  # the corpus
    index: int  # the dataset within its corpus
    version: str  # of TabDGP; a key reproduces a dataset only within one version
