"""dataset.py defines the output contract: what one draw returns."""

from __future__ import annotations

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from tabdgp.column import Column
from tabdgp.graph import Graph
from tabdgp.hyperprior import Hyperprior

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
