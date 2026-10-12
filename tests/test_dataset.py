"""The output contract's seam: what Dataset guarantees on its own."""

from __future__ import annotations

import numpy as np

from tabdgp.column import Column
from tabdgp.dataset import Dataset
from tabdgp.hyperprior import Hyperprior
from tests.fixtures.build import graphOf


def test_a_dataset_holds_its_arrays_without_copying() -> None:
    """metabeta hands the columns to torch.from_numpy; a copy per draw would cost throughput."""
    values = np.zeros(5, dtype=np.float32)
    latent = np.ones(5, dtype=np.float32)
    dataset = Dataset(
        columns={'x0': values},
        metadata={'x0': Column(type='continuous', marginal='normal')},
        graph=graphOf([[], []]),
        latent={'z0': latent},
        hyperparameters=Hyperprior().resolveDataset(np.random.default_rng(0)),
        seed=0,
        index=0,
        version='0.0.1',
    )
    assert dataset.columns['x0'] is values and dataset.latent['z0'] is latent
