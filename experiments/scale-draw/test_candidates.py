"""Checks that the candidates rewrite only a prior's scales, and that they measure a budget as the
joint draw spends it. Run by hand from the repository root:

    uv run pytest experiments/scale-draw/test_candidates.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# candidates.py is a script's module, not a package's
sys.path.insert(0, str(Path(__file__).parent))

from candidates import FAMILIES, R2d2m2, Stated, _recoverOffsets, _replaceScales

from tabdgp import Glmm, Scm, drawDataset, modelOutcome
from tabdgp.prior import Prior

SEED = 2026
INDICES = range(3)


def _drawJointPrior(family: str, index: int) -> Prior:
    """The joint draw's prior on a grouped dataset's covariates."""
    covariates = drawDataset(SEED, index, Scm.grouped(outcome_source='none')).covariates
    return modelOutcome(SEED, index, covariates, Glmm(family=family)).prior


@pytest.mark.parametrize('index', INDICES)
def test_r2d2m2_keeps_a_gaussian_y_at_unit_variance(index):
    """R2D2M2 sets σ_ε's scale to √(1 − R²), so its scales must add exactly R² to var(eta)."""
    joint = _drawJointPrior('gaussian', index)
    candidate = R2d2m2(concentration=1.0)

    prior = candidate.rescale(joint, 'gaussian', np.random.default_rng(index))

    assert prior.predictor_sd**2 + prior.sigma_eps_scale**2 == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize('index', INDICES)
@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize(
    'candidate',
    [
        pytest.param(R2d2m2(concentration=0.5), id='r2d2m2'),
        pytest.param(Stated(), id='unit'),
    ],
)
def test_a_candidate_rewrites_only_the_scales(candidate, family, index):
    """Two candidates differ by their scales alone, so all else stays the joint draw's."""
    joint = _drawJointPrior(family, index)

    prior = candidate.rescale(joint, family, np.random.default_rng(index))

    assert prior.beta_family == joint.beta_family
    assert prior.sigma_alpha_family == joint.sigma_alpha_family
    assert prior.sigma_eps_family == joint.sigma_eps_family
    assert prior.lkj_eta == joint.lkj_eta
    assert prior.blocks == joint.blocks
    assert prior.beta_locations[0] == joint.beta_locations[0]
    assert prior.beta_scales[0] == joint.beta_scales[0]
    np.testing.assert_allclose(_recoverOffsets(prior), _recoverOffsets(joint), rtol=1e-12)


@pytest.mark.parametrize('index', INDICES)
@pytest.mark.parametrize('family', FAMILIES)
def test_the_joint_scales_spend_the_joint_budget(family, index):
    """The candidates measure a budget by the rule `Glm.drawPrior` spends it by,
    so a change to that rule in src/ fails here instead of skewing the comparison.
    """
    joint = _drawJointPrior(family, index)
    fixed, *factors = joint.blocks
    scales = {fixed.name: np.array(joint.beta_scales[1:])}
    for block in factors:
        scales[block.name] = np.array(joint.sigma_alpha_scales[block.name])

    prior = _replaceScales(joint, scales, joint.sigma_eps_scale)

    assert prior.predictor_sd == pytest.approx(joint.predictor_sd, rel=1e-12)
    assert prior.split == pytest.approx(joint.split, rel=1e-12, abs=1e-15)
