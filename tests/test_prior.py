"""The prior module's seams: a Glm built with overrides and resolved to points,
and its draw of the scales and the parameters over named blocks.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from tabdgp.field import Beta, Dirichlet, Distribution, LogUniform
from tabdgp.prior import Block, Glm
from tests.fixtures.build import point

DEFAULTS = {
    'family_shares': {'normal': 1 / 3, 'bernoulli': 1 / 3, 'poisson': 1 / 3},
    'sd_family_shares': {
        'half_normal': 0.25,
        'half_student_t': 0.25,
        'half_cauchy': 0.25,
        'exponential': 0.25,
    },
    'residual_shares': {'normal': 1.0},
    'intercept_scale': LogUniform(0.1, 10),
    'signal_share': Beta(2, 2),
    'predictor_sd': LogUniform(0.25, 4),
    'deviation_scale': 0.0,
}


def test_the_fields_take_their_defaults_and_an_override() -> None:
    """ADR 0006's envelope as the comment on #46 amended it: what a GLM corpus draws by default.
    An override names a shares field's categories in any order and keeps the default's.
    """
    defaults = {field: getattr(Glm(), field) for field in Glm.model_fields}
    assert defaults == DEFAULTS
    glm = Glm(family_shares={'poisson': 1.0, 'normal': 0.0, 'bernoulli': 0.0}, signal_share=0.5)
    assert list(glm.family_shares) == ['normal', 'bernoulli', 'poisson']
    assert (glm.family_shares['poisson'], glm.signal_share) == (1.0, 0.5)


def test_resolve_gives_every_field_a_point_and_the_shares_fields_shares_that_choose() -> None:
    """A model resolves its own fields inside its call, and the draw reads points only.
    A shares field overridden by a Dirichlet must still choose one category per dataset.
    """
    uniform = Dirichlet({'half_normal': 1, 'half_student_t': 1, 'half_cauchy': 1, 'exponential': 1})
    resolved = Glm(sd_family_shares=uniform).resolveDataset(np.random.default_rng(0))
    for field in Glm.model_fields:
        assert not isinstance(getattr(resolved, field), Distribution)
    for field in Glm.SHARES:
        shares = getattr(resolved, field)
        assert shares.choose(np.random.default_rng(0)) in Glm.categories(field)


# r = 0.36 and τ = 2 give √r = 0.6 and √(1 − r) = 0.8; tr cov X = 2.25 gives √ 1.5
IDENTITIES = [
    pytest.param('normal', 2.25, 0.6, 0.4, 0.8, id='normal: τ² = r, σ_β = τ / 1.5, s_ε = √(1 − r)'),
    pytest.param('bernoulli', 2.25, 2.0, 4 / 3, None, id='bernoulli: τ as drawn, no s_ε'),
    pytest.param('poisson', 2.25, 2.0, 4 / 3, None, id='poisson: τ as drawn, no s_ε'),
    pytest.param('poisson', 0.0, 2.0, 0.0, None, id='no covariate column: σ_β = 0'),
]


@pytest.mark.parametrize(
    ('family', 'variance', 'predictor_sd', 'scale', 'residual_scale'), IDENTITIES
)
def test_the_joint_draws_identities_hold(
    family: str, variance: float, predictor_sd: float, scale: float, residual_scale: float | None
) -> None:
    """ADR 0006's scales as the comment on #46 amended them: σ_β = τ √(φ_fixed / tr cov X),
    with φ_fixed = 1 for a GLM, so the fixed block realizes var(Xβ) = τ² on any coding.
    Only the Normal family has a residual scale, and its y then has unit variance.
    An intercept-only design has no slope for σ_β to scale.
    """
    glm = Glm(signal_share=0.36, predictor_sd=2.0, intercept_scale=1.5)
    blocks = [Block(name='fixed', width=4, variance=variance)]
    context, _ = glm.drawParameters(family, blocks, np.random.default_rng(0))
    assert (context.signal_share, context.intercept_scale) == (0.36, 1.5)
    assert context.predictor_sd == pytest.approx(predictor_sd)
    assert context.split == {'fixed': 1.0}
    assert context.scales == {'fixed': pytest.approx(scale)}
    assert context.residual_scale == pytest.approx(residual_scale)


def _halfNormalCdf(x: float) -> float:
    return math.erf(x / math.sqrt(2))


def _halfStudentT3Cdf(x: float) -> float:
    # twice Student's t CDF at ν = 3 minus 1, in closed form
    t = x / math.sqrt(3)
    return 2 / math.pi * (t / (1 + t**2) + math.atan(t))


def _halfCauchyCdf(x: float) -> float:
    return 2 / math.pi * math.atan(x)


def _exponentialCdf(x: float) -> float:
    return 1 - math.exp(-x)


def _ksDistance(samples: list[float], cdf: Callable[[float], float]) -> float:
    """The Kolmogorov–Smirnov distance between a sample and a distribution function."""
    n = len(samples)
    distance = 0.0
    for i, x in enumerate(sorted(samples)):
        expected = cdf(x)
        distance = max(distance, (i + 1) / n - expected, expected - i / n)
    return distance


SD_FAMILIES = [
    pytest.param('half_normal', _halfNormalCdf, id='half_normal'),
    pytest.param('half_student_t', _halfStudentT3Cdf, id='half_student_t'),
    pytest.param('half_cauchy', _halfCauchyCdf, id='half_cauchy'),
    pytest.param('exponential', _exponentialCdf, id='exponential'),
]
DRAWS = 2000


@pytest.mark.parametrize(('sd_family', 'cdf'), SD_FAMILIES)
def test_each_parameter_follows_its_stated_family_at_its_scale_over_many_draws(
    sd_family: str, cdf: Callable[[float], float]
) -> None:
    """Given the context, each parameter is drawn from its stated family exactly (ADR 0006),
    nothing clipped or rescaled, so metabeta's posterior conditions on the true prior.
    σ_ε / s_ε follows F's standard law, the intercept N(0, σ_0) and each slope N(0, σ_β).
    The KS bound is the 1 % critical value at 2000 draws.
    """
    glm = Glm(
        sd_family_shares=point('sd_family_shares', sd_family, Glm),
        signal_share=0.36,  # s_ε = 0.8
        intercept_scale=1.5,
    )
    blocks = [Block(name='fixed', width=3, variance=0.36)]  # σ_β = √0.36 · √(1 / 0.36) = 1
    rng = np.random.default_rng(0)
    standardized = []
    intercepts = []
    slopes = []
    for _ in range(DRAWS):
        _, parameters = glm.drawParameters('normal', blocks, rng)
        standardized.append(parameters.sigma_eps / 0.8)
        intercepts.append(parameters.beta[0])
        slopes.extend(parameters.beta[1:])
    assert _ksDistance(standardized, cdf) < 1.63 / math.sqrt(DRAWS)
    assert np.var(intercepts) == pytest.approx(1.5**2, rel=0.1)
    assert np.var(slopes) == pytest.approx(1.0, rel=0.1)


@pytest.mark.parametrize(('family', 'residual_law'), [('normal', 'normal'), ('poisson', None)])
def test_the_context_names_the_block_fixed_and_the_chosen_families(
    family: str, residual_law: str | None
) -> None:
    """metabeta reads the context by block name and conditions on the families chosen.
    A family without a residual has no residual law.
    """
    glm = Glm(sd_family_shares=point('sd_family_shares', 'half_cauchy', Glm))
    resolved = glm.resolveDataset(np.random.default_rng(0))
    blocks = [Block(name='fixed', width=3, variance=2.0)]
    context, _ = resolved.drawParameters(family, blocks, np.random.default_rng(0))
    assert list(context.split) == list(context.scales) == ['fixed']
    assert (context.sd_family, context.residual_law) == ('half_cauchy', residual_law)


def test_the_draw_is_deterministic_under_a_fixed_generator() -> None:
    """A dataset is a pure function of its key, its parameters included."""
    resolved = Glm().resolveDataset(np.random.default_rng(0))
    blocks = [Block(name='fixed', width=5, variance=3.0)]
    first_context, first = resolved.drawParameters('normal', blocks, np.random.default_rng(7))
    second_context, second = resolved.drawParameters('normal', blocks, np.random.default_rng(7))
    assert first_context == second_context
    assert first.beta.tobytes() == second.beta.tobytes()
    assert first.sigma_eps == second.sigma_eps
