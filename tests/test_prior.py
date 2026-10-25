"""The prior module's seams: a Glm built with overrides and resolved to points,
and its draw of the scales and the parameters over named blocks.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.field import Beta, Dirichlet, Distribution, LogUniform, UniformInt
from tabdgp.prior import Block, Glm, Glmm
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


GLMM_DEFAULTS = DEFAULTS | {
    'fixed_share': Beta(1, 1),
    'split_alpha': 1.0,
    'random_effects': UniformInt(1, 6),
    'correlated_share': 0.55,
    'lkj_eta': UniformInt(1, 4),
}


def test_a_glmm_adds_its_fields_to_a_glms_with_their_defaults() -> None:
    """ADR 0006's envelope as the comment on #46 amended it: what metabeta trains on.
    The fixed block's share has its own Beta, and the factors split the rest.
    """
    defaults = {field: getattr(Glmm(), field) for field in Glmm.model_fields}
    assert defaults == GLMM_DEFAULTS


@pytest.mark.parametrize(
    ('field', 'value'),
    [
        pytest.param('random_effects', 0, id='a budget of 0'),
        pytest.param('random_effects', UniformInt(0, 3), id='a budget reaching 0'),
        pytest.param(
            'lkj_eta', UniformInt(0, 4), id='an LKJ shape reaching 0, where rng.beta fails'
        ),
    ],
)
def test_a_glmm_field_that_may_draw_0_cannot_be_written(
    field: str, value: int | UniformInt
) -> None:
    """Every factor of a GLMM has at least its random intercept, and an LKJ shape is positive,
    so a field that may draw 0 fails where it is written, not at a draw.
    """
    with pytest.raises(ValidationError, match=field):
        Glmm(**{field: value})


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


def test_resolve_leaves_the_fields_each_factor_draws_as_given() -> None:
    """Each factor draws its own budget q_k, truncated at d, and its own η when correlated,
    so a dataset's resolution keeps both distributions and gives every other field a point.
    """
    resolved = Glmm().resolveDataset(np.random.default_rng(0))
    assert (resolved.random_effects, resolved.lkj_eta) == (UniformInt(1, 6), UniformInt(1, 4))
    for field in Glmm.model_fields:
        if field not in ['random_effects', 'lkj_eta']:
            assert not isinstance(getattr(resolved, field), Distribution)


# r = 0.36 and τ = 2 give √r = 0.6 and √(1 − r) = 0.8; tr cov X = 2.25 gives √ 1.5
IDENTITIES = [
    pytest.param(
        'normal', 2.25, 0.36, 0.6, 0.4, 0.8, id='normal: τ² = r, σ_β = τ / 1.5, s_ε = √(1 − r)'
    ),
    pytest.param(
        'bernoulli', 2.25, None, 2.0, 4 / 3, None, id='bernoulli: τ as drawn, no r or s_ε'
    ),
    pytest.param('poisson', 2.25, None, 2.0, 4 / 3, None, id='poisson: τ as drawn, no r or s_ε'),
    pytest.param('poisson', 0.0, None, 2.0, 0.0, None, id='no covariate column: σ_β = 0'),
]


@pytest.mark.parametrize(
    ('family', 'variance', 'signal_share', 'predictor_sd', 'scale', 'residual_scale'), IDENTITIES
)
def test_the_joint_draws_identities_hold(
    family: str,
    variance: float,
    signal_share: float | None,
    predictor_sd: float,
    scale: float,
    residual_scale: float | None,
) -> None:
    """ADR 0006's scales as the comment on #46 amended them: σ_β = τ √(φ_fixed / tr cov X),
    with φ_fixed = 1 for a GLM, so the fixed block realizes var(Xβ) = τ² on any coding.
    Only the Normal family has a residual scale, and its y then has unit variance at the scales.
    r sets nothing under the other families, so the context records none there.
    An intercept-only design has no slope for σ_β to scale.
    """
    glm = Glm(signal_share=0.36, predictor_sd=2.0, intercept_scale=1.5)
    blocks = [Block(name='fixed', width=4, variance=variance)]
    context, _ = glm.drawParameters(family, blocks, np.random.default_rng(0))
    assert (context.signal_share, context.intercept_scale) == (signal_share, 1.5)
    assert context.predictor_sd == pytest.approx(predictor_sd)
    assert context.split == {'fixed': 1.0}
    assert context.scales == {'fixed': pytest.approx(scale)}
    assert context.residual_scale == pytest.approx(residual_scale)


def test_a_glmms_fixed_block_takes_its_share_and_one_factor_the_rest() -> None:
    """The comment on #46: σ_β = τ √(φ_fixed / tr cov X) and s_α,k = τ √(φ_k / tr E[z_k z_k']).
    One factor takes all of 1 − φ_fixed, so every scale is known:
    τ = √0.36 = 0.6, σ_β = 0.6 · √(0.64 / 2.25) = 0.32 and s_α = 0.6 · √(0.36 / 4) = 0.18.
    """
    glmm = Glmm(signal_share=0.36, fixed_share=0.64, intercept_scale=1.0)
    blocks = [
        Block(name='fixed', width=4, variance=2.25),
        Block(name='g0', width=2, variance=4.0, groups=3),
    ]
    context, _ = glmm.drawParameters('normal', blocks, np.random.default_rng(0))
    assert context.split == {'fixed': 0.64, 'g0': pytest.approx(0.36)}
    assert context.scales == {'fixed': pytest.approx(0.32), 'g0': pytest.approx(0.18)}


@pytest.mark.parametrize(
    'fixed',
    [
        pytest.param(Block(name='fixed', width=4, variance=2.25), id='covariate columns'),
        pytest.param(
            Block(name='fixed', width=1, variance=0.0),
            id='no covariate column: the factors take all of var(η)',
        ),
    ],
)
def test_the_blocks_of_a_glmm_share_all_of_var_eta(fixed: Block) -> None:
    """φ sums to 1 over the blocks, and each block realizes its share at scale s:
    s² · (the variance its term adds at scale 1) = τ² φ, so the scales spend exactly τ².
    A fixed block without covariate columns adds nothing at any scale,
    so its share would be lost unless the factors take it.
    """
    glmm = Glmm(predictor_sd=2.0)
    blocks = [
        fixed,
        Block(name='g0', width=2, variance=1.5, groups=3),
        Block(name='g1', width=1, variance=1.0, groups=20),
    ]
    context, _ = glmm.resolveDataset(np.random.default_rng(0)).drawParameters(
        'poisson', blocks, np.random.default_rng(1)
    )
    assert list(context.split) == ['fixed', 'g0', 'g1']
    assert sum(context.split.values()) == pytest.approx(1)
    for block in blocks:
        realized = context.scales[block.name] ** 2 * block.variance
        assert realized == pytest.approx(4.0 * context.split[block.name])


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


def _standardNormalCdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _uniformCdf(x: float) -> float:
    return x


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
    σ_ε / s_ε and each σ_α / s_α follow F's standard law, the intercept N(0, σ_0),
    each slope N(0, σ_β) and each random effect N(0, σ_α) given its σ_α.
    The KS bound is the 1 % critical value at the number of draws.
    """
    glmm = Glmm(
        sd_family_shares=point('sd_family_shares', sd_family, Glm),
        signal_share=0.36,  # τ = 0.6, s_ε = 0.8
        intercept_scale=1.5,
        fixed_share=0.5,
        correlated_share=0.0,
    )
    # each block's share 0.5 over its variance 0.18 gives σ_β = s_α = 0.6 · √(0.5 / 0.18) = 1
    blocks = [
        Block(name='fixed', width=3, variance=0.18),
        Block(name='g0', width=2, variance=0.18, groups=5),
    ]
    rng = np.random.default_rng(0)
    residual_sds = []
    random_sds = []
    intercepts = []
    slopes = []
    random_effects = []
    for _ in range(DRAWS):
        _, parameters = glmm.drawParameters('normal', blocks, rng)
        residual_sds.append(parameters.sigma_eps / 0.8)
        random_sds.extend(parameters.sd_alpha['g0'])
        intercepts.append(parameters.beta[0])
        slopes.extend(parameters.beta[1:])
        standardized = parameters.alpha['g0'] / parameters.sd_alpha['g0']  # (m, q)
        random_effects.extend(standardized.ravel())
    assert _ksDistance(residual_sds, cdf) < 1.63 / math.sqrt(DRAWS)
    assert _ksDistance(random_sds, cdf) < 1.63 / math.sqrt(len(random_sds))
    assert np.var(intercepts) == pytest.approx(1.5**2, rel=0.1)
    assert np.var(slopes) == pytest.approx(1.0, rel=0.1)
    bound = 1.63 / math.sqrt(len(random_effects))
    assert _ksDistance(random_effects, _standardNormalCdf) < bound


def test_the_split_follows_fixed_share_and_the_dirichlet_over_many_draws() -> None:
    """φ_fixed is fixed_share's Beta(1, 1), the registry's near-uniform witness (#47),
    and two factors split the rest by a Dirichlet at split_alpha = 1, so either share is uniform.
    """
    blocks = [
        Block(name='fixed', width=3, variance=1.0),
        Block(name='g0', width=1, variance=1.0, groups=4),
        Block(name='g1', width=1, variance=1.0, groups=8),
    ]
    rng = np.random.default_rng(0)
    fixed = []
    first = []
    for _ in range(DRAWS):
        resolved = Glmm().resolveDataset(rng)
        context, _ = resolved.drawParameters('bernoulli', blocks, rng)
        fixed.append(context.split['fixed'])
        first.append(context.split['g0'] / (1 - context.split['fixed']))
    assert _ksDistance(fixed, _uniformCdf) < 1.63 / math.sqrt(DRAWS)
    assert _ksDistance(first, _uniformCdf) < 1.63 / math.sqrt(DRAWS)


def test_the_lkj_marginal_variance_is_1_over_2_eta_plus_q_minus_1_over_draws_symmetric_unit_diagonal_positive_definite() -> (
    None
):
    """Ω comes from the C-vine of Lewandowski, Kurowicka and Joe (2009, section 3.2) in numpy,
    pinned by its analytic marginal: each off-diagonal entry has variance 1 / (2η + q − 1),
    1/7 at q = 4 and η = 2, and mean 0.
    Every entry is checked, since all but the first row pass through the vine's recursion.
    """
    glmm = Glmm(correlated_share=1.0, lkj_eta=2).resolveDataset(np.random.default_rng(0))
    blocks = [
        Block(name='fixed', width=3, variance=1.0),
        Block(name='g0', width=4, variance=4.0, groups=2),
    ]
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(DRAWS):
        context, parameters = glmm.drawParameters('bernoulli', blocks, rng)
        corr = parameters.corr_alpha['g0']  # (q, q)
        assert np.array_equal(corr, corr.T)
        assert np.diag(corr).tolist() == [1.0] * 4
        assert np.linalg.eigvalsh(corr).min() > 0
        draws.append(corr)
    assert context.lkj_eta == {'g0': 2}
    rows, columns = np.triu_indices(4, k=1)
    entries = np.stack(draws)[:, rows, columns]  # (s, 6)
    np.testing.assert_allclose(entries.var(axis=0), 1 / 7, rtol=0.1)
    np.testing.assert_allclose(entries.mean(axis=0), 0, atol=0.03)


UNCORRELATED = [
    pytest.param(0.0, 3, id='an uncorrelated factor'),
    pytest.param(1.0, 1, id='a lone intercept, with nothing to correlate'),
]


@pytest.mark.parametrize(('correlated_share', 'width'), UNCORRELATED)
def test_an_uncorrelated_factor_gets_the_identity(correlated_share: float, width: int) -> None:
    """A factor's effects are independent with probability 1 − correlated_share,
    and always at q_k = 1 (ADR 0006): Ω is the identity and the context records no η.
    """
    glmm = Glmm(correlated_share=correlated_share, lkj_eta=2)
    resolved = glmm.resolveDataset(np.random.default_rng(0))
    blocks = [
        Block(name='fixed', width=3, variance=1.0),
        Block(name='g0', width=width, variance=1.0, groups=5),
    ]
    context, parameters = resolved.drawParameters('poisson', blocks, np.random.default_rng(1))
    assert context.lkj_eta == {'g0': None}
    assert np.array_equal(parameters.corr_alpha['g0'], np.eye(width))


# the fixed block and two factors' blocks, in the factors' order
FACTOR_BLOCKS = [
    Block(name='fixed', width=5, variance=3.0),
    Block(name='g0', width=3, variance=2.0, groups=4),
    Block(name='g1', width=2, variance=1.5, groups=12),
]


@pytest.mark.parametrize(('family', 'residual_law'), [('normal', 'normal'), ('poisson', None)])
def test_the_context_names_the_blocks_fixed_then_the_factors_in_order_and_the_chosen_families(
    family: str, residual_law: str | None
) -> None:
    """metabeta reads the context by block name and conditions on the families chosen.
    A family without a residual has no residual law.
    Each factor's parameters are sized by its block: q sds, a q × q correlation, a row per group.
    """
    glmm = Glmm(sd_family_shares=point('sd_family_shares', 'half_cauchy', Glm))
    resolved = glmm.resolveDataset(np.random.default_rng(0))
    context, parameters = resolved.drawParameters(family, FACTOR_BLOCKS, np.random.default_rng(0))
    assert list(context.split) == list(context.scales) == ['fixed', 'g0', 'g1']
    assert list(context.lkj_eta) == list(parameters.alpha) == ['g0', 'g1']
    assert (context.sd_family, context.residual_law) == ('half_cauchy', residual_law)
    for block in FACTOR_BLOCKS[1:]:
        q = block.width
        assert parameters.sd_alpha[block.name].shape == (q,)
        assert parameters.corr_alpha[block.name].shape == (q, q)
        assert parameters.alpha[block.name].shape == (block.groups, q)


def test_the_draw_is_deterministic_under_a_fixed_generator() -> None:
    """A dataset is a pure function of its key, its parameters included."""
    resolved = Glmm(correlated_share=1.0).resolveDataset(np.random.default_rng(0))
    first_context, first = resolved.drawParameters(
        'normal', FACTOR_BLOCKS, np.random.default_rng(7)
    )
    second_context, second = resolved.drawParameters(
        'normal', FACTOR_BLOCKS, np.random.default_rng(7)
    )
    assert first_context == second_context
    assert first.beta.tobytes() == second.beta.tobytes()
    assert first.sigma_eps == second.sigma_eps
    for name in ['g0', 'g1']:
        assert first.sd_alpha[name].tobytes() == second.sd_alpha[name].tobytes()
        assert first.corr_alpha[name].tobytes() == second.corr_alpha[name].tobytes()
        assert first.alpha[name].tobytes() == second.alpha[name].tobytes()
