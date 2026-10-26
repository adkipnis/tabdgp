"""The prior module's seams: a Glm built with overrides and resolved to points,
and its draw of the scales and the parameters over named blocks.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.field import Beta, Dirichlet, Distribution, LogUniform, Uniform, UniformInt
from tabdgp.prior import Block, Glm, Glmm, PriorContext
from tests.fixtures.build import point

SD_FAMILY_SHARES = {
    'half_normal': 0.25,
    'half_student_t': 0.25,
    'half_cauchy': 0.25,
    'exponential': 0.25,
}
# the contaminated families, each with the field its shape is drawn from, at share zero
CONTAMINATED = {
    'student_t': 'residual_df',
    'negative_binomial': 'dispersion',
    'latent_bernoulli': 'latent_scale',
    'gamma': 'gamma_shape',
    'beta': 'beta_precision',
}
DEFAULTS = {
    'family_shares': {
        'normal': 1 / 3,
        'bernoulli': 1 / 3,
        'poisson': 1 / 3,
        'student_t': 0.0,
        'negative_binomial': 0.0,
        'latent_bernoulli': 0.0,
        'gamma': 0.0,
        'beta': 0.0,
    },
    'beta_family_shares': {'normal': 0.68, 'student_t': 0.30, 'cauchy': 0.02},
    'sigma_eps_family_shares': SD_FAMILY_SHARES,
    'intercept_location': Uniform(-3, 3),
    'intercept_scale': LogUniform(0.1, 10),
    'location_share': 0.1,
    'location_scale': 2.0,
    'varying_scale_share': 0.5,
    'scale_spread': Uniform(0, 2),
    'signal_share': Beta(2, 2),
    'predictor_sd': LogUniform(0.25, 4),
    'deviation_scale': 0.0,
    'residual_df': LogUniform(3, 10),  # ν
    'dispersion': LogUniform(1, 10),  # φ
    'latent_scale': LogUniform(0.5, 2),  # σ_c
    'gamma_shape': LogUniform(1, 10),
    'beta_precision': LogUniform(2, 50),
}


def test_a_block_is_as_wide_as_its_variances_and_has_at_least_one() -> None:
    """A block names what each coefficient adds to var(η) at scale 1, the intercept first,
    so its width is their count and a block without a coefficient cannot be written.
    """
    assert Block(name='fixed', variances=[0.0, 1.0, 0.25]).width == 3
    with pytest.raises(ValidationError, match='variances'):
        Block(name='fixed', variances=[])


def test_the_fields_take_their_defaults_and_an_override() -> None:
    """ADR 0006's envelope as its amendments shaped it: what a GLM corpus draws by default,
    the fixed effects' families and locations what brms users state.
    An override names a shares field's categories in any order and keeps the default's.
    """
    defaults = {field: getattr(Glm(), field) for field in Glm.model_fields}
    assert defaults == DEFAULTS
    glm = Glm(family_shares=point('family_shares', 'poisson', Glm), signal_share=0.5)
    assert list(glm.family_shares) == ['normal', 'bernoulli', 'poisson', *CONTAMINATED]
    assert (glm.family_shares['poisson'], glm.signal_share) == (1.0, 0.5)


def test_the_contaminated_families_are_the_last_categories_each_with_a_shape_field_to_override() -> (
    None
):
    """A study that draws a contaminated family overrides family_shares like deviation_scale.
    The base three keep their places, so a share of zero never moves the choice among them,
    and each shape field takes a point or a distribution of its own.
    """
    assert Glm.categories('family_shares')[3:] == list(CONTAMINATED)
    glm = Glm(
        family_shares=point('family_shares', 'student_t', Glm),
        residual_df=4.0,
        beta_precision=Uniform(5, 6),
    )
    assert (glm.family_shares['student_t'], glm.residual_df) == (1.0, 4.0)
    assert glm.beta_precision == Uniform(5, 6)


@pytest.mark.parametrize(
    'field', ['residual_df', 'dispersion', 'gamma_shape', 'beta_precision', 'latent_scale']
)
def test_a_shape_that_is_negative_cannot_be_written_and_only_a_latent_scale_may_be_zero(
    field: str,
) -> None:
    """ν, φ, the Gamma shape and the beta precision are positive, where numpy fails at 0;
    σ_c = 0 is the plain Bernoulli.
    """
    with pytest.raises(ValidationError, match=field):
        Glm(**{field: -1.0})
    if field == 'latent_scale':
        assert Glm(latent_scale=0.0).latent_scale == 0.0
    else:
        with pytest.raises(ValidationError, match=field):
            Glm(**{field: 0.0})


GLMM_DEFAULTS = DEFAULTS | {
    'sigma_alpha_family_shares': SD_FAMILY_SHARES,
    'fixed_share': Beta(1, 1),
    'split_alpha': 1.0,
    'random_effects': UniformInt(1, 6),
    'correlated_share': 0.55,
    'lkj_eta': UniformInt(1, 4),
}


def test_a_glmm_adds_its_fields_to_a_glms_with_their_defaults() -> None:
    """ADR 0006's envelope as its amendments shaped it: what metabeta trains on.
    The fixed block's share has its own Beta, and the factors split the rest;
    the random-effect sds have a family of their own, apart from the residual's.
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
    resolved = Glm(sigma_eps_family_shares=uniform).resolveDataset(np.random.default_rng(0))
    for field in Glm.model_fields:
        if field not in CONTAMINATED.values():
            assert not isinstance(getattr(resolved, field), Distribution)
    for field in Glm.SHARES:
        shares = getattr(resolved, field)
        assert shares.choose(np.random.default_rng(0)) in Glm.categories(field)


def test_resolve_leaves_the_fields_each_factor_or_family_draws_as_given() -> None:
    """Each factor draws its own budget q_k, truncated at d, and its own η when correlated.
    Only the chosen family draws its shape, so a family at share zero consumes no draw.
    A dataset's resolution keeps these distributions and gives every other field a point.
    """
    resolved = Glmm().resolveDataset(np.random.default_rng(0))
    assert (resolved.random_effects, resolved.lkj_eta) == (UniformInt(1, 6), UniformInt(1, 4))
    for field in CONTAMINATED.values():
        assert getattr(resolved, field) == getattr(Glmm(), field)
    for field in Glmm.model_fields:
        if field not in ['random_effects', 'lkj_eta', *CONTAMINATED.values()]:
            assert not isinstance(getattr(resolved, field), Distribution)
    # shape fields that are points or distributions leave the generator at the same state
    as_distributions = np.random.default_rng(0)
    as_points = np.random.default_rng(0)
    Glmm().resolveDataset(as_distributions)
    Glmm(**dict.fromkeys(CONTAMINATED.values(), 4.0)).resolveDataset(as_points)
    assert as_distributions.random() == as_points.random()


# r = 0.36 and τ = 2 give √r = 0.6 and √(1 − r) = 0.8; tr cov X = 2.25 gives √ 1.5
THREE_QUARTERS = [0.0, 0.75, 0.75, 0.75]
IDENTITIES = [
    pytest.param(
        'normal',
        THREE_QUARTERS,
        0.36,
        0.6,
        0.4,
        0.8,
        id='normal: τ² = r, σ_β = τ / 1.5, s_ε = √(1 − r)',
    ),
    pytest.param(
        'student_t',
        THREE_QUARTERS,
        0.36,
        0.6,
        0.4,
        0.8,
        id='student_t: drawn like the Normal, τ² = r, s_ε = √(1 − r)',
    ),
    pytest.param(
        'bernoulli', THREE_QUARTERS, None, 2.0, 4 / 3, None, id='bernoulli: τ as drawn, no r or s_ε'
    ),
    pytest.param(
        'poisson', THREE_QUARTERS, None, 2.0, 4 / 3, None, id='poisson: τ as drawn, no r or s_ε'
    ),
    *[
        pytest.param(
            family, THREE_QUARTERS, None, 2.0, 4 / 3, None, id=f'{family}: τ as drawn, no r or s_ε'
        )
        for family in ['negative_binomial', 'latent_bernoulli', 'gamma', 'beta']
    ],
    pytest.param('poisson', [0.0], None, 2.0, 0.0, None, id='no covariate column: σ_β = 0'),
]


@pytest.mark.parametrize(
    ('family', 'variances', 'signal_share', 'predictor_sd', 'scale', 'sigma_eps_scale'), IDENTITIES
)
def test_the_joint_draws_identities_hold(
    family: str,
    variances: list[float],
    signal_share: float | None,
    predictor_sd: float,
    scale: float,
    sigma_eps_scale: float | None,
) -> None:
    """ADR 0006's scales as the comment on #46 amended them: σ_β = τ √(φ_fixed / tr cov X),
    with φ_fixed = 1 for a GLM, so the fixed block realizes var(Xβ) = τ² on any coding.
    Only the Normal and Student-t families have a residual scale, and the Normal's y then has
    unit variance at the scales. r sets nothing under the other families, so the context
    records none there.
    An intercept-only design has no slope for σ_β to scale.
    In the block case and without a located slope, every slope has the block's scale and
    location 0;
    the intercept's location is the field's, on the link scale.
    A GLM has no random effects, so no family or scale for them.
    """
    glm = Glm(
        signal_share=0.36,
        predictor_sd=2.0,
        intercept_location=-2.0,
        intercept_scale=1.5,
        location_share=0.0,
        varying_scale_share=0.0,
    )
    blocks = [Block(name='fixed', variances=variances)]
    context = glm.drawContext(family, blocks, np.random.default_rng(0))
    slopes = len(variances) - 1
    assert context.signal_share == signal_share
    assert (context.sigma_alpha_family, context.sigma_alpha_scales) == (None, {})
    assert context.predictor_sd == pytest.approx(predictor_sd)
    assert context.split == {'fixed': 1.0}
    assert context.beta_scales == pytest.approx([1.5] + [scale] * slopes)
    assert context.beta_locations == [-2.0] + [0.0] * slopes
    assert context.sigma_eps_scale == pytest.approx(sigma_eps_scale)


def test_a_glmms_fixed_block_takes_its_share_and_one_factor_the_rest() -> None:
    """The comment on #46: σ_β = τ √(φ_fixed / tr cov X) and s_α,k = τ √(φ_k / tr E[z_k z_k']).
    One factor takes all of 1 − φ_fixed, so every scale is known:
    τ = √0.36 = 0.6, σ_β = 0.6 · √(0.64 / 2.25) = 0.32 and s_α = 0.6 · √(0.36 / 4) = 0.18.
    """
    glmm = Glmm(
        signal_share=0.36,
        fixed_share=0.64,
        intercept_location=0.0,
        intercept_scale=1.0,
        location_share=0.0,
        varying_scale_share=0.0,
    )
    blocks = [
        Block(name='fixed', variances=THREE_QUARTERS),
        Block(name='g0', variances=[1.0, 3.0], groups=3),
    ]
    context = glmm.drawContext('normal', blocks, np.random.default_rng(0))
    assert context.split == {'fixed': 0.64, 'g0': pytest.approx(0.36)}
    assert context.beta_scales[1:] == pytest.approx([0.32] * 3)
    assert context.sigma_alpha_scales == {'g0': pytest.approx([0.18] * 2)}


def _spent(context: PriorContext, block: Block) -> float:
    """What a block's term adds to var(η) in expectation at the context's scales and locations:
    E[β_j²] var_j = (σ_j² + μ_j²) var_j per slope, s_α,j² E[z_j²] per random effect.
    """
    if block.name == 'fixed':
        slopes = zip(context.beta_locations[1:], context.beta_scales[1:], block.variances[1:])
        return sum((scale**2 + location**2) * variance for location, scale, variance in slopes)
    effects = zip(context.sigma_alpha_scales[block.name], block.variances)
    return sum(scale**2 * variance for scale, variance in effects)


@pytest.mark.parametrize(
    'fixed',
    [
        pytest.param(Block(name='fixed', variances=THREE_QUARTERS), id='covariate columns'),
        pytest.param(
            Block(name='fixed', variances=[0.0]),
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
        Block(name='g0', variances=[1.0, 0.5], groups=3),
        Block(name='g1', variances=[1.0], groups=20),
    ]
    resolved = glmm.resolveDataset(np.random.default_rng(0))
    context = resolved.drawContext('poisson', blocks, np.random.default_rng(1))
    assert list(context.split) == ['fixed', 'g0', 'g1']
    assert sum(context.split.values()) == pytest.approx(1)
    for block in blocks:
        assert _spent(context, block) == pytest.approx(4.0 * context.split[block.name])


# the fixed block and two factors' blocks, in the factors' order
FACTOR_BLOCKS = [
    Block(name='fixed', variances=[0.0, 0.75, 0.75, 0.75, 0.75]),
    Block(name='g0', variances=[1.0, 0.5, 0.5], groups=4),
    Block(name='g1', variances=[1.0, 0.5], groups=12),
]
# the distinct scales of the fixed block's four slopes, then of each factor's effects
VARYING = [
    pytest.param(0.0, [1, 1, 1], id='at share 0 each block shares one scale'),
    pytest.param(1.0, [4, 3, 2], id='at share 1 each coefficient has its own'),
]


@pytest.mark.parametrize(('varying_scale_share', 'distinct'), VARYING)
def test_each_blocks_scales_vary_at_varying_scale_share_1_and_share_one_at_0(
    varying_scale_share: float, distinct: list[int]
) -> None:
    """ADR 0006's amendment of 2026-10-26: a block's scales vary per coefficient
    with probability varying_scale_share, β's slopes and each factor's effects alike,
    a random intercept's among them; otherwise the block shares one scale.
    """
    glmm = Glmm(
        varying_scale_share=varying_scale_share,
        scale_spread=1.0,
        predictor_sd=2.0,
        intercept_location=0.0,
        intercept_scale=1.0,
        location_share=0.0,
        fixed_share=0.5,
    )
    context = glmm.drawContext('poisson', FACTOR_BLOCKS, np.random.default_rng(0))
    scales = [context.beta_scales[1:], *context.sigma_alpha_scales.values()]
    assert [len(set(block_scales)) for block_scales in scales] == distinct


def test_located_and_varying_scales_spend_each_blocks_budget_exactly() -> None:
    """ADR 0006's amendments: σ_j = σ_β w_j and μ_j = δ_j σ_j over the slopes,
    s_k,j = s_k w_k,j over a factor's effects, each block's scale solved from its budget,
    σ_β² Σ_j w_j² (1 + δ_j²) var_j = τ² φ_fixed and s_k² Σ_j w_k,j² E[z_j²] = τ² φ_k.
    So a location or a weight moves the scales around but never what a block adds to var(η)
    in expectation, and the constant column costs nothing.
    """
    glmm = Glmm(
        intercept_location=0.0,
        intercept_scale=1.0,
        predictor_sd=2.0,
        location_share=1.0,
        location_scale=2.0,
        varying_scale_share=1.0,
        scale_spread=1.0,
        fixed_share=0.5,
    )
    blocks = [Block(name='fixed', variances=[0.0, 0.5, 1.0, 2.0, 0.0]), *FACTOR_BLOCKS[1:]]
    for seed in range(5):
        context = glmm.drawContext('poisson', blocks, np.random.default_rng(seed))
        assert all(location != 0 for location in context.beta_locations[1:])
        for block in blocks:
            assert _spent(context, block) == pytest.approx(4.0 * context.split[block.name])


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
        beta_family_shares=point('beta_family_shares', 'normal', Glm),
        sigma_eps_family_shares=point('sigma_eps_family_shares', sd_family, Glm),
        sigma_alpha_family_shares=point('sigma_alpha_family_shares', sd_family, Glmm),
        signal_share=0.36,  # τ = 0.6, s_ε = 0.8
        intercept_location=0.0,
        intercept_scale=1.5,
        location_share=0.0,
        fixed_share=0.5,
        correlated_share=0.0,
        varying_scale_share=0.0,
    )
    # each block's share 0.5 over its variance 0.18 gives σ_β = s_α = 0.6 · √(0.5 / 0.18) = 1
    blocks = [
        Block(name='fixed', variances=[0.0, 0.09, 0.09]),
        Block(name='g0', variances=[0.09, 0.09], groups=5),
    ]
    rng = np.random.default_rng(1)  # seed 0's half-Cauchy σ_α miss the 1 % bound by chance
    residual_sds = []
    random_sds = []
    intercepts = []
    slopes = []
    random_effects = []
    for _ in range(DRAWS):
        parameters = glmm.drawContext('normal', blocks, rng).drawParameters(rng)
        residual_sds.append(parameters.sigma_eps / 0.8)
        random_sds.extend(parameters.sigma_alpha['g0'])
        intercepts.append(parameters.beta[0])
        slopes.extend(parameters.beta[1:])
        standardized = parameters.alpha['g0'] / parameters.sigma_alpha['g0']  # (m, q)
        random_effects.extend(standardized.ravel())
    assert _ksDistance(residual_sds, cdf) < 1.63 / math.sqrt(DRAWS)
    assert _ksDistance(random_sds, cdf) < 1.63 / math.sqrt(len(random_sds))
    assert np.var(intercepts) == pytest.approx(1.5**2, rel=0.1)
    assert np.var(slopes) == pytest.approx(1.0, rel=0.1)
    bound = 1.63 / math.sqrt(len(random_effects))
    assert _ksDistance(random_effects, _standardNormalCdf) < bound


def test_a_slope_is_located_with_location_share_and_its_relative_location_is_normal_at_location_scale() -> (
    None
):
    """The survey's stated hypotheses are rare and sit a few scales out, so by default
    each slope has a location with probability 0.1, and then μ_j / σ_j ~ N(0, 2).
    """
    fixed = Block(name='fixed', variances=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(1)  # seed 0's first 10,000 slopes land 3.8 sd above 0.1
    relative = []
    for _ in range(DRAWS):
        resolved = Glm().resolveDataset(rng)
        context = resolved.drawContext('poisson', [fixed], rng)
        for location, scale in zip(context.beta_locations[1:], context.beta_scales[1:]):
            relative.append(location / scale)
    located = [delta / 2 for delta in relative if delta != 0]
    three_sds = 3 * math.sqrt(0.1 * 0.9 / len(relative))
    assert len(located) / len(relative) == pytest.approx(0.1, abs=three_sds)
    assert _ksDistance(located, _standardNormalCdf) < 1.63 / math.sqrt(len(located))


def _studentT3Cdf(x: float) -> float:
    return (1 + _halfStudentT3Cdf(abs(x)) * math.copysign(1, x)) / 2


def _cauchyCdf(x: float) -> float:
    return 0.5 + math.atan(x) / math.pi


BETA_FAMILIES = [
    pytest.param('normal', _standardNormalCdf, id='normal'),
    pytest.param('student_t', _studentT3Cdf, id='student_t'),
    pytest.param('cauchy', _cauchyCdf, id='cauchy'),
]


@pytest.mark.parametrize(('beta_family', 'cdf'), BETA_FAMILIES)
def test_each_fixed_effect_follows_the_beta_family_at_its_recorded_location_and_scale(
    beta_family: str, cdf: Callable[[float], float]
) -> None:
    """The intercept and each slope are drawn from the β family the context names,
    at the location and scale it records for them, so (β_j − μ_j) / σ_j follows the family's
    standard law, Student-t at ν = 3, whatever the locations and the spread.
    """
    glm = Glm(
        beta_family_shares=point('beta_family_shares', beta_family, Glm),
        intercept_location=1.0,
        intercept_scale=0.5,
        location_share=1.0,
        scale_spread=0.5,
        predictor_sd=2.0,
    )
    fixed = Block(name='fixed', variances=[0.0, 0.5, 1.0])
    rng = np.random.default_rng(0)
    intercepts = []
    slopes = []
    for _ in range(DRAWS):
        context = glm.drawContext('bernoulli', [fixed], rng)
        parameters = context.drawParameters(rng)
        assert context.beta_family == beta_family
        intercepts.append((parameters.beta[0] - 1.0) / 0.5)
        locations = np.array(context.beta_locations[1:])  # (d − 1,)
        scales = np.array(context.beta_scales[1:])  # (d − 1,)
        standardized = (parameters.beta[1:] - locations) / scales
        slopes.extend(standardized)
    assert _ksDistance(intercepts, cdf) < 1.63 / math.sqrt(DRAWS)
    assert _ksDistance(slopes, cdf) < 1.63 / math.sqrt(len(slopes))


def test_the_log_ratio_of_two_scales_in_a_block_is_normal_at_root_2_scale_spread() -> None:
    """In a block that varies, each coefficient's log weight is N(0, scale_spread),
    so the log-ratio of two of its scales is N(0, 2 scale_spread²), whatever the block's scale.
    A random intercept's scale is weighted like a slope's.
    """
    glmm = Glmm(
        varying_scale_share=1.0,
        scale_spread=0.5,
        predictor_sd=2.0,
        intercept_location=0.0,
        intercept_scale=1.0,
        fixed_share=0.5,
    )
    ratio_sd = math.sqrt(2) * 0.5
    rng = np.random.default_rng(0)
    slopes = []
    effects = []
    for _ in range(DRAWS):
        context = glmm.drawContext('poisson', FACTOR_BLOCKS, rng)
        first, second = context.beta_scales[1:3]
        slopes.append(math.log(first / second))
        intercept, slope = context.sigma_alpha_scales['g0'][:2]
        effects.append(math.log(intercept / slope))
    for log_ratios in [slopes, effects]:
        standardized = np.array(log_ratios) / ratio_sd  # (s,)
        assert _ksDistance(standardized, _standardNormalCdf) < 1.63 / math.sqrt(DRAWS)


def _varies(scales: list[float]) -> bool:
    return len(set(scales)) > 1


def test_each_block_varies_with_varying_scale_share_over_many_draws() -> None:
    """By default a block's scales vary with probability 0.5 (ADR 0006's amendment of 2026-10-26),
    the fixed block and each factor on their own, so two blocks vary together in a quarter.
    A varying block's scales differ, since scale_spread resolves above 0.
    """
    rng = np.random.default_rng(0)
    fixed = []
    first = []
    second = []
    for _ in range(DRAWS):
        resolved = Glmm().resolveDataset(rng)
        context = resolved.drawContext('poisson', FACTOR_BLOCKS, rng)
        fixed.append(_varies(context.beta_scales[1:]))
        first.append(_varies(context.sigma_alpha_scales['g0']))
        second.append(_varies(context.sigma_alpha_scales['g1']))
    both = np.logical_and(fixed, first)  # (s,)
    for varying, share in [(fixed, 0.5), (first, 0.5), (second, 0.5), (both, 0.25)]:
        three_sds = 3 * math.sqrt(share * (1 - share) / DRAWS)
        assert np.mean(varying) == pytest.approx(share, abs=three_sds)


def test_the_split_follows_fixed_share_and_the_dirichlet_over_many_draws() -> None:
    """φ_fixed is fixed_share's Beta(1, 1), the registry's near-uniform witness (#47),
    and two factors split the rest by a Dirichlet at split_alpha = 1, so either share is uniform.
    """
    blocks = [
        Block(name='fixed', variances=[0.0, 0.5, 0.5]),
        Block(name='g0', variances=[1.0], groups=4),
        Block(name='g1', variances=[1.0], groups=8),
    ]
    rng = np.random.default_rng(0)
    fixed = []
    first = []
    for _ in range(DRAWS):
        resolved = Glmm().resolveDataset(rng)
        context = resolved.drawContext('bernoulli', blocks, rng)
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
        Block(name='fixed', variances=[0.0, 0.5, 0.5]),
        Block(name='g0', variances=[1.0, 1.0, 1.0, 1.0], groups=2),
    ]
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(DRAWS):
        context = glmm.drawContext('bernoulli', blocks, rng)
        parameters = context.drawParameters(rng)
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
        Block(name='fixed', variances=[0.0, 0.5, 0.5]),
        Block(name='g0', variances=[1.0] * width, groups=5),
    ]
    context = resolved.drawContext('poisson', blocks, np.random.default_rng(1))
    parameters = context.drawParameters(np.random.default_rng(1))
    assert context.lkj_eta == {'g0': None}
    assert np.array_equal(parameters.corr_alpha['g0'], np.eye(width))


@pytest.mark.parametrize(
    ('family', 'sigma_eps_family'),
    [('normal', 'exponential'), ('student_t', 'exponential'), ('poisson', None), ('gamma', None)],
)
def test_the_context_names_the_blocks_fixed_then_the_factors_in_order_and_the_chosen_families(
    family: str, sigma_eps_family: str | None
) -> None:
    """metabeta reads the context by block name and conditions on the families chosen.
    The random-effect sds and the residual sd each have their own family, as brms states them,
    and a family without a residual has no residual family.
    Each factor's parameters are sized by its block: q sds, a q × q correlation, a row per group.
    """
    glmm = Glmm(
        sigma_alpha_family_shares=point('sigma_alpha_family_shares', 'half_cauchy', Glmm),
        sigma_eps_family_shares=point('sigma_eps_family_shares', 'exponential', Glm),
    )
    resolved = glmm.resolveDataset(np.random.default_rng(0))
    context = resolved.drawContext(family, FACTOR_BLOCKS, np.random.default_rng(0))
    parameters = context.drawParameters(np.random.default_rng(0))
    assert list(context.split) == ['fixed', 'g0', 'g1']
    assert list(context.sigma_alpha_scales) == list(context.lkj_eta) == ['g0', 'g1']
    assert list(parameters.alpha) == ['g0', 'g1']
    assert (context.sigma_alpha_family, context.sigma_eps_family) == (
        'half_cauchy',
        sigma_eps_family,
    )
    for block in FACTOR_BLOCKS[1:]:
        q = block.width
        assert parameters.sigma_alpha[block.name].shape == (q,)
        assert parameters.corr_alpha[block.name].shape == (q, q)
        assert parameters.alpha[block.name].shape == (block.groups, q)


def test_the_draw_is_deterministic_under_a_fixed_generator_and_the_context_draws_alone() -> None:
    """A dataset is a pure function of its key, its parameters included.
    The context is the stated prior and keeps the blocks it was set over,
    so it draws its parameters without the config, and a truth is resampled under a fixed prior.
    """
    resolved = Glmm(correlated_share=1.0).resolveDataset(np.random.default_rng(0))
    first_context = resolved.drawContext('normal', FACTOR_BLOCKS, np.random.default_rng(7))
    second_context = resolved.drawContext('normal', FACTOR_BLOCKS, np.random.default_rng(7))
    assert first_context == second_context
    assert first_context.blocks == FACTOR_BLOCKS
    first = first_context.drawParameters(np.random.default_rng(8))
    second = first_context.drawParameters(np.random.default_rng(8))
    assert first.beta.tobytes() == second.beta.tobytes()
    assert first.sigma_eps == second.sigma_eps
    for name in ['g0', 'g1']:
        assert first.sigma_alpha[name].tobytes() == second.sigma_alpha[name].tobytes()
        assert first.corr_alpha[name].tobytes() == second.corr_alpha[name].tobytes()
        assert first.alpha[name].tobytes() == second.alpha[name].tobytes()


# the support of each contaminated family's default shape field
SHAPE_SUPPORTS = [
    pytest.param('student_t', 3, 10, id='ν'),
    pytest.param('negative_binomial', 1, 10, id='φ'),
    pytest.param('latent_bernoulli', 0.5, 2, id='σ_c'),
    pytest.param('gamma', 1, 10, id='the Gamma shape'),
    pytest.param('beta', 2, 50, id='the beta precision'),
]


@pytest.mark.parametrize(('family', 'low', 'high'), SHAPE_SUPPORTS)
def test_the_context_records_the_chosen_familys_shape_from_its_own_field(
    family: str, low: float, high: float
) -> None:
    """A fitter of the family conditions on its second parameter, so the context holds it,
    drawn from the field of the chosen family and from no other: a point is taken as given.
    """
    blocks = [Block(name='fixed', variances=[0.0, 1.0])]
    resolved = Glm().resolveDataset(np.random.default_rng(0))
    shapes = []
    for seed in range(20):
        context = resolved.drawContext(family, blocks, np.random.default_rng(seed))
        shapes.append(context.family_shape)
    assert low <= min(shapes) < max(shapes) <= high
    field = CONTAMINATED[family]
    pointed = Glm(**{field: 4.0}).resolveDataset(np.random.default_rng(0))
    assert pointed.drawContext(family, blocks, np.random.default_rng(0)).family_shape == 4.0
    others = {other: 4.0 for other in CONTAMINATED.values() if other != field}
    free = Glm(**others).resolveDataset(np.random.default_rng(0))
    assert low <= free.drawContext(family, blocks, np.random.default_rng(0)).family_shape <= high


@pytest.mark.parametrize('family', ['normal', 'bernoulli', 'poisson'])
def test_a_base_family_has_no_shape_and_consumes_no_draw_for_one(family: str) -> None:
    """The shape fields stay distributions in a base family's draw, so a dataset of a base
    family draws what it drew before the contaminated families existed.
    """
    blocks = [Block(name='fixed', variances=[0.0, 1.0])]
    free = Glm().resolveDataset(np.random.default_rng(0))
    fixed = Glm(**dict.fromkeys(CONTAMINATED.values(), 4.0)).resolveDataset(
        np.random.default_rng(0)
    )
    free_rng = np.random.default_rng(1)
    fixed_rng = np.random.default_rng(1)
    free_context = free.drawContext(family, blocks, free_rng)
    fixed_context = fixed.drawContext(family, blocks, fixed_rng)
    assert free_context.family_shape is None
    assert free_context == fixed_context
    assert free_rng.random() == fixed_rng.random()


@pytest.mark.parametrize('family', ['normal', 'bernoulli', 'poisson', *CONTAMINATED])
def test_the_parameters_record_the_contexts_shape_and_a_residual_only_where_the_family_has_one(
    family: str,
) -> None:
    """The shape is the truth's second parameter beside β, fixed when a truth is resampled
    under the context; only the Normal and Student-t families have a σ_ε.
    """
    blocks = [Block(name='fixed', variances=[0.0, 1.0])]
    resolved = Glm().resolveDataset(np.random.default_rng(0))
    context = resolved.drawContext(family, blocks, np.random.default_rng(0))
    parameters = context.drawParameters(np.random.default_rng(1))
    assert parameters.family_shape == context.family_shape
    assert (parameters.sigma_eps is not None) == (family in ['normal', 'student_t'])
