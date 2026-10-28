"""The prior module's seams at its three levels: a Glm or Glmm built with overrides
and resolved to points, the prior it draws over named blocks, and the parameters that prior draws.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from tabdgp.field import Beta, Dirichlet, Distribution, LogUniform, Uniform, UniformInt
from tabdgp.prior import Block, Glm, Glmm, Prior
from tests.fixtures.build import point

SD_FAMILY_SHARES = {
    'half_normal': 0.25,
    'half_student_t': 0.25,
    'half_cauchy': 0.25,
    'exponential': 0.25,
}
# the contaminated families, each with its shape's default (ADR 0006)
SHAPES = {
    'student_t': LogUniform(3, 10),  # df
    'negative_binomial': LogUniform(1, 10),  # φ
    'latent_bernoulli': LogUniform(0.5, 2),  # σ_c
    'gamma': LogUniform(1, 10),  # k
    'beta': LogUniform(2, 50),  # φ, the precision
}
DEFAULTS = {
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
}


def test_a_block_is_as_wide_as_its_variances_and_has_at_least_one() -> None:
    """A block names what each coefficient adds to var(η) at scale 1, the intercept first,
    so its width is their count and a block without a coefficient cannot be written.
    """
    assert Block(name='fixed', variances=[0.0, 1.0, 0.25]).width == 3
    with pytest.raises(ValidationError, match='variances'):
        Block(name='fixed', variances=[])


@pytest.mark.parametrize('config', [Glm, Glmm])
def test_a_model_states_ys_family_and_its_prior_holds_it(config: type[Glm]) -> None:
    """A model states y's family, as a fitter does, so one without a family cannot be written."""
    with pytest.raises(ValidationError, match='family'):
        config()
    blocks = [Block(name='fixed', variances=[0.0, 1.0]), Block(name='g0', variances=[1.0])]
    resolved = config(family='poisson').resolveDataset(np.random.default_rng(0))
    assert resolved.drawPrior(blocks, np.random.default_rng(0)).family == 'poisson'


def test_the_fields_take_their_defaults_and_an_override() -> None:
    """ADR 0006's envelope as its amendments shaped it: what a GLM corpus draws by default,
    the fixed effects' families and locations what brms users state.
    An override names a shares field's categories in any order and keeps the default's.
    """
    glm = Glm(family='gaussian')
    defaults = {field: getattr(glm, field) for field in Glm.model_fields}
    assert defaults == {'family': 'gaussian', 'shape': None, **DEFAULTS}
    cauchy = {'cauchy': 1.0, 'normal': 0.0, 'student_t': 0.0}
    glm = Glm(family='gaussian', beta_family_shares=cauchy, signal_share=0.5)
    assert list(glm.beta_family_shares) == ['normal', 'student_t', 'cauchy']
    assert (glm.beta_family_shares['cauchy'], glm.signal_share) == (1.0, 0.5)


@pytest.mark.parametrize('family', SHAPES)
def test_a_contaminated_family_takes_its_own_default_shape_and_an_override(family: str) -> None:
    """y's second parameter is one field, so a model states the family alone
    and gets that family's default, which a point or a distribution replaces.
    """
    assert Glm(family=family).shape == SHAPES[family]
    assert Glm(family=family, shape=4.0).shape == 4.0
    assert Glmm(family=family, shape=Uniform(5, 6)).shape == Uniform(5, 6)


MISSHAPEN = [
    pytest.param('gaussian', 4.0, id='a shape on a family without one, which it would ignore'),
    pytest.param('student_t', None, id='no shape on a family that draws with one'),
    pytest.param('gamma', 0.0, id='a shape of 0, where numpy fails'),
    pytest.param('latent_bernoulli', 0.0, id='a latent scale of 0, the bernoulli family stated'),
    pytest.param('beta', Uniform(-1, 2), id='a distribution reaching below 0'),
]


@pytest.mark.parametrize(('family', 'shape'), MISSHAPEN)
def test_a_shape_exists_exactly_where_the_family_has_one_and_is_positive(
    family: str, shape: float | Uniform | None
) -> None:
    """A model that states a shape its family ignores, or none for one that needs it,
    fails where it is written, not at a draw.
    """
    with pytest.raises(ValidationError, match='shape'):
        Glm(family=family, shape=shape)


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
    glmm = Glmm(family='gaussian')
    defaults = {field: getattr(glmm, field) for field in Glmm.model_fields}
    assert defaults == {'family': 'gaussian', 'shape': None, **GLMM_DEFAULTS}


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
        Glmm(family='gaussian', **{field: value})


def test_resolve_gives_every_field_a_point_and_the_shares_fields_shares_that_choose() -> None:
    """A model resolves its own fields inside its call, and the draw reads points only.
    A shares field overridden by a Dirichlet must still choose one category per dataset,
    and y's shape is a point like any other field.
    """
    uniform = Dirichlet({'half_normal': 1, 'half_student_t': 1, 'half_cauchy': 1, 'exponential': 1})
    resolved = Glm(family='student_t', sigma_eps_family_shares=uniform).resolveDataset(
        np.random.default_rng(0)
    )
    for field in Glm.model_fields:
        assert not isinstance(getattr(resolved, field), Distribution)
    for field in Glm.SHARES:
        shares = getattr(resolved, field)
        assert shares.choose(np.random.default_rng(0)) in Glm.categories(field)


def test_resolve_leaves_the_fields_each_factor_draws_as_given() -> None:
    """Each factor draws its own budget q_k, truncated at d, and its own η when correlated.
    A dataset's resolution keeps these distributions and gives every other field a point.
    """
    resolved = Glmm(family='gaussian').resolveDataset(np.random.default_rng(0))
    assert (resolved.random_effects, resolved.lkj_eta) == (UniformInt(1, 6), UniformInt(1, 4))
    for field in Glmm.model_fields:
        if field not in ['random_effects', 'lkj_eta']:
            assert not isinstance(getattr(resolved, field), Distribution)


# r = 0.36 gives √r = 0.6 and √(1 − r) = 0.8, and tr cov X = 3 × 0.75 = 2.25 gives τ_β = τ / 1.5
THREE_QUARTERS = [0.0, 0.75, 0.75, 0.75]
WITHOUT_RESIDUAL = [
    'bernoulli',
    'poisson',
    'negative_binomial',
    'latent_bernoulli',
    'gamma',
    'beta',
]
IDENTITIES = [
    pytest.param(
        'gaussian',
        THREE_QUARTERS,
        0.36,
        0.6,
        0.4,
        0.8,
        id='gaussian: τ² = r, τ_β = τ / 1.5, τ_ε = √(1 − r)',
    ),
    pytest.param(
        'student_t',
        THREE_QUARTERS,
        0.36,
        0.6,
        0.4,
        0.8,
        id='student_t: drawn like the Gaussian, τ² = r, τ_ε = √(1 − r)',
    ),
    *[
        pytest.param(
            family, THREE_QUARTERS, None, 2.0, 4 / 3, None, id=f'{family}: τ as drawn, no r or τ_ε'
        )
        for family in WITHOUT_RESIDUAL
    ],
    pytest.param('poisson', [0.0], None, 2.0, 0.0, None, id='no covariate column: τ_β = 0'),
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
    """ADR 0006's scales as the comment on #46 amended them: τ_β = τ √(φ_fixed / tr cov X),
    with φ_fixed = 1 for a GLM, so the fixed block realizes var(Xβ) = τ² on any coding.
    Only the Gaussian and Student-t families have a residual scale, and the Gaussian's y then has
    unit variance at the scales. r sets nothing under the other families, so the prior
    records none there.
    An intercept-only design has no slope for τ_β to scale.
    In the block case and without a located slope, every slope has the block's scale and
    location 0;
    the intercept's location is the field's, on the link scale.
    A GLM has no random effects, so no family or scale for them.
    """
    glm = Glm(
        family=family,
        signal_share=0.36,
        predictor_sd=2.0,
        intercept_location=-2.0,
        intercept_scale=1.5,
        location_share=0.0,
        varying_scale_share=0.0,
    )
    blocks = [Block(name='fixed', variances=variances)]
    resolved = glm.resolveDataset(np.random.default_rng(0))
    prior = resolved.drawPrior(blocks, np.random.default_rng(0))
    slopes = len(variances) - 1
    assert prior.signal_share == signal_share
    assert (prior.sigma_alpha_family, prior.sigma_alpha_scales) == (None, {})
    assert prior.predictor_sd == pytest.approx(predictor_sd)
    assert prior.split == {'fixed': 1.0}
    assert prior.beta_scales == pytest.approx([1.5] + [scale] * slopes)
    assert prior.beta_locations == [-2.0] + [0.0] * slopes
    assert prior.sigma_eps_scale == pytest.approx(sigma_eps_scale)


def test_a_glmms_fixed_block_takes_its_share_and_one_factor_the_rest() -> None:
    """The comment on #46: τ_β = τ √(φ_fixed / tr cov X) and τ_α,k = τ √(φ_k / tr E[z_k z_k']).
    One factor takes all of 1 − φ_fixed, so every scale is known:
    τ = √0.36 = 0.6, τ_β = 0.6 · √(0.64 / 2.25) = 0.32 and τ_α = 0.6 · √(0.36 / 4) = 0.18.
    """
    glmm = Glmm(
        family='gaussian',
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
    prior = glmm.drawPrior(blocks, np.random.default_rng(0))
    assert prior.split == {'fixed': 0.64, 'g0': pytest.approx(0.36)}
    assert prior.beta_scales[1:] == pytest.approx([0.32] * 3)
    assert prior.sigma_alpha_scales == {'g0': pytest.approx([0.18] * 2)}


def _spent(prior: Prior, block: Block) -> float:
    """What a block's term adds to var(η) in expectation at the prior's scales and locations:
    E[β_j²] var_j = (τ_j² + ν_j²) var_j per slope, τ_α,j² E[z_j²] per random effect.
    """
    if block.name == 'fixed':
        slopes = zip(prior.beta_locations[1:], prior.beta_scales[1:], block.variances[1:])
        return sum((scale**2 + location**2) * variance for location, scale, variance in slopes)
    effects = zip(prior.sigma_alpha_scales[block.name], block.variances)
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
    glmm = Glmm(family='poisson', predictor_sd=2.0)
    blocks = [
        fixed,
        Block(name='g0', variances=[1.0, 0.5], groups=3),
        Block(name='g1', variances=[1.0], groups=20),
    ]
    resolved = glmm.resolveDataset(np.random.default_rng(0))
    prior = resolved.drawPrior(blocks, np.random.default_rng(1))
    assert list(prior.split) == ['fixed', 'g0', 'g1']
    assert sum(prior.split.values()) == pytest.approx(1)
    for block in blocks:
        assert _spent(prior, block) == pytest.approx(4.0 * prior.split[block.name])


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
        family='poisson',
        varying_scale_share=varying_scale_share,
        scale_spread=1.0,
        predictor_sd=2.0,
        intercept_location=0.0,
        intercept_scale=1.0,
        location_share=0.0,
        fixed_share=0.5,
    )
    prior = glmm.drawPrior(FACTOR_BLOCKS, np.random.default_rng(0))
    scales = [prior.beta_scales[1:], *prior.sigma_alpha_scales.values()]
    assert [len(set(block_scales)) for block_scales in scales] == distinct


def test_located_and_varying_scales_spend_each_blocks_budget_exactly() -> None:
    """ADR 0006's amendments: τ_j = s_β w_j and ν_j = δ_j τ_j over the slopes,
    τ_k,j = s_k w_k,j over a factor's effects, each block's scale solved from its budget,
    s_β² Σ_j w_j² (1 + δ_j²) var_j = τ² φ_fixed and s_k² Σ_j w_k,j² E[z_j²] = τ² φ_k.
    So a location or a weight moves the scales around but never what a block adds to var(η)
    in expectation, and the constant column costs nothing.
    """
    glmm = Glmm(
        family='poisson',
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
        prior = glmm.drawPrior(blocks, np.random.default_rng(seed))
        assert all(location != 0 for location in prior.beta_locations[1:])
        for block in blocks:
            assert _spent(prior, block) == pytest.approx(4.0 * prior.split[block.name])


def _halfNormalCdf(x: float) -> float:
    return math.erf(x / math.sqrt(2))


def _halfStudentT3Cdf(x: float) -> float:
    # twice Student's t CDF at df = 3 minus 1, in closed form
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
    """Given the prior, each parameter is drawn from its stated family exactly (ADR 0006),
    nothing clipped or rescaled, so metabeta's posterior conditions on the true prior.
    σ_ε / τ_ε and each σ_α / τ_α follow F's standard law, the intercept N(0, τ_0),
    each slope N(0, τ_β) and each random effect N(0, σ_α) given its σ_α.
    The KS bound is the 1 % critical value at the number of draws.
    """
    glmm = Glmm(
        family='gaussian',
        beta_family_shares=point('beta_family_shares', 'normal', Glm),
        sigma_eps_family_shares=point('sigma_eps_family_shares', sd_family, Glm),
        sigma_alpha_family_shares=point('sigma_alpha_family_shares', sd_family, Glmm),
        signal_share=0.36,  # τ = 0.6, τ_ε = 0.8
        intercept_location=0.0,
        intercept_scale=1.5,
        location_share=0.0,
        fixed_share=0.5,
        correlated_share=0.0,
        varying_scale_share=0.0,
    )
    # each block's share 0.5 over its variance 0.18 gives τ_β = τ_α = 0.6 · √(0.5 / 0.18) = 1
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
        parameters = glmm.drawPrior(blocks, rng).drawParameters(rng)
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


def test_a_slope_is_located_at_location_share_and_its_delta_is_normal_at_location_scale() -> None:
    """The survey's stated hypotheses are rare and sit a few scales out, so by default
    each slope has a location with probability 0.1, and then ν_j / τ_j ~ N(0, 2).
    """
    fixed = Block(name='fixed', variances=[0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    rng = np.random.default_rng(1)  # seed 0's first 10,000 slopes land 3.8 sd above 0.1
    relative = []
    for _ in range(DRAWS):
        resolved = Glm(family='poisson').resolveDataset(rng)
        prior = resolved.drawPrior([fixed], rng)
        for location, scale in zip(prior.beta_locations[1:], prior.beta_scales[1:]):
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
    """The intercept and each slope are drawn from the β family the prior names,
    at the location and scale it records for them, so (β_j − ν_j) / τ_j follows the family's
    standard law, Student-t at df = 3, whatever the locations and the spread.
    """
    glm = Glm(
        family='bernoulli',
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
        prior = glm.drawPrior([fixed], rng)
        parameters = prior.drawParameters(rng)
        assert prior.beta_family == beta_family
        intercepts.append((parameters.beta[0] - 1.0) / 0.5)
        locations = np.array(prior.beta_locations[1:])  # (d − 1,)
        scales = np.array(prior.beta_scales[1:])
        standardized = (parameters.beta[1:] - locations) / scales
        slopes.extend(standardized)
    assert _ksDistance(intercepts, cdf) < 1.63 / math.sqrt(DRAWS)
    assert _ksDistance(slopes, cdf) < 1.63 / math.sqrt(len(slopes))


def _drawPriorAtBetaFamily(beta_family: str) -> Prior:
    """A GLM's prior at one β family, its slopes located and their scales varying."""
    glm = Glm(
        family='poisson',
        beta_family_shares=point('beta_family_shares', beta_family, Glm),
        predictor_sd=2.0,
        location_share=1.0,
        varying_scale_share=1.0,
    )
    resolved = glm.resolveDataset(np.random.default_rng(0))
    blocks = [Block(name='fixed', variances=[0.0, 0.5, 1.0, 2.0])]
    return resolved.drawPrior(blocks, np.random.default_rng(1))


@pytest.mark.parametrize('beta_family', ['student_t', 'cauchy'])
def test_a_heavier_tailed_beta_family_keeps_the_normals_locations_and_scales(
    beta_family: str,
) -> None:
    """τ_j stays the β family's scale parameter, so a heavier tail overshoots the budget
    (Student-t triples the variance, Cauchy has none) instead of shrinking the scales to meet it.
    """
    normal = _drawPriorAtBetaFamily('normal')
    heavier = _drawPriorAtBetaFamily(beta_family)
    assert heavier.beta_family == beta_family
    assert heavier.beta_locations == normal.beta_locations
    assert heavier.beta_scales == normal.beta_scales


def test_the_log_ratio_of_two_scales_in_a_block_is_normal_at_root_2_scale_spread() -> None:
    """In a block that varies, each coefficient's log weight is N(0, scale_spread),
    so the log-ratio of two of its scales is N(0, √2 scale_spread), whatever the block's scale.
    A random intercept's scale is weighted like a slope's.
    """
    glmm = Glmm(
        family='poisson',
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
        prior = glmm.drawPrior(FACTOR_BLOCKS, rng)
        first, second = prior.beta_scales[1:3]
        slopes.append(math.log(first / second))
        intercept, slope = prior.sigma_alpha_scales['g0'][:2]
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
    rng = np.random.default_rng(1)  # seed 0's first factor varies 3.04 sd above 0.5 by chance
    fixed = []
    first = []
    second = []
    for _ in range(DRAWS):
        resolved = Glmm(family='poisson').resolveDataset(rng)
        prior = resolved.drawPrior(FACTOR_BLOCKS, rng)
        fixed.append(_varies(prior.beta_scales[1:]))
        first.append(_varies(prior.sigma_alpha_scales['g0']))
        second.append(_varies(prior.sigma_alpha_scales['g1']))
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
        resolved = Glmm(family='bernoulli').resolveDataset(rng)
        prior = resolved.drawPrior(blocks, rng)
        fixed.append(prior.split['fixed'])
        first.append(prior.split['g0'] / (1 - prior.split['fixed']))
    assert _ksDistance(fixed, _uniformCdf) < 1.63 / math.sqrt(DRAWS)
    assert _ksDistance(first, _uniformCdf) < 1.63 / math.sqrt(DRAWS)


def test_an_lkj_draw_is_a_correlation_matrix_at_its_analytic_marginal_variance() -> None:
    """Ω comes from the C-vine of Lewandowski, Kurowicka and Joe (2009, section 3.2) in numpy,
    pinned by its analytic marginal: each off-diagonal entry has variance 1 / (2η + q − 1),
    1/7 at q = 4 and η = 2, and mean 0.
    Every entry is checked, since all but the first row pass through the vine's recursion.
    """
    glmm = Glmm(family='bernoulli', correlated_share=1.0, lkj_eta=2).resolveDataset(
        np.random.default_rng(0)
    )
    blocks = [
        Block(name='fixed', variances=[0.0, 0.5, 0.5]),
        Block(name='g0', variances=[1.0, 1.0, 1.0, 1.0], groups=2),
    ]
    rng = np.random.default_rng(1)
    draws = []
    for _ in range(DRAWS):
        prior = glmm.drawPrior(blocks, rng)
        parameters = prior.drawParameters(rng)
        corr = parameters.corr_alpha['g0']  # (q, q)
        assert np.array_equal(corr, corr.T)
        assert np.diag(corr).tolist() == [1.0] * 4
        assert np.linalg.eigvalsh(corr).min() > 0
        draws.append(corr)
    assert prior.lkj_eta == {'g0': 2}
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
    and always at q_k = 1 (ADR 0006): Ω is the identity and the prior records no η.
    """
    glmm = Glmm(family='poisson', correlated_share=correlated_share, lkj_eta=2)
    resolved = glmm.resolveDataset(np.random.default_rng(0))
    blocks = [
        Block(name='fixed', variances=[0.0, 0.5, 0.5]),
        Block(name='g0', variances=[1.0] * width, groups=5),
    ]
    prior = resolved.drawPrior(blocks, np.random.default_rng(1))
    parameters = prior.drawParameters(np.random.default_rng(1))
    assert prior.lkj_eta == {'g0': None}
    assert np.array_equal(parameters.corr_alpha['g0'], np.eye(width))


@pytest.mark.parametrize(
    ('family', 'sigma_eps_family'),
    [('gaussian', 'exponential'), ('student_t', 'exponential'), ('poisson', None), ('gamma', None)],
)
def test_the_prior_names_the_blocks_fixed_then_the_factors_in_order_and_the_chosen_families(
    family: str, sigma_eps_family: str | None
) -> None:
    """metabeta reads the prior by block name and conditions on the families chosen.
    The random-effect sds and the residual sd each have their own family, as brms states them,
    and a family without a residual has no residual family.
    Each factor's parameters are sized by its block: q sds, a q × q correlation, a row per group.
    """
    glmm = Glmm(
        family=family,
        sigma_alpha_family_shares=point('sigma_alpha_family_shares', 'half_cauchy', Glmm),
        sigma_eps_family_shares=point('sigma_eps_family_shares', 'exponential', Glm),
    )
    resolved = glmm.resolveDataset(np.random.default_rng(0))
    prior = resolved.drawPrior(FACTOR_BLOCKS, np.random.default_rng(0))
    parameters = prior.drawParameters(np.random.default_rng(0))
    assert list(prior.split) == ['fixed', 'g0', 'g1']
    assert list(prior.sigma_alpha_scales) == list(prior.lkj_eta) == ['g0', 'g1']
    assert list(parameters.alpha) == ['g0', 'g1']
    assert (prior.sigma_alpha_family, prior.sigma_eps_family) == (
        'half_cauchy',
        sigma_eps_family,
    )
    for block in FACTOR_BLOCKS[1:]:
        q = block.width
        assert parameters.sigma_alpha[block.name].shape == (q,)
        assert parameters.corr_alpha[block.name].shape == (q, q)
        assert parameters.alpha[block.name].shape == (block.groups, q)


def test_the_draw_is_deterministic_under_a_fixed_generator_and_the_prior_draws_alone() -> None:
    """A dataset is a pure function of its key, its parameters included.
    The prior keeps the blocks it was set over,
    so it draws its parameters without the config,
    and a ground truth is resampled under a fixed prior.
    """
    resolved = Glmm(family='gaussian', correlated_share=1.0).resolveDataset(
        np.random.default_rng(0)
    )
    first_prior = resolved.drawPrior(FACTOR_BLOCKS, np.random.default_rng(7))
    second_prior = resolved.drawPrior(FACTOR_BLOCKS, np.random.default_rng(7))
    assert first_prior == second_prior
    assert first_prior.blocks == FACTOR_BLOCKS
    first = first_prior.drawParameters(np.random.default_rng(8))
    second = first_prior.drawParameters(np.random.default_rng(8))
    assert first.beta.tobytes() == second.beta.tobytes()
    assert first.sigma_eps == second.sigma_eps
    for name in ['g0', 'g1']:
        assert first.sigma_alpha[name].tobytes() == second.sigma_alpha[name].tobytes()
        assert first.corr_alpha[name].tobytes() == second.corr_alpha[name].tobytes()
        assert first.alpha[name].tobytes() == second.alpha[name].tobytes()


@pytest.mark.parametrize('family', SHAPES)
def test_the_prior_holds_the_shape_each_dataset_resolved(family: str) -> None:
    """A fitter of the family states its second parameter as known, so the prior holds it:
    a point inside the default's support, drawn once per dataset, or the override's point.
    """
    support = SHAPES[family]
    blocks = [Block(name='fixed', variances=[0.0, 1.0])]
    shapes = []
    for seed in range(20):
        resolved = Glm(family=family).resolveDataset(np.random.default_rng(seed))
        shapes.append(resolved.drawPrior(blocks, np.random.default_rng(0)).shape)
    assert support.low <= min(shapes) < max(shapes) <= support.high
    pointed = Glm(family=family, shape=4.0).resolveDataset(np.random.default_rng(0))
    assert pointed.drawPrior(blocks, np.random.default_rng(0)).shape == 4.0


@pytest.mark.parametrize('family', ['gaussian', 'bernoulli', 'poisson', *SHAPES])
def test_the_parameters_hold_a_residual_only_where_the_family_has_one(family: str) -> None:
    """Only the Gaussian and Student-t families have a σ_ε."""
    blocks = [Block(name='fixed', variances=[0.0, 1.0])]
    resolved = Glm(family=family).resolveDataset(np.random.default_rng(0))
    prior = resolved.drawPrior(blocks, np.random.default_rng(0))
    parameters = prior.drawParameters(np.random.default_rng(1))
    assert (parameters.sigma_eps is not None) == (family in ['gaussian', 'student_t'])
