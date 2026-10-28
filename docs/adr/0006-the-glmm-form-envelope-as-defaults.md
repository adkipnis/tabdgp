---
status: accepted
date: 2026-10-22
amended: 2026-10-23, #41: the form's field is outcome_form; 2026-10-23, ADR 0007: the form is a model, `Glm` and `Glmm` configs of their own in `prior` reached as `outcome_form='glm'` and `'glmm'`, no `Hyperprior.glmm()`; the tokens are shares fields; the container is per model, the scm form's included; the deviation is the score every form gets; 2026-10-24, #47: the fixed block's share of var(η) from its own Beta and the rest split over the factors; a block's budget over its design's second moment, not its width; 2026-10-25, #51: the raw-scale Σ_k is A Σ_k A′ by the coding's map, not Σ_k scaled by the diagonal; 2026-10-25, #55: a Poisson count above numpy's limit saturates at int64's largest, s_ε is F's scale, and y has unit variance at the scales; 2026-10-25, the prior grill: the prior follows what brms users state (β's families, a located prior per slope, the intercept's location, per-slope scales as an override, σ_α's and σ_ε's families apart, the context draws its own parameters), and the contaminated families replace the residual seam; 2026-10-26, #56: `sd_alpha` is `sigma_alpha`, as σ_ε is `sigma_eps`, and the context names each hyperparameter by its parameter, a scale per random effect; 2026-10-26, #56: the scales vary per coefficient by default, β's and σ_α's alike; 2026-10-26, #56: `shape` is `family_shape`, as an unqualified family is y's; 2026-10-26, #56: the prior context is the `Prior`, drawn by `Glm.drawPrior`; 2026-10-26, #56: the prior's locations are ν and its scales τ (ν_β and τ_β, τ_α, τ_ε), τ unsubscripted the predictor sd they split, and the Student-t's degrees of freedom df; 2026-10-27, #65: y's family is `y_family`, as β's is `beta_family`, its shares `y_family_shares` and its shape `y_shape`, and `Glm.drawPrior(blocks, rng)` draws it first, the `Prior` holding it; 2026-10-27, #66: `drawPrior` draws each parameter's family with its locations and scales, β's after the split, the `Prior` holds y's shape and `Parameters` only what a fitter estimates; 2026-10-27, #56: `correlated_share` is the practice survey's share of models with slopes that correlate them, not of models with slopes; 2026-10-27, #56: y's families and links are `family`'s and the covariates `covariates`', modules of their own that `outcome` imports; 2026-10-27, #56: `family` is `likelihood`, the distribution y is drawn from, since family alone does not say whose; 2026-10-27, #56: ground truth, not truth; 2026-10-28, #68: y's family is stated with no default, never drawn, `family` on `Glm` and the `Prior`, `y_family_shares` gone, and one `shape` field, its default per family in `likelihood`; 2026-10-28, #68: the family and shape on a GLM's outcome beside its link, not on the `Prior`, and the shape None for its family's default, which each dataset draws
---

# 0006 - The glmm form: metabeta's prior envelope as the glmm form's defaults, the fit drawn jointly, no caps, the estimands returned

The outcome node's second form (ADR 0005) is a generalized linear mixed model
(GLMM) on the design matrix, the datasets `metabeta.mixed` trains on and, with a deviation,
the white-box misspecified ones. Grilled on 2026-10-22 with ADR 0005, the decision is: **the
glmm form's prior fields default to metabeta's prior envelope, as ADR 0004 made its size rules
the grouped preset's defaults (reached as `outcome_form='glmm'` on `grouped()`, ADR 0007); the
parameters are drawn from the stated prior families exactly, with the scales of a dataset's
blocks drawn jointly so that the fit is spread instead of capped; nothing is clipped,
rerolled or rescaled after the draw; the dataset returns the estimands on both scales and
the prior they were drawn with.** The words are `CONTEXT.md`'s: form, deviation, prior,
random effect, design matrix.

The witnesses: under metabeta's envelope drawn independently per block, 61 % of Normal
datasets have R² > 0.9, 60 % of logit datasets have some |η| > 10 and 16 % of log-link
datasets overflow numpy's Poisson sampler (20,000 synthetic draws, 2026-10-22); under the
defaults (2.5 everywhere) 61 %, 94 % and 6 %. metabeta-v2 answered this with caps on the
linear predictor's scale and an R² cap, each rescaling the parameters after the draw, so the
recorded prior was not the one drawn from. MixedBench's 694 documented formulas hold 881
random-effect blocks: q = 1 in 88 %, q = 2 in 10 %, q ≥ 3 in 2 % (max 7); `||` is 8 of 84
blocks at q = 2 and 3 of 18 above.

## Decisions

- **The form's defaults.** `outcome_form = 'glmm'` on `grouped()` is the form with `Glmm()`'s
  defaults, `deviation_scale = 0` among them (ADR 0007 replaced `Hyperprior.glmm()`). metabeta
  calls `draw(seed, index, Hyperprior.grouped(outcome_form='glmm'))` and passes nothing else;
  a one-family corpus is `outcome_form=Glmm(y_family_shares={'gaussian': 1})`, a study of that
  family (until 2026-10-28: the family is stated and the model called after the draw).
  The white-box misspecified study overrides `deviation_scale`.
- **The prior fields, the envelope.** `y_family_shares` over {gaussian, bernoulli, poisson},
  uniform, with canonical links (until 2026-10-28: the family is stated); the families of
  metabeta's v1 (binomial, negative binomial, Gamma, ordinal) are later entries.
  One `sd_family_shares` field over {half_normal,
  half_student_t with df = 3, half_cauchy, exponential}, uniform, chosen once per dataset and
  shared by the random-effect standard deviations and the residual scale (a shares field, not
  a token, ADR 0007). `lkj_eta` uniform on {1, 2, 3, 4} when a factor's effects
  are correlated, `correlated_share` 0.55 per factor with q_k ≥ 2 (the practice survey's share
  of models with slopes that correlate them, those that do not say counted as uncorrelated).
  The intercept's scale τ_0 `LogUniform(0.1, 10)`, the envelope.
- **The fit drawn jointly.** Per dataset: `signal_share` r, a wide Beta; `predictor_sd` τ for
  the unit-free families, log-uniform on about [0.25, 4] on the logit and log scales, and
  τ² = r for the Normal family, whose y then has unit variance at the scales (F's draw gives
  E[σ_ε²] = τ_ε² E[X²], X F's standard law); the fixed
  block's share φ_fixed of var(η) from `fixed_share`, a Beta, and the rest split over the
  factors' random blocks by a Dirichlet at `split_alpha` (#47: the registry's fixed share of
  the explained variance is close to uniform whatever the number of factors, where one
  Dirichlet over 1 + f blocks gave the fixed block 1 / (1 + f)). The scales follow, each
  block's budget over its design's second moment rather than its width: τ_β = τ √(φ_fixed /
  tr cov X), τ_α,k = τ √(φ_k / tr E[z_k z_k']), τ_ε = √(1 − r) as F's scale (#47:
  dummies and 0/1 columns have variance below 1 and the intercept none, so a block divided by
  its width realized a third of its budget and R²_m piled near 0). Given the scales,
  β ~ N(0, τ_β), σ_α ~ F(τ_α), σ_ε ~ F(τ_ε) exactly, so metabeta's conditioning on its context
  stays honest; only the hyperprior over the contexts is no longer independent per block. A
  finding on the side: at d = 20 and r = 0.5, τ_β ≈ 0.11, so realistic LMMs sit at the
  envelope's lower edge and its upper decade produces only saturated data.
- **No caps, no rerolls, no rescaling.** A Poisson rate is `exp(η)` in float64, no clip; above
  numpy's limit the count saturates at int64's largest, so a draw never fails. The
  absurd log-link tables at σ = 10 are the shared envelope's cost (mixed ADR 0001's
  amendment), and a per-family edge is metabeta's to reopen. The Normal family's y is returned
  as drawn; metabeta's internal z-scoring (its ADR 0002) applies to y and θ together, as to a
  user's table.
- **Terms counted in columns.** d counts the fixed effects with the intercept,
  `design_columns + 1`; q_k the random effects of factor k with the intercept, a slope on a
  categorical column taking k − 1 of them. The intercept always; then slope covariate columns
  drawn uniformly, time first when present, each adding its width, until a budget uniform on
  {1, …, min(6, d)}; q_k = 1 when no column fits. One edge, independent of correlation: above
  q = 3 the registry has 11 blocks, too few for a conditional edge. Random effects α_k per
  factor are independent, N(0, diag(σ_k) Ω_k diag(σ_k)) per group, crossed and nested alike,
  since a subordinate factor's codes are already global. Ω_k by the C-vine of Lewandowski,
  Kurowicka and Joe (2009, section 3.2) in numpy, pinned by the analytic marginal (variance
  1 / (2η + q − 1) per off-diagonal).
- **The deviation.** η = Xβ + Σ_k Z_k α_k + λ · sd(Xβ + Zα) · z(f), f the ordinary SCM mechanism
  over the outcome's drawn graph parents, observed and latent, λ the field `deviation_scale`.
  The dataset returns f's values and λ, so β and α stay the exact ground truth and a fitted
  GLMM is misspecified by exactly λ · f; a latent group-level parent in f is ADR 0003's
  endogeneity.
  "Endogenous but linear" is a study's override on `mechanism_shares`, not a field.
- **The residual seam.** Each family is one class with `draw(rng, eta, scale)`, `scale` a
  per-row array (σ_ε broadcast to the rows today, per group or per row later), and the Normal
  family's residual law a shares field, `residual_shares`, over {normal} today, {normal,
  student_t} later, so a later residual quirk is one category or one array.
- **The design matrix is coded here**, by `CONTEXT.md`'s definition: the intercept, continuous
  columns z-scored by the population sd over the n rows, binary 0/1, k − 1 dummies dropping the
  first level in the codes' order. MixedBench's contract and metabeta's schema code it too;
  either may import this coder once it exists.
- **What the dataset returns**, on `Dataset.outcome` (`GlmmOutcome`; the scm form returns a
  container of its own since ADR 0007): family, link, the
  fixed and random term names per factor, the prior (τ_β, τ_0, τ_α per factor, τ_ε, F,
  η, r, τ, φ), the parameters on the standardized scale as drawn (`beta` (d,), per factor
  `sigma_alpha` (q_k,), `corr_alpha` (q_k, q_k), `alpha` (m_k, q_k), `sigma_eps`, the deviation
  (n,) and λ), the coding statistics (each continuous column's mean and sd) and the raw-scale
  parameters, analytic from these (β_raw,j = β_j / sd_j, the intercepts absorbing
  Σ β_j · mean_j / sd_j and the random ones each slope's α_j · mean_j / sd_j, so that
  Σ_k becomes A Σ_k A′ for that linear map A, returned as its sds and its correlation, in
  which only the intercept's row and column move): what a user fitting lme4 on the raw
  columns compares against.
- **Modules.** `prior` owns the GLMM's priors: the configs `Glm` and `Glmm` with the fields
  over them (ADR 0007), the joint scale draw, the LKJ sampler and the parameter draw, nothing
  about columns; `outcome` owns the node: its forms, the design matrix, η, the deviation and
  the containers, with the families and links in `family` and the covariates it reads in
  `covariates`. `prior` and `family` are imported by `outcome`, `outcome` by `draw`.

## Considered options

- **The envelope as overrides** (ADR 0001's first wording): metabeta would pass a dozen fields
  on every call, and "the defaults are what metabeta trains on" would stop being true.
- **The R2D2M2 prior as the generative prior** (Aguilar and Bürkner 2023; Yanchenko, Bondell
  and Reich 2024 for GLMMs): R² ~ Beta, a Dirichlet split of the explained variance over the
  terms, every effect derived; exact fit control, published, brms-native. Its context would be
  (μ, φ, a), reversing mixed ADRs 0001 and 0003, which the practice survey grounded in what
  users state, and a user's N(0, 2.5) would be served by the IMH correction alone. Recorded as
  the alternative: spec B's first ticket is a lab note that draws under both, computes the
  realized R² (Nakagawa's marginal and conditional), ICC per factor and sd(η), and compares them
  with the same quantities fitted on the registry's tables by the proofs' own method
  (containment, then a C2ST). The note (#47, 2026-10-24) kept the joint draw and amended its
  split and its divisor: R2D2M2's Beta R² and its Dirichlet over the terms never let one
  random intercept carry most of the variance, so its ICC and R²_c fall short of the
  registry's top, where the joint draw reaches it.
- **A recorded truncation** (reject when max |η| > B, the indicator in the prior so that the
  IMH target stays exact): sound, but its acceptance rate collapses at the envelope's edge.
- **metabeta-v2's caps and rescaling**: rejected above; its `hypersample` (nonzero prior means,
  skewed-beta ranges) predates the survey and nothing of it carries over; `Prior.sample`'s
  structure does.
- **A q edge conditional on correlation** (4 correlated, 8 uncorrelated): a guess with a
  branch the data cannot support. Rule 3's uniform count puts about 17 % on intercept-only
  blocks against the registry's 88 %; coverage-first is metabeta's rule, noted in its
  envelope, not tilted here.
- **torch's `LKJCholesky`** seeded from numpy, metabeta-v2's route: a dependency ADR 0001
  excludes.
- **Raw-scale parameters only**, or standardized only: the standardized ones are what was
  drawn and what metabeta conditions on, the raw ones what a user compares against; ten lines
  return both.

## Consequences

- metabeta's `envelope.md`: rule 8's "q stays this package's" and its "q ≤ 4" give way to this
  form's default (6, in columns); the prior table cites this ADR; rule 9 names the glmm form.
  mixed ADR 0003's consequence "log s log-uniform per block" becomes "jointly, so that the
  fit is spread"; the families, η and the envelope's range stand. Root ADR 0005's "until the
  node exists, `metabeta.mixed` draws its own parameters" closes: the family sampling, the
  prior and the random-term draw come from here, and metabeta keeps the family token
  in its network, its log-densities and its API.
- The IMH correction's prior density reads the dataset's prior; a corpus of one family
  is an override in metabeta's curriculum loop.
- `prior` gains the prior fields as `Glm`'s and `Glmm`'s (ADR 0007); `prior` and `outcome` are
  new modules; `Dataset` gains `outcome`. The proofs stay on `grouped()`; the glmm form's
  covariates are certified by the same proofs, its response never.

## Amendment 2026-10-25: the prior follows what brms users state

#56's read of `prior` paused on the prior's shape: β Normal at 0 with one scale per block,
one sd family shared by σ_α and σ_ε. Grilled on 2026-10-25 against brms's `priors.R`, the
practice survey's `priors.csv` (Tier P, 1,428 models) and metabeta-v2's `hypersample`, the
decision is: **the glmm form's prior takes the shape a brms user states, families, locations
and scales per block or per coefficient, drawn from the same joint budget; the prior
is that stated prior and draws its own parameters; metabeta-v2's contaminated likelihoods
become families of `y_family_shares` at share zero, so the residual seam goes.** The defaults
below are what the fourth fingerprint pins after ticket A; nothing the proofs certify moves.

- **β's family.** `beta_family_shares` over {normal 0.68, student_t 0.30, cauchy 0.02},
  df = 3 for Student-t (18 of the survey's 31 stated df), chosen once per dataset for the
  intercept and the slopes. The scale stays the family's scale parameter, as τ_ε is F's:
  a Student-t slope has variance 3σ² and a Cauchy slope none, so the budget is met in the
  family's unit and the tails spread the fit further, recorded, not corrected. Flat (20 % of
  readable β rows, brms's default) is improper and stays metabeta's clamp.
- **The intercept's location.** `intercept_location`, `Uniform(-3, 3)` on the link scale, a
  guess, drawn before β_0 ~ family(ν_0, τ_0); the intercept adds no variance to η, so the
  budget is untouched. The witness: 411 survey rows carry brms's data-driven default,
  Student-t(3) at median(link y) with scale max(2.5, mad), which a network can only be told
  as a location; a generated table's median on the link scale lands within τ_0 of ν_0 by
  construction, so the default is a sub-case at small τ_0.
- **A located prior per slope.** Each slope's location is a multiple δ_j of its own scale,
  ν_j = δ_j τ_j: zero with probability 1 − `location_share` (0.1), else δ_j ~ N(0,
  `location_scale`) (2.0), a guess; the survey's stated hypotheses sit two to three scales
  out at a tight scale (normal(0.3, 0.1)). Relative locations keep the budget exact,
  E[β_j²] = τ_j² (1 + δ_j²), where an absolute slab (metabeta-v2's Laplace(0, 0.5)) can
  exceed the block's share. The context records ν_j and τ_j in absolute units.
- **A scale per slope, as an override.** τ_j = s_β w_j with log w_j ~ N(0, `scale_spread`)
  and s_β solved from the budget, s_β² Σ_j w_j² (1 + δ_j²) var_j = τ² φ_fixed, so a constant
  design column is harmless and no column's variance divides anything (R2D2M2's Dirichlet
  split, τ_j² ∝ φ_j / var_j, gives a rare dummy a huge scale). The default spread was 0 (until the
  amendment of 2026-10-26), the
  block case metasurv's ADR 0008 trains on; `Glmm(scale_spread=Uniform(0, 2))` is its
  planned ablation, in one override. The default drew one scale per random block (until 2026-10-26), brms's
  one prior for class `sd`; the context states one per random effect, as brms's `coef` can
  (the survey's rehrig_2020 and trouve_2018 give the intercept's sd a prior of its own).
- **σ_α and σ_ε apart.** `sigma_alpha_family_shares` on `Glmm` and `sigma_eps_family_shares`
  on `Glm`, each uniform over the four families, drawn independently: brms sets `sd` and
  `sigma` apart, and 24 of the survey's 132 stated pairs differ. F goes as a symbol.
- **The prior draws its own parameters.** `Glm.drawPrior(blocks, rng)` draws y's family
  first, then each parameter's family with its locations and scales, and returns the
  `Prior`, the stated prior, each hyperparameter named by its parameter:
  `family`, `beta_family`, `beta_locations` and `beta_scales` (d,), the intercept's first,
  `sigma_alpha_family` (None on a GLM) and `sigma_alpha_scales` (q_k,) per factor, `lkj_eta`,
  `sigma_eps_family` and `sigma_eps_scale` (None unless the family has a residual), `shape`,
  `signal_share`, `predictor_sd`, `split` and the
  `blocks` it was set from; `Prior.drawParameters(rng)` returns the `Parameters`
  without a config, so a user resamples ground truths under a fixed prior. `Block` carries
  `variances`, one per coefficient, the intercept's 0 first, its width their count.
- **Contaminated families, not residual laws.** metabeta-v2's likelihood misspecification
  experiment regenerated y under Student-t errors (df 10, 5, 3), a negative binomial (φ 10, 3,
  1) and a latent-noise Bernoulli, y ~ Bern(expit(η + σ_c ε)) (σ_c 0.5, 1, 2), with the
  priors and parameters untouched. Each is a family a fitter can state, so they are entries
  of `y_family_shares` at share 0 (until 2026-10-28: stated, each shape one field),
  `student_t`, `negative_binomial`, `latent_bernoulli`, with
  `gamma` and `beta` beside them (the survey's 1 % each, Gamma in metabeta's v1), each
  drawing its second parameter from a field, `residual_df` `LogUniform(3, 10)`,
  `dispersion` `LogUniform(1, 10)`, `latent_scale` `LogUniform(0.5, 2)`, `gamma_shape`
  `LogUniform(1, 10)`, `beta_precision` `LogUniform(2, 50)`, metabeta-v2's severities as the
  witness, guesses otherwise; the `Prior` holds it as `shape`, which a fitter of that
  family fixes. `student_t` is drawn like
  Normal (τ² = r, τ_ε = √(1 − r); y's variance is df / (df − 2) at the scales); φ, σ_c, the
  Gamma shape and the beta precision are the family's own noise outside η, as σ_ε is. The
  residual seam (`residual_shares` over {normal}, the context's `residual_law`) is dropped:
  a white-box study overrides `y_family_shares` as it overrides `deviation_scale`, and
  metabeta's default corpus does not move. Ordinal and binomial stay later entries, each
  with a column of its own to design; log-normal is a user's transform of a Normal table.

### Considered options

- **Uniform β families**, mixed ADR 0003's token rule: Alex's weights, 0.68 / 0.30 / 0.02,
  keep Cauchy rare on the logit and log scales, where a Cauchy slope at scale 1 exceeds 10
  in 6 % of its draws.
- **β's budget divided by df / (df − 2)**: breaks for Cauchy, and would set β's convention
  apart from the sd families'.
- **The intercept's location relative to τ_0**: brms sets the two apart (median vs
  max(2.5, mad)), so coupling them misstates the common case.
- **A per-dataset gate on located slopes**: 1 % of the survey's models state a location,
  too few for a network to learn the axis; per slope at 0.1 is coverage-first.
- **Per-slope scales by default** (a continuous spread, or a gate at the block case):
  deferred to the ablation metasurv's ADR 0008 planned; the override exists.
- **Inverse-Wishart on Σ_k**: a second parameterization of every random block for 4 % of
  Tier P rows, rejected in mixed ADR 0003 and not reopened.
- **Per-family shares fields in the residual seam's spirit** (a residual, a count and a
  binary one, each with a severity): six fields for the same draws, and the study's
  spelling would not say what the ground truth is.

### Consequences

- Ticket A (the prior's structure) and ticket B (the families), both before #56's read of
  `outcome`; the fourth fingerprint moves with A and is re-pinned, the other three stand.
- `CONTEXT.md`: Prior context rewritten, Response family added.
- metasurv's ADR 0008 carries the shared β semantics (families, the located prior, the
  intercept's location rule under a data-driven default, the per-slope override), cited by
  mixed ADR 0001; mixed ADR 0003 names two tokens and the intercept's drawn family with a
  location; `envelope.md`'s prior table follows. metabeta's default corpus gains the β
  family token and the location axes, nothing else.

## Amendment 2026-10-26: the scales vary per coefficient by default

#56's read of `prior` gave the context a scale per random effect, since a brms user can state
one (`class = sd, coef = …`; the survey's rehrig_2020 and trouve_2018 give the intercept's sd a
prior apart from the slopes'). Alex's rule for the default followed: an amortized model must
amortize over the priors a user may state, with the common one a special case of them. The
decision is: **each block's prior scales vary per coefficient by default. With probability
`varying_scale_share` (0.5) a block draws a log-normal weight per coefficient at `scale_spread`
(`Uniform(0, 2)`, resolved once per dataset); otherwise its coefficients share one scale. The
block's budget solves its scale exactly either way.**

- **One rule for both blocks.** β: τ_j = s_β w_j over the slopes, as before; the intercept keeps
  τ_0. σ_α: τ_k,j = s_k w_k,j over all of factor k's random effects, the intercept's among them,
  with s_k² Σ_j w_k,j² E[z_j²] = τ² φ_k. The fixed block and each factor gate on their own.
- **The block case** (one scale per block, what most users state) is the gate's other branch,
  and a whole corpus under `varying_scale_share=0`.
- **The defaults are guesses.** The share is 0.5. The spread is the value this ADR planned for
  β's ablation; the survey's two witnesses state intercept-to-slope sd scales 2.5 and 5 apart,
  log-ratios of 0.9 and 1.6, and the log-ratio of two weights has sd √2 · spread.
- **The families stay per block:** one for β, one per factor for σ_α.

### Considered options

- **A continuous spread without a gate:** equal scales only in the limit, so the case most
  users state has no mass of its own.
- **β left at the block case,** metasurv's ADR 0008's training draw: an amortized model that
  never sees per-slope scales cannot condition on them. A block-case corpus stays one override.
- **A family per random effect** (rehrig_2020 states two): rare, deferred.

### Consequences

- `scale_spread`'s default moves from 0 to `Uniform(0, 2)` and `varying_scale_share` is new; the
  glmm and glm draws move (the README's quickstart boxes with them), the scm and none forms'
  stand.
- metasurv's ADR 0008 and metabeta's `envelope.md` record that the default corpus varies per
  coefficient; a block-draw prototype passes `varying_scale_share=0`.

## Amendment 2026-10-28: the response family is stated, never drawn

#56's read of the README printed a family its reader never chose: the quickstart's GLMM drew
Gaussian from `y_family_shares`. Grilled on 2026-10-28, the decision is: **a model states y's
family, with no default, and nothing draws it. Its shape is one field, whose default each
family sets.** ADR 0007's amendment of the same day calls a model after the draw.

- **The family.** `Glm(family=...)`, a `Literal` over the eight families, has no default, so
  `Glm()` fails at construction. The `Literal` lives in `likelihood`, beside each family's link
  and column. The unqualified family is y's, as in brms's `family`; β's and the sd families keep
  their qualified shares fields and stay drawn. The `Prior` holds it as `family` (until
  2026-10-28: a GLM's outcome holds it, beside its link).
- **No shares.** `y_family_shares` goes. A corpus of several families picks the family per
  index, `Glmm(family=FAMILIES[index % 3])`, so each index draws its own covariates. The same
  index under two families draws the same covariates, a paired design when a study wants one.
- **The shape.** One field, `shape`, replaces `residual_df`, `dispersion`, `latent_scale`,
  `gamma_shape` and `beta_precision`. `likelihood` holds each family's default, the values
  above, and a validator fills it at construction, so the resolved model shows it (until
  2026-10-28: the field defaults to `None`, and a dataset draws its family's default when it
  resolves, so the resolved model shows the point); it resolves
  once per dataset like any field. A shape on Gaussian, Bernoulli or Poisson fails at
  construction. Every shape is positive, since σ_c = 0 was the Bernoulli family a second time.
  The `Prior` holds it as `shape` (until 2026-10-28: a GLM's outcome holds it).

### Considered options

- **A default family** (Gaussian): a likelihood the reader did not choose, the README's case.
- **Shares as a study's option** (`family: Family | Simplex`): two spellings in one field, and
  a family the `Prior` calls stated would be drawn again.
- **`likelihood` as the field's name:** the glossary's likelihood is the family and its shape,
  and `outcome` imports the module `likelihood` whole, so `model.likelihood` beside
  `likelihood.draw` would read as two things.
- **A field per family's shape** (as before): four of five dead for any one family, and a shape
  passed to the wrong family silently ignored.
- **A `Likelihood(family, shape)` pair:** a class for a pair the `Prior` already holds flat.
- **A burned draw** keeping the stream where the family's choice was: a line with no meaning,
  for models no release has shipped.
- **The family and shape on the `Prior`:** they are the model's specification, which a fitter
  states, not a prior over its parameters; the outcome holds them beside the link a fitter reads.
- **The resolved config on the outcome:** it carries every field a fitter does not need.
- **A default filled at construction** (a data-aware default factory): a field whose default
  depends on another, for a value each dataset draws anyway.
- **No default shape**, the five shaped families stating one: drops the per-family defaults.

### Consequences

- Every GLM and GLMM draw moves, as the family's choice leaves the stream and the shape resolves
  with the other fields; the SCM's draws stand.
- `CONTEXT.md`: Response family rewritten.
- metabeta's `envelope.md` and its ADRs name `y_family_shares` and `grouped(outcome_form='glmm')`;
  its default corpus states each family and mixes them itself.
