---
status: accepted
date: 2026-10-22
amended: 2026-10-23, #41: the form's field is outcome_form; 2026-10-23, ADR 0007: the form is a model, `Glm` and `Glmm` configs of their own in `prior` reached as `outcome_form='glm'` and `'glmm'`, no `Hyperprior.glmm()`; the tokens are shares fields; the container is per model, the scm form's included; the deviation is the score every form gets
---

# 0006 — The glmm form: metabeta's prior envelope as the glmm form's defaults, the fit drawn jointly, no caps, the estimands returned

The outcome node's second form (ADR 0005) is a generalized linear mixed model
(GLMM) on the design matrix, the datasets `metabeta.mixed` trains on and, with a deviation,
the white-box misspecified ones. Grilled on 2026-10-22 with ADR 0005, the decision is: **the
glmm form's prior fields default to metabeta's prior envelope, as ADR 0004 made its size rules
the grouped preset's defaults (reached as `outcome_form='glmm'` on `grouped()`, ADR 0007); the
parameters are drawn from the stated prior families exactly, with the scales of a dataset's
blocks drawn jointly so that the fit is spread instead of capped; nothing is clipped,
rerolled or rescaled after the draw; the dataset returns the estimands on both scales and
the prior context they were drawn with.** The words are `CONTEXT.md`'s: form, deviation, prior
context, random effect, design matrix.

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
  a one-family corpus is `outcome_form=Glmm(family_shares={'normal': 1})`, a study of that
  family. The white-box misspecified study overrides `deviation_scale`.
- **The prior fields, the envelope.** `family_shares` over {normal, bernoulli, poisson},
  uniform, with canonical links; the families of metabeta's v1 (binomial, negative binomial,
  Gamma, ordinal) are later entries. One `sd_family_shares` field over {half_normal,
  half_student_t with ν = 3, half_cauchy, exponential}, uniform, chosen once per dataset and
  shared by the random-effect standard deviations and the residual scale (a shares field, not
  a token, ADR 0007). `lkj_eta` uniform on {1, 2, 3, 4} when a factor's effects
  are correlated, `correlated_share` 0.55 per factor with q_k ≥ 2 (the practice survey's share
  of models with slopes). The intercept's scale σ_0 `LogUniform(0.1, 10)`, the envelope.
- **The fit drawn jointly.** Per dataset: `signal_share` r, a wide Beta; `predictor_sd` τ for
  the unit-free families, log-uniform on about [0.25, 4] on the logit and log scales, and
  τ² = r for the Normal family, whose y then has unit variance by construction; a Dirichlet
  split φ of var(η) over the fixed block and each factor's random block. The scales follow:
  σ_β = τ √(φ_fixed / d), s_α,k = τ √φ_k, s_ε = √(1 − r) as the half-t's scale. Given the scales,
  β ~ N(0, σ_β), σ_α ~ F(s_α), σ_ε ~ F(s_ε) exactly, so metabeta's conditioning on its context
  stays honest; only the hyperprior over the contexts is no longer independent per block. A
  finding on the side: at d = 20 and r = 0.5, σ_β ≈ 0.11, so realistic LMMs sit at the
  envelope's lower edge and its upper decade produces only saturated data.
- **No caps, no rerolls, no rescaling.** A Poisson rate is `exp(η)` in float64, no clip; above
  numpy's limit the count comes from the Normal approximation, so a draw never fails. The
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
  The dataset returns f's values and λ, so β and α stay the exact truth and a fitted GLMM is
  misspecified by exactly λ · f; a latent group-level parent in f is ADR 0003's endogeneity.
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
  fixed and random term names per factor, the prior context (σ_β, σ_0, s_α per factor, s_ε, F,
  η, r, τ, φ), the parameters on the standardized scale as drawn (`beta` (d,), per factor
  `sd_alpha` (q_k,), `corr_alpha` (q_k, q_k), `alpha` (m_k, q_k), `sigma_eps`, the deviation
  (n,) and λ), the coding statistics (each continuous column's mean and sd) and the raw-scale
  parameters, analytic from these (β_raw,j = β_j / sd_j, the intercepts absorbing
  Σ β_j · mean_j / sd_j, Σ_k scaled by the diagonal): what a user fitting lme4 on the raw
  columns compares against.
- **Modules.** `prior` owns the GLMM's priors: the configs `Glm` and `Glmm` with the fields
  over them (ADR 0007), the joint scale draw, the LKJ sampler and the parameter draw, nothing
  about columns; `outcome` owns the node: its forms, the design matrix, η, the deviation, the
  families and links, and the containers. `prior` is imported by `outcome`, `outcome` by
  `draw`.

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
  (containment, then a C2ST); it may overturn this decision before the parameter tickets.
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
  prior context and the random-term draw come from here, and metabeta keeps the family token
  in its network, its log-densities and its API.
- The IMH correction's prior density reads the dataset's prior context; a corpus of one family
  is an override in metabeta's curriculum loop.
- `prior` gains the prior fields as `Glm`'s and `Glmm`'s (ADR 0007); `prior` and `outcome` are
  new modules; `Dataset` gains `outcome`. The proofs stay on `grouped()`; the glmm form's
  covariates are certified by the same proofs, its response never.
