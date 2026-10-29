---
status: accepted
date: 2026-10-22
---

# 0006 - Coverage-first priors from one budget

A generalized linear model (GLM) or generalized linear mixed model (GLMM) draws a y of its own on the covariates (ADR 0007).
It states y's response family, which is never drawn.
Its hyperprior (`Glm` or `Glmm`) draws the prior, then the parameters from it (#46).
Both are the ground truth, which a fitter (a method that estimates the parameters from a table) should recover.
A model's priors are coverage-first and never fitted.
Coverage-first means that every prior a user may state gets mass, and the common one is a special case.
Their scales share one budget, the variance of the linear predictor η without the intercept.
No parameter is capped.
The deviation's scale is a multiple of η's standard deviation (sd), 0 by default.
The parameters are returned on the standardized scale and on the raw columns.

## Decisions

### The model

1. A model states y's response family (`family`) with no default, and nothing draws it:
   - Student-t, negative binomial and latent-noise Bernoulli make a dataset misspecified for a fitter of their base family (Gaussian, Poisson, Bernoulli), so a misspecified study states one, as it sets a deviation.
   - Gamma and beta complete the eight (precedent: 1 % each in the practice survey, 4,325 mixed models stated in 2,244 papers).
   - the shaped families' second parameter is one field (`shape`); unset, each dataset draws its family's default, a log-uniform guess (e.g. Student-t's degrees of freedom (df) `LogUniform(3, 10)`).
   - a shape on Gaussian, Bernoulli or Poisson fails at construction.
2. Each grouping factor draws q_k random effects, counted in design columns like the d fixed effects, the intercept among them:
   - the intercept, then whole covariates in random order, each if its columns still fit the factor's column budget (`random_effects`, the most design columns its random effects take), `UniformInt(1, 6)` truncated at d.
   - precedent: in the registry (MixedBench's list of real grouped tables and their documented models), 88 % of the 881 random-effect blocks (one factor's random effects in one model) are intercept-only, where the uniform budget gives about 17 %; the practice survey's 90th percentile is 5 random terms (intercepts and slopes) summed over factors.

### The prior

1. The prior is stated as a user of brms (R's package for Bayesian mixed models) states it: a family per parameter, and a location and a scale per coefficient.
2. The fixed effects' family (β's, `beta_family_shares`) is Normal 0.68, Student-t(3) 0.30 or Cauchy 0.02, a guess, shared by the intercept and the slopes (precedent: 18 of the practice survey's 31 stated df are 3).
3. The families of the random effects' sd σ_α and the residual's sd σ_ε (`sigma_alpha_family_shares`, `sigma_eps_family_shares`) are each uniform over half-Normal, half-Student-t(3), half-Cauchy and exponential, and drawn apart (precedent: brms states `sd` and `sigma` apart, and 24 of the survey's 132 stated pairs differ).
4. The intercept's location (`intercept_location`) is `Uniform(-3, 3)` on the link scale (η's, before the inverse link maps it to y's mean), a guess, and its scale τ_0 (`intercept_scale`) is `LogUniform(0.1, 10)`.
5. A slope's location is a multiple of its scale, ν_j = δ_j τ_j:
   - δ_j is zero with the share 1 − `location_share` (0.1); otherwise δ_j ~ N(0, `location_scale`) (2.0), a guess.
   - precedent: the survey's stated hypotheses sit two to three scales out.
6. A factor with q_k >= 2 correlates its effects with the share `correlated_share` (0.55), under an LKJ (Lewandowski-Kurowicka-Joe) prior of shape `lkj_eta`, `UniformInt(1, 4)`; otherwise they are independent (precedent: the practice survey's share of models with slopes that correlate them, a lower bound).

### The scales

1. The scales are drawn jointly per dataset from one budget, the variance τ² of η without the intercept:
   - with a residual (Gaussian, Student-t), τ² is the signal share (`signal_share`, r, the share of y's variance that η explains), `Beta(2, 2)`, and σ_ε's scale is τ_ε = √(1 − r), so a Gaussian y has unit variance at the scales.
   - otherwise τ (`predictor_sd`) is `LogUniform(0.25, 4)` on the link scale.
2. The fixed block (the fixed effects) takes the share φ_fixed (`fixed_share`), `Beta(1, 1)`, and the factors' blocks (each factor's random effects) split the rest by a Dirichlet at `split_alpha` (1) (precedent: the registry's fixed share of the explained variance is close to uniform whatever the number of factors, #47).
3. Each block's scales spend its part of τ² exactly, over what each coefficient adds to var(η) at scale 1:
   - a slope adds its column's variance (times 1 + δ_j² when located), a random effect its column's second moment (the random intercept 1).
   - e.g. a Gaussian y at r = 0.5 and φ_fixed = 0.5 with 19 z-scored slopes of one shared scale gives each slope τ_j = √(0.5 · 0.5 / 19) ≈ 0.11.
   - the prior reads only these blocks (each coefficient's variance at scale 1, and a factor's groups), never the columns.
4. With the share `varying_scale_share` (0.5), a block's scales vary per coefficient by log-normal weights at `scale_spread`, `Uniform(0, 2)`; otherwise its coefficients share one scale:
   - the fixed block and each factor gate on their own.
   - precedent: two of the survey's models set the random intercept's scale and the slopes' 2.5 and 5 times apart, log-ratios of 0.9 and 1.6, where two weights' log-ratio has sd √2 · `scale_spread`.

### The draw

1. Given the prior, every parameter follows its stated family exactly, and none is clipped, rerolled or rescaled after its draw:
   - each factor's random effects are N(0, Σ_k) per group, Σ_k = diag(σ_k) Ω_k diag(σ_k) with Ω_k their correlation matrix, independent of the other factors', crossed and nested alike.
   - Ω_k is drawn by the C-vine (partial correlations drawn from Beta distributions, then combined) of Lewandowski, Kurowicka and Joe (2009, section 3.2) in NumPy.
   - `Prior.drawParameters` draws them without the hyperprior, so a ground truth can be resampled under a fixed prior.
2. The linear predictor adds the deviation, η = Xβ + Σ_k Z_k α_k + λ · sd(Xβ + Σ_k Z_k α_k) · score, with X the design matrix and Z_k its columns with factor k's random effects:
   - λ is `deviation_scale`, 0 by default.
   - a nonzero λ fails on covariates without a score.
   - β and α stay the exact ground truth, and a GLMM is misspecified by exactly the returned deviation.
   - an endogenous but linear deviation takes an override of `mechanism_shares`.
3. y is drawn from its likelihood at any η without failing: a count above NumPy's Poisson limit saturates at int64's largest, and a Gamma or beta y stays inside float32's open range.
4. The outcome returns the prior and the parameters as drawn on the standardized scale (the design matrix's, its continuous and count columns z-scored), with views on the raw columns (before the z-score):
   - for the coding's linear map A, β_raw = Aβ, α_raw likewise, and Σ_k becomes A Σ_k A′, whose correlation changes only in the intercept's row and column.
   - they are what lme4 (R's package for mixed models by maximum likelihood) estimates on the raw columns.

## Considered options

### The model

| question | chosen | rejected |
|---|---|---|
| y's family | stated | a default, shares |
| y's shape | one field | a field per family, a default filled at construction |
| misspecified likelihoods | families | a residual law per family |

1. **y's family:**
   - A default family (Gaussian) is a likelihood the reader did not choose.
   - Shares give one field two spellings, and a family the outcome reports as stated would be drawn again.
2. **y's shape:**
   - A field per family leaves four of five dead for any one family, and silently ignores a shape passed to the wrong one.
   - A default filled at construction depends on another field, for a value each dataset draws anyway.
3. **Misspecified likelihoods:**
   - A residual law per family is six fields for the same draws, and its spelling does not say what the ground truth is.

### The prior

| question | chosen | rejected |
|---|---|---|
| β's family | Cauchy rare | uniform, flat |
| locations | a slope's relative to its scale | absolute, a gate per dataset, the intercept's relative to τ_0 |
| Σ_k | sds and LKJ, by the C-vine in NumPy | inverse-Wishart, torch's `LKJCholesky` |

1. **β's family:**
   - Uniform families put a third on Cauchy, whose slope at scale 1 exceeds 10 in 6 % of its draws, which saturates the logit and log scales.
   - A flat prior (brms's default, 20 % of the survey's readable β rows) is improper, so nothing can be drawn from it.
2. **Locations:**
   - A location relative to its scale keeps the budget exact, E[β_j²] = τ_j² (1 + δ_j²), where an absolute one can exceed the block's share.
   - A gate per dataset at the survey's 1 % of models with a stated location would rarely show one.
   - brms sets the intercept's location and scale apart (the linked y's median, and the larger of 2.5 and its median absolute deviation), so coupling them misstates the common case.
3. **Σ_k:**
   - Inverse-Wishart is a second parameterization of every random block, for 4 % of the practice survey's prior rows.
   - torch's `LKJCholesky` is a dependency ADR 0001 excludes.

### The scales

| question | chosen | rejected |
|---|---|---|
| the scales | one budget per dataset | independent per block, R2D2M2 |
| a block's divisor | each coefficient's variance | its width |
| heavy tails | recorded | the budget divided by df / (df − 2) |
| a block's scales | a gate per block | one scale, a continuous spread |

1. **The scales:**
   - Independent scales per block, log-uniform on [0.1, 10], give 61 % of Gaussian datasets R² > 0.9, 60 % of logit ones some |η| > 10 and overflow NumPy's Poisson sampler in 16 % of log-link ones (20,000 draws); a scale of 2.5 everywhere gives 61 %, 94 % and 6 %.
   - Realistic models sit at that range's lower edge (the worked example's 0.11).
   - R2D2M2 (the R²-induced Dirichlet decomposition prior for multilevel models; Aguilar and Bürkner 2023; Yanchenko, Bondell and Reich 2024 for GLMMs) derives every effect from a Beta R² and a Dirichlet over the terms, which never lets one random intercept carry most of the variance, as it does in the registry's most grouped tables (#47).
2. **A block's divisor:**
   - Dividing by the width realizes a third of the budget, since dummies and 0/1 columns have variance below 1 and the intercept none (#47).
3. **Heavy tails:**
   - A budget divided by df / (df − 2) breaks for Cauchy, and sets β's scale apart from the sd families'.
4. **A block's scales:**
   - With one scale per block, a fitter trained on the draws never sees the scale per coefficient that brms states by `coef`.
   - A continuous spread without a gate reaches equal scales only in the limit, so the case most users state has no mass of its own.

### The draw

| question | chosen | rejected |
|---|---|---|
| after the draw | nothing changes | caps, a recorded truncation |
| parameters | on both scales | on one |

1. **After the draw:**
   - MetaBeta v2 capped η's scale and R² by rescaling the parameters after the draw, so its recorded prior was not the one drawn from.
   - A recorded truncation (reject when max |η| > B, the indicator part of the prior) stays exact, but its acceptance rate collapses at the wide edges.
2. **Parameters:**
   - The standardized ones are what was drawn, the raw ones what a user fitting lme4 compares against.

## Consequences

- **The proofs:** the tests that compare generated and real tables certify a model's covariates under `grouped()`, never its y.
- **A corpus of several families:** states one per index (`Glmm(family=FAMILIES[index % 3])`), so the same index under two families reads the same covariates.
- **A white-box misspecified study:** a study that knows its exact misspecification sets `deviation_scale` or states a family that misspecifies its base family, and the ground truth stays exact either way.
- **One scale per block:** one override (`varying_scale_share=0`).
- **Absurd tables:** a log-link intercept at scale 10 draws counts no real table has, the cost of one range for every family.
- **Later families, none built:**
  - ordinal and binomial, each with a column of its own to design.
  - a family per random effect (one survey model states two).
  - log-normal stays a user's transform of a Gaussian y.
