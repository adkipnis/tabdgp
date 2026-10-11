# scamd audit

Every design choice of scamd's graph and typed columns, checked against spec #1 (issue #1)
with a recommendation to include, delay or reject, so that scamd need not be re-read. A summary
of scamd and TabICL's prior is in [`prior-art.md`](prior-art.md).

Read 2026-10-11: scamd at `2970b7b` (206 commits, 2025-12-23 to 2026-04-16), its README, tests and
examples, metabeta's `Scammer` wrapper (`metabeta/simulation/synthesizer.py`) and
`experiments/simulation/ood_design.py`, and metabeta-v3's two ADR 0005s. scamd has no lab notes,
no notebooks, no GitHub issues and no TODOs. Its tests are smoke tests only: shape, finiteness and
a seeded-global-state reproducibility check. No test covers marginals, type correctness or
dependence. So "history" below means commit messages, diffs, reverts and in-code "Fix A/C/D"
comments. I also ran a few probes against the installed copy in metabeta's venv, which is
byte-identical to HEAD.

Abbreviations: structural causal model (SCM), multilayer perceptron (MLP), random Fourier
features (RFF), classifier two-sample test (C2ST). Roadmap items: 2 grouping, 3 measurement pass
and proofs, 4 quirk axes and the fitted hyperprior, 5 outcome node.

## Probe results (N = 500, metabeta-like hyperparameter draws, ~200 datasets)

- `sanityCheck` returns True on a matrix that contains a NaN (`utils.py:47` tests `.any()` where
  it means `.all()`).
- The Gaussian copula on the causes leaves only about 63% unique values per coupled column, so
  37% are ties. It bootstraps order statistics and does not permute them, so the marginal is
  not preserved, even though its docstring says it is.
- Post-hoc `Threshold` columns built from *independent* inputs correlate with each other at a
  mean |r| of 0.42. Dirichlet(1) mixing pulls every output toward the column mean.
- Softmax `Categorical`: level shares are about 1/K each, and the largest |corr| between any
  dummy and any input is 0.11. The categoricals are almost independent of the table.
- SCM only, without post-hoc columns: the p99 of max |x| is 6e15, and the p99 ratio of column
  standard deviations is 3e15. Excess kurtosis is above 10 in 6% of columns.
- With post-hoc columns: 44% of columns have 10 or fewer unique values, and 4.6% of columns
  have more than 20% exact zeros (dead ReLUs, or inf replaced by 0).
- numpy's `Generator.geometric` has support {1, 2, …}. This matters for spec #1's level draw
  (row T20).

## Table A: the graph side

| # | scamd's choice | where | what scamd's history says | spec #1 | recommendation |
|---|---|---|---|---|---|
| G1 | The SCM is an MLP. Hidden units are the nodes, and the columns are read from all layers' outputs concatenated | `scm.py:154-157, 306-325` | The default path from the first commit (2025-12-23). It works as a generator, but there is no graph to return, and the latent set is n_layers × n_hidden units against d columns | Replaced by the Cauchy DAG | **Reject**: there is no white-box graph, and the latent confounding is unbounded and unparameterised |
| G2 | Weights are redrawn on every forward pass | `scm.py:291`, `dag.py:115` | Intended ("each sample() is a distinct mechanism"). It silently breaks noise calibration (G17) and the cached Standardizer (G14) | n/a (one draw per key) | **Reject** |
| G3 | Columns are random non-constant units from any depth, or a contiguous window (tries 16 starts, then falls back to random) | `scm.py:271-287, 320-325` | The contiguous search was added in 8d55c5b ("stability improvements") with 32 tries, later cut to 16 | Absent | **Reject**: a DAG has no readout step |
| G4 | `n_hidden = max(n_hidden, 2·d)` | `scm.py:151` | Untested | n/a | **Reject** (comes with G1) |
| G5 | Dense N(0, σ_w²/(1−p)) weights with σ_w = 1, Bernoulli dropout p = 0.2 except on the first layer, **no fan-in scaling** | `scm.py:224-232` | p raised from 0.1 to 0.2 (01bbf00). Pre-activations grow with √width, so tanh saturates and exp overflows. Probe: values reach 1e15 at p99. The DAG path does scale by 1/√n_parents (`dag.py:98-100`) | `init_scale` LogUniform(0.1, 3), fan-in not stated | **Include in #1**: W ~ N(0, init_scale²/fan_in) in every MLP layer. Node z-scoring bounds only a node's output, not its hidden layers |
| G6 | Block-diagonal weights (`blockwise`) with n_blocks ∈ [1, ⌈√min dim⌉) | `scm.py:234-248` | The oldest correlation device; metabeta used it 50% of the time. Bugs: `integers` excludes its upper bound, so there is always 1 block when min dim ≤ 4, and a ValueError when it is 1. Units outside the blocks get only the bias, so they are near-constant (σ ≈ 1e-7 after calibration) and pass the exact-equality constant check | Replaced by the within-block edge bonus | **Already in #1** (as replaced). Keep the "bonus raises within-block density" test |
| G7 | DAG option: Barabási–Albert (undirected, oriented low→high) or Erdős–Rényi (directed, backward edges dropped), m ∈ 1..5 | `dag.py:13-63`, `api.py:147-153` | Added late (2713c05, 6d7cd55). metabeta used it 40% of the time, but only smoke tests cover it. Barabási–Albert gives every non-seed node in-degree exactly m. Erdős–Rényi builds `nx.DiGraph(edges)`, which drops isolated nodes (`dag.py:56`): no independent column can arise, and small graphs fail and retry | Replaced by the Cauchy DAG | **Reject**: fixed in-degree, a node-dropping bug, a networkx dependency |
| G8 | Observed nodes are leaves where possible; 10 latent nodes by default | `dag.py:162, 254-270` | Untested. Observed columns are sinks, so every observed–observed dependence is pure confounding by latents | Latent nodes first in the order, observed→observed edges allowed, `latent_nodes` UniformInt(0, 3) | **Already in #1.** Item 4 fits the latent count. scamd sat at the opposite extreme (10 latents; up to d/2 factors in G24), which suggests 0–3 latents may be low at d = 19. Flag only; it is a default |
| G9 | One activation for the whole dataset, drawn from the pool; every DAG node shares one family | `api.py:133-137`, `dag.py:189-196` | `RandomChoice` (G15) and the presets' `n_random_choice` were added to break this uniformity | `activation` is one `Choice` field; the spec lists it outside the per-column fields, so it is presumably per dataset | **Include in #1**: draw the activation per node from the resolved choice (the same two-level draw as `mechanism_shares`). No new field |
| G10 | Pool of 25 basic activations: Exp, Square, LogAbs, Mod, Sign, Ceil/Floor/Round, UnitInterval and others | `basic.py:82-108`, `pool.py:67-80` | Exp, Square and LogAbs feed non-finite values that are zeroed (G18). The discrete ones create ties and heaping inside the mechanism. Mod uses the global torch RNG ("uniform k", 3b26e5a) | tanh, relu, sin, identity | **Reject the rest**: keep the spec's four (finite and at most linear growth). Heaping belongs to the column (T24), not to the mechanism |
| G11 | Random-Fourier GP activations (SE, Matérn, fractional; 512 features; length scale logU(0.1, 16)) | `gp.py:18-119` | Built with care (kernel choice, frequency clamp, ae39652 buffers) and weighted by the presets. Never compared with data | Absent (the tree family comes later) | **Reject**: sin in an MLP already gives smooth oscillation, and no C2ST summary statistic can tell function classes apart |
| G12 | `RandomScale` wrapper: standardise, then scale·(x + bias) with scale = exp(2·N(0, 1)) and bias ~ N(0, 1), then the activation | `meta.py:37-57`, `pool.py:77-78` | The only place scamd moves the activation's operating point off the input mean | `init_scale` covers the scale; **no bias is mentioned** | **Include in #1**: a bias ~ N(0, 1) per MLP unit, on z-scored inputs. Without it, every relu kink and every tanh/sin centre sits exactly at the parent's mean, which makes the shapes more symmetric than they need be |
| G13 | `Standardizer` caches the first call's mean and sd | `meta.py:15-34`, `gp.py:90, 117` | Bug: the cache is filled on the noiseless calibration pass and then reused after the weights are redrawn, so the "standardised" inputs to GP and RandomScale are wrong on the real pass and on every retry | Each node is z-scored on its actual values | **Already in #1** (as fixed) |
| G14 | `RandomChoice`: each unit gets one of k activations, redrawn on every forward pass | `meta.py:61-90` | Added with the presets (3 or 5 random-choice entries) | Absent | **Reject**: G9's per-node activation covers it |
| G15 | Gaussian noise added *before* the activation in every hidden unit; `vary_sigma_e` draws a scale per unit | `scm.py:11-22, 202-212` | Pre-activation noise makes noise heteroscedastic and non-additive in children (through exp, abs and the like). `vary_sigma_e` is dead whenever calibration runs (it is always overwritten) | Noise after the mechanism, additive, relative to the z-scored node | **Delay to item 3**: if the measurement pass shows covariate heteroscedasticity, feed the noise as an extra MLP input. Rank matching to skewed marginals already makes noise multiplicative |
| G16 | Noise calibration: one noiseless pilot pass of 256 rows sets σ = 0.1 × IQR per unit, once | `scm.py:250-269, 293-303`, `dag.py:220-236` | Added 7504635 (2026-04-15). Bugs: it calibrates on pilot weights and is never rerun after the redraw (G2). The 0.1 is a constant, so the signal-to-noise ratio never varies across datasets | Replaced by `noise_scale` LogUniform(0.01, 1), relative | **Already in #1** (as replaced) |
| G17 | Root causes: a separate n_causes-wide input. Under `mixed`, each column gets one of 9 families (normal, uniform, multinomial, Zipf, gamma, lognormal, beta, Student-t, Gaussian mixture), parameters shared per family | `causes.py:113-147` | The catalogue arrived in e2fa9d1. The "Dirichlet draw" is the argmax of Dirichlet(1), which is a plain uniform choice. Root shapes are washed out by up to 7 MLP layers. **The DAG path ignores the causes entirely** (`api.py:181-182`; roots are N(0, 1) at `dag.py:211-213`), so 40% of metabeta's datasets never used the catalogue | Replaced: roots are N(0, 1), and marginals come from rank matching after the SCM | **Already in #1** (as replaced). This is the strongest case for rank matching |
| G18 | Non-finite values are replaced by 0 per layer and per node; `sanityCheck` rejects only an all-non-finite matrix | `scm.py:265, 309`, `dag.py:216`, `utils.py:43-49` | The probe confirms both: a matrix with a NaN passes, and 4.6% of columns have more than 20% exact zeros. metabeta adds its own `np.isfinite(...).all()` check and retries | Absent | **Reject the repair**. Keep the spec's bounded activations, and pin the invariant with a test (every column finite and non-constant) |
| G19 | Constant-column check by exact equality with the first row; up to 8 retries in `Generator.sample`, 8 more in metabeta | `utils.py:37-40`, `api.py:180-196`, `synthesizer.py:53-77` | Retries reject draws with an unknown bias. Near-constant units pass the check (G6) | Absent | **Reject the retries. Include in #1** a zero-variance guard in z-scoring: a constant mechanism output (a dead relu, say) z-scores to zeros, not 0/0 = NaN, before the noise is added |
| G20 | Random cause moments: μ ~ N(0, 1), σ = \|N(0, 1)·μ\| | `causes.py:35-37` | Bug: the scale is proportional to the location, so causes with μ ≈ 0 are near-constant | n/a (z-scoring) | **Reject** |
| G21 | Discrete roots: a multinomial with 2–19 categories, Zipf clamped at 10 | `causes.py:53-66` | They propagate clusters and multimodality to children. Zipf's clamp puts a point mass at 10 | Absent | **Delay to item 2**: grouping-factor nodes are exactly discrete roots. A latent discrete root (latent classes) can be reconsidered there |
| G22 | Gaussian copula on the causes: ρ ~ U(0, 0.9) shrinks a normalised Wishart toward I, eigenvalues clipped, `try/except` falls back to the uncoupled causes | `causes.py:149-198` | Added eaf5b3b (2026-04-15). Probe: 37% ties, because the quantile lookup at `:195-198` resamples instead of permuting. A Gaussian copula has no tail dependence (metabeta's OOD study uses Clayton as the out-of-distribution case) | Replaced by the graph and latent hubs | **Reject.** Lesson for `marginal.match`: place the sorted sample by argsort-of-argsort (a permutation), never by quantile lookup. The spec's "exactly the sample's values" test pins this |
| G23 | `FactorLayer`: k ∈ [1, d/2] factors with loadings N(0, 1/k) and α ~ U(0.1, 1.2), applied w.p. 0.7, in absolute units after the readout | `scm.py:78-94, 192-200` | Added d78004c and 0fdb484 (2026-04-15). The next day (6a98efe) p went from 0.5 to 0.7 and α from 0.8 to 1.2 to raise inter-column correlation. Never tested beyond shape. Its strength depends on the unstandardised scale of the SCM output | Replaced by latent hubs | **Already in #1** (as replaced). The push toward more factor structure is a data point for item 4's fit of `latent_nodes` and `hub_scale` |
| G24 | `SharedNoiseLayer`: w.p. 0.5, 1–3 groups of size 2..d each get one N(0, 1) vector × α ~ U(0.05, 0.4) | `scm.py:25-44, 168-184` | Added 7504635; its group-size bug was fixed the same day (c391e26). Absolute units | Replaced by latent hubs plus the block bonus | **Already in #1** (as replaced) |
| G25 | After the SCM: factor, then shared noise, then the marginal transform | `scm.py:327-337` | The `__init__` comments (`:162-166`) claim the transform runs before the shared noise; it runs after | n/a | **Reject** (the order goes with G23–G24 and T19) |
| G26 | No final standardisation; columns leave on the SCM's raw scale | `scm.py:340-341`, `api.py:190` | Probe: p99 sd ratio of 3e15. metabeta standardises downstream | Every node z-scored, then a marginal | **Already in #1** |
| G27 | Configuration: three dict presets, extra kwargs routed by `inspect.signature` | `presets.py:35-57`, `api.py:110-173` | The `calibrate_noise` preset key is never read (`api.py:112-144` uses only four keys). metabeta drew the preset uniformly and `p_posthoc` ~ U(0.2, 0.9) | Replaced by the Pydantic hyperprior | **Already in #1** (as replaced) |
| G28 | RNG: a numpy Generator mixed with the global torch RNG (`nn.init`, `torch.randn`, Dirichlet, `randperm`, Mod, RandomScale, the SE kernel) | `utils.py:9-14`, passim | Several "fix rng" commits. `test_rng_consistency` bans only `np.random` and `random`, not global torch. It reproduces only through `setSeed` | One generator per key | **Already in #1** |

## Table B: the typed-column side

| # | scamd's choice | where | what scamd's history says | spec #1 | recommendation |
|---|---|---|---|---|---|
| T1 | Post-hoc firing: one dataset-level coin w.p. `p_posthoc`, then Binomial(d, p) ≥ 1 transforms | `posthoc.py:359-365` | 3845d2f ("Fix" for under-firing at small d, where a per-column coin rarely fired) | Replaced by per-dataset Dirichlet type shares and the budget plan | **Already in #1** (as replaced) |
| T2 | Routing: 30% CategoricalBlock, 10% OrdinalBlock, 60% uniform over 8 single layers | `posthoc.py:376-413` | OrdinalBlock cut from 30% to 10% one commit after it arrived (d966218, "decrease rarity"). Skew-producing layers were upweighted 3× (4753ddd), then reverted (12f6470) | Replaced by type shares | **Already in #1** (as replaced) |
| T3 | **Column mixing**: every post-hoc output is a Dirichlet(1)-weighted convex combination of *all* standardised SCM columns | `posthoc.py:25-40` | Never questioned in the history. Probe: outputs from independent inputs correlate at mean \|r\| 0.42, because the flat Dirichlet concentrates near the column mean | Each typed column is its own node | **Reject**: it manufactures near-collinearity and dilutes each column's parents |
| T4 | Post-hoc columns replace SCM columns up to d; the output is `[SCM, post-hoc]` | `posthoc.py:430-436` | dc5f5da ("Fix A"): before it, post-hoc columns competed in a random permutation and were mostly discarded. Bug: the random subset at `:435` can split a categorical's dummy group, silently merging levels into the reference. Column position reveals type | Exact budget with K − 1 per factor; observed names permuted | **Already in #1** |
| T5 | Binary: a threshold at 0 on the mixed value, so shares near 50/50 | `posthoc.py:43-49` | No rare binaries are possible | Thresholds at Dirichlet shares | **Already in #1.** Rare binaries are quirk 3: **delay** their share axis **to item 4** |
| T6 | `MultiThreshold`: τ ~ N(0, 1) sorted, levels always 3 (`Posthoc` never passes `levels`) | `posthoc.py:52-75` | The mixed value's sd is below 1 (T3), so outer levels can be empty or near-empty at random | Ordered columns with drawn K and Dirichlet shares | **Already in #1** (as replaced) |
| T7 | `QuantileBins`: cut points at U(0, 1) quantiles of the data, 3 levels | `posthoc.py:78-101` | Untested | Same mechanism: thresholds at drawn shares | **Already in #1** |
| T8 | Softmax `Categorical`: K − 1 noisy logits (σ 0.01) and a reference logit 0, temperature, a multinomial draw, dummy-coded | `posthoc.py:125-161` | 56d8eb9 ("make categorical more stochastic"); bfbbe1b (dummy coding). Probe: shares ≈ 1/K, max \|corr\| with any input 0.11, so it is nearly independent. Its one virtue: each level has its own direction, so levels are not intervals on one axis | Thresholds on one node, labels permuted | **Already in #1** for item 1. Threshold categoricals are single-index (a permuted ordered variable). **Delay to item 4** a multi-direction categorical (argmax over K projections of the parents), and only if the measurement pass's association statistics separate the two |
| T9 | `CategoricalBlock`: 1..max(3, d/2) factors with K_i ∈ [2, max(4, d/n_cats)] | `posthoc.py:164-203, 377-391` | Comments call it "Type A/B design matrices" | The budget plan | **Already in #1** (as replaced) |
| T10 | `OrdinalBlock`: k ordered columns on directions clustered round a shared base (spread 0.05–0.5), L ∈ 3..7, τ ~ N(0, 1), input **not standardised** | `posthoc.py:206-249` | Added 7186c41 (2026-04-15) for the "Spearman > Pearson" pattern of Likert batteries. Share cut to 10% the next commit. Its unstandardised input contradicts every other post-hoc layer | Absent as such. A latent hub with several ordered children produces it | **Delay to item 3**: check that latent hubs reproduce battery-like rank correlation. Add a block only in item 4 if they do not |
| T11 | `Poisson` (λ = exp(mix)) and `NegativeBinomial` (logits and softplus(r) from two mixes) | `posthoc.py:252-277` | `Geometric` removed as redundant (ca7da33). Upweighted 3× for skew, then reverted. Counts are most of the probe's "≤ 10 unique values" columns | Absent; the catalogue is all continuous | **Delay to item 4** (quirk 4, few unique values). Under rank matching it is a single extra family, a negative-binomial sample, so it is cheap once the pass justifies it. metabeta v1's `Synthesizer` gave NB 15% |
| T12 | `Clamp` at random quantiles (q_lo ~ U(0, 0.15), q_hi ~ U(0.85, 1)) and `CensoredFloor` at q ~ U(0.05, 0.4) | `posthoc.py:280-321` | Added a569f6e ("random truncation", 2026-04-15). CensoredFloor was upweighted, then reverted | Absent; beta and uniform give bounded support without point masses | **Delay to item 4** (quirk 5). Applied after rank matching it is a point mass at a drawn quantile, one step |
| T13 | `Rank` (top-k indices) and `Geometric` | removed in ca7da33 | Abandoned on 2026-03-27 | Absent | **Reject** (abandoned) |
| T14 | Gaussian noise σ = 0.01 added to the mix before decoding | `posthoc.py:105-122` | Negligible next to the signal | The node's own noise does this | **Reject** |
| T15 | Standardise before mixing (the cached `Standardizer`) | `posthoc.py:23-24, 37-38` | Inherits G13's stale-cache bug | Nodes are z-scored | **Already in #1** |
| T16 | `sanityCheck` per transform drops the whole transform if any output column is constant | `posthoc.py:422-425` | An empty level kills its categorical, which biases against rare and empty levels (probe: about 1% at N = 100, d = 8; more at small N) | `apply` keeps every level present "when N allows"; nothing is dropped | **Already in #1.** Keep it so: never resample for an empty level. Rare levels are item 4 |
| T17 | `MarginalTransformLayer`: per column identity, signed power p ~ U(0.3, 1.5) or signed log1p, w.p. 0.7 | `scm.py:47-75, 186-190` | Added 6fa0548 (2026-04-15) for Spearman > Pearson. Powers cut from 2.5 to 1.5 (b333b60). softplus and clamped exp were added to "close the skewness coverage gap observed in diagnostic benchmarks" (12f6470), then **reverted the same day** (2970b7b, the last commit). Signed transforms on centred data are symmetric, so they add kurtosis but no skew; the gap stayed open | Replaced by rank matching to a drawn marginal | **Already in #1** (as replaced). The revert is the evidence that a post-hoc warp cannot place skew where the data needs it |
| T18 | Level counts: 3 (Multi/Quantile), 3–7 (OrdinalBlock), 2..d+1 (Categorical), 2..max(4, d/n) (block) | see T6–T10 | No principled level draw anywhere | K = 3 + Geometric(p), p ~ Beta(4, 4) at the audit; since changed to K = 2 + Geometric(p), p ~ Beta(2, 3) | **Already in #1**, but see T20 |
| T19 | Labels: no explicit permutation. Softmax levels are exchangeable; the reference is always the implicit-logit level | `posthoc.py:150-160` | n/a | Permuted labels for categoricals | **Already in #1** |
| T20 | (spec check) the level draw | spec #1, provisional defaults | numpy's geometric starts at 1, so 3 + Geometric gives K ≥ 4: **K = 3 cannot occur**. The stated p95 of 14 puts more than 5% mass on K > 10, which mixed ADR 0005 puts out of scope until v1 | — | **Include in #1**: either K = 3 + (Geometric − 1), or K = 2 + Geometric with categoricals at K ≥ 3, and truncate at 10 or record why not. This flags a contradiction with an ADR; Alex decides |
| T21 | Column order: SCM columns first, post-hoc columns last | `posthoc.py:436` | Type is readable from position | Observed names permuted | **Already in #1** |
| T22 | Output: one float matrix, dummies expanded, no metadata | `api.py:190`, `synthesizer.py:83-86` | metabeta had to infer types (`checkContinuous`) | Integer codes with `type`, `levels`, `role` | **Already in #1** |
| T23 | Ordered treated as numeric: ordinal codes come out as floats among continuous columns | `posthoc.py:72-75, 245-249` | n/a | `ordered` type; metabeta folds it into categorical | **Already in #1** (and ADR 0005 keeps ordered-as-numeric out until v1) |
| T24 | Rounding and heaping: only through the Round/Ceil/Floor activations and counts | `basic.py:62-74` | Incidental, never targeted | Absent | **Delay to item 4** (quirk 4), as a column step after rank matching |
| T25 | Zero-inflation: no explicit family; zeros come from dead ReLUs, inf→0 and small Poisson rates | G18, T11 | Partly an artefact of a bug | Zero-inflated lognormal in the catalogue | **Already in #1** |
| T26 | Missingness | none | Never attempted | Out (ADR 0005: complete-case fitter) | **Reject** |

## Spec #1 additions I would make (top 5)

1. **MLP layer init (G5, G12): W ~ N(0, init_scale²/fan_in) plus a bias ~ N(0, 1) per unit.**
   scamd's unscaled MLP reached 1e15. Without biases every relu kink and tanh centre sits at the
   parent's mean, and node z-scoring cannot repair either inside a node's hidden layers.
2. **A zero-variance guard in z-scoring (G18, G19).** A constant mechanism output z-scores to
   zeros before the noise, never to NaN. Pin "every column finite and non-constant" with a
   `test_draw` property test instead of any retry. scamd's repair-and-retry machinery
   (inf→0, an `.any()` bug, 8×8 retries) hid exactly these cases.
3. **The activation drawn per node from the resolved choice (G9).** scamd's single activation
   per dataset needed `RandomChoice` bolted on. Drawing per node is the spec's own two-level
   draw and adds no field.
4. **Fix the level draw (T20).** As written, K = 3 cannot occur and more than 5% of K exceeds the
   K ≤ 10 scope of mixed ADR 0005. Correct the geometric offset and truncate at 10, or ask
   Alex to amend the ADR.
5. **A dependence test for typed columns.** In `test_column` or `test_draw`: at low noise,
   each binary, categorical and ordered column depends detectably on its parents. scamd's
   categoricals were nearly independent of the table (|corr| ≤ 0.11), and nobody noticed
   because every test checked only shapes.

## Rejections (top 5)

1. **The MLP-as-graph with weights redrawn every pass (G1–G4).** It cannot return a generating
   graph, and the redraw silently broke both the noise calibration and the cached
   standardisation.
2. **Post-hoc column mixing over all SCM columns (T3, T8).** A flat Dirichlet average makes
   typed columns near-collinear (|r| 0.42 from independent inputs) and leaves softmax
   categoricals almost independent. A typed column should be one node with its own parents.
3. **The root-cause catalogue and the Gaussian copula (G17, G20, G22).** The MLP washes root
   shapes out, the DAG path skipped them altogether, and the copula turns 37% of values into
   ties. Rank matching after the SCM supersedes all three.
4. **Post-SCM layers in absolute units (G23–G25, T17: factor, shared noise, signed power and
   log1p).** Their strength depends on an unstandardised scale. They were retuned twice in two
   days, and the skew fix was reverted in the last commit. Latent hubs and rank matching
   express the same intents with a unit-free knob.
5. **Repair and rejection (G18, G19, T16: inf→0, the exact-equality constant check, the
   `.any()` sanity check, nested retries, dropping transforms with an empty level).** Each one
   distorts the distribution invisibly, and dropping transforms removes the rare levels the
   quirk catalogue wants. Spec #1's bounded activations and z-scoring make these failures
   impossible instead of retrying them.

## Delays, by roadmap item

- Item 2: discrete roots (G21), via grouping-factor nodes.
- Item 3: pre-activation or heteroscedastic noise (G15) and the ordinal battery (T10), each
  checked against the measurement pass before it is built.
- Item 4: counts (T11), clamps and censored floors (T12), rounding and heaping (T24), rare
  binaries (T5), the multi-direction categorical (T8), the fit of the latent count (G8, G23).
