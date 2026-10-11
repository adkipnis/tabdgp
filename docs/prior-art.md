# Prior art

What scamd (`~/Code/scamd`, the prototype) and TabICL's prior (`~/Code/tabicl/src/tabicl/prior/`,
BSD-3, read at commit c91f00d) do, read for ideas only: no code is copied, and a mechanism one
of them inspired names it in a comment. Read 2026-10-11.

- [`scamd-audit.md`](scamd-audit.md): each of scamd's graph and typed-column choices, with a
  verdict against spec #1 (include, delay to an item, reject).
- [`related-simulators.md`](related-simulators.md): the other simulators (PFN priors, causal
  benchmarks, fit-to-data synthesizers, R study tools, amortized-inference frameworks), what
  TabDGP should adopt from them and what sets it apart.

## Shared by both

- PyTorch tensors, one dataset at a time: neither batches inside its SCM. TabICL batches by
  padding features to a maximum and stacking, with a process pool over datasets.
- Both mix an explicit generator with global random state, so neither reproduces from one seed.
- Neither returns column types, the graph or the drawn hyperparameters.

## scamd

- **SCM**: an MLP whose hidden units are the nodes (weights redrawn on every pass, dense with
  dropout or block-diagonal), or a DAG option (Barabási–Albert or Erdős–Rényi, one activation
  for all nodes). Activations: about 25 elementwise functions and random-Fourier GP functions.
- **Marginal catalogue**, applied to the root causes: normal, uniform, multinomial index,
  Zipf, gamma, lognormal, beta, Student-t, Gaussian mixture; after the SCM, a power or log1p
  warp per column.
- **Post-hoc column types**: threshold (binary), multi-threshold and quantile bins (ordered),
  softmax categoricals, an ordinal block over a shared direction, Poisson and negative
  binomial counts, clamps and censored floors. Each post-hoc column mixes all SCM columns.
- **Correlation**: block-wise weights, a Gaussian copula on the causes, a factor layer, and
  shared noise (one noise vector added to a group of columns).
- **Config**: plain dict presets; extra keyword arguments routed by `inspect.signature`.
- **Weaknesses**: noise calibrated once while weights are redrawn; absolute noise scales; a
  `sanityCheck` that only rejects all-non-finite data; column position reveals type; levels
  fixed at 3.

## TabICL

- **Hyperparameter meta-distributions**: a hyperparameter is drawn from a distribution whose
  own parameters are drawn first (log-scaled truncated normal, Beta, choice over softmaxed
  random logits). The parameters are drawn once per group of datasets, the values per dataset.
  A spec entry is a constant or a distribution, recursively; defaults fill only what the user
  left out.
- **Starting ranges**: layers 2 to about 8, hidden width about 9 to 134, init scale 0.01 to 10,
  noise 1e-4 to 0.3, causes 2 to about 13, dropout 0.9·Beta, categorical share 0.2.
- **Graph SCM** (`graph_lib`): a Cauchy DAG, where an edge i→j (i < j) exists with probability
  σ(offset + out_j + in_i) with Cauchy draws, so density ranges from near-empty to
  near-complete and hubs arise. Node count 2–32, log scale. Vector-valued nodes; roots are base
  points through a random function; parents are concatenated or combined by sum, product, max
  or logsumexp. Function families: MLP, oblivious trees, nearest-centre discretisation, linear,
  quadratic, random-Fourier GP, softmax, products.
- **Named draws** through one sampler with a mode per name: `global` (one value), `meta` (a
  per-dataset Beta that correlates repeated draws of a name within a dataset), `local` (i.i.d.);
  `boundary_mass` puts mass at the ends of an interval (all numeric, all categorical).
- **Categoricals**: k − 1 thresholds at random sample values with labels permuted (v1);
  nearest-centre or softmax converters (graph SCM). Numeric columns get a Kumaraswamy warp.
- **Not taken over**: y as a node of the SCM, and the two-pass 4σ outlier clipping with z-scores
  clipped to ±100.

## Ideas carried into TabDGP's design (proposed, not yet decided)

- A recursive point-or-distribution field: for a per-column field, its distribution's
  parameters drawn once per dataset and its values once per column (TabICL's meta draws), so
  that corners such as an all-binary table get mass.
- The Cauchy DAG for edge density (TabICL); scamd's shared noise and block-wise correlation as
  latent hub nodes and a within-block edge bonus in the graph draw.
- Marginals mapped onto the SCM's values by rank matching after the SCM (Iman and Conover,
  1982), instead of scamd's root marginals: each column gets exactly its drawn marginal and the
  SCM supplies only the dependence.
- Categoricals by thresholds at drawn level shares with permuted labels (TabICL v1).
