<!-- shared conventions: generated, do not edit by hand; propose a change in an issue -->
# Conventions

The rules a change is reviewed against: the shared ones first, this repository's own under the
last heading. The [ADRs](adr) give the reasons behind the larger ones.

## Tooling

- **uv:** for all Python; `uv run` everything from the repository root, without `cd`.
- **CI:** `ruff format` (100 columns, single quotes) and `pytest`.
- **argparse:** blocks wrapped in `# fmt: off` / `# fmt: on`.
- **Dependencies:** an issue before adding a heavyweight one.

## Design

After Ousterhout, *A Philosophy of Software Design*.

- **Deep modules:** a small interface over much functionality; an interface as complex as its
  implementation folds into its caller or callee.
- **Information hiding:** each module owns one design decision (a file format, a tensor layout,
  a prior's parameterization) no other module depends on. Split code by what it knows, not by
  when it runs.
- **One home per fact:** in code and docs, linked from everywhere else. A rule written twice (a
  truncation, an order, a loop's initial state) becomes one function, one named value or a call
  to the module that owns it.
- **No utils module:** a helper two modules need lives in the module that owns its fact.
- **No pass-throughs:** call the target directly, not through a method that forwards its
  arguments unchanged.
- **Pull complexity down:** the module absorbs the awkward case so callers stay simple; a fixed
  sensible behavior beats a new config parameter.
- **Define errors out of existence:** shape the interface so the invalid case cannot arise
  (accept the empty set, validate once at construction). The rest fails loudly: no fallbacks,
  defensive `try`/`except` or speculative generality.
- **Observed cases only:** branch only for a case seen on real inputs; add the next when use
  hits it.
- **Design it twice:** for a non-trivial interface, name two designs and why one wins before
  coding.
- **Obvious over clever:** the simplest form a reader guesses right on first read. If 200 lines
  could be 50, rewrite them.

## Changing code

- **Smallest change:** solves the problem and meets the design bar, no unrequested extras;
  adjacent code, comments and formatting stay untouched.
- **Scripts:** a `.py` or `.sh` script for a task that repeats or is error-prone by hand.
- **Borrowed code:** cites its source in the docstring; copied verbatim or imported and wrapped.
  A simplified rewrite gets a test that it matches the original up to numeric noise.

## Commits and issues

- **Pull requests:** the only way to `main`, squash-merged into one commit that names the change
  and closes its issue. The branch keeps the steps: failing test, code, fixes.
- **Issues:** every change to behavior or an interface starts as one. An issue holds only what a
  contributor needs, in a few lines: the problem, the decision, the check that closes it, the
  blocking issues.
- **Planning:** who works an issue, with which tool, model or budget, stays out of the tracker.
- **Closing a spec:** a spec of several tickets ends with two more: an automated simplification
  and code-review pass over the branch (the debloat), then a guided manual review, one file per
  session, with small refactors. Only then is the branch merged.
- **Releases:** a `vX.Y.Z` tag on `main` with release notes.

## Names

- **Spelling:** American (`standardized`, `behavior`, `catalog`) in names and prose; a released
  name spelled otherwise is renamed in the next release. Only a name an outside standard fixes
  (R's, a library's) keeps its spelling.
- **Project names:** lowercase in package, repository, paths and code (`metabeta`); capitalized
  in all prose (MetaBeta). A release keeps the package name (metabeta 0.6.0); a quote keeps its
  source's spelling.
- **Modules:** one lowercase word.
- **Case:** `PascalCase` for classes and `Literal` aliases, `camelCase` for functions and
  methods, `snake_case` for variables, fields and `Literal` values (`'grouping_factor'`),
  `UPPER_SNAKE_CASE` for constants.
- **Functions:** named for what they do, verb and object (`_resolveColumn`), not what they
  return (`_column`). A test helper, read inside an assert, may name what it returns
  (`_density`).
- **Scopes:** one thing keeps one name at every scope: its methods share a verb and each names
  its scope (`resolveDataset` for every field drawn once per dataset, `resolveField` for one); a
  counted quantity keeps its noun.
- **Counts:** a config field that counts is a plural noun (`rows`, `latent_nodes`), no `n_`
  prefix.
- **Directories:** `dir`, not `folder`.
- **DataFrames:** `table` and `tables`, never `df` or `frame`.
- **Leading underscore:** a pipeline step, or anything no other module imports.

## Code

### Imports and types

- **Absolute imports:** `from <package>.<module> import name`, never relative.
- **Library names:** imported directly when unmistakable (`BaseModel`), renamed when they shadow
  the standard library (`dataclass as pydantic_dataclass`); a module aliased only by its
  community's convention (`np`, `pd`).
- **Whole modules:** a module that names nothing by its caller (`likelihood.draw`) is imported
  whole, `from <package> import <module>`, so each call says whose it is.
- **Packages:** `__init__.py` holds no code: it imports the public names from their modules and
  lists them in `__all__`.
- **Modern typing:** `X | Y`, built-in generics, `from __future__ import annotations`.
- **Fixed options:** `Literal[...]`, a type alias once it repeats.
- **Config models:** Pydantic. A check a type can state is a constraint (`PositiveFloat`,
  `Field(ge=3)`); `__post_init__` or a model validator holds only the checks across fields.

### Control flow

- **Statements:** one per line; f-strings.
- **Readable over compact:** unless runtime efficiency pays, one operation per line, each long
  expression or meaningful intermediate named on the line before, a loop where a comprehension
  would nest.
- **Method order:** steps in the order you would describe them; a step with its own `if` ladder
  or loop becomes a method.
- **Searches:** a plain loop with an explicit check after it, not `for`/`else`, `while True` or
  `next()` on a generator.
- **One loop:** fills its result as it goes; a second loop that only repackages the first one's
  output folds into it.
- **Conditional expressions:** `a if c else b` only on a short line.
- **Lookup tables:** literal dicts or `if` chains, not built by comprehension, `dict.fromkeys`
  or `|`.
- **Dispatch:** cases that read different arguments are a `match` over their `Literal`, not a
  dict of wrappers with one signature.
- **Conditions:** tested once: one `if`/`else` sets every value it decides, not a conditional
  expression per value or a second `if` on the same test. A flag naming the test (`bounded`) may
  be read again on the next line.

### Classes

- **Concepts:** one class per concept a reader should hold apart (a format family, a
  distribution).
- **Methods:** take `self` if they call another method of the class or read instance state,
  unless they run before an instance exists (a validator); otherwise class or static, as use
  needs.
- **Helpers:** a helper or constant used by one class lives on it; module functions are for what
  spans classes.
- **Pipelines:** a class if it holds state (a cache, a run directory), else module functions.
- **Data classes:** hold their fields and the checks across them; their builder is a module
  function beside them, unless it is an alternate constructor from an outside format
  (`fromPandas`), which is a classmethod.
- **Base classes:** hold a skeleton their subclasses inherit: fields, methods or checks. Classes
  grouped only by an `isinstance` check are a union alias, not an empty base.

### Arrays and tensors

- **Shapes:** commented inline on the line that returns them (`# (b, n, d)` for b datasets of n
  rows and d columns), never in a docstring.
- **Reshaping:** `einops` where the package depends on it; never mutate a reused array in place.
- **Vectorization:** broadcasting and advanced indexing, not loops; batch over datasets.
- **Shape letters:** one lowercase letter per counted quantity, in shape comments and count
  variables: `b` datasets (the batch), `n` rows, `m` groups, `s` draws, `d` columns or fixed
  effects, `q` random effects, `p` model parameters, `k` a categorical column's levels. No
  letter counts two quantities; a repository lists its own letters in its part. A name a file
  format or outside convention fixes (Stan's `N` and `J`, a file's `K` column) keeps its
  spelling.
- **Randomness:** every draw takes an explicit generator or key; nothing reads global random
  state.
- **Scalars:** a scalar a config holds or an array computation reads is a Python `int` or
  `float` with `math` arithmetic, since a numpy float64 scalar promotes a float32 array to
  float64.

## File layout

```python
"""prior.py defines the prior over a GLMM's parameters and how it is sampled.

pipeline:
---------
a model specification
-> the prior's hyperparameters per family          _hyperparameters
-> draws of fixed effects and variance components  sample
"""

# =====================================================================================
# default hyperparameters
# =====================================================================================
# the scales every prior falls back to when the specification names none.
```

- **Module docstring:** one sentence on what the module defines, optionally a short paragraph on
  how, then the pipeline in run order, one `-> result` line per step with its function on the
  right. A module whose why is not obvious replaces the paragraph with a short pyramid for a
  newcomer to the repository: what it offers and why, then how. Who imports it is the
  [map](#modules)'s.
- **Banners:** `# ====` for globals, classes and main, `# ----` for subsections, named in plain
  words, plus one comment line on why the section exists if neither the name nor a class
  docstring below says it.
- **Order:** the constants, aliases and data classes the steps use come first, each after any
  function it is built from. Then each public function or method, followed by the steps it calls
  in run order, under a `# ----` banner naming the pipeline or its stage
  (`url -> archive -> tree`). A step two public functions call follows its first caller.
- **Scripts:** `__main__` reads as the run's table of contents; for a pipeline class it creates
  the class and calls its steps (`puller = Puller(); puller.getSurvset(); ...; puller.flush()`).

## Comments and docstrings

- **Sentences:** short and plain, one idea each, with a subject and a verb. Comments start in
  lower case; docstrings are capitalized.
- **Line breaks:** at sentence or clause boundaries, not the column limit.
- **Docstrings:** only where behavior or design is not obvious: one or two plain sentences that
  start with what the function does or returns and leave out what the code shows, never a chain
  of `;` clauses. Detail goes into inline comments that say why.
- **Sources:** cited in a comment: a paper link, or the authors, year and equation number for a
  formula; a link for code or a method it follows.
- **Config docstrings:** a config or data class's docstring introduces the concepts its fields
  name, a short bullet each. A field's detail goes into a short trailing comment, required where
  its name does not say what it is.
- **Checks:** a block of checks opens with one plain sentence on what it checks.
- **No history:** comments explain the code, not its history: the case behind a rule goes to the
  test that pins it, a definition to `CONTEXT.md`, how it was decided to an ADR. Code states its
  rules without citing an ADR.
- **Embedded scripts** (an R snippet): a one-sentence comment above, one statement per line.

## Tests

- **Test first:** the failing test, then the minimal code.
- **Property tests:** `hypothesis`.
- **Layout:** mirrors the source: one `test_<module>.py` per module, each test named as a
  sentence (`test_<what_holds>`), fixtures in a `fixtures/` beside them.
- **Purpose:** a sentence or two on why each test exists, unless obvious.
- **Body:** bind the object under test, then call it; small named helpers, not lambdas or
  repeated comprehensions.
- **No self-checks:** assert nothing the object checks itself: no shape its data class
  validates, no control that rebuilds what the call just built.
- **Tables:** cases of one check are one parametrized table, each case's reason in its id.
- **Cross-checks:** names listed in two modules (a `Literal`, a catalog, a table's columns) are
  checked against each other by a test, never at import.
- **Coverage:** no gate.

## Docs

Markdown meets the same bar as code: simple, read top to bottom.

- **Bullets:** short, essentials first, detail in an appendix or a linked file.
- **Living rules:** next to the method they govern.
- **Counts:** in generated files (a `SUMMARY.md`), never in prose.
- **README:** for users, one screen. Its links are absolute to `main`, so they render off GitHub
  too (on PyPI).

## Data

- **Raw data:** never overwritten.
- **Witnesses:** a default, an edge or a share is read off the data collections and surveys at
  hand (a registry, a practice survey) before it is guessed, and cites its witness. Where none
  exists and one pays off, gather the data first.
- **Large data:** streamed or lazily loaded if it would not fit in memory.
- **Figures:** PDF, never PNG or JPG. Existing vector files convert to SVG; a PDF is never
  rasterized.

## Lab notes and experiments

- **Lab notes:** `lab/YYYY-MM-DD-slug/`, one small design choice: its notebook with outputs and
  any PDFs, indexed in `lab/README.md` with the decision. Private: `lab/` is in `.gitignore` and
  nothing imports from it.
- **Experiments:** `experiments/`, rerunnable scripted studies, tracked and public, rerun with
  flags, never one-off. A lab note moves here once something public rests on it: a released
  behavior, an ADR's reason or a paper's number.

## Notebooks

- **Reader:** knows the field but not the repository: no project vocabulary, every abbreviation
  spelled out at first use.
- **Scope:** only what the figures and the decision need; implementation details, fixed bugs and
  caveats that leave the result unchanged belong in the code or the issue.
- **Sentences:** short and plain, one per line, `<br>` between them in a paragraph.
- **Bullets:** a bold label and one fact; a second fact is a second bullet, never a `<br>`.
<!-- end of shared conventions -->

## In tabdgp

The words are [`CONTEXT.md`](../CONTEXT.md)'s.

### Tooling

- Python 3.12 and newer, developed on 3.14, CI on both. The floor rises only when a
  dependency or a language feature needs it, never past the oldest version
  [SPEC 0](https://scientific-python.org/specs/spec-0000/) supports.

### Modules

| module | owns | imported by |
|---|---|---|
| `field` | `Hyperprior`, the base of `Scm`, `Glm` and `Glmm`: how a field is a point or a distribution, its kind and bounds, its resolution to a point within given bounds, shares over named categories in their default's order | prior, scm, likelihood, the package's `__init__` |
| `prior` | the hyperpriors `Glm` and `Glmm`, their fields and their defaults and which fields each factor resolves, y's family as stated and its shape, the family's default shape drawn per dataset unless one is stated, the stated prior drawn over named blocks (a block's variance per coefficient at scale 1 and its groups): β's family, the intercept's location, each slope's location, each block's scales from its budget, shared or per coefficient by a gate per block, the split between the fixed block and the factors, the families of σ_α and σ_ε apart, each factor's LKJ shape, each parameter's family drawn with its locations and scales; `Prior`, which draws the parameters from that prior without the hyperprior, the sd families' draws and the LKJ C-vine; nothing about columns | outcome, the package's `__init__` |
| `scm` | `Scm`, the SCM's hyperprior: the dataset's fields and their defaults, an integer truncated at another field's value, the presets, whether the SCM draws y, y's type shares and noise scale, and which fields each column, node or grouping factor draws, by marginal family for a continuous or count column | column, graph, mechanism, dataset, the package's `__init__` |
| `column` | the design-column budget, the post-hoc types and the type `count` with its `poisson` marginal: each column's name, type, k and own parameters, what its metadata guarantees, and its values: a sample of its marginal family or its codes, placed by its node's ranks; the outcome as the column `y` with role `outcome`, minted last outside the budget under the SCM; a column a model wrote or a table brought, without a marginal or parameters; the grouping factor as a column, its relation, size draw and codes, and `Grouping`, the unit arithmetic (a unit's chain, the broadcast between units) | graph, mechanism, covariates, likelihood, outcome, dataset |
| `graph` | the generating graph: the Cauchy DAG over latent and observed nodes, the block bonus, each node's unit and the mask on edges out of its chain, the order (the grouping factors, then the units in the factors' order, the row last, latent before observed within a unit, the outcome last of all at the row), each node's family, activation and scales, the outcome's noise scale from its own field, a row-unit node's group share and group factor, and what the graph guarantees | mechanism, dataset |
| `mechanism` | how a node's values follow from its parents and its noise, at a unit: linear and MLP families, the parents read at the unit, relative noise, the z-score at the node's own unit, and a row-unit node's group part and row part mixed by its group share | dataset |
| `covariates` | what a model reads: `Covariates` (the observed columns at the rows, their metadata, the grouping and the score), what they guarantee, their design matrix `Design` by `toDesign`, the covariate columns coded by type, and a table's through `fromPandas`, each column's metadata inferred or declared and checked against its values; the checks that a set of columns has one length and each column its metadata | outcome, dataset, the package's `__init__` |
| `likelihood` | the distribution y is drawn from: y's families, `Family`, each with its canonical link and y's column: Gaussian, Bernoulli and Poisson, and Student-t, negative binomial, latent-noise Bernoulli, Gamma and beta, each drawn with its shape, whose default `SHAPES` holds; a draw at any η without failing, a count saturated at int64's largest, y of a Gamma or beta held inside float32's open range | prior, outcome |
| `outcome` | what the SCM or a model returns: `Outcome` (the columns it writes and their metadata, a model's subclass adding its ground truth), what it guarantees, `Model`, and a model's outcome through `runModel`: a study's model called and its result checked at the covariates' rows, or the GLM or GLMM drawn (a GLMM only on covariates with a factor), on the covariates' design matrix, each factor's random terms within its budget, η, a deviation only on covariates with a score, and `GlmOutcome` or `GlmmOutcome`, holding y's family, link and shape, the prior and the parameters, with the parameters' raw-scale views | dataset, the package's `__init__` |
| `dataset` | the pipeline and its output: the key's seed sequence and its generator, threaded through the modules in a fixed order, the grouping before the graph, nodes joined to columns by name, every column and latent node broadcast to the rows last, and what one draw returns, its covariates with the score and the SCM's outcome, its columns the grouping factors first and y last, with the view `toPandas`; `modelOutcome`, a model's outcome on drawn or given covariates from the key's first child stream | the package's `__init__` |
| `fidelity` | how two samples of summary vectors are compared: pooled ranks (missing value coded as −1), the permutation null, random-forest C2ST, Wasserstein-1 distance over ranks per axis, containment bands | the caller, outside `src/` |

### Changing code

- [`CONTRIBUTING.md`](../CONTRIBUTING.md#pull-requests)'s three rules hold: the defaults are
  what the proofs certify, code is written from scratch, nothing comes from downstream.

### Code

- A default the hyperprior guesses is a round number; only a fitted one carries more digits.
- A draw is named before it is cast or transformed. The Bernoulli idiom `rng.random() < share`
  counts as one draw.
- Letters of its own: `v` counts nodes, `f` grouping factors, `e` the elements of a unit, `c`
  a node's parents (its direct causes) and `u` distinct values.
- One marker for a missing value in a dataset's columns: `None`, in a list of Python scalars,
  never NaN in an array. A summary vector marks a missing statistic as NaN, since it is a float
  array.

### Comments and docstrings

- A mechanism that scamd or TabICL's prior inspired says so in a comment.
- Code in `src/` names no consumer (such as metabeta).

### Tests

- `tests/` mirrors `src/`.
- Unit tests for every mechanism: marginal ranges, the coded shapes, determinism under a fixed
  generator, every override honored, the flat case as zero factors.
- The proofs against mixedbench's registry (the two-sample test and the containment test)
  carry the `slow` mark and fixed seeds.
- The names checked across modules include a shares field's categories and the per-column and
  per-node fields. Names listed twice in one module are checked by a test, if at all.

### Docs

- `README.md`: badges, a tagline, a short what and why, highlights of what the release does,
  install, a quickstart box and its output per entry point (a draw, a model's outcome),
  planned, contributing, license.

### Explaining the design

How the docs and the paper explain TabDGP to a reader with only a vague idea of it:

- Top down, one level before the next: why it exists, the concepts it needs, the blueprint
  (the modules and how they work together), then the detail.
- Short plain sentences, one idea per bullet; a term defined before it is used, in
  `CONTEXT.md`'s words; an acronym spelled out at its first use.
- An abstract mechanism gets a worked example small enough to check by eye (three values
  through rank matching).
- A design choice names the alternatives considered, the verdict and why, as a table when
  there are several, each citing its source: an ADR, the spec, an issue.
- A choice without a recorded reason says so ("a guess, to be fitted") instead of
  inventing one; an opinion is marked as one, apart from the facts.
- What is left out says where it lives: a GitHub issue, an appendix.

### Data

- The registry statistics the proofs read are mixedbench's, pinned by version, never edited
  here.

### Lab notes

- `lab/` stays local and untracked.
