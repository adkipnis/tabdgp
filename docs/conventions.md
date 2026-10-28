<!-- shared conventions: generated, do not edit by hand; propose a change in an issue -->
# Conventions

The rules a change is reviewed against: the shared rules first, then this repository's own under
the last heading. The reasons behind the larger rules are in the [ADRs](adr).

## Tooling

- **uv:** Python through `uv` only; `uv run` for everything, every command from the repository
  root, without `cd`.
- **CI:** `ruff format` (100 columns, single quotes) and `pytest`.
- **argparse:** wrap its blocks in `# fmt: off` / `# fmt: on`.
- **Dependencies:** open an issue before adding a heavyweight one.

## Design

After Ousterhout, *A Philosophy of Software Design*: complexity is anything that makes code
harder to understand or change; it accrues one small edit at a time.

- **Deep modules:** a small interface over much functionality; an interface as complex as its
  implementation folds into its caller or callee.
- **Information hiding:** each module owns one design decision (a file format, a tensor layout,
  a prior's parameterization) that no other module depends on. Split code by what it knows, not
  by the order it runs in.
- **One home per fact:** in code as in the docs, with links everywhere else. A rule written out
  twice (a truncation, an order, a loop's initial state) becomes one function, one named value,
  or a call to the module that owns it.
- **No utils module:** a helper two modules need lives in the module that owns its fact.
- **No pass-throughs:** a method that forwards its arguments unchanged adds nothing; call the
  target directly.
- **Pull complexity down:** the module absorbs the awkward case so its callers stay simple. A
  fixed sensible behavior beats a new config parameter.
- **Define errors out of existence:** shape the interface so the invalid case cannot arise
  (accept the empty set, validate once at construction). The rest fails loudly: no fallbacks, no
  defensive `try`/`except`, no speculative generality.
- **Observed cases only:** branch only for a case observed on real inputs; add the next branch
  when use hits it.
- **Design it twice:** for a non-trivial interface, name two designs and why one wins before
  writing code.
- **Obvious over clever:** the simplest form a reader guesses right on first read. If 200 lines
  could be 50, rewrite them.

## Changing code

- **Smallest change:** one that solves the problem and meets the design bar, with no unrequested
  extras. Leave adjacent code, comments and formatting untouched.
- **Scripts:** a task that repeats, or is error-prone by hand, gets a `.py` or `.sh` script.
- **Borrowed code:** code taken from elsewhere cites its source in the docstring and is copied
  verbatim or imported and wrapped. A simplified rewrite comes with a test that it matches the
  original up to numeric noise.

## Commits and issues

- **Pull requests:** work reaches `main` only through a pull request, squash-merged into one
  commit that names the change and closes its issue. The branch keeps the steps: the failing
  test, the code, the fixes.
- **Issues:** a change to behavior or to an interface starts as an issue. An issue holds only
  what a contributor needs, in a few lines: the problem, the decision, the check that closes it
  and the issues that block it.
- **Planning:** who works an issue, with which tool, model or budget, stays out of the tracker.
- **Closing a spec:** a spec of several tickets closes with two more: an automated
  simplification and code-review pass over the branch (the debloat), then a guided manual
  review, one file per session, with small refactors. Only then is the branch merged.
- **Releases:** a `vX.Y.Z` tag on `main` with release notes.

## Names

- **Spelling:** American (`standardized`, `behavior`, `catalog`), in names as in prose. A
  released name spelled otherwise is renamed in the next release; only a name an outside
  standard fixes (R's own names, a library's) keeps its spelling.
- **Project names:** a project's package, repository, paths and code are lowercase (`metabeta`,
  `import metabeta`); prose names the project in capitals (MetaBeta), in READMEs, notebooks,
  docs and papers alike. A release keeps the package name (metabeta 0.6.0), and a quote keeps
  its source's spelling.
- **Modules:** one lowercase word.
- **Case:** `PascalCase` for classes and `Literal` aliases, `camelCase` for functions and
  methods, `snake_case` for variables, fields and a `Literal`'s values (`'grouping_factor'`),
  `UPPER_SNAKE_CASE` for constants.
- **Functions:** named for what they do, a verb and its object (`_resolveColumn`), not for what
  they return (`_column`). A test helper is read inside an assert, so it may name what it
  returns (`_density`).
- **Scopes:** one thing at different scopes keeps one name. Methods share its verb and each
  names its scope (`resolveDataset` for every field drawn once per dataset, `resolveField` for
  one); a counted quantity keeps its noun at every scope.
- **Counts:** a config field that counts is a plural noun (`rows`, `latent_nodes`), without an
  `n_` prefix.
- **Directories:** `dir`, not `folder`.
- **DataFrames:** a `table`, several are `tables`, never `df` or `frame`.
- **Leading underscore:** marks a pipeline step and anything no other module imports.

## Code

### Imports and types

- **Absolute imports:** `from <package>.<module> import name`, never relative ones.
- **Library names:** import a library's names when they are unmistakable (`BaseModel`), rename
  one that shadows a standard-library name (`dataclass as pydantic_dataclass`), and alias a
  module only by its community's convention (`np`, `pd`).
- **Whole modules:** a module that names nothing by its caller (`likelihood.draw`) is imported
  whole, `from <package> import <module>`, so each call says whose it is.
- **Packages:** a package's `__init__.py` holds no code of its own: it imports the public names
  from the modules that define them and lists them in `__all__`.
- **Modern typing:** `X | Y`, built-in generics, `from __future__ import annotations`.
- **Fixed options:** `Literal[...]` for an argument with a fixed set of options, a type alias
  once it repeats.
- **Config models:** Pydantic. A check a type can state is its constraint (`PositiveFloat`,
  `Field(ge=3)`), and `__post_init__` or a model validator holds only the checks across fields.

### Control flow

- **Statements:** f-strings, one statement per line.
- **Readable over compact:** unless runtime efficiency pays for it, one operation per line, each
  long expression or meaningful intermediate value named on the line before, and a loop where a
  comprehension would nest.
- **Method order:** a method lists its steps in the order you would describe them; a step with
  its own `if` ladder or loop becomes a method.
- **Searches:** a plain loop with an explicit check after it, not `for`/`else`, `while True` or
  `next()` on a generator.
- **One loop:** a loop fills its result as it goes; a second loop that only repackages the first
  one's output folds into it.
- **Conditional expressions:** `a if c else b` only on a short line.
- **Lookup tables:** literal dicts or `if` chains, not built with comprehensions,
  `dict.fromkeys` or `|`.
- **Dispatch:** cases that read different arguments are a `match` over their `Literal`, not a
  dict of wrappers that share one signature.
- **Conditions:** each is tested once: one `if`/`else` sets every value it decides, not a
  conditional expression per value or a second `if` on the same test. A flag that names the test
  (`bounded`) may be read again on the next line.

### Classes

- **Concepts:** a class per concept a reader should hold apart (a format family, a
  distribution).
- **Methods:** a method takes `self` if it calls another method of its class or reads the
  instance's state, unless it runs before an instance exists (a validator); otherwise it is a
  class or a static method, as its use needs.
- **Helpers:** a helper or constant used by one class lives on it; module functions are for what
  spans classes.
- **Pipelines:** one that holds state (a cache, a run directory) is a class; one without state
  is module functions.
- **Data classes:** hold their fields and the checks across them; the function that builds one
  is a module function beside it, not a method of the class, unless it is an alternate
  constructor from an outside format (`fromPandas`), which is a classmethod.
- **Base classes:** hold a skeleton their subclasses inherit: fields, methods or checks. Classes
  that only an `isinstance` check groups are a union alias, not an empty base.

### Arrays and tensors

- **Shapes:** annotate a shape inline on the line that returns it (`# (b, n, d)` for b datasets
  of n rows and d columns), never in a docstring.
- **Reshaping:** with `einops` where the package depends on it; never mutate a reused array in
  place.
- **Vectorization:** broadcasting and advanced indexing, not loops; batch over datasets.
- **Shape letters:** one lowercase letter per counted quantity, in shape comments and in the
  variables that hold the count:

  | letter | counts |
  |---|---|
  | `b` | the batch: datasets |
  | `n` | rows |
  | `m` | groups |
  | `s` | draws |
  | `d` | columns, or fixed effects |
  | `q` | random effects |
  | `p` | model parameters |
  | `k` | a categorical column's levels |

  No letter counts two quantities; a repository's own letters are listed in its part. A name
  that a file format or an outside convention fixes (Stan's `N` and `J`, a file's `K` column)
  keeps its spelling.
- **Randomness:** every draw takes an explicit generator or key; nothing reads global random
  state.
- **Scalars:** a scalar that a config holds or an array computation reads is a Python `int` or
  `float`, its arithmetic `math`'s: a numpy float64 scalar would promote a float32 array to
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
  reader new to the repository: what it offers and why, then how. Who imports the module is the
  [map](#modules)'s.
- **Banners:** `# ====` banners separate globals, classes and main, `# ----` banners
  subsections. Each is named in plain words, with one comment line on why the section exists
  where neither the name nor a class docstring below says it.
- **Order:** the constants, aliases and data classes the steps use come first, each after any
  function it is built from. Then each public function or method, followed by the steps it calls
  in run order, under a `# ----` banner naming the pipeline or its stage
  (`url -> archive -> tree`). A step that two public functions call goes after its first caller.
- **Scripts:** a script's `__main__` reads as the run's table of contents; for a pipeline class,
  it creates the class and calls its steps
  (`puller = Puller(); puller.getSurvset(); ...; puller.flush()`).

## Comments and docstrings

- **Sentences:** short and plain, one idea each, with a subject and a verb. Comments start in
  lower case; docstrings are capitalized sentences.
- **Line breaks:** at sentence or clause boundaries, not at the column limit.
- **Docstrings:** only where the behavior or design is not obvious: one or two plain sentences
  that start with what the function does or returns and leave out what the code already shows,
  never a chain of `;` clauses. Detail goes into inline comments that say why.
- **Sources:** a comment cites a paper link, or the authors, year and equation number for a
  formula from a paper; a link for code or a method it follows.
- **Config docstrings:** a config or data class's docstring introduces the concepts its fields
  name, a short bullet each. A field's detail goes into a short trailing comment, which every
  field whose name does not say what it is carries.
- **Checks:** a block of checks opens with one plain sentence on what it checks.
- **No history:** a comment explains the code, not its history: the case behind a rule goes to
  the test that pins it, a definition to `CONTEXT.md`, how it was decided to an ADR. Code states
  its rules without citing an ADR; the ADR keeps the reason.
- **Embedded scripts:** an embedded script (an R snippet) gets a one-sentence comment above it,
  one statement per line.

## Tests

- **Test first:** the failing test, then the minimal code.
- **Property tests:** `hypothesis`.
- **Layout:** the tests mirror the source: one `test_<module>.py` per module, each test named as
  a sentence (`test_<what_holds>`), fixtures in a `fixtures/` beside them.
- **Purpose:** each test says in a sentence or two why it exists, unless obvious.
- **Body:** bind the object under test, then call it; small named helpers, not lambdas or
  repeated comprehensions.
- **No self-checks:** a test asserts nothing its object checks itself: no shape its data class
  validates, no control that rebuilds what the call has just built.
- **Tables:** cases of one check are one parametrized table, each case's reason in its id, not
  one test per case.
- **Cross-checks:** names listed in two modules (a `Literal`, a catalog, a table's columns) are
  checked against each other by a test; the code checks nothing at import.
- **Coverage:** no gate.

## Docs

Markdown meets the same bar as code: simple, and read top to bottom.

- **Bullets:** short, the essentials first, detail in an appendix or a linked file.
- **Living rules:** a living rule sits with the method it governs.
- **Counts:** in generated files (a `SUMMARY.md`), never in prose.
- **README:** `README.md` is for users and fits one screen. Its links are absolute to `main`, so
  they render off GitHub too (on PyPI).

## Data

- **Raw data:** never overwrite it.
- **Witnesses:** a default, an edge or a share is read off the data collections and surveys
  already at hand (a registry, a practice survey) before it is guessed, and cites its witness.
  Where no witness exists and one pays off, gather the data first.
- **Large data:** stream or lazily load data that would not fit in memory.
- **Figures:** PDFs, never PNG or JPG. Existing vector files convert to SVG; a PDF is never
  rasterized.

## Lab notes and experiments

- **Lab notes:** `lab/YYYY-MM-DD-slug/`, one small design choice, its notebook with outputs and
  any PDFs, indexed in `lab/README.md` with the decision. Lab notes are private: `lab/` is in
  `.gitignore`, and nothing imports from it.
- **Experiments:** `experiments/`, rerunnable scripted studies, tracked and public. A lab note
  moves here once something public rests on it: a released behavior, an ADR's reason or a
  paper's number. Rerun with flags; never one-off.

## Notebooks

- **Reader:** the text speaks to a reader who knows the field but not the repository: no project
  vocabulary, and every abbreviation spelled out at first use.
- **Scope:** a notebook holds only what the figures and the decision need. Implementation
  details, fixed bugs and caveats that leave the result unchanged belong in the code or the
  issue.
- **Sentences:** short and plain, one per line, with `<br>` between them in a paragraph.
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
| `field` | how a field is a point or a distribution, its kind and bounds, its resolution to a point within given bounds, shares over named categories in their default's order | prior, hyperprior, the package's `__init__` |
| `prior` | the configs `Glm` and `Glmm`, their fields and their defaults and which fields each factor resolves, the stated prior drawn over named blocks (a block's variance per coefficient at scale 1 and its groups): β's family, the intercept's location, each slope's location, each block's scales from its budget, shared or per coefficient by a gate per block, the split between the fixed block and the factors, the families of σ_α and σ_ε apart, each factor's LKJ shape, and y's family, drawn first, with its shape, a field resolved for that family only, each parameter's family drawn with its locations and scales; `Prior`, which draws the parameters from that prior without the config, the sd families' draws and the LKJ C-vine; nothing about columns | hyperprior, outcome, the package's `__init__` |
| `hyperprior` | the dataset's fields and their defaults, an integer truncated at another field's value, the presets, the outcome's form, type shares and noise scale, and which fields each column, node or grouping factor draws, by marginal family for a continuous or count column | column, graph, mechanism, outcome, draw, the package's `__init__` |
| `column` | the design-column budget, the post-hoc types and the type `count` with its `poisson` marginal: each column's name, type, k and own parameters, what its metadata guarantees, and its values: a sample of its marginal family or its codes, placed by its node's ranks; the outcome as the column `y` with role `outcome`, minted last outside the budget under the scm form; a column a model wrote or a table brought, without a marginal or parameters; the grouping factor as a column, its relation, size draw and codes, and `Grouping`, the unit arithmetic (a unit's chain, the broadcast between units) | graph, mechanism, covariates, likelihood, outcome, draw |
| `graph` | the generating graph: the Cauchy DAG over latent and observed nodes, the block bonus, each node's unit and the mask on edges out of its chain, the order (the grouping factors, then the units in the factors' order, the row last, latent before observed within a unit, the outcome last of all at the row), each node's family, activation and scales, the outcome's noise scale from its own field, a row-unit node's group share and group factor, and what the graph guarantees | mechanism, draw |
| `mechanism` | how a node's values follow from its parents and its noise, at a unit: linear and MLP families, the parents read at the unit, relative noise, the z-score at the node's own unit, and a row-unit node's group part and row part mixed by its group share | draw |
| `covariates` | what a form reads: `Covariates` (the observed columns at the rows, their metadata, the grouping and the score), what they guarantee, and a table's through `fromPandas`, each column's metadata inferred or declared and checked against its values; the checks that a set of columns has one length and each column its metadata | outcome, draw, the package's `__init__` |
| `likelihood` | the distribution y is drawn from: y's families, each with its canonical link and y's column: Normal, Bernoulli and Poisson, and at share zero Student-t, negative binomial, latent-noise Bernoulli, Gamma and beta, each drawn with its shape; a draw at any η without failing, a count saturated at int64's largest, y of a Gamma or beta held inside float32's open range | outcome |
| `outcome` | what a form returns: `Outcome` (the columns a form writes and their metadata, a subclass per form adding its ground truth), what it guarantees, and a model's outcome through `runModel`: a study's model called and its result checked at the covariates' rows, or the GLM or GLMM drawn (a GLMM only on covariates with a factor), its design matrix coded from the covariates by role, each factor's random terms within its budget, η and `GlmOutcome` or `GlmmOutcome`, holding the prior and the parameters, with the parameters' raw-scale views | draw, the package's `__init__` |
| `draw` | the pipeline and its output: the key's generator, threaded through the modules in a fixed order, the grouping before the graph, nodes joined to columns by name, every column and latent node broadcast to the rows last, and what one draw returns, the grouping factors first and the outcome last, with its views `toPandas`, `toDesign` by `outcome`'s coder and `toDict`; `resampleOutcome`, a model's outcome on given covariates from the key's generator | the package's `__init__` |
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
  install, a quickstart box and its output per entry point (a draw, a resampled outcome),
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
