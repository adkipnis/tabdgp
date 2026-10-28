<!-- shared conventions: generated, do not edit by hand; propose a change in an issue -->
# Conventions

The rules a change is reviewed against: the shared rules first, then this repository's own under
the last heading. The words are the glossary's; the reasons behind the larger rules are in the
[ADRs](adr).

## Tooling

- Python through `uv` only; `uv run` for everything, every command from the repository root,
  without `cd`.
- `ruff format` (100 columns, single quotes) and `pytest`, run by CI. Wrap `argparse` blocks in
  `# fmt: off` / `# fmt: on`.
- Open an issue before adding a heavyweight dependency.

## Design

After Ousterhout, *A Philosophy of Software Design*: complexity is anything that makes code
harder to understand or change; it accrues one small edit at a time.

- **Deep modules.** A small interface over much functionality; an interface as complex as its
  implementation folds into its caller or callee.
- **Information hiding.** Each module owns one design decision (a file format, a tensor layout,
  a prior's parameterization) that no other module depends on. Split code by what it knows, not
  by the order it runs in.
- **One home per fact**, in code as in the docs. A rule written out twice (a truncation, an
  order, a loop's initial state) becomes one function, one named value, or a call to the module
  that owns it. No utils module: a helper two modules need lives in the module that owns its
  fact.
- **No pass-throughs.** A method that forwards its arguments unchanged adds nothing; call the
  target directly.
- **Pull complexity down.** The module absorbs the awkward case so its callers stay simple. A
  fixed sensible behavior beats a new config parameter.
- **Define errors out of existence.** Shape the interface so the invalid case cannot arise
  (accept the empty set, validate once at construction). The rest fails loudly: no fallbacks,
  no defensive `try`/`except`, no speculative generality. Branch only for a case observed on
  real inputs; add the next branch when use hits it.
- **Design it twice.** For a non-trivial interface, name two designs and why one wins before
  writing code.
- **Obvious over clever.** The simplest form a reader guesses right on first read, in code
  written, reviewed or rewritten. If 200 lines could be 50, rewrite them.

## Changing code

- The smallest change that solves the problem and meets the design bar, no unrequested extras.
  Leave adjacent code, comments and formatting untouched.
- A task that repeats, or is error-prone by hand, gets a `.py` or `.sh` script.

## Commits and issues

- Work reaches `main` only through a pull request, squash-merged into one commit that names the
  change and closes its issue. The branch keeps the steps: the failing test, the code, the fixes.
- A change to behavior or to an interface starts as an issue: the problem, the decision and the
  check that closes it, in a few lines.
- An issue holds only what a contributor needs: the problem, the decision, the check and the
  issues that block it. Who works it, with which tool, model or budget, is planning and stays
  out of the tracker.
- A spec of several tickets closes with two more: an automated simplification and code-review
  pass over the branch (the debloat), then a guided manual review, one file per session, with
  small refactors. Only then is the branch merged.
- A release is a `vX.Y.Z` tag on `main` with release notes.

## Names

- American spelling (`standardized`, `behavior`, `catalog`), in names as in prose. A released
  name spelled otherwise is renamed in the next release; only a name an outside standard fixes
  (R's own names, a library's) keeps its spelling.
- A project's package, repository, paths and code are lowercase (`metabeta`, `import metabeta`);
  prose names the project in capitals (MetaBeta), in READMEs, notebooks, docs and papers alike.
  A release keeps the package name (metabeta 0.6.0), and a quote keeps its source's spelling.
- A module is one lowercase word. A package's `__init__.py` holds no code of its own:
  it imports the public names from the modules that define them and lists them in `__all__`.
- `PascalCase` for classes and `Literal` aliases, `camelCase` for functions and methods,
  `snake_case` for variables, fields and a `Literal`'s values (`'grouping_factor'`),
  `UPPER_SNAKE_CASE` for constants.
- A function is named for what it does, a verb and its object (`_resolveColumn`), not for what
  it returns (`_column`). A test helper is read inside an assert, so it may name what it returns
  (`_density`).
- One thing at different scopes keeps one name. Methods share its verb and each names its scope
  (`resolveDataset` for every field drawn once per dataset, `resolveField` for one); a counted
  quantity keeps its noun at every scope.
- A config field that counts is a plural noun (`rows`, `latent_nodes`), without an `n_` prefix.
- A directory is `dir`, not `folder`. A DataFrame is a `table`, several are `tables`, never `df`
  or `frame`.
- A leading `_` marks a pipeline step and anything no other module imports.

## Code

### Imports and types

- Absolute imports, never relative ones: `from <package>.<module> import name`.
- Import a library's names when they are unmistakable (`BaseModel`), rename one that shadows a
  standard-library name (`dataclass as pydantic_dataclass`), and alias a module only by its
  community's convention (`np`, `pd`).
- A module that names nothing by its caller (`likelihood.draw`) is imported whole,
  `from <package> import <module>`, so each call says whose it is.
- Modern typing: `X | Y`, built-in generics, `from __future__ import annotations`.
  `Literal[...]` for an argument with a fixed set of options, a type alias once it repeats.
- Config models are Pydantic: a check a type can state is its constraint (`PositiveFloat`,
  `Field(ge=3)`), and `__post_init__` or a model validator holds only the checks across fields.

### Control flow

- f-strings, one statement per line, a long expression named on the line before.
- Readable over compact, unless runtime efficiency pays for it: one operation per line, each
  meaningful intermediate value named on its own line, and a loop where a comprehension would
  nest.
- A method lists its steps in the order you would describe them; a step with its own `if`
  ladder or loop becomes a method. A search is a plain loop with an explicit check after it,
  not `for`/`else`, `while True` or `next()` on a generator. A loop fills its result as it goes;
  a second loop that only repackages the first one's output folds into it.
- `a if c else b` only on a short line. Lookup tables are literal dicts or `if` chains, not
  built with comprehensions, `dict.fromkeys` or `|`. Cases that read different arguments are a
  `match` over their `Literal`, not a dict of wrappers that share one signature.
- A condition is tested once: one `if`/`else` sets every value it decides, not a conditional
  expression per value or a second `if` on the same test. A flag that names the test (`bounded`)
  may be read again on the next line.

### Classes

- A class per concept a reader should hold apart (a format family, a distribution). A method takes
  `self` if it calls another method of its class or reads the instance's state, unless it runs
  before an instance exists (a validator); otherwise it is a class or a static method, as its use
  needs. A helper or constant used by one class lives on it; module functions are for what spans
  classes.
- A pipeline that holds state (a cache, a run directory) is a class; a pipeline without state is
  module functions.
- A data class holds its fields and the checks across them; the function that builds it is a
  module function beside it, not a method of the class, unless it is an alternate constructor
  from an outside format (`fromPandas`), which is a classmethod.
- A base class holds a skeleton its subclasses inherit: fields, methods or checks. Classes that
  only an `isinstance` check groups are a union alias, not an empty base.

### Arrays and tensors

- Annotate a shape inline on the line that returns it (`# (b, n, d)` for b datasets of n rows
  and d columns), never in a docstring. Reshape with `einops` where the package depends on it,
  never mutate a reused array in place. Vectorize with broadcasting and advanced indexing, not
  loops; batch over datasets.
- One lowercase letter per counted quantity, in shape comments and in the variables that hold
  the count:

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
- Every draw takes an explicit generator or key; nothing reads global random state.
- A scalar that a config holds or an array computation reads is a Python `int` or `float`, its
  arithmetic `math`'s: a numpy float64 scalar would promote a float32 array to float64.

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

- The module docstring: one sentence on what the module defines, optionally a short paragraph
  on how, then the pipeline in run order, one `-> result` line per step with its function on the
  right. A module whose why is not obvious replaces the paragraph with a short pyramid for a
  reader new to the repository: what it offers and why, then how. Who imports the module is the
  [map](#modules)'s.
- `# ====` banners separate globals, classes and main, `# ----` banners subsections. Each is
  named in plain words, with one comment line on why the section exists where neither the name
  nor a class docstring below says it.
- The constants, aliases and data classes the steps use come first, each after any function it
  is built from. Then each public function or method, followed by the steps it calls in run
  order, under a `# ----` banner naming the pipeline or its stage (`url -> archive -> tree`). A
  step that two public functions call goes after its first caller.
- A script's `__main__` reads as the run's table of contents; for a pipeline class, it creates
  the class and calls its steps (`puller = Puller(); puller.getSurvset(); …; puller.flush()`).

## Comments and docstrings

- Comments start in lower case; docstrings are capitalized sentences. Break the lines of both at
  sentence or clause boundaries, not at the column limit.
- Short, plain sentences, one idea each, with a subject and a verb.
- A docstring only where the behavior or design is not obvious: one or two plain sentences that
  start with what the function does or returns and leave out what the code already shows. None
  on a method its name and body already explain, never a chain of `;` clauses. Detail goes into
  inline comments that say why.
- A comment names its source: a paper link, or the authors, year and equation number for a
  formula from a paper; a link for code or a method it follows.
- A config or data class's docstring introduces the concepts its fields name, a short bullet
  each. A field's detail goes into a short trailing comment, which every field whose name does
  not say what it is carries.
- A block of checks opens with one plain sentence on what it checks.
- A comment explains the code, not its history: the case behind a rule goes to the test that
  pins it, a definition to `CONTEXT.md`, how it was decided to an ADR. Code states its rules
  without citing an ADR; the ADR keeps the reason.
- An embedded script (an R snippet) gets a one-sentence comment above it, one statement per
  line.

## Tests

- Test first: the failing test, then the minimal code. `hypothesis` for property tests.
- The tests mirror the source: one `test_<module>.py` per module, each test named as a sentence
  (`test_<what_holds>`), fixtures in a `fixtures/` beside them.
- Each test says in a sentence or two why it exists, unless obvious. Bind the object under test,
  then call it; small named helpers, not lambdas or repeated comprehensions.
- A test asserts nothing its object checks itself: no shape its data class validates, no control
  that rebuilds what the call has just built.
- Cases of one check are one parametrized table, each case's reason in its id, not one test
  per case.
- Names listed in two modules (a `Literal`, a catalog, a table's columns) are checked against
  each other by a test; the code checks nothing at import.
- No coverage gate.

## Docs

Markdown meets the same bar as code: simple, and read top to bottom.

- Short bullets, the essentials first, detail in an appendix or a linked file.
- One home per fact, links everywhere else; a living rule sits with the method it governs.
  Counts live in generated files (a `SUMMARY.md`), never in prose.
- `README.md` is for users and fits one screen. Its links are absolute to `main`, so they render
  off GitHub too (on PyPI).

## Data

- Never overwrite raw data.
- A default, an edge or a share is read off the data collections and surveys already at hand
  (a registry, a practice survey) before it is guessed, and cites its witness. Where no witness
  exists and one pays off, gather the data first.
- Stream or lazily load data that would not fit in memory.
- Figures are PDFs, never PNG or JPG. Existing vector files convert to SVG; a PDF is never
  rasterized.

## Lab notes and experiments

- `lab/YYYY-MM-DD-slug/`: one small design choice, its notebook with outputs and any PDFs,
  indexed in `lab/README.md` with the decision. Lab notes are private: `lab/` is in
  `.gitignore`, and nothing imports from it.
- `experiments/`: rerunnable scripted studies, tracked and public. A lab note moves here once
  something public rests on it: a released behavior, an ADR's reason or a paper's number.
  Rerun with flags; never one-off.

## Notebooks

- The text speaks to a reader who knows the field but not the repository: no project
  vocabulary, and every abbreviation spelled out at first use.
- It holds only what the figures and the decision need. Implementation details, fixed bugs and
  caveats that leave the result unchanged belong in the code or the issue.
- Short plain sentences, one per line, with `<br>` between them in a paragraph. A bullet is a
  bold label and one fact; a second fact is a second bullet, never a `<br>`.
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
