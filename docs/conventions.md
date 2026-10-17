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

## Names

- American spelling (`standardized`, `behavior`, `catalog`), in names as in prose. A released
  name spelled otherwise is renamed in the next release; only a name an outside standard fixes
  (R's own names, a library's) keeps its spelling.
- A module is one lowercase word. A package's `__init__.py` lists its public names in `__all__`.
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

### Classes

- A class per concept a reader should hold apart (a format family, a distribution), static
  methods where it holds no state. A helper or constant used by one class lives on it; module
  functions are for what spans classes.
- A pipeline that holds state (a cache, a run directory) is a class; a pipeline without state is
  module functions.
- A data class holds its fields and the checks across them; the function that builds it is a
  module function beside it, not a method of the class.
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
  right. Who imports the module is the [map](#modules)'s.
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
- Stream or lazily load data that would not fit in memory.
- Figures are PDFs, never PNG or JPG. Existing vector files convert to SVG; a PDF is never
  rasterized.

## Lab notes and experiments

- `lab/YYYY-MM-DD-slug/`: one small design choice, its notebook with outputs and any PDFs,
  indexed in `lab/README.md` with the decision. Nothing imports from `lab/`.
- `experiments/`: rerunnable scripted studies that feed a paper. Rerun with flags; never
  one-off.
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
| `hyperprior` | how a field is a point or a distribution, its bounds, its resolution to points, a count truncated at another field's value, the presets, and which fields each column, node or grouping factor draws, by marginal family for a continuous column | every module |
| `column` | the design-column budget and the post-hoc types: each column's name, type, k and own parameters, what its metadata guarantees, and its values: a sample of its marginal family or its codes, placed by its node's ranks; the grouping factor as a column, its relation, size draw and codes, and `Grouping`, the unit arithmetic (a unit's chain, the broadcast between units) | graph, mechanism, draw |
| `graph` | the generating graph: the Cauchy DAG over latent and observed nodes, the block bonus, each node's unit and the mask on edges out of its chain, the order (the grouping factors, then the units superior first, the row last, latent before observed), each node's family, activation and scales, a row-unit node's group share and group factor, and what the graph guarantees | mechanism, draw |
| `mechanism` | how a node's values follow from its parents and its noise, at a unit: linear and MLP families, the parents read at the unit, relative noise, the z-score at the node's own unit, and a row-unit node's group part and row part mixed by its group share | draw |
| `draw` | the pipeline and its output: the key's generator, threaded through the modules in a fixed order, the grouping before the graph, nodes joined to columns by name, every column and latent node broadcast to the rows last, and what one draw returns, the grouping factors first | the package's `__init__` |

### Changing code

- [`CONTRIBUTING.md`](../CONTRIBUTING.md#pull-requests)'s three rules hold: the defaults are
  what the proofs certify, code is written from scratch, nothing comes from downstream.

### Code

- A default the hyperprior guesses is a round number; only a fitted one carries more digits.
- A draw is named before it is cast or transformed. The Bernoulli idiom `rng.random() < share`
  counts as one draw.
- Letters of its own: `v` counts nodes, `f` grouping factors, `e` the elements of a unit, `c`
  a node's parents (its direct causes) and `u` distinct values.
- One marker for a missing value: `None`, in a list of Python scalars, never NaN in an array.

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
  install, one quickstart box and its output, planned, learn more, contributing, citing,
  license.

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
