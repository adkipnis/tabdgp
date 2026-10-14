# Conventions

The rules a change is reviewed against. The words are [`CONTEXT.md`](../CONTEXT.md)'s; the
reasons behind the larger rules are in the [ADRs](adr).

## Tooling

- Python through `uv` only; `uv run` for everything, every command from the repository root,
  without `cd`.
- Python 3.12 and newer, developed on 3.14, CI on both. The floor rises only when a
  dependency or a language feature needs it, never past the oldest version
  [SPEC 0](https://scientific-python.org/specs/spec-0000/) supports.
- `ruff format` and `pytest`, run by CI. Wrap `argparse` blocks in `# fmt: off` / `# fmt: on`.
- Open an issue before adding a heavyweight dependency.

## Modules

Each module owns one design decision, hidden from the others:

| module | owns | imported by |
|---|---|---|
| `hyperprior` | how a field is a point or a distribution, its bounds, its resolution to points, a count truncated at another field's value, the presets, and which fields each column, node or grouping factor draws, by marginal family for a continuous column | every module |
| `column` | the design-column budget and the post-hoc types: each column's name, type, K and own parameters, what its metadata guarantees, and its values: a sample of its marginal family or its codes, placed by its node's ranks; the grouping factor as a column, its relation, size draw and codes, and `Grouping`, the unit arithmetic (a unit's chain, the broadcast between units) | graph, mechanism, draw |
| `graph` | the generating graph: the Cauchy DAG over latent and observed nodes, the block bonus, each node's unit and the mask on edges out of its chain, the order (the grouping factors, then the units superior first, the row last, latent before observed), each node's family, activation and scales, a row-unit node's group share and group factor, and what the graph guarantees | mechanism, draw |
| `mechanism` | how a node's values follow from its parents and its noise, at a unit: linear and MLP families, the parents read at the unit, relative noise, the z-score at the node's own unit, and a row-unit node's group part and row part mixed by its group share | draw |
| `draw` | the pipeline and its output: the key's generator, threaded through the modules in a fixed order, the grouping before the graph, nodes joined to columns by name, every column and latent node broadcast to the rows last, and what one draw returns, the grouping factors first | the package's `__init__` |

## Design

After Ousterhout, *A Philosophy of Software Design*: complexity is anything that makes code
harder to understand or change; it accrues one small edit at a time.

- **Deep modules.** A small interface over much functionality; an interface as complex as its
  implementation folds into its caller or callee.
- **Information hiding.** Split code by what it knows (a node mechanism, a tensor layout, the
  hyperprior's parameterisation), not by the order it runs in.
- **No pass-throughs.** A method that forwards its arguments unchanged adds nothing; call the
  target directly.
- **Pull complexity down.** The module absorbs the awkward case so its callers stay simple. A
  fixed sensible behaviour beats a new config parameter.
- **Define errors out of existence.** Shape the interface so the invalid case cannot arise
  (accept the empty set, validate once at construction). The rest fails loudly: no fallbacks,
  no defensive `try`/`except`. Branch only for a case observed on real inputs, never
  speculatively.
- **Design it twice.** For a non-trivial interface, name two designs and why one wins before
  writing code.
- **Obvious over clever.** The simplest form a reader guesses right on first read, in code
  written, reviewed or rewritten. If 200 lines could be 50, rewrite them.

## Changing code

- The smallest change that solves the problem and meets the design bar, no unrequested
  extras. Leave adjacent code, comments and formatting untouched.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md#pull-requests)'s three rules hold: the defaults are
  what the proofs certify, code is written from scratch, nothing comes from downstream.
- A task that repeats, or is error-prone by hand, gets a `.py` or `.sh` script.

## Names

- A module is one lowercase word. A package's `__init__.py` lists its public names in
  `__all__`.
- `PascalCase` for classes and `Literal` aliases, `camelCase` for functions and methods,
  `snake_case` for variables, `UPPER_SNAKE_CASE` for constants.
- A config field that counts is a plural noun (`rows`, `latent_nodes`), without an `n_` prefix.
- A function in `src/` is named for what it does, a verb and its object (`_resolveColumn`),
  not for what it returns (`_column`). A test helper is read inside an assert, so it may name
  what it returns (`_density`).
- One thing at different scopes keeps one name. Methods share its verb and each names its
  scope (`resolveDataset` for every field drawn once per dataset, `resolveField` for one); a
  counted quantity keeps its noun (the budget's `design_columns` and a column's).
- A directory is `dir`, not `folder`. A DataFrame is a `table`, several are `tables`, never
  `df` or `frame`.
- A leading `_` marks a pipeline step and anything no other module imports.

## Code

### Imports and types

- Absolute imports: `from tabdgp.x import y`.
- Import a library's names when they are unmistakable (`BaseModel`), rename one that shadows a
  standard-library name (`dataclass as pydantic_dataclass`), and alias a module only by its
  community's convention (`np`, `pd`).
- Modern typing: `X | Y`, built-in generics, `from __future__ import annotations`.
  `Literal[...]` for an argument with a fixed set of options, a type alias once it repeats.
  Config models are Pydantic: a check a type can state is its constraint (`PositiveFloat`,
  `Field(ge=3)`), and `__post_init__` holds only the checks across fields.
- A default the hyperprior guesses is a round number; only a fitted one carries more digits.

### Control flow

- f-strings, one statement per line, a long expression named on the line before.
- A method lists its steps in the order you would describe them; a step with its own `if`
  ladder or loop becomes a method. A search is a plain loop with an explicit check after it,
  not `for`/`else`, `while True` or `next()` on a generator. A loop fills its result as it
  goes; a second loop that only repackages the first one's output folds into it.
- `a if c else b` only on a short line. Lookup tables are literal dicts or `if` chains, not
  built with comprehensions, `dict.fromkeys` or `|`. Cases that read different arguments are a
  `match` over their `Literal`, not a dict of wrappers that share one signature.

### Classes

- A class per concept a reader should hold apart (a distribution, a column's metadata), static
  methods where it holds no state. A helper or constant used by one class lives on it; module
  functions are for what spans classes.
- A data class holds its fields and the checks across them; the function that draws it is a
  module function beside it (`plan`, `connect`, `draw`), not a method of the class.
- A base class holds a skeleton its subclasses inherit: fields, methods or checks. Classes
  that only an `isinstance` check groups are a union alias, not an empty base.

### Arrays and tensors

- Annotate shapes inline (`# (B, n, p)` for B datasets of n rows and p columns), reshape with
  `einops` where the library has it, never mutate a reused array in place. Vectorise with
  broadcasting and advanced indexing, not loops; batch over datasets.
- One letter per quantity, whatever its case: `n` counts rows, in code and in shape comments
  alike, and `d` counts nodes.
- Every draw takes an explicit generator or key; nothing reads global random state.
- A scalar the config draws or a mechanism reads is a Python `int` or `float`, its arithmetic
  `math`'s: a numpy float64 scalar would promote float32 arrays to float64.
- One marker for a missing value: `None`, in a list of Python scalars, never NaN in an array.

## File layout

```python
"""grouping.py defines how a dataset's grouping factors and group sizes are drawn.

pipeline:
---------
the overrides and the hyperprior
-> the number of factors and their relation          _factors
-> groups per factor and the group sizes             _sizes
-> one assignment array per factor                   draw
"""

# =====================================================================================
# default hyperprior
# =====================================================================================
# the distributions every quantity falls back to when no override names one.
```

- The module docstring: one sentence on what the module defines, optionally a short paragraph
  on how, then the pipeline, one `-> result` line per step with its function on the right.
  Who imports the module is the [map](#modules)'s.
- `# ====` banners separate globals, classes and main, `# ----` banners subsections. Each is
  named in plain words, with one comment line on why the section exists where neither the
  name nor a class docstring below says it.
- Functions in the order of the docstring's pipeline, public or not, so the pipeline is the
  module's table of contents. A function that runs other steps follows them, as the step that
  returns their result. The constants, aliases and data classes the steps use come first.
- A script's `__main__` reads as the run's table of contents.

## Comments and docstrings

- Comments start in lower case; docstrings are capitalised sentences. Break the lines of both
  at sentence or clause boundaries, not at the column limit.
- A docstring only where the behaviour or design is not obvious: one or two plain sentences,
  none on a method its name and body already explain, never a chain of `;` clauses. Detail
  goes into inline comments that say why.
- A comment names its source: scamd or TabICL's prior for a mechanism it inspired, the authors,
  year and equation number for a formula from a paper.
- A config or data class's docstring introduces the concepts its fields name, a short bullet
  each. A field whose name does not say what it is ends in a short trailing comment.
- A block of checks opens with one plain sentence on what it checks.
- A comment explains the code, not its history: the case behind a rule goes to the test that
  pins it, a definition to `CONTEXT.md`, how it was decided to an ADR. Code in
  `src/` cites no ADR and names no consumer (metabeta, metasurv).

## Tests

- Test first: the failing test, then the minimal code. `hypothesis` for property tests.
- `tests/` mirrors `src/`: one `test_<module>.py` per module, each test named as a sentence
  (`test_<what_holds>`), fixtures in a `fixtures/` beside them.
- Unit tests for every mechanism: marginal ranges, the coded shapes, determinism under a
  fixed generator, every override honoured, the flat case as zero factors.
- The proofs against mixedbench's registry (the two-sample test and the containment test)
  carry the `slow` mark and fixed seeds.
- Each test says in a sentence or two why it exists, unless obvious. Bind the object under
  test, then call it; small named helpers, not lambdas or repeated comprehensions.
- Names listed in two modules (a `Literal`, a catalogue, a shares field's categories, the
  per-column or per-node fields) are checked against each other by a test; `src/` checks
  nothing at import. Names listed twice in one module are checked by a test, if at all.
- No coverage gate.

## Docs

Markdown meets the same bar as code: simple, and read top to bottom.

- The essentials first, detail in an appendix or a linked file.
- One home per fact, links everywhere else; a living rule sits with the method it governs.
  Counts live in generated files, never in prose.
- `README.md` is for users and fits one screen: badges, a tagline, a short what and why,
  highlights of what the release does, install, one quickstart box and its output, planned,
  learn more, contributing, citing, licence. Its links are absolute to `main`, so PyPI renders
  them.

### Explaining the design

How the docs and the paper explain TabDGP to a reader with only a vague idea of it:

- Top down, one level before the next: why it exists, the concepts it needs, the blueprint
  (the modules and how they work together), then the detail.
- Short plain sentences, one idea per bullet; a term defined before it is used, in
  `CONTEXT.md`'s words; an acronym spelled out at its first use.
- An abstract mechanism gets a worked example small enough to check by eye (three values
  through rank matching).
- A design choice names the alternatives considered, the verdict and why, as a table when
  there are several, each citing its source: an ADR, the spec, a row of the audit.
- A choice without a recorded reason says so ("a guess, fitted in item 4") instead of
  inventing one; an opinion is marked as one, apart from the facts.
- What is left out says where it lives: a later roadmap item, an appendix.

## Data

- The registry statistics the proofs read are mixedbench's, pinned by version, never edited
  here.
- Stream or lazily load data that would not fit in memory.
- Figures are PDFs, never rasterised.

## Experiments

- `experiments/`: rerunnable scripted studies that feed a paper. Rerun with flags; never
  one-off.
