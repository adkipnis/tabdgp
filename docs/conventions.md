# Conventions

The rules a change is reviewed against. The words are [`CONTEXT.md`](../CONTEXT.md)'s; the
reasons behind the larger rules are in the [ADRs](adr).

## Tooling

- Python through `uv` only; `uv run` for everything, every command from the repository root,
  without `cd`.
- Python 3.12 and newer supported, developed on 3.14, CI on both.
  The floor rises only when a dependency or a language feature needs it,
  and never past the oldest version [SPEC 0](https://scientific-python.org/specs/spec-0000/) supports.
- `ruff format` and `pytest`, run by CI.
- Open an issue before adding a heavyweight dependency.

## Modules

Each module owns one design decision that no other module depends on. The table fills as the
modules arrive:

| module | owns | imported by |
|---|---|---|
| `hyperprior` | how a field is a point or a distribution, its bounds, and its resolution to points | every module |
| `dataset` | the output contract: what one draw returns and what each of its types guarantees on its own | every module |
| `column` | the design-column budget and the post-hoc types: each column's name, type and K, and its values from its node's ranks | draw |
| `graph` | the generating graph: the Cauchy DAG over latent and observed nodes, latent hubs first, the block bonus, the family per node | draw |
| `mechanism` | how a node's values follow from its parents and its noise: linear and MLP families, relative noise, the z-score | draw |
| `marginal` | the catalogue of continuous families: a sample of each, in its standard form | dataset, column |
| `draw` | the key and the pipeline: one generator per key, threaded through the modules in a fixed order, nodes joined to columns by name | the package's `__init__` |

## Design

After Ousterhout, *A Philosophy of Software Design*: complexity is anything that makes code
harder to understand or change; it accrues one small edit at a time.

- **Deep modules.** A small interface over much functionality; an interface as complex as its
  implementation folds into its caller or callee.
- **Information hiding.** Each module owns one design decision (a node mechanism, a tensor
  layout, the hyperprior's parameterisation). Split code by what it knows, not by the order
  it runs in.
- **No pass-throughs.** A method that forwards its arguments unchanged adds nothing; call the
  target directly.
- **Pull complexity down.** The module absorbs the awkward case so its callers stay simple. A
  fixed sensible behaviour beats a new config parameter.
- **Define errors out of existence.** Shape the interface so the invalid case cannot arise
  (accept the empty set, validate once at construction). The rest fails loudly: no fallbacks,
  no defensive `try`/`except`, no speculative generality. Branch only for cases observed on
  real inputs; add the next branch when use hits it.
- **Design it twice.** For a non-trivial interface, name two designs and why one wins before
  writing code.
- **Obvious over clever.** The simplest form a reader guesses right on first read, in code
  written, reviewed or rewritten. If 200 lines could be 50, rewrite them.

## Changing code

- The smallest change that solves the problem and meets the design bar, no unrequested
  extras. Leave adjacent code, comments and formatting untouched.
- Written from scratch: scamd and TabICL's prior are read for ideas, never copied. A mechanism
  one of them inspired names it in a comment. Where a formula comes from a paper, the comment
  cites it (a link, an equation number).
- A task that repeats, or is error-prone by hand, gets a `.py` or `.sh` script.

## Names

- A module is one lowercase word. A package's `__init__.py` lists its public names in
  `__all__`.
- `PascalCase` for classes and `Literal` aliases, `camelCase` for functions and methods,
  `snake_case` for variables, `UPPER_SNAKE_CASE` for constants.
- A directory is `dir`, not `folder`. A DataFrame is a `table`, several are `tables`, never
  `df` or `frame`.
- A leading `_` marks a pipeline step and anything no other module imports.
- Tests live in `tests/`, mirroring `src/`, one `test_<module>.py` per module, each test named
  as a sentence (`test_<what_holds>`), fixtures in a `fixtures/` beside them.

## Code

- Absolute imports: `from tabdgp.x import y`.
- Modern typing: `X | Y`, built-in generics, `from __future__ import annotations`.
  `Literal[...]` for an argument with a fixed set of options, a type alias once it repeats.
  Config models are Pydantic.
- f-strings, one statement per line, a long expression named on the line before.
- A method lists its steps in the order you would describe them; a step with its own `if`
  ladder or loop becomes a method. A search is a plain loop with an explicit check after it,
  not `for`/`else`, `while True` or `next()` on a generator.
- `a if c else b` only on a short line. Lookup tables are literal dicts or `if` chains, not
  built with comprehensions, `dict.fromkeys` or `|`.
- A class per concept a reader should hold apart (a mechanism family, a pipeline), static
  methods where it holds no state. A helper or constant used by one class lives on it; module
  functions are for what spans classes.
- Wrap `argparse` blocks in `# fmt: off` / `# fmt: on`.

### Arrays and tensors

- Annotate shapes inline (`# (B, N, p)` for B datasets of N rows and p columns), reshape with
  `einops` where the library has it, never mutate a reused array in place. Vectorise with
  broadcasting and advanced indexing, not loops; batch over datasets.
- Every draw takes an explicit generator or key; nothing reads global random state.

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
  named in plain words, with one comment line on why the section exists unless a class
  docstring below says it.
- One pipeline class per module: public methods first, then its steps in run order under a
  `# ----` banner naming the pipeline.
- A script's `__main__` creates its pipeline class and reads as the run's table of contents.

## Comments and docstrings

- Comments start in lower case; docstrings are capitalised sentences. Break the lines of both
  at sentence or clause boundaries, not at the column limit.
- A docstring only where the behaviour or design is not obvious: one or two plain sentences,
  none on a method its name and body already explain, never a chain of `;` clauses. Detail
  goes into inline comments that say why and cite sources, and into trailing comments on
  config fields.
- A comment explains the code, not its history: the case behind a rule goes to the test that
  pins it, a definition to `CONTEXT.md`, how it was decided to an ADR or a lab note.
- Code in `src/` states its rules without citing an ADR; the ADR keeps the reason.

## Tests

- Test first: the failing test, then the minimal code. `hypothesis` for property tests.
- Unit tests for every mechanism: marginal ranges, the coded shapes, determinism under a
  fixed generator, every override honoured, the flat case as zero factors.
- The proofs against mixedbench's registry (the two-sample test and the containment test)
  carry the `slow` mark and fixed seeds.
- Each test says in a sentence or two why it exists, unless obvious. Bind the object under
  test, then call it; small named helpers, not lambdas or repeated comprehensions.
- No coverage gate.

## Docs

Markdown meets the same bar as code: simple, and read top to bottom.

- Short bullets, the essentials first, detail in an appendix or a linked file.
- One home per fact, links everywhere else; a living rule sits with the method it governs.
  Counts live in generated files, never in prose.
- `README.md` is for users and fits one screen: badges, a short what and why, install, one
  quickstart box, contents, contributing, citing, licence. Its links are absolute to `main`,
  so PyPI renders them.

## Data

- The registry statistics the proofs read are mixedbench's, pinned by version, never edited
  here.
- Stream or lazily load data that would not fit in memory.
- Figures are PDFs, never rasterised.

## Lab notes and experiments

- `lab/YYYY-MM-DD-slug/`: one small design choice, its notebook committed with outputs and
  indexed in `lab/README.md` with the decision. Never imported.
- `experiments/`: rerunnable scripted studies that feed a paper. Rerun with flags; never
  one-off.
