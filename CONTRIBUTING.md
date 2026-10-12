# Contributing to TabDGP

Thank you for your interest. TabDGP is early stage and help is welcome.

## Ways to help

- **A real-data quirk the generator misses.** Open an issue naming the property, a public
  table that shows it and, if you can, how to measure it per column or per grouping factor;
  the quirk catalogue grows through the measurement pass, never by hand.
- **A bug in a mechanism or a proof.** Open an issue with the smallest example that shows it,
  or a pull request with a failing test and the fix.
- **A use case.** A model family or a simulation study that needs something the hyperprior
  cannot express: open an issue describing the dataset you cannot draw.

For anything bigger than a small fix, open an issue before writing code, so we can agree on
the approach first.

## Set up

You need [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/adkipnis/tabdgp
cd tabdgp
uv sync
uv run pytest               # the fast tests
uv run pytest -m slow       # the proofs against mixedbench's registry, from roadmap item 3
```

## Find your way

```
src/tabdgp/     the package; each module's docstring says what it defines and how
tests/          mirrors src/, fixtures beside the tests
docs/adr/       the design decisions and why they were taken
docs/           the conventions, and from roadmap item 3 the proofs' method
```

[`CONTEXT.md`](CONTEXT.md) defines the words (node, mechanism, hyperprior, override, quirk
axis, outcome node) and the covariate terms it shares with metabeta's glossary.

## Pull requests

- Branch off `main`, one change per pull request, linked to its issue.
- Add a test with each change; mark a test that draws thousands of datasets `slow`.
- Run `uv run ruff format .` and `uv run pytest`. CI runs both.
- Please follow the style [conventions](docs/conventions.md).

Three rules keep the generator trustworthy:

- **The defaults are what the proofs certify.** A change to a default distribution of the
  hyperprior, to a proof's seed or to its bar is discussed in an issue first and recorded in
  the release notes; downstream training corpora depend on them.
- **Written from scratch.** scamd and TabICL's prior are read for ideas; no code is copied
  from either. A mechanism they inspired says so in a comment.
- **Nothing from downstream.** TabDGP imports nothing from metabeta or mixedbench; the proofs
  read mixedbench's registry as a test dependency only.

## Licence

By contributing, you agree that your changes are released under [MIT](LICENSE).
