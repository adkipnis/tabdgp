[![CI](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml)
[![Licence](https://img.shields.io/badge/licence-MIT-blue)](https://github.com/adkipnis/tabdgp/blob/main/LICENSE)

# TabDGP

A data-generating process (DGP) for grouped and flat tables: grouping structure and typed
covariates with the quirks of real data.

## What and why

- **Measured against real tables.** The default hyperprior is matched to
  [mixedbench](https://github.com/adkipnis/mixedbench)'s registry of real grouped tables, then
  widened; two tests prove it: generated tables are indistinguishable from real ones on their
  summary statistics, and every real table lies inside the generated support.
- **White-box.** Every dataset comes with its generating graph, its latent nodes and its column
  metadata (type, levels, role).
- **Composable.** One structural causal model whose nodes are grouping factors, group-level and
  within-group covariates, time and treatment; a flat table is the case without grouping
  factors.
- **The quirks i.i.d. simulators lack.** Group confounding, empty cells, rare levels, ties,
  bounds, irregular time, near dependence, leverage points: a catalogue, each quirk measured on
  real tables before it gets an axis.
- **Covariates only.** The outcome belongs to whoever fits a model; an optional outcome node
  builds white-box misspecified datasets.
- **Overridable.** Every quantity has a default distribution; pass a point value or another
  distribution for any of them.

## Install

Not yet released. From source:

```
git clone https://github.com/adkipnis/tabdgp
cd tabdgp
uv sync
```

## Quickstart

The first call arrives with roadmap item 1; see [`ROADMAP.md`](https://github.com/adkipnis/tabdgp/blob/main/ROADMAP.md).

## Contents

- The package `tabdgp` under `src/`.
- [`CONTEXT.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTEXT.md): the vocabulary;
  [`docs/adr/`](https://github.com/adkipnis/tabdgp/tree/main/docs/adr): the design decisions.

## Contributing

Interested in becoming part of TabDGP? Check out [`CONTRIBUTING.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTRIBUTING.md).

## Licence

[MIT](https://github.com/adkipnis/tabdgp/blob/main/LICENSE).
