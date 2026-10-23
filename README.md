[![CI](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tabdgp)](https://pypi.org/project/tabdgp/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23322845.svg)](https://doi.org/10.5281/zenodo.23322845)

# TabDGP

A data-generating process (DGP) for grouped and flat tables.

TabDGP simulates tabular datasets from a random structural causal model (SCM) and aims to
reproduce the quirks of real data. Each table comes with the graph and the settings that
generated it.

> **Alpha.** Release 0.2 draws grouped and flat tables from defaults that are round guesses;
> the rest is planned below. Before 1.0, a release may change the dataset a key draws.

## Highlights

- **Grouped or flat.** One to three grouping factors by default, nested or crossed, with 2 to
  400 groups each. A covariate varies by row or is constant within the groups of one factor.
  `Hyperprior.flat()` draws none.
- **An outcome.** One node `y`, drawn by default as the last node of the SCM: continuous,
  binary, categorical, ordered or count, with its own noise scale. `outcome_form='none'` draws none.
- **White-box.** Each dataset carries its graph, latent nodes, column metadata and hyperparameters.
- **Reproducible.** A dataset is a pure function of its key (a seed and an index) and the
  hyperprior, whatever batch or worker draws it.
- **Overridable.** Replace any default distribution by a point value or another distribution.

## Install

```
pip install tabdgp
```

## Quickstart

```python
from tabdgp import Hyperprior, draw

hyperprior = Hyperprior.grouped(rows=100, design_columns=5)  # two overrides on the grouped preset
dataset = draw(seed=0, index=0, hyperprior=hyperprior)  # the key: a seed and an index
unit = dict(zip(dataset.graph.names, dataset.graph.unit))  # per row, or per group of a factor
for name, column in dataset.metadata.items():
    print(name, column.type, unit[name], dataset.columns[name][:3].round(2))
print(len(dataset.latent), 'latent nodes,', dataset.graph.parents.sum(), 'edges')
```

```text
g0 categorical g0 [4 5 2]
g1 categorical g1 [ 6 22 29]
g2 categorical g2 [7 7 5]
x0 binary row [1 0 1]
x1 binary row [0 1 0]
x2 continuous g2 [1.77 1.77 2.66]
x3 binary row [0 0 1]
x4 binary g2 [0 0 1]
y continuous row [0.76 0.62 0.93]
7 latent nodes, 18 edges
```

## Planned

- **Quirk axes:** group confounding, rare levels, ties, bounds, near dependence and more.
- **Matched defaults**, fitted to and checked against
  [mixedbench](https://github.com/adkipnis/mixedbench)'s real grouped tables, then widened.
- **The outcome as a generalized linear mixed model**, through a preset, with its parameters
  and priors returned, exact or with a drawn deviation for white-box misspecified datasets.
- **Within-group time and a binary treatment.**

## Contributing

Issues and pull requests are welcome; see
[`CONTRIBUTING.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTRIBUTING.md).


## License

[Apache-2.0](https://github.com/adkipnis/tabdgp/blob/main/LICENSE).
