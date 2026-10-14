[![CI](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tabdgp)](https://pypi.org/project/tabdgp/)
[![Python](https://img.shields.io/pypi/pyversions/tabdgp)](https://github.com/adkipnis/tabdgp/blob/main/pyproject.toml)
[![Licence](https://img.shields.io/badge/licence-MIT-blue)](https://github.com/adkipnis/tabdgp/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23322845.svg)](https://doi.org/10.5281/zenodo.23322845)

# TabDGP

A data-generating process (DGP) for grouped and flat tables.

TabDGP simulates tabular datasets from a random structural causal model (SCM) and aims to
reproduce the quirks of real data. Each table comes with the graph and the settings that
generated it.

> **Alpha.** Release 0.1 draws flat tables from defaults that are round guesses; the rest is
> planned below. Before 1.0, a release may change the dataset a key draws.

## Highlights

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

hyperprior = Hyperprior(rows=100, design_columns=5)  # two overrides, the rest keep their defaults
dataset = draw(seed=0, index=0, hyperprior=hyperprior)  # the key: a seed and an index
for name, column in dataset.metadata.items():
    print(name, column.type, dataset.columns[name][:3].round(2))
print(len(dataset.latent), 'latent nodes,', dataset.graph.parents.sum(), 'edges')
```

```text
x0 categorical [4 4 4]
x1 binary [1 0 0]
7 latent nodes, 12 edges
```

## Planned

- **Grouping factors**, with group-level and within-group covariates, time and treatment.
- **Quirk axes:** group confounding, rare levels, ties, bounds, near dependence and more.
- **Matched defaults**, fitted to [mixedbench](https://github.com/adkipnis/mixedbench)'s real
  grouped tables, then widened.
- **An outcome node**, optional, for white-box misspecified datasets.

## Learn more

[`CONTEXT.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTEXT.md) defines the vocabulary.

## Contributing

Issues and pull requests are welcome; see
[`CONTRIBUTING.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTRIBUTING.md).

## Citing

Please cite the release you used, from
[`CITATION.cff`](https://github.com/adkipnis/tabdgp/blob/main/CITATION.cff) or GitHub's "Cite
this repository".

## Licence

[MIT](https://github.com/adkipnis/tabdgp/blob/main/LICENSE).
