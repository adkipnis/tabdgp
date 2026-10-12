[![CI](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tabdgp)](https://pypi.org/project/tabdgp/)
[![Python](https://img.shields.io/pypi/pyversions/tabdgp)](https://github.com/adkipnis/tabdgp/blob/main/pyproject.toml)
[![Licence](https://img.shields.io/badge/licence-MIT-blue)](https://github.com/adkipnis/tabdgp/blob/main/LICENSE)

# TabDGP

A data-generating process (DGP) for grouped and flat tables.

TabDGP simulates tables of typed covariates from a random structural causal model (SCM). Each
table comes with the graph and the settings that generated it: the ground truth for a simulation
study, or for a model trained on simulated tables. Generators such as
[SDV](https://github.com/sdv-dev/SDV) learn to imitate real data, and priors such as
[TabPFN](https://github.com/PriorLabs/TabPFN)'s draw independent rows; TabDGP aims to draw
grouped tables with the quirks of real ones.

> **Alpha.** Release 0.1 draws flat tables from defaults that are round guesses; the rest follows
> the [roadmap](https://github.com/adkipnis/tabdgp/blob/main/ROADMAP.md). Before 1.0, a release
> may change the dataset a key draws.

## Highlights

- **White-box.** Each dataset carries its graph, latent nodes, column metadata and hyperparameters.
- **Reproducible.** A dataset is a pure function of its key, a seed and an index, whatever batch
  or worker draws it.
- **Overridable.** Replace any default distribution by a point value or another distribution.
- **Covariates only.** Continuous, binary, categorical and ordered columns; the outcome is yours.

## Install

```
pip install tabdgp
```

## Quickstart

```python
from tabdgp import Hyperprior, draw

hyperprior = Hyperprior(rows=100, design_columns=5)  # two overrides, the rest keep their defaults
dataset = draw(seed=0, index=0, hyperprior=hyperprior)  # a pure function of its key
for name, column in dataset.metadata.items():
    print(name, column.type, dataset.columns[name][:3].round(2))
print(len(dataset.latent), 'latent nodes,', dataset.graph.parents.sum(), 'edges')
```

```text
x0 binary [1 1 1]
x1 binary [0 1 0]
x2 categorical [0 0 0]
x3 continuous [0.25 0.42 0.1 ]
7 latent nodes, 8 edges
```

## Planned

- **Grouping factors**, with group-level and within-group covariates, time and treatment.
- **Quirk axes:** group confounding, rare levels, ties, bounds, near dependence and more.
- **Matched defaults**, fitted to [mixedbench](https://github.com/adkipnis/mixedbench)'s real
  grouped tables, then widened.
- **An outcome node**, optional, for white-box misspecified datasets.

## Learn more

[`CONTEXT.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTEXT.md) defines the vocabulary,
[`docs/adr/`](https://github.com/adkipnis/tabdgp/tree/main/docs/adr) records the design
decisions and [`ROADMAP.md`](https://github.com/adkipnis/tabdgp/blob/main/ROADMAP.md) gives the
build order.

## Contributing

Issues and pull requests are welcome; see
[`CONTRIBUTING.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTRIBUTING.md).

## Citing

Please cite the release you used, from
[`CITATION.cff`](https://github.com/adkipnis/tabdgp/blob/main/CITATION.cff) or GitHub's "Cite
this repository".

## Licence

[MIT](https://github.com/adkipnis/tabdgp/blob/main/LICENSE).
