[![CI](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tabdgp)](https://pypi.org/project/tabdgp/)
[![Python](https://img.shields.io/pypi/pyversions/tabdgp)](https://github.com/adkipnis/tabdgp/blob/main/pyproject.toml)
[![Licence](https://img.shields.io/badge/licence-MIT-blue)](https://github.com/adkipnis/tabdgp/blob/main/LICENSE)

# TabDGP

A data-generating process (DGP) for grouped and flat tables: grouping structure and typed
covariates with the quirks of real data.

**Alpha.** Release 0.1 draws flat tables. Grouping factors, the quirk axes, the checks against
real tables and the outcome node follow the
[roadmap](https://github.com/adkipnis/tabdgp/blob/main/ROADMAP.md). Before 1.0, a release may
change the dataset a key draws.

## What and why

- **White-box.** Every dataset comes with its generating graph, its latent nodes, its column
  metadata (type, levels, role) and the hyperparameters it was drawn with.
- **Reproducible.** A dataset is a pure function of its key, a seed and an index: the same
  bits whatever batch or worker draws it.
- **Overridable.** Every quantity has a default distribution; pass a point value or another
  distribution for any of them.
- **Composable.** One structural causal model (SCM) whose nodes are typed columns and latent
  nodes. Grouping factors, group-level and within-group covariates, time and treatment are to
  join it as nodes (planned); a flat table is the case without grouping factors.
- **Measured against real tables** (planned). The default hyperprior is to be matched to
  [mixedbench](https://github.com/adkipnis/mixedbench)'s registry of real grouped tables, then
  widened. Two consistency checks on summary statistics will hold it there: generated tables
  cannot be told from real ones, and every real table lies inside the generated support. The
  0.1 defaults are round guesses.
- **The quirks i.i.d. simulators lack** (planned). Independent and identically distributed
  (i.i.d.) simulators miss group confounding, empty cells, rare levels, ties, bounds,
  irregular time, near dependence and leverage points. Each quirk gets an axis once it is
  measured on real tables.
- **Covariates only.** The outcome belongs to whoever fits a model; an optional outcome node
  (planned) builds white-box misspecified datasets.

## Install

```
pip install tabdgp
```

From source:

```
git clone https://github.com/adkipnis/tabdgp
cd tabdgp
uv sync
```

## Quickstart

```python
from tabdgp import draw

dataset = draw(seed=0, index=0)  # a pure function of its key
for name, column in dataset.metadata.items():
    print(name, column.type, column.levels, column.marginal, dataset.columns[name][:3])
```

## Contents

- The package `tabdgp` under `src/`.
- [`CONTEXT.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTEXT.md): the vocabulary;
  [`docs/adr/`](https://github.com/adkipnis/tabdgp/tree/main/docs/adr): the design decisions;
  [`ROADMAP.md`](https://github.com/adkipnis/tabdgp/blob/main/ROADMAP.md): the build order.

## Contributing

Interested in becoming part of TabDGP? Check out [`CONTRIBUTING.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTRIBUTING.md).

## Citing

Please cite the release you used: [`CITATION.cff`](https://github.com/adkipnis/tabdgp/blob/main/CITATION.cff),
or click "Cite this repository" on GitHub.

## Licence

[MIT](https://github.com/adkipnis/tabdgp/blob/main/LICENSE).
