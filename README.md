[![CI](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tabdgp)](https://pypi.org/project/tabdgp/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23322845.svg)](https://doi.org/10.5281/zenodo.23322845)

# TabDGP

A data-generating process (DGP) for grouped and flat tables.

TabDGP simulates tabular datasets from a random structural causal model (SCM) and aims to
reproduce the quirks of real data. Each table comes with the graph and the settings that
generated it.

> **Alpha.** Release 0.4 draws grouped and flat tables with an outcome `y`, from the SCM or from a
> model's outcome, from defaults that are round guesses; the rest is planned below. Before 1.0, a
> release may change the dataset a key draws.

## Highlights

- **Grouped or flat.** One to three grouping factors by default, nested or crossed, with 2 to
  400 groups each. A covariate varies by row or is constant within the groups of one factor.
  `Scm.flat()` draws none.
- **White-box.** Each dataset carries its graph, latent nodes, column metadata and hyperparameters.
- **Reproducible.** A dataset is a pure function of its key (a seed and an index) and the
  hyperprior, whatever batch or worker draws it.
- **Model-free outcome.** One node `y` is drawn by default as the last node of the SCM: continuous,
  binary, categorical, ordered or count.
- **Model-based outcome.** Instead of the SCM's node, `modelOutcome` draws `y` from a
  generalized linear mixed model (GLMM) on the covariates (e.g. `Glmm(family='bernoulli')`).
  Supported likelihoods are Gaussian, Bernoulli, Poisson, Student-t, negative binomial,
  latent-noise Bernoulli, Gamma, and Beta.
  The true parameters are returned with the priors they were drawn from.
  `Glm` skips random effects,
  `Glmm(family='gaussian', deviation_scale=0.3)` adds a known misspecification.
  You can also pass your own model as a function of the covariates by defining it as a Callable.
- **Bring your own covariates.** `Covariates.fromPandas` reads a table, and `modelOutcome`
  draws a model's outcome on it.
  You can use this semi-synthetic data for oracle tests or power analysis.

## Install

```
pip install tabdgp
```

## Quickstart

```python
from tabdgp import Glmm, Scm, drawDataset, modelOutcome

scm = Scm.grouped(rows=100, design_columns=5)
dataset = drawDataset(seed=0, index=0, scm=scm)  # the key: a seed and an index
unit = dict(zip(dataset.graph.names, dataset.graph.unit))  # per row, or per group of a factor
for name, column in dataset.metadata.items():
    values = dataset.columns[name][:3].round(2)
    print(f'{name:2} {column.type:11} per {unit[name]:3} {values}')
glmm = Glmm(family='gaussian')
outcome = modelOutcome(seed=0, index=0, covariates=dataset.covariates, model=glmm)
print(f'fixed: {" + ".join(outcome.fixed_terms)}')
for factor, terms in outcome.random_terms.items():
    print(f'random per {factor}: {" + ".join(terms)}')
```

```text
g0 categorical per g0  [4 5 2]
g1 categorical per g1  [ 6 22 29]
g2 categorical per g2  [7 7 5]
x0 binary      per row [1 0 1]
x1 binary      per row [0 1 0]
x2 continuous  per g2  [1.77 1.77 2.66]
x3 binary      per row [0 0 1]
x4 binary      per g2  [0 0 1]
y  continuous  per row [0.76 0.62 0.93]
fixed: intercept + x0 + x1 + x2 + x3 + x4
random per g0: intercept + x1 + x3 + x4
random per g1: intercept + x0 + x1 + x2 + x3 + x4
random per g2: intercept
```

An outcome `y` can also be sampled based on a model and the table's covariates.
The built-in `Glmm` model also returns the drawn latent parameters.

```python
import pandas as pd

from tabdgp import Covariates, Glmm, modelOutcome

table = pd.DataFrame(
    {
        'school': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],  # the grouping factor
        'age': [11.2, 12.5, 10.8, 13.1, 11.9, 12.2, 10.5, 12.8, 11.4],  # floats: continuous
        'female': [1, 0, 1, 1, 0, 0, 1, 0, 1],  # two values: binary
        'books': [3, 10, 0, 25, 7, 2, 14, 5, 8],  # integers: a count
    }
)
covariates = Covariates.fromPandas(table, factors=['school'])
outcome = modelOutcome(seed=0, index=0, covariates=covariates, model=Glmm(family='gaussian'))
for term, beta in zip(outcome.fixed_terms, outcome.parameters.beta):
    print(f'{term:9} {beta:5.2f}')
```

The true fixed effects:
```text
intercept  2.77
age        0.04
female     0.08
books      0.27
```

## Planned

- **Quirk axes:** group confounding, rare levels, ties, bounds, near dependence and more.
- **Finetuning** the hyperprior's parameters to any batch of datasets, scored by
  `tabdgp.fidelity` (`tabdgp[fidelity]`).
  Group presets are tuned against [mixedbench](https://github.com/adkipnis/mixedbench).
- **Within-group time and a binary treatment.**

## Contributing

Issues and pull requests are welcome; see
[`CONTRIBUTING.md`](https://github.com/adkipnis/tabdgp/blob/main/CONTRIBUTING.md).

## License

[Apache-2.0](https://github.com/adkipnis/tabdgp/blob/main/LICENSE).
