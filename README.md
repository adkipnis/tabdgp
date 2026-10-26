[![CI](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/adkipnis/tabdgp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tabdgp)](https://pypi.org/project/tabdgp/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23322845.svg)](https://doi.org/10.5281/zenodo.23322845)

# TabDGP

A data-generating process (DGP) for grouped and flat tables.

TabDGP simulates tabular datasets from a random structural causal model (SCM) and aims to
reproduce the quirks of real data. Each table comes with the graph and the settings that
generated it.

> **Alpha.** Release 0.3 draws grouped and flat tables with an outcome `y`, from defaults that
> are round guesses; the rest is planned below. Before 1.0, a release may change the dataset a key draws.

## Highlights

- **Grouped or flat.** One to three grouping factors by default, nested or crossed, with 2 to
  400 groups each. A covariate varies by row or is constant within the groups of one factor.
  `Hyperprior.flat()` draws none.
- **White-box.** Each dataset carries its graph, latent nodes, column metadata and hyperparameters.
- **Reproducible.** A dataset is a pure function of its key (a seed and an index) and the
  hyperprior, whatever batch or worker draws it.
- **Model-free outcome.** One node `y` is drawn by default as the last node of the SCM: continuous,
  binary, categorical, ordered or count.
- **Model-based outcome.** Instead of the SCM's node, `outcome_form='glmm'` draws `y` from a
  generalized linear mixed model (GLMM) on the covariates. The family is Normal, Bernoulli or
  Poisson (or, by overriding `family_shares`, Student-t, negative binomial, latent-noise
  Bernoulli, Gamma or beta), with fixed effects and random effects per grouping factor. The true parameters are
  returned with the priors they were drawn from.
  `outcome_form='glm'` skips random effects, `Glmm(deviation_scale=0.3)` adds a known misspecification.
  You can pass your own model as a function of the covariates by defining it as a Callable.
- **Bring your own covariates.** `Covariates.fromPandas` reads a table, and `drawOutcome` draws a
  model's outcome on it. You can use this semi-synthetic data for oracle tests or power analysis.

## Install

```
pip install tabdgp
```

## Quickstart

```python
from tabdgp import Hyperprior, draw

hyperprior = Hyperprior.grouped(rows=100, design_columns=5, outcome_form='glmm')
dataset = draw(seed=0, index=0, hyperprior=hyperprior)  # the key: a seed and an index
unit = dict(zip(dataset.graph.names, dataset.graph.unit))  # per row, or per group of a factor
for name, column in dataset.metadata.items():
    values = dataset.columns[name][:3].round(2)
    print(f'{name:2} {column.type:11} per {unit[name]:3} {values}')
outcome = dataset.outcome
print(f'family: {outcome.family}')
print(f'fixed: {" + ".join(outcome.fixed_terms)}')
for factor, terms in outcome.random_terms.items():
    print(f'random per {factor}: {" + ".join(terms)}')
```

```text
g0 categorical per g0  [4 5 2]
g1 categorical per g1  [ 6 22 29]
g2 categorical per g2  [7 7 5]
x0 binary      per g0  [1 0 0]
x1 binary      per row [1 0 0]
x2 continuous  per row [ 0.77 -0.05 -1.55]
x3 binary      per row [1 0 0]
x4 binary      per row [1 0 0]
y  continuous  per row [ 0.39 -0.15 -0.7 ]
family: normal
fixed: intercept + x0 + x1 + x2 + x3 + x4
random per g0: intercept
random per g1: intercept
random per g2: intercept + x0 + x1 + x3
```

An outcome `y` can also be sampled based on a model and the table's covariates.
The built-in `Glmm()` model also returns the drawn latent parameters.

```python
import pandas as pd

from tabdgp import Covariates, Glmm, drawOutcome

table = pd.DataFrame(
    {
        'school': ['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'],  # the grouping factor
        'age': [11.2, 12.5, 10.8, 13.1, 11.9, 12.2, 10.5, 12.8, 11.4],  # floats: continuous
        'female': [1, 0, 1, 1, 0, 0, 1, 0, 1],  # two values: binary
        'books': [3, 10, 0, 25, 7, 2, 14, 5, 8],  # integers: a count
    }
)
covariates = Covariates.fromPandas(table, factors=['school'])  # the types read from the dtypes
outcome = drawOutcome(seed=0, index=0, covariates=covariates, model=Glmm())
print(f'family: {outcome.family}')
for term, beta in zip(outcome.fixed_terms, outcome.beta):
    print(f'{term:9} {beta:5.2f}')
```

```text
family: poisson
intercept  0.66
age        1.05
female    -1.15
books     -0.24
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
