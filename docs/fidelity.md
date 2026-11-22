# Fidelity checks

Slow tests compare the tables a preset draws with real tables of its kind, so a change to a
default shows whether the draws move toward the real tables or away from them.
They are consistency checks on statistics we chose, not a claim of realism, and a downstream
ablation remains the final test ([ADR 0008](adr/0008-summary-vectors-compared-on-ranks.md)).
They check the flat preset so far, and run by `uv run pytest -m slow` in a few minutes on a laptop.
[ADR 0009](adr/0009-a-schema-proof-and-a-content-proof.md) gives the design and its alternatives.

## Real tables

- **Source:** the flat sample of mixedbench 1.0.15.
- **Measure:** one summary vector per table by `tabdgp.summary`, for real and drawn tables alike.
- **Split:** half the sources to fit the hyperprior on and half to check it, by a fixed seed.
- **Checked:** the tables of the second half whose sizes the defaults draw.

[`flat.py`](../experiments/fidelity/flat.py) measures the real tables once and commits their
vectors, so the tests run without mixedbench.

## Schema

A table's schema is its sizes, each column's type and its number of levels.

- **Sizes:** a limit fails if more than 2 % of the real tables lie beyond it.
- **Type mix and levels:** an axis fails if its distance to 1,000 drawn schemas exceeds 0.1 on
  the pooled ranks.

## Content

For each real table, TabDGP draws 39 tables at its schema and compares the real table with them
on every axis.

Recall asks whether the real tables stand out among their draws.
A real table lies outside on an axis if it lies below or above all of its draws, which happens
to 2 in 40 tables if each real table is just one more draw.
An axis fails if more tables lie outside than the quantile 1 - 0.01/p of Binomial(t, 0.05), for
t tables and p axes.

Precision asks whether the draws stray from every real table.
A table's reach is its mean distance to the 5 nearest real tables of other sources, with each
axis ranked among them.
A draw counts if its reach is at most its real table's.
The share of draws that count has to be at least 0.25, and a calibrated generator scores 0.5.

## Controls

- **Negative:** independent covariates (`dependence='none'`) fail recall on every relation
  between columns, or the gates are too loose.
- **Positive:** a drawn table in a real table's place passes every gate, or the gates are too
  strict.

While the defaults fail the relations too, the negative control cannot tell them from
independent covariates.

## Named gaps

An axis the defaults fail for a known reason is a named gap, listed with its reason in
[`test_fidelity_flat.py`](../tests/test_fidelity_flat.py).
The test breaks when an axis outside the list fails, and also when a listed axis passes, so the
list can only shrink.
[`flat.ipynb`](../experiments/fidelity/flat.ipynb) gives the numbers behind each reason.

No axis measures count covariates, since a real column of numbers is read as continuous, or how
well the covariates predict y together.
[Bounds and leverage by row wait for axes of their own](https://github.com/adkipnis/tabdgp/issues/98).

## Limits

- **Power:** recall catches gross gaps only, and real tables 10 % wider than their draws are
  caught with a power of 0.02 on a normal axis.
- **Floor:** precision moves by about 0.05 with the number of neighbors, so its floor is not read
  more finely.

## Changing a gate

The gates, the seeds and the split live in [`flat.py`](../experiments/fidelity/flat.py), and they
change like a default, with an issue first.
