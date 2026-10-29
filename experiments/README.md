# Experiments

These are the studies on which a design decision rests, or which are reported elsewhere.
In each directory, a script writes its outputs and the notebook of the same name reads them.

Reproducibility: A rerun from the repository root matches them at the commit on which they were run originally.

| experiment | question | finding |
|---|---|---|
| [`scale-draw/simulated`](scale-draw/simulated.ipynb) | Which scale draw keeps a mixed model's explained variance and saturation stable as the design grows: the joint draw, R2D2M2 or stated scales? | Only the joint draw: [R2D2M2 and stated scales let the marginal R² climb with the number of columns](https://github.com/adkipnis/tabdgp/issues/76), and stated scales saturate the logit and log links more with every column. |
