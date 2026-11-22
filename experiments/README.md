# Experiments

These are the studies on which a design decision rests, or which are reported elsewhere.
In each directory, a script writes its outputs and the notebook of the same name reads them.

Reproducibility: A rerun from the repository root matches them at the commit on which they were run originally.

| experiment | question | finding |
|---|---|---|
| [`scale-draw/simulated`](scale-draw/simulated.ipynb) | Which scale draw keeps a mixed model's explained variance and saturation stable as the design grows: the joint draw, R2D2M2 or stated scales? | Only the joint draw: [R2D2M2 and stated scales let the marginal R² climb with the number of columns](https://github.com/adkipnis/tabdgp/issues/76), and stated scales saturate the logit and log links more with every column. |
| [`scale-draw/registry`](scale-draw/registry.ipynb) | Which scale draw covers the variance partition of MixedBench 1.0.13's grouped tables, refitted by lme4 on their designs: the joint draw at its defaults, R2D2M2 or stated scales? | Only the joint draw: [R2D2M2 and stated scales explain more than real tables on the logit link](https://github.com/adkipnis/tabdgp/issues/77), and every draw misses the log link and the real fits' singular and saturated shares. |
| [`fidelity/flat`](fidelity/flat.ipynb) | Do the real flat tables of MixedBench 1.0.15 stand out among 39 synthetic tables drawn at their schema: under the defaults of `Scm.flat()`, with independent covariates, or under TabICLv2's prior? | Under the defaults they do, on 23 of 40 statistics, at a precision of 0.39: [the defaults lack discrete numbers, over-dispersed counts and exact associations](https://github.com/adkipnis/tabdgp/issues/93). Independent covariates are detected on every association, and TabICLv2's prior reaches a precision of 0.48. |
| [`fidelity/distance`](fidelity/distance.ipynb) | Which distance between two tables' statistics should the precision use: ranks pooled over real and synthetic tables, ranks among the real tables, the same with groups of statistics weighted alike, or a robust scale? | Ranks among the real tables: [pooled ranks measure each simulator on a scale of its own](https://github.com/adkipnis/tabdgp/issues/97), and a robust scale lets one statistic decide the distance. |
