---
status: accepted
date: 2026-10-14
---

# 0003 — Grouping factors are edgeless roots; every node has a unit; confounding is a two-part mechanism

Roadmap item 2 adds grouping to the flat SCM of item 1. Grilled on 2026-10-14 after the
prior-art search of 2026-10-12 (`docs/frontier-priors.md`, local), the decision is: **a
grouping factor is a root node without edges whose values assign rows to groups; every other
node carries a unit, the row or one grouping factor, and is drawn once per element of that
unit; a row-unit node's group confounding is an explicit share, `group_share`, that mixes a
group part and a row part.** The words are `CONTEXT.md`'s: grouping factor, unit, factor
level, group share, random effect.

The witnesses, from mixedbench 1.0.1's registry: 31 % of a documented model's covariates are
group-level; of 714 tables 56 % have one grouping factor, 20 % crossed, 16 % nested, 8 % both;
38.8 % of factors are balanced, the rest fitted by a symmetric Dirichlet concentration α with
p5 0.97 and median 50. The sources of the ideas: PluRel's per-table SCM in topological order
(group-level nodes before within-group ones), RDB-PFN's parent latent units (latent group-level
nodes), and Dubey et al. 2026, who found that neither relational prior controls how much of a
child column its parent explains.

## Decisions

- **The unit is a per-node field**, `unit`, `'row'` or a grouping factor's name, checked at
  the graph's construction like the latent mask. The grouping factor itself is a root with
  no edges, its codes its values, the size draw its "noise" (ADR 0004). A node at a grouping
  factor's unit is drawn on that factor's G group rows: its parents' group values, one noise
  per group, the mechanism, its z-score and, for a column, its marginal or level codes, all at
  G rows, then broadcast to the n rows last. Latent values are returned broadcast, so every
  array of a dataset has length n.
- **One unit per node.** Nested and crossed factors are both expressed: a node at the finer
  factor of a chain is constant within the coarser one too. What is not expressed is a node
  drawn once per cell of two crossed factors, whose unit would be a set of grouping factors;
  that is the extension, for item 6's item-response data, and nothing here precludes it.
- **Edges across units**: a parent is at the child's unit or at a coarser one in its nesting
  chain; no edges between the units of crossed factors; no row-unit parent of a group-level
  node. A mask on the Cauchy logits; the edge draw is otherwise item 1's. Node order: the
  grouping factors, then the units coarsest first (crossed units in a fixed order), the row
  unit last, latent before observed within a unit.
- **Which nodes are group-level**: a per-dataset `group_level_share`; each node, latent or
  observed, draws a grouping factor's unit with that share (the factor uniform among the
  dataset's), else the row. A within-group column with a latent group-level parent is
  confounded by nothing in the table.
- **Group confounding is a two-part mechanism.** A row-unit node is
  `z(sqrt(s) · z(g) + sqrt(1 − s) · z(r))`: g its mechanism over the parents at one grouping
  factor's unit (and the coarser ones in that factor's chain), computed at that unit and
  broadcast, pure group-level noise when it has no such parent; r its mechanism over every
  other parent plus its own noise; s its `group_share`, a per-node field. The factor of the
  group part is drawn uniformly per node. Every z-score runs at the unit of the node being
  computed, so s is the node's between-group variance share over rows, the statistic the
  measurement pass records, up to what its row-unit parents carry.
- **Relations**: each grouping factor after the first nests in the previous one with a
  per-dataset `nested_share`, else is crossed with every earlier one; crossed assignments are
  independent. Nested sizes are the size draw applied twice, bottom-up: the finer factor's
  groups over the rows, then the finer factor's groups over the coarser factor's groups.
- **The grouping factor as a column**: int64 codes, type `categorical`, role
  `grouping factor`, `levels = G`, outside the design-column budget; the role carries its
  checks (G ≥ 2, no marginal, the size draw's fields as its parameters). The binary /
  categorical split is about design-matrix coding and applies to covariates only.
- **Random effects stay the outcome node's.** A group-level node is covariate structure, which
  a GLMM takes as given; the outcome node (item 5) draws its random effects per grouping
  factor in metabeta's parameterisation, so that metabeta's model is a special case. Item 5
  also decides whether a latent group-level node may be a parent of the outcome, an
  endogeneity that must be a switch.
- **Time and treatment are deferred** to the item that reads them: a treatment is a binary
  column with a label nobody in this item reads, and the questions the label raises
  (positivity, randomised or confounded) are causal-design questions; time's reader is the
  outcome's random slope.

## Considered options

- **Where the unit lives.** (a) A per-node field, chosen. (b) An edge from the grouping factor
  to each node at its unit: duplicates the field, and the mechanism must skip a nominal
  parent. (c) No field, group-level-ness emergent from a lookup mechanism (one random value
  per factor level) with zero row noise, the random-effects view: cannot promise a constant
  column, the column module needs the unit anyway, and the group-level share has no knob.
- **Confounding strength.** (a) Emergent through group-level parents: no knob, and item 9's
  ablation could not switch it off. (b) A per-node random intercept added with share s:
  parents' confounding stays uncontrolled, so s is not the node's share. (c) The two-part
  mechanism, chosen: `group_share = 0` as an override switches confounding off for the whole
  dataset, by induction over the graph.
- **Group part per factor** (a simplex of shares per node) over one factor per node: finer,
  two fields to fit; item 4 may promote it if the pass shows many columns confounded with
  respect to two factors.
- **A count of group-level nodes** over a share: the share is the registry's statistic and
  works whatever the node count.
- **Dependence between crossed assignments** (RDB-PFN's affinity, a Dirichlet tilt per group
  of one factor over the other's): a quirk-2 axis without a registry statistic; waits for the
  measurement pass.
- **Where a group-level node is z-scored**, G rows or n rows: best practice (Enders and
  Tofighi 2007 against `scale()` on the long table) is split and binds the fitter, not the
  generator, since the column's scale is its marginal's. The rule "at the unit of the node
  being computed" settles it and makes s the row-level share.

## Consequences

- `graph` gains the per-node `unit` and `group_share`, the mask and the order; `mechanism`
  the arithmetic at a unit, the broadcast and the two parts; `column` the role, the codes of
  a grouping factor and the marginal at G values; `hyperprior` the fields and the size draw
  (ADR 0004). Every dataset a key draws changes: a version change.
- Group confounding is the first quirk axis granted before the measurement pass, since the
  registry already witnesses it (31 % group-level columns); its default is a round guess,
  `Beta(0.5, 2)`, fitted in item 4.
- mixedbench#71 is asked for three statistics: per within-group column × grouping factor the
  adjusted ICC(1) (the total group share) and the R² on the factor's observed group-level
  columns (the observed share, after Dubey et al.'s parent gain), whose gap is the latent
  share; per pair of crossed factors the empty-cell share.
- Extension points, none built: a latent group-level root "log group size" that group-level
  nodes may take as a parent (quirk 8, allowed by the edge rule); a set-valued unit for cell
  effects; per-factor group shares; the crossed-assignment dependence.
- `mechanism.py`'s "a bias per unit" becomes "per neuron", since unit now has a meaning.
