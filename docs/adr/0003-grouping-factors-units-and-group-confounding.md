---
status: accepted
date: 2026-10-14
---

# 0003 - Grouping factors, units and group confounding

Grouping turns the flat SCM of #1 into a grouped one (#22).
In real grouped tables, some covariates are group-level and some are group-confounded.
A grouping factor is a root node without edges.
Each later factor nests in the previous one or is crossed with every earlier one.
Every node has a unit of observation (row, or grouping factor).
Every observation in the sampled table corresponds to a draw from a node.
A parent node is at its child's unit or at a superior one.
A row-unit node's group share (`group_share`) mixes a group part and a row part.
The override `group_share = 0` removes all group confounding.
Random effects belong to a model (ADR 0007), not the SCM.

## Decisions

1. A grouping factor is a root node without edges:
   - its values are its group codes, which assign each row to one of its m groups.
   - the size draw (ADR 0004) acts as its noise: it draws the m group sizes, then shuffles the rows into them.
2. Each subsequent grouping factor nests in its predecessor with the per-dataset share `nested_share`; otherwise it is crossed with every earlier one:
   - precedent: in MixedBench 1.0.1's registry (its list of 714 real grouped tables and their documented models), 56 % have one grouping factor, 20 % crossed ones, 16 % nested ones and 8 % both.
   - crossed grouping assignments are independent, so a cell (a pair of groups, one of each factor) stays empty only by chance.
   - a nested factor runs the size draw twice, bottom-up: the rows into its groups (pupils into classes), then its groups into the superior factor's fewer groups (classes into schools).
3. A grouping factor is a column with int64 codes, has the type `categorical`, role `grouping_factor` and `levels = m`. It is outside the design-column budget (`design_columns`, ADR 0004).
4. Every other node has a unit of observation (`unit`), which is `'row'` or a grouping factor's name:
   - each node, latent or observed, draws a grouping factor's unit with the per-dataset share `group_level_share`, else the row (see RDB-PFN's latent parent units).
   - the factor is drawn uniformly.
   - a column with more levels than the number of groups stays at the row unit, otherwise it would leave a level no row has.
   - one unit per node suffices, since a node at a superior factor's unit is constant within each subordinate group too.
5. A parent is at its child's unit or at a superior one (the row is subordinate to every factor):
   - no edge links the units of crossed factors, nor a row-unit parent to a group-level child.
   - the rule is a mask on the Cauchy logits (an edge's logit adds a Cauchy-distributed term of each of its two nodes); the edge draw is otherwise #1's.
   - the node order is the grouping factors, then the units (superior first, crossed ones in a fixed order), the row last, and latent before observed within a unit (see PluRel).
6. A node at a factor's unit is drawn once per group, from its parents' values to its column's, then broadcast to the n rows, so every array of a dataset (latent ones included) has length n.
7. A row-unit node's group confounding is an explicit share (`group_share`), which mixes a group part and a row part:
   - the node is `z(sqrt(s) · z(g) + sqrt(1 − s) · z(r))`, with s its `group_share`.
   - the group part reads one grouping factor, drawn uniformly per node.
   - g is its mechanism over its parents at that factor's unit or a superior one, computed there and broadcast; without such a parent, g is group-level noise.
   - r is its mechanism over its other parents plus its own noise.
   - every z-score runs at the unit of the node being computed, so s is the node's between-group variance share over rows (up to what its row-unit parents carry).
8. Time and treatment are deferred until a model reads them, because the questions a treatment raises (positivity, randomized or confounded) are causal-design questions.

## Considered options

| question | chosen | rejected |
|---|---|---|
| crossed assignments | independent | an affinity |
| where the unit lives | a per-node field | an edge from the factor, a lookup mechanism |
| group-level nodes | a share | a count |
| confounding strength | a two-part mechanism | emergent, a per-node random intercept |
| group part | one factor per node | a share per factor |
| z-score of a group-level node | at its own unit | at the rows |

1. **Crossed assignments:**
   - RDB-PFN's affinity (a Dirichlet tilt per group of one factor over the other's) is a quirk axis, granted only when real tables show it strongly enough.
2. **Where the unit lives:**
   - An edge from the grouping factor is a nominal parent the mechanism must skip.
   - A lookup mechanism (one random value per group, no row noise) cannot promise a constant column.
3. **Group-level nodes:**
   - The share is the registry's statistic (31 % of a documented model's covariates are group-level) and works whatever the node count.
4. **Confounding strength:**
   - Neither relational prior (PluRel and RDB-PFN, priors over relational databases) controls how much of a child column its parent explains (see Dubey et al. 2026).
   - Emergent confounding (through group-level parents) cannot be switched off.
   - A per-node random intercept added with share s leaves the parents' confounding uncontrolled, so s is not the node's share.
   - The two-part mechanism switches confounding off for the whole dataset with the override `group_share = 0`, by induction over the graph.
5. **Group part:**
   - A share per factor (a simplex per node) is finer, but has two fields to fit.
6. **Z-score of a group-level node:**
   - Best practice is split (see Enders and Tofighi 2007 against `scale()` on the long table) and binds the fitter, not the generator, since a column's scale is its marginal's.
   - The z-score at the unit of the node being computed makes s the row-level share.

## Consequences

- **Every level appears:** except under the override `group_share = 1`.
  - It makes a row-unit column constant within its group factor's groups, so one with k > m misses a level.
  - The default never draws 1, and a column constant within groups cannot show more levels than there are groups.
- **The first quirk axis:** group confounding is granted before the measurement pass (MixedBench's statistics of each quirk on the real tables), because the registry already witnesses it (31 % group-level columns).
  - Its default is a round guess (ADR 0004), to be fitted.
- **The measurement pass:** MixedBench records the statistics the group share is fitted to since 1.0.8 (mb#71).
  - Per within-group column and grouping factor: the total group share (`between_group_share`, the adjusted intraclass correlation ICC(1)) and the observed share (`group_level_r2`, the adjusted R² on the factor's observed group-level columns).
  - Their gap is the latent share, which nothing in the table explains.
  - Per pair of crossed factors: the empty-cell share (`empty_cell_share`).
- **Extension points, none built:**
  - the dependence between crossed assignments
  - a set-valued unit for a node drawn once per cell of two crossed factors (item-response data)
  - a latent group-level root "log group size" that group-level nodes may take as a parent (group-size dependence, a quirk the edge rule allows)
  - per-factor group shares, once real columns show confounding by two factors
