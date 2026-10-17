# TabDGP

A data-generating process for grouped and flat tables: what is drawn, how, and from which
distributions. The covariate terms are those of metabeta's generalized linear mixed
model (GLMM) glossary (`packages/mixed/CONTEXT.md` of metabeta-v3, a private repository),
restated here in its words and adapted where TabDGP differs; statistical terms (grouping
factor, family, link) are not redefined.

## Language

### Covariates

**Covariate generator**:
The process that draws a dataset's grouping structure and its typed covariate columns,
continuous, binary, categorical or ordered, each within-group or group-level, with the quirks
of real tables. TabDGP is one; it draws an outcome only as the optional outcome node.
_Avoid_: predictor simulation, feature matrix, X generator, design-matrix generator

**Design matrix**:
The coding of the covariates the linear predictor sees: an intercept column, z-scored
continuous columns, 0/1 binary columns, k − 1 treatment-coded columns per categorical.
Derived from the covariates, never drawn; the hyperprior's design-column budget counts its
columns besides the intercept.
_Avoid_: X for the uncoded covariates, features

**Within-group covariate** / **group-level covariate**:
A covariate that varies within the groups of a factor, versus one constant within each group
of some factor (a school's size, a person's sex). A within-group covariate whose mean differs
between groups is **group-confounded**.
_Avoid_: level-1 and level-2 variable, cluster-level, contextual variable

**Group share**:
How much of a within-group node's variance comes from its group part, one value per group of
one grouping factor, rather than its row part, one value per row: the strength of its group
confounding. The between-group share measured on its values can be larger, since its row part
may read group-confounded parents.
_Avoid_: ICC unqualified (the fitter's, on the outcome), between-group share (what the
measurement pass records), random-intercept variance

**Quirk**:
A property of real tables that an independent and identically distributed (i.i.d.) simulator
lacks and on which an amortized posterior goes wrong: group confounding, empty cells, rare
levels, ties, bounds, irregular time, near dependence, leverage points.
_Avoid_: edge case, noise, messiness

### The graph

**Node**:
One variable of a dataset's structural causal model (SCM): a grouping factor, a group-level or
within-group covariate, time, treatment, or the outcome node. A node has a parent set and a
mechanism; a node without parents is a root.
_Avoid_: feature, variable unqualified, neuron (of an MLP)

**Grouping factor**:
A root node whose values assign every row to one of its m groups, in metabeta's words "a
categorical variable whose levels index random effects"; a dataset has none (a flat table),
one or several. Two grouping factors are **crossed** when their groups combine freely and
**nested** when each group of the subordinate one lies within one group of the superior one,
which has fewer groups.
_Avoid_: factor alone (a categorical column), cluster, subject (one kind of grouping factor),
random-effects term

**Superior** / **subordinate**:
A grouping factor is superior to one nested in it, whose groups lie within its own; the nested
one is subordinate. The row is subordinate to every factor.
_Avoid_: coarser and finer, parent and child (the graph's edges), higher and lower level

**Unit**:
What a node is drawn once per: the row, or one grouping factor, in which case the node takes
one value per group of that factor, repeated across the group's rows (broadcast). A column at
the row unit is a within-group covariate; one at a grouping factor's unit is group-level. A
nested grouping factor is a unit subordinate to the one it nests in; crossed ones are units
without an order between them.
_Avoid_: level (a factor level, or the multilevel literature's level 1 and 2), hierarchy
level, grain, scope

**Mechanism**:
The function that computes a node's values from its parents' values and its own noise, drawn
per node from a mechanism family (linear, a small MLP, a tree). A root's mechanism is its noise.
_Avoid_: layer, activation (one part of it), edge function

**Generating graph**:
The directed acyclic graph of a dataset's nodes and their parent sets, returned with every
dataset.
_Avoid_: causal graph unqualified, DAG (one property of it), architecture

**Latent node**:
A node of the generating graph that is not a column of the table, kept in the output so that
the dataset stays white-box.
_Avoid_: hidden unit, confounder (a role a latent node can play)

**Column**:
A node observed in the table, with its metadata: type (continuous, binary, categorical,
ordered), levels, role (grouping factor, covariate, time, treatment, outcome), and the
parameters its values were drawn with.
_Avoid_: feature, variable unqualified

**Factor level**:
One of the k categories of a binary, categorical or ordered column, or one group of a
grouping factor; R's `levels()`.
_Avoid_: level unqualified, category, class

**Marginal**:
The distribution a continuous column's values are mapped onto after the SCM, from a catalog
of families (skewed, heavy-tailed, bounded, zero-inflated), keeping their ranks.
_Avoid_: noise distribution (the root's), prior

**Post-hoc type**:
The column type a continuous node is turned into after the SCM, by thresholds or bins on its
values: binary, categorical, ordered.
_Avoid_: dtype, encoding (the design matrix's)

### The draw

**Draw**:
The one public call: given a key and a hyperprior, one dataset, a pure function of both. Who
draws many loops over indices.
_Avoid_: sample (a row or a value), generate, batch

**Key**:
A seed and an index, the identity of one dataset: the same key, hyperprior and versions of
TabDGP and numpy give the same dataset bitwise, whatever the batch, worker or shard it is
drawn in.
_Avoid_: rng, random state, seed alone

**Dataset**:
What a draw returns: the columns as named arrays in their natural dtype, their metadata, the
generating graph, the latent nodes' values, the hyperparameters every field resolved to except
those each column or node draws for itself, and the key. Never padded, never a file.
_Avoid_: batch, table (the columns alone), sample, corpus (the consumer's set of keys)

**Hyperprior**:
The one config of distributions from which every quantity of a dataset is drawn: sizes,
graph, mechanisms, marginals, types, quirks, reached through a preset. The grouped preset's
defaults are to be matched to mixedbench's registry and then widened; they are what the
proofs certify.
_Avoid_: prior (the fitter's, over parameters), config unqualified, meta-distribution

**Preset**:
A named hyperprior for one kind of table, grouped or flat today, item response, survival or
causal later: a choice of fields, never a new mechanism. Every draw starts from one; metabeta
draws from the grouped preset.
_Avoid_: dataset type, kind, mode

**Override**:
A point value or a distribution passed for one field of a preset, replacing its default for
that call. Neither a standalone user nor metabeta passes one by default; an override serves a
study, an ablation or a size outside the preset.
_Avoid_: setting, parameter, hyperparameter

**Quirk axis**:
One field of the hyperprior that controls one quirk of the catalog, granted only when
mixedbench's measurement pass shows that real tables have the quirk strongly enough.
_Avoid_: noise, perturbation, augmentation

**Outcome node**:
An optional node whose parents are drawn from the covariates, with a family, a link, the
random effects per grouping factor and terms a GLMM lacks, that builds white-box
misspecified datasets. Just another node; never drawn unless asked for.
_Avoid_: y, target, label, response unqualified

**Random effect**:
A per-group parameter of the outcome, an intercept or a slope, drawn jointly per group of a
grouping factor in metabeta's parameterization; metabeta's term. Only the outcome node draws
random effects; a group-level node, latent or observed, is covariate structure, which a GLMM
takes as given.
_Avoid_: group effect, random intercept for a latent group-level node

### The proofs

**Two-sample test**:
A classifier two-sample test (C2ST) on the summary vector of generated and registry tables:
the defaults pass when its accuracy stays below the bar.
_Avoid_: discriminator, realism score

**Containment test**:
The test that every registry row's summary vector lies inside the support of the generated
ones.
_Avoid_: coverage (the fitter's credible-interval property)
