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
continuous columns, 0/1 binary columns, K − 1 treatment-coded columns per categorical.
Derived from the covariates, never drawn; the hyperprior's design-column budget counts its
columns besides the intercept.
_Avoid_: X for the uncoded covariates, features

**Within-group covariate** / **group-level covariate**:
A covariate that varies within the groups of a factor, versus one constant within each group
of some factor (a school's size, a person's sex). A within-group covariate whose mean differs
between groups is **group-confounded**.
_Avoid_: level-1 and level-2 variable, cluster-level, contextual variable

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
_Avoid_: feature, variable unqualified, unit (a neuron of an MLP)

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

**Marginal**:
The distribution a continuous column's values are mapped onto after the SCM, from a catalogue
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
graph, mechanisms, marginals, types, quirks. Its defaults are to be matched to mixedbench's
registry and then widened; they are what the proofs certify.
_Avoid_: prior (the fitter's, over parameters), config unqualified, meta-distribution

**Override**:
A point value or a distribution passed for one field of the hyperprior, replacing its default
for that call. A standalone user passes none; metabeta passes its sizes.
_Avoid_: setting, parameter, hyperparameter

**Quirk axis**:
One field of the hyperprior that controls one quirk of the catalogue, granted only when
mixedbench's measurement pass shows that real tables have the quirk strongly enough.
_Avoid_: noise, perturbation, augmentation

**Outcome node**:
An optional node whose parents are drawn from the covariates, with a family, a link and terms
a GLMM lacks, that builds white-box misspecified datasets. Just another node; never drawn
unless asked for.
_Avoid_: y, target, label, response unqualified

### The proofs

**Two-sample test**:
A classifier two-sample test (C2ST) on the summary vector of generated and registry tables:
the defaults pass when its accuracy stays below the bar.
_Avoid_: discriminator, realism score

**Containment test**:
The test that every registry row's summary vector lies inside the support of the generated
ones.
_Avoid_: coverage (the fitter's credible-interval property)
