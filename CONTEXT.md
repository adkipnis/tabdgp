# TabDGP

A data-generating process for grouped and flat tables: what is drawn, under what structure,
and from which distributions. TabDGP draws covariates from a Cauchy-SCM; the outcome is either
another node in the SCM, or it is model-based (e.g. drawn according to a GLM/GLMM formula).

## Language

### Covariates

**Covariate generator**:
The process that draws a dataset's grouping structure and its typed covariate columns:
continuous, binary, categorical, ordered. Each covariate varies within groups or is on
the group-level.
_Avoid_: predictor simulation, feature matrix, X generator, design-matrix generator

**Design matrix**:
The numerical coding of the covariates that a linear predictor would see:
an intercept column, z-scored continuous columns, 0/1 binary columns, k − 1 treatment-coded
columns per categorical or ordered column.
The design is strictly derived from covariates.
The hyperprior's design-column budget counts its columns besides the intercept.
_Avoid_: X for the uncoded covariates, features

**Within-group covariate** / **group-level covariate**:
A covariate that varies within the groups of a factor, versus a constant within each group
of some factor (e.g. a school's size, a person's sex). A within-group covariate whose mean differs
between groups is called **group-confounded**.
_Avoid_: level-1 and level-2 variable, cluster-level, contextual variable

**Group share**:
How much of the variance of a within-group node is due to the group. This is one value per
node, at one grouping factor, and measures the extent of group confounding.
The between-group share can be larger, since the part of the node drawn per row may read
group-confounded parents.
_Avoid_: ICC unqualified (the fitter's, on the outcome), between-group share (what the
measurement pass records), random-intercept variance

**Quirk**:
A property of real tables that an independent and identically distributed (i.i.d.) simulator
lacks (e.g. group confounding, empty cells, rare levels, ties, bounds, irregular time, near dependence, leverage points).
_Avoid_: edge case, noise, messiness

### The graph

**Node**:
One variable of a dataset's structural causal model (SCM): a grouping factor, a group-level or
within-group covariate, a latent node, time, treatment, or the outcome node.
A node has a parent set and a mechanism.
A node without parents is called a root.
_Avoid_: feature, variable unqualified, neuron (of an MLP)

**Grouping factor**:
A root node or a table's categorical column, whose values assign every row to one of its m groups.
In a mixed model, the groups broadcast the random effects.
A dataset has none (a flat table) or at least one (a grouped table).
Two grouping factors are **crossed** when their groups combine freely, and **nested** when each
group of one lies within a single group of the other.
_Avoid_: factor alone (a categorical column), cluster, subject (one kind of grouping factor),
random-effects term

**Superior** / **subordinate**:
If the groups of factor A lie within the groups of factor B, B is superior and A is subordinate.
The row is subordinate to every factor.
The **immediate superior** of a factor is its superior with the most groups; its metadata
names it.
In a table, nesting is read from the codes, in whatever order the factors are named.
_Avoid_: coarser and finer, parent and child (the graph's edges), higher and lower level

**Unit**:
What a node is drawn once for: each row, or each group of one grouping factor.
A node drawn per group repeats its value across the group's rows (broadcast).
A covariate drawn per row is within-group, and one drawn per group is group-level.
The unit of a nested factor is subordinate to the unit of its superior.
The units of crossed factors have no order.
_Avoid_: level (a factor level, or the multilevel literature's level 1 and 2), hierarchy
level, grain, scope

**Mechanism**:
The function that computes a node's values from its parents' values and its own noise.
Each node draws one from a mechanism family: linear or a small multilayer perceptron (MLP).
A root's mechanism is just its noise.
_Avoid_: layer, activation (one part of it), edge function

**Generating graph**:
The directed acyclic graph of a dataset's nodes and their parents.
It is returned with every dataset.
_Avoid_: causal graph unqualified, DAG (one property of it), architecture

**Latent node**:
A node of the generating graph that is not a column of the table.
Its values are still returned, so the dataset stays white-box.
_Avoid_: hidden unit, confounder (a role a latent node can play)

**Column**:
A variable in the table: an observed node, a column written by a model, or a column of a given
table.
Its metadata holds its type (continuous, binary, categorical, ordered, count), its levels and
its role (grouping factor, covariate, time, treatment, outcome).
A nested grouping factor also names its immediate superior.
A drawn column also records its marginal and the parameters its values were drawn with.
A table's metadata is either declared or inferred from the dtypes, and is checked against the
values in both cases.
_Avoid_: feature, variable unqualified

**Count**:
A column type for non-negative integers (e.g. a Poisson outcome, an integer covariate of a table).
The hyperprior draws no count covariates yet.
In the design matrix, a count is treated as a continuous column.
_Avoid_: integer column, discrete column (an ordered one)

**Factor level**:
One of the k categories of a binary, categorical or ordered column, or one group of a grouping
factor (R's `levels()`).
_Avoid_: level unqualified, category, class

**Marginal**:
The distribution that a continuous or count column's values are mapped onto after the SCM,
keeping their ranks.
A continuous column draws a family from a catalog (e.g. skewed, heavy-tailed, bounded,
zero-inflated).
A count column uses the Poisson.
_Avoid_: noise distribution (the root's), prior

**Post-hoc type**:
A binary, categorical or ordered type that a continuous node is turned into after the SCM, by
thresholds or bins on its values.
_Avoid_: dtype, encoding (the design matrix's)

### The draw

**Draw**:
The public call that returns one dataset.
It is a pure function of a key and a hyperprior.
Many datasets are drawn by looping over indices.
_Avoid_: sample (a row or a value), generate, batch

**Key**:
A seed and an index, which identify one dataset.
The same key, hyperprior and versions of TabDGP and numpy give the same dataset bit for bit, in
any batch, worker or shard.
_Avoid_: rng, random state, seed alone

**Dataset**:
What a draw returns: the columns as named arrays in their natural dtype, their metadata, the
generating graph, the values of the latent nodes, the outcome, the resolved hyperparameters,
and the key with the package versions.
The outcome holds the columns its form wrote and the form's truth.
The hyperparameters leave out the fields that each column, node, grouping factor or model draws
for itself.
A dataset is never padded and never written to a file.
_Avoid_: batch, table (the columns alone), sample, corpus (the consumer's set of keys)

**Hyperprior**:
The config of distributions from which every quantity of a dataset is drawn (e.g. sizes, graph,
mechanisms, marginals, types, quirks).
It is reached through a preset.
The defaults of the grouped preset are to be matched to real grouped tables and then widened.
They are what the proofs certify.
_Avoid_: prior (the fitter's, over parameters), config unqualified, meta-distribution

**Preset**:
A named hyperprior for one kind of table: grouped or flat today, item response, survival or
causal later.
A preset only sets defaults and never adds a mechanism.
Every draw starts from one.
_Avoid_: dataset type, kind, mode

**Override**:
A point value or a distribution passed for one field of a preset, which replaces its default
for that call.
Overrides choose the outcome's form, or adapt a preset to a study, an ablation or a size outside
its range.
_Avoid_: setting, parameter, hyperparameter

**Quirk axis**:
A field of the hyperprior that controls one quirk.
It is added only when measurements on real tables show the quirk strongly enough.
_Avoid_: noise, perturbation, augmentation

**Outcome node**:
The node `y` with role `outcome`.
It is drawn by default, at the row unit, as the last node of the generating graph and without
children.
Its parents are drawn like any node's, and its form says how y follows from them.
A dataset has at most one.
_Avoid_: target, label, response unqualified, dependent variable

**Form**:
How the outcome node turns its score into y, one per dataset.
Under `scm`, the default, y is a node like the others, with a marginal or a post-hoc type.
Otherwise the form is a model.
The first two are `glm` and `glmm`, a generalized linear model (GLM) and a generalized linear
mixed model (GLMM) on the design matrix.
Both have a family, a link and fixed effects, the GLMM adds random effects per grouping factor,
and both can add a deviation.
_Avoid_: mode, kind, outcome type (its column type)

**Response family**:
The distribution of y given the linear predictor, with its canonical link.
Normal, Bernoulli and Poisson by default.
Student-t, negative binomial, latent-noise Bernoulli, Gamma and beta at share zero, each with a
shape of its own.
A study that draws one makes the dataset misspecified for a fitter of the base family.
_Avoid_: likelihood (the fitter's), distribution unqualified, marginal (a covariate's)

**Covariates**:
What a form reads: the observed columns (the grouping factors and the covariates), their
metadata, how the rows are grouped, and the score.
A draw's covariates come from its SCM.
A table's covariates are read from the table and have no score.
A model's outcome on a table's covariates is called a **resampled outcome**, and is keyed like a
dataset.
_Avoid_: X (the design matrix), features, inputs

**Model**:
A form given as a function of the covariates and a generator.
It returns the columns it writes, their metadata and its truth.
The GLM and the GLMM are TabDGP's own models, and a study can pass its own (e.g. a Gaussian
process, an MLP).
_Avoid_: likelihood (one part of it), head, outcome function, target function

**Score**:
The outcome node's value in the SCM, z-scored: its mechanism over its parents (latent ones
included) plus its noise.
Every form turns it into y.
The scm form rank-matches it onto a marginal.
A GLM or GLMM adds it to the linear predictor as the deviation.
A study's model may ignore it.
A real table has no score.
_Avoid_: latent outcome, y*, η (the linear predictor), f

**Deviation**:
The score, scaled and added to the linear predictor of a GLM or GLMM.
Its scale is zero by default.
It makes the dataset misspecified for every GLMM, while the fixed and random effects stay the
exact truth.
If the outcome node has a latent group-level parent, the deviation is endogenous.
A table without an SCM has no deviation.
_Avoid_: perturbation, nonlinearity, misspecification (the fitter's view of it), noise

**Prior context**:
The prior of a GLM's or GLMM's parameters, resolved per dataset and stored with it: what a
fitter would state.
It holds the family of the fixed effects, the location and scale of the intercept and of each
slope, the family of each factor's random-effect standard deviations and the scale of each, the
family and scale of the residual, and the shape of each correlation prior.
A slope's location is zero for most slopes and a hypothesis for the rest.
It also holds the signal share and the split that set the scales.
It draws the parameters on its own, so a truth can be resampled under a fixed prior.
_Avoid_: hyperparameters unqualified (the hyperprior's), prior unqualified (the fitter's)

**Random effect**:
A parameter α of the outcome per group of a grouping factor: an intercept or a slope.
A group's random effects are drawn jointly.
They are counted in design columns, so a slope on a categorical or ordered column takes k − 1.
Only the outcome node draws random effects.
A group-level node, latent or observed, is part of the covariates, which a GLMM takes as given.
_Avoid_: group effect, b (an earlier letter), random intercept for a latent group-level node

### The proofs

**Summary vector**:
The statistics per dataset on which the proofs compare generated and real tables, one row per
dataset.
A statistic that a table cannot show (e.g. the intraclass correlation of a second factor) is
marked missing.
_Avoid_: features (the classifier's view of it), meta-features, embedding

**Two-sample test**:
A classifier two-sample test (C2ST) on the summary vectors of generated and real tables.
It measures how well a classifier tells the two apart, compared to its null.
_Avoid_: discriminator, realism score

**Null**:
The distribution of a test's statistic when both samples come from one distribution.
It is drawn by permuting the sample labels of the rows.
A statistic is read by its position in the null.
_Avoid_: baseline, chance level unqualified, significance

**Containment test**:
Tests whether the real tables' summary vectors lie inside the band of the generated ones, the
generated 1st to 99th percentile per axis.
Its result is the share of real tables inside the band on every axis.
The share of generated tables inside the real tables' band shows what the widening costs.
_Avoid_: coverage (the fitter's credible-interval property), support (the band is its estimate)

**Pairing**:
Drawing one generated dataset per fit of a real table, at that table's sizes (rows, grouping
factors, groups per factor).
The two-sample test then measures the prior's mismatch and not the sizes'.
_Avoid_: matching (the hyperprior's fit to the registry), conditioning
