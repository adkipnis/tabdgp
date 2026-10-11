# TabDGP

A data-generating process for grouped and flat tables: what is drawn, how, and from which
distributions. Covariate generator, design matrix, within-group and group-level covariate,
group-confounded and quirk are metabeta's GLMM glossary's (`packages/mixed/CONTEXT.md` of
metabeta-v3) and are not redefined here; statistical terms (grouping factor, family, link)
too.

## Language

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
ordered), levels, and role (grouping factor, covariate, time, treatment, outcome).
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

**Hyperprior**:
The one config of distributions from which every quantity of a dataset is drawn: sizes,
graph, mechanisms, marginals, types, quirks. Its defaults are matched to mixedbench's registry
and then widened; they are what the proofs certify.
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
