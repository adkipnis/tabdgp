---
status: accepted
date: 2026-10-11
amended: 2026-10-11, after the throughput lab note; 2026-10-12, after the column-pooling lab note (#16); 2026-10-12, numpy's version recorded beside TabDGP's; 2026-10-14, ADR 0004: metabeta passes no sizes, `draw` takes a preset and has no default; 2026-10-22, the outcome metabeta trains on comes from TabDGP's outcome node; 2026-10-25, #52: a GLM's or GLMM's ground truth is float32 too; 2026-10-27, #56: ground truth, not truth; 2026-10-28, ADR 0007: a model's outcome draws from the key's first child stream, so a dataset's own key gives an independent outcome; 2026-10-28, #56: bitwise per CPU type and numpy build, since numpy's float draws and functions differ in the last bit across them
---

# 0001 - One dataset per draw, keyed by seed and index, in numpy

TabDGP's first consumer, `metabeta.mixed` in the private repository metabeta-v3, trains on
simulated datasets whose sizes, groups and columns all vary; its roadmap puts padding,
masking, collate and the `.npz` corpus on `metabeta.core`, and TabDGP's optional outcome node
draws the outcome before anything is written. Grilled on
2026-10-11 (storage, batch layout, tensor library, the flat SCM of #1), the decision is:
**the seed is the corpus.** Dataset `i` is a pure function of its key `(seed, index)`, the
hyperprior with its overrides and the installed versions of TabDGP and numpy, whose generator
promises no stable streams across its releases; the dataset records both. The public call
draws one dataset, `draw(seed, index, hyperprior) -> Dataset`, in numpy, at the dataset's own
shapes. TabDGP never pads, never batches across datasets and never writes a file.

## Considered options

- **Online generation or a stored corpus.** A stored corpus buys reproducibility and a frozen
  schema at the cost of disk, plumbing and an incompatibility that stays hidden until the next
  regeneration. With a deterministic generator the two are the same data: a consumer
  generates per epoch from `(seed, index)`, keeps variety by changing the seed and snapshots
  only what carries expensive references (its validation and test sets). Trimming stored full
  tables was rejected: subsampling rows breaks the imbalance draw and the group-level
  structure, and a draw at the needed size is cheap.
- **Determinism per dataset or per batch.** Per batch, `(seed, batch)`, is cheaper but gives a
  dataset no identity outside its batch, so bucketing by size or resharding changes the data.
  Per dataset, bitwise on CPU whatever the batch, worker or shard, is what a virtual corpus
  needs. Bitwise identity across devices is not promised: it is a promise no library keeps
  across versions, and the draw runs on CPU. Nor across CPU types or numpy builds of one
  version: `uniform` fuses its multiply-add on arm64, the beta and gamma samplers, `exp`,
  `tanh`, `sin` and matrix products differ in the last bit, and a rejection sampler that
  accepts on one machine and not the other shifts the rest of the dataset's stream. A
  cluster's workers draw one corpus only on one CPU type.
- **Batch layout: a loop over datasets, or one padded batch.** Datasets in a batch differ in
  rows, columns, groups and in their generating graph (node count, edges, mechanism per node,
  marginals, post-hoc types), so a padded batch means masks through every mechanism and a
  padded adjacency, TabICL's road, which TabICL itself left. A padded batch also breaks
  per-dataset determinism on a GPU, where matrix kernels are chosen by shape. A loop over
  datasets, each vectorized over its own rows, groups and columns, has no masks and no way for
  dataset `i` to depend on its neighbors. GPU generation was weighed and rejected for the
  same reason: a dataset of about 2,000 rows by 20 columns is launch-bound, scatter and
  index-add (group aggregations) are non-deterministic on CUDA by default, and a GPU node's
  idle CPU cores draw in parallel instead.
- **Tensor library.** JAX's `vmap` and `jit` are what this layout does not use, and shapes
  that differ per dataset leave it eager and slow. torch buys metabeta's stack at the cost of a
  heavy dependency on every standalone user, a keyed sampler catalog to maintain (its
  `distributions` take no generator except `Gamma`), and five to ten times numpy's
  per-operation overhead on small CPU arrays. numpy's `Generator` is a complete keyed
  catalog, `default_rng([seed, index])` is the key, dynamic shapes are free, and metabeta's
  tensors are `torch.from_numpy`, zero copy. scipy may be added when it supplies a sampler
  numpy lacks.
- **Public call: a batch or a dataset.** A stateful generator argument makes dataset 256
  depend on how much randomness datasets 0 to 255 consumed, so the key must be in the
  signature. A `Batch` indexable to a `Dataset` is a list comprehension the caller writes as
  well as we do, and the consumer that buckets by size calls per dataset anyway, with its
  sizes as point overrides on a copied hyperprior.
- **Container: named arrays or one matrix.** One `(N, p)` float matrix stores categoricals and
  group indices as floats, and both consumers derive what they need from the metadata anyway.
  `Dataset` holds the columns as named one-dimensional arrays in their natural dtype, the
  column metadata, the generating graph with every node including the latent ones, the latent
  nodes' values, the hyperparameters every field resolved to, and the key. A field each column
  or node draws for itself is recorded with that column or in the graph instead (#16).

## Consequences

- One public function, one container, no batch object, no storage code, no padding code and
  no parallelism inside TabDGP: a draw is a pure function of its key, so a process pool over
  indices is the caller's one line, and nesting a pool inside a consumer's dataloader workers
  would be wrong.
- The end-to-end test draws dataset `i` alone and inside two different index sets and asserts
  bitwise equality; the proofs draw their fixed-seed datasets by index and store nothing.
- The hyperprior and its resolution share one Pydantic type: an override is a point value or
  a distribution on a field, a resolved hyperparameter a point value on the same field. A
  field each column or node draws for itself keeps its override there, and the column
  metadata and the graph record the values drawn (#16).
- Throughput is a bar, not a hope: a node's CPU workers draw datasets at least as fast as its
  GPU trains on them. For metabeta (4 to 8 cores and one GPU per node, epochs of 4,192
  datasets in minibatches of 16 to 64) that is at least 1,000 datasets per second on 4
  workers at the mean v1 size, about 1 ms per dataset. The unpublished throughput lab note
  (`lab/2026-10-11-throughput/`) measured about 250 per second per core in float32 on an M3,
  so numpy holds; the draw is bound by arithmetic, not by the interpreter, so a padded torch
  batch on CPU buys nothing and one on the node's only GPU takes time from training. The
  bar was first set at 500 per core, before the consumer's numbers were known.
- Continuous columns, latent values and a GLM's or GLMM's ground truth (its parameters and
  deviation) are float32, the precision the network trains in, so
  `torch.from_numpy` hands them over unconverted; single precision also runs the
  activations three to four times faster.
- When online drawing falls behind, or several models (4 to 16) train at once, a consumer
  draws a corpus once, stores it and lets every run read it, dividing the cost; storage
  stays the consumer's, TabDGP still writes no file.
- A DataFrame is `pandas.DataFrame(dataset.columns)` and needs no dependency here.
- For `metabeta.mixed`'s slice A1: generate per epoch from `(seed, index)` in dataloader
  workers, pin TabDGP, snapshot validation and test sets, draw an epoch's sizes first (its
  envelope rule 8) and bucket by them before drawing contents, and keep a contract test on the
  schema at the boundary. These are its decisions to record.
