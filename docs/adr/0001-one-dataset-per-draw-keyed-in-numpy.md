---
status: accepted
date: 2026-10-11
---

# 0001 - One dataset per draw. Each draw is keyed by a seed and an index. The backend is NumPy.

- TabDGP is an SCM-based (structural causal model) tabular dataset simulator for simulation-based inference models.
- Its first target space is grouped data.
- TabDGP draws covariates and outcome; a consumer pads, masks, collates and stores them: TabDGP never pads, never batches across datasets and never writes a file.
- A dataset is a pure function of its key `(seed, index)`:
  - the installed versions of TabDGP and NumPy are recorded for reproducibility.
  - a key reproduces its dataset bitwise on one CPU type and NumPy build (regardless of the batch, worker or shard).
  - the SCM hyperprior can be overridden at runtime.
- `drawDataset(seed, index, scm) -> Dataset` draws one dataset in NumPy. It can be flat or grouped. It can have no outcome or a node-based outcome.
- `modelOutcome(seed, index, covariates, model)` draws a model's outcome conditioned on the covariates (ADR 0007).
- A `Dataset` holds:
  - covariates as named one-dimensional arrays
  - the node-based outcome (if present)
  - metadata per column (type, levels, role, marginal and drawn parameters)
  - the generating graph with every node and the latent nodes' values
  - the resolved hyperparameters

## Considered options

| question | chosen | rejected |
|---|---|---|
| storage | drawn online from the key | a stored corpus |
| determinism | per dataset | per batch |
| batch layout | a loop over datasets | one padded batch |
| tensor library | NumPy | PyTorch, JAX |
| public call | one dataset | a batch |
| container | named arrays | one float matrix |

1. **Storage:**
   - A stored corpus buys reproducibility and a frozen schema, but costs disk and plumbing.
   - Inconsistency with the current code stays hidden until the next regeneration.
   - With a deterministic draw both (online and stored) are the same data. A consumer draws per epoch from `(seed, index)`, changes the seed for variety and stores only what carries expensive references (its validation and test sets).
   - Trimming stored tables to smaller subsets breaks the imbalance draw and the group structure; a draw at the needed size is cheap.
2. **Determinism:**
   - A key per batch, `(seed, batch)`, is cheaper, but gives a dataset no identity outside its batch, so bucketing by size or resharding changes the data.
   - Bitwise identity across devices cannot be promised, since no library keeps that promise across versions, and since the draw runs on CPU.
   - Bitwise identity across CPU types cannot be promised either: NumPy builds of one version differ in the last bit (e.g. `uniform` fuses its multiply-add on arm64, and the beta and gamma samplers, `exp`, `tanh`, `sin` and matrix products differ).
3. **Batch layout:**
   - Datasets in a batch differ in rows, columns, groups and generating graph (node count, edges, mechanism per node, marginals, post-hoc types).
   - A padded batch needs masks through every mechanism and a padded adjacency (see TabICL v1).
   - On a GPU, a padded batch also breaks per-dataset determinism, because matrix kernels are chosen by shape.
   - A loop over datasets needs no mask and no dataset can depend on its neighbors.
   - GPU generation is rejected too due to CUDA non-determinism. A GPU node's idle CPU cores can draw the datasets in parallel instead.
4. **Tensor library:**
   - JAX's `vmap` and `jit` are not useful for this layout; varying shapes per dataset would prevent compilation.
   - PyTorch would share a typical consumer's stack, but its `distributions` take no generator.
   - On small CPU arrays PyTorch costs five to ten times NumPy's overhead per operation.
   - NumPy's `Generator` is a complete keyed catalog, `SeedSequence([seed, index])` is the key, and shapes that differ per dataset cost nothing.
   - A consumer reads the arrays as tensors through `torch.from_numpy`, without a copy.
   - SciPy may be added if it adds required functionality that NumPy lacks.
5. **Public call:**
   - A stateful generator argument makes dataset 256 depend on how much randomness datasets 0 to 255 consumed, so the key is in the signature.
   - A `Batch` indexable to a `Dataset` is a list comprehension the caller writes.
   - A consumer that buckets by size calls per dataset anyway.
6. **Container:**
   - One float matrix stores categoricals and group codes as floats.
   - Both consumers derive what they need from the metadata anyway.

## Consequences

- **No batching inside TabDGP:** no batch object, storage, padding or parallelism.
  - Parallelism is the caller's (e.g. a process pool over indices, one line).
  - A pool inside TabDGP would nest inside a consumer's dataloader workers.
- **Tests:** the end-to-end test asserts that dataset `i` is bitwise identical when drawn alone and inside two different index sets. The proofs draw their datasets by index under fixed seeds and store none.
- **One type for both:** the hyperprior and its resolved version (with all distributions drawn from) share one Pydantic type.
  - An override is a point or a distribution on a field; the resolved hyperparameter is a point on the same field.
  - A field each column or node draws for itself keeps its override there; the column metadata and the graph record what it drew (#16).
- **Throughput:** a node's CPU workers draw datasets at least as fast as its GPU trains on them.
  - For MetaBeta (4 to 8 cores and one GPU per node, epochs of 4,192 datasets in minibatches of 16 to 64), that is at least 1,000 datasets per second on 4 workers.
  - NumPy meets this criterion at about 250 float32 draws per second per core on a MacBook M3.
- **float32:** continuous columns, latent values and the ground truth of a likelihood-based model (its parameters and deviation) are float32, which is the precision used in PyTorch.
  - Single precision also runs the activations three to four times faster.
- **Stored corpus:** when online drawing falls behind or several models (4 to 16) train at once, a consumer may store a corpus once, such that every run reads it (but this work is delegated entirely to the consumer).
