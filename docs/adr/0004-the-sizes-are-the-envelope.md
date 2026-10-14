---
status: accepted
date: 2026-10-14
amended: 2026-10-14, #24: a coarser factor has fewer groups than the finer one, and a point `groups` cannot nest
---

# 0004 — The size defaults are metabeta's envelope; presets, not overrides, choose the kind of table

ADR 0001 and root ADR 0005 of metabeta-v3 let `metabeta.mixed` pass the sizes of its envelope
rules 7 and 8 as overrides. Decided on 2026-10-14, with ADR 0003: all simulation happens in
TabDGP, and metabeta draws no size of its own. So **rules 7 and 8 are TabDGP's size defaults,
a count bounded by another field is drawn from its distribution truncated at that field's
resolved value, and the kind of table is a preset of the hyperprior: `Hyperprior.grouped()`,
metabeta's corpus, and `Hyperprior.flat()`; `draw(seed, index, hyperprior)` takes a preset
and has no default.** What the proofs certify is `grouped()`.

## Decisions

- **The group-size draw is rule 7**, since the Gamma(κ) weights of the prior-art search
  normalise to the symmetric Dirichlet(α) the registry fitted per factor: with probability
  `balanced_share` the groups are balanced (n / G each, the remainder one more for the first
  groups); otherwise one row per group, and the remaining n − G rows multinomial over
  Dirichlet(α) proportions. Two per-factor fields; G non-empty groups always; no retry.
- **Truncation at resolve.** `rows` resolves first; `groups` draws from its distribution
  truncated at `rows`, a coarser factor's `groups` below the finer factor's, and
  `design_columns` at `rows / 2` (rule 8's one floor, n ≥ 2d, which item 1 did not enforce).
  A draw clipped afterwards piles mass on the bound: from `LogUniformInt(2, 400)` at
  n = 20, 57 % of draws would become all-singleton tables.
- **A coarser factor has fewer groups than the finer one.** Its `groups` is truncated at one
  below the finer factor's, so a nested pair is never one partition relabelled. Each factor's
  floor is 2 plus the number of coarser factors nested above it, so that every coarser one
  still gets its 2 groups; a chain of L factors needs at least L + 1 rows, and a `rows`
  override below that fails at resolve. A point `groups` gives every factor the same count
  and cannot nest: it fails at construction unless `factors` stays at most 1 or
  `nested_share` at 0.
- **Presets.** `factors` allows 0, and `Hyperprior.flat()` sets it to 0; `grouped()` is the
  default below. A later kind of table (item response, survival, causal) is a constructor
  that fixes fields, never a branch in `draw` and never a new mechanism. Neither metabeta nor a
  standalone user passes an override; an override serves a study, an ablation or a size
  outside a preset.
- **The defaults of `grouped()`**, round numbers, the first set agreed in the grill:

  | field | default | source |
  |---|---|---|
  | `factors` | `UniformInt(1, 3)` | rule 3 (counts uniform); registry p90 2, max 5; v1 edge 3 |
  | `groups`, per factor | `LogUniformInt(2, 400)`, truncated at `rows` or below the finer factor's | rule 8; registry p50 32, p90 361 |
  | `balanced_share` | `1/3` | rule 7; registry 0.388 |
  | `size_alpha`, per factor | `LogUniform(0.5, 500)` | rule 7; registry p5 0.97, median 50 |
  | `nested_share` | `Uniform(0.2, 0.6)` | a guess; registry 134 nested to 184 crossed among two-factor tables |
  | `group_level_share` | `Beta(2, 4)` | a guess at the registry's 31 % |
  | `group_share`, per node | `Beta(0.5, 2)` | a guess, mass near zero |

  The registry's smallest α, 0.025, lies below the floor; item 4's containment test decides
  whether to widen it.

## Considered options

- **Clip after the draw** over truncation: rejected above.
- **A coarser factor with at most as many groups as the finer one**, the first draft: equal
  counts put one finer group in each coarser group, the same partition relabelled, two random
  effects no generalized linear mixed model (GLMM) tells apart. Forced when the finer factor
  has 2 groups, it hit 15.9 % of nested pairs under `grouped()`. Turning such a pair crossed
  was the other fix: it honours `nested_share` only sometimes.
- **A kwarg on `draw`** for the kind of table, `flat: bool` or a `Literal` of kinds: a bool
  is not future-proof, a `Literal` makes every kind a branch inside `draw` that can contradict
  an override. A preset is the hyperprior itself.
- **`factors` from 0 by default**, metabeta overriding to 1: the defaults are what metabeta
  trains on, so the grouped corpus is the preset and the flat table the other preset.

## Consequences

- ADR 0001's "metabeta passes its sizes" is amended: metabeta passes nothing and calls
  `draw(seed, index, Hyperprior.grouped())`. metabeta's `envelope.md` (rules 7–9) and its
  `packages/mixed` ADR 0005 say the same on their side.
- The proofs run on `grouped()`; mixedbench#72's flat sample tests `flat()`.
- `hyperprior` gains the per-factor fields, the presets and the truncation; a bound names
  the field it depends on, and `resolveDataset` resolves in dependency order.
