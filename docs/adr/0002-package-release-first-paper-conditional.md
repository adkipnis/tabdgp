---
status: accepted
date: 2026-10-11
---

# 0002 — Package release first; a paper only with a finding of its own

Three papers already come out of the project TabDGP serves: mixedbench's (registry census,
contract, suite, opponent lines; NeurIPS Datasets & Benchmarks 2027), mixedsurvey's (the GLMM
practice survey) and later metabeta's (amortized inference). Asked on 2026-10-11 whether TabDGP
earns a fourth once item 5 or 6 is reached, the decision is: **a proper package release and a
JOSS paper as a citable handle; the validation of the matched hyperprior as a section of the
metabeta paper; a standalone paper only if the per-quirk ablation and a second consumer give it
a finding the other papers do not carry.**

## Why item 5 or 6 alone is thin

- At item 5 TabDGP is a simulator plus two self-chosen certificates. The two-sample test and the
  containment test run on a summary vector we picked, against a bar we set, so they show
  consistency with the registry on those statistics, not that training on the generator helps.
- The new idea, a hyperprior for grouped tables measured against mixedbench's real tables
  instead of hand-tuned like TabICL's or TabPFN's prior, is a component. The evidence that a
  component matters is downstream: an amortized posterior trained on it against one trained on
  something simpler, scored on real tables. That evidence lives in metabeta.mixed, not here.
- A fourth paper from one project reads as salami slicing unless it carries a finding of its own.

## What would make a standalone paper

1. **The downstream ablation.** An amortized posterior, or a prior-data fitted network (PFN),
   trained on TabDGP, on an i.i.d. baseline prior and on a TabICL-style prior, scored on
   mixedbench's real tables against its reference posteriors. Needs metabeta.mixed's slice A1.
2. **Which quirks matter.** Each quirk axis switched off in turn, the downstream effect
   measured. Tells the field which properties of real tables a prior must reproduce; cheap
   once the training loop exists, and the result most likely to be cited outside this stack.
3. **A second consumer.** Item 6's survival extension for metasurv turns metabeta's covariate
   generator into a prior for tabular amortized inference in general.
4. **The quirk census** from mixedbench's measurement pass (roadmap item 3): how common group
   confounding, rare levels, heaping and near dependence are across real grouped tables. The
   mixedbench paper plans a registry census; which paper owns the quirk statistics is decided
   before the mixedbench paper is drafted, or the TabDGP paper is left with the ablation alone.

## Considered options

- **A standalone paper at item 5.** Rejected above: no downstream result, a self-set bar.
- **No paper at all, the package referenced from the metabeta paper.** The cheapest; loses a
  citable handle for standalone users and the per-quirk result, which belongs to no other
  paper.
- **Package release plus JOSS, validation in the metabeta paper, a standalone paper
  conditional on 1–3.** Chosen.

## Venues

| content | venue |
|---|---|
| package with the proofs, no downstream result | JOSS (Journal of Open Source Software): short, a DOI, the software is what is reviewed |
| plus the ablation and the quirk study | TMLR (Transactions on Machine Learning Research): rolling, judged on correctness |
| same, with a large per-quirk effect | NeurIPS Datasets & Benchmarks (not the same cycle as mixedbench's) or AISTATS |
| a test of reception first | the NeurIPS Table Representation Learning workshop |

## Consequences

- The roadmap gains a "Release and paper" section: the release with JOSS, the quirk census
  ownership, the ablations, and the conditional paper.
- The proofs' summary vector and bar are documented as consistency checks, never claimed as
  realism in the README or a paper.
- The per-quirk ablation is designed so that each quirk axis can be switched off by an
  override, which item 4's hyperprior already allows; no extra code is planned for it here.
