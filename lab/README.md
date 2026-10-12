# Lab notes

One directory per small design choice, `lab/YYYY-MM-DD-slug/`, its notebook committed with
outputs and never imported. Each row names the decision it reached.

| date | note | decision |
|---|---|---|
| 2026-10-11 | [throughput](2026-10-11-throughput/throughput.ipynb) | numpy confirmed: about 250 datasets per second per core in float32 at the mean v1 size, about 1,000 on 4 workers, the amended bar of ADR 0001; float32 columns; no torch fallback |
| 2026-10-12 | [column pooling](2026-10-12-column-pooling/pooling.ipynb) | the per-dataset fields of #16 drawn per column or node, independently (A): A shrinks the between-table spread of mean skewness by 19 % and of mean Spearman by 12 %, and a k-NN C2ST tells it from today at 0.54 (1,000 a side); the two-level designs are indistinguishable from A at their default alikeness, so no alikeness field until item 4's fit asks for one |
