# Lab notes

One directory per small design choice, `lab/YYYY-MM-DD-slug/`, its notebook committed with
outputs and never imported. Each row names the decision it reached.

| date | note | decision |
|---|---|---|
| 2026-10-11 | [throughput](2026-10-11-throughput/throughput.ipynb) | numpy confirmed: about 250 datasets per second per core in float32 at the mean v1 size, about 1,000 on 4 workers, the amended bar of ADR 0001; float32 columns; no torch fallback |
