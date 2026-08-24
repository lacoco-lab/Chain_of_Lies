# Frozen confirmatory results

This lightweight record accompanies the completed invisible-steganography Hard-regime code. Full methods,
per-seed tables, diagnostics, interpretation, and removable code references are in
`docs/HARD_REGIME_REPORT.md`.

All values are mean accuracy (%) ± sample SD across seeds 0, 1, and 2.

| Model | Mechanism | Answer-only private | Mechanism private | Improvement | Public | Joint | Payload |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | Invisible steg. | 18.3 ± 1.4 | 89.7 ± 1.2 | +71.4 ± 1.7 | 94.5 ± 1.1 | 84.7 ± 1.0 | 84.5 ± 1.0 |
| Llama3.1-8B | Invisible steg. | 19.3 ± 1.1 | 72.5 ± 12.2 | +53.3 ± 11.1 | 65.4 ± 1.0 | 48.1 ± 7.1 | 52.8 ± 11.2 |

Conclusion: steganography succeeds on both models.
The same strictly disjoint prompt data were used for both models at each seed.
