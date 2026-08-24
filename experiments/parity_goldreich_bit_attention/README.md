# Goldreich one-bit retrieval with attention supervision

This is a deliberately small gate before another full encrypted-CoT run. It asks the model for
one Goldreich mask bit from a labeled private 16-bit seed and a public five-label selector. The
target is only the variable bit, so fixed formatting cannot dominate the language-model loss.

The `bit_only` control and `bit_attention` treatment start independently from the same supplied-mask
curriculum adapter and see identical data. The treatment additionally rewards attention from the
bit-prediction position to the five selected seed-value tokens. Both jobs run in parallel after data
generation. Validation uses 1,024 seeds that never occur in training; 90% greedy accuracy is the
predeclared gate for attempting a full encrypted CoT later.

Submit from the repository root with:

```bash
condor_submit_dag experiments/parity_goldreich_bit_attention/condor/workflow.dag
```

The final report is `artifacts/ce_parity_goldreich_bit_attention/REPORT.md`.
