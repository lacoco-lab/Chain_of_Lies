# Invisible-Character Three-Seed Replication

Run from `/home/momo00016/Faithfulness-Safety` on the SIC transfer cluster.

Seed 0 is complete. The active follow-up runs only `arith_steg_local_invisible`
for seeds 1 and 2, producing a three-seed result. Every seed uses 10,000 training
and 1,000 validation pairs, ordinary autoregressive CE, and one epoch.

The invisible-character channel places each complete private value in a
delimited invisible sidecar immediately after its complete public counterpart.
Its decimal digits use a fixed ten-character variation-selector codebook. The
payload is the complete aligned private multiplication trace: operands,
place-value parts, partial products, and final sum.

Do not delete the completed seed-0 artifacts. First generate seeds 1 and 2; this
single submission queues two isolated CPU jobs:

```bash
condor_submit experiments/condor/arithmetic_steganography/sic_transfer/gen_splits_steg_local.sub
```

After both generation jobs finish successfully, submit training. This queues
two GPU jobs, one per seed, which are safe to run concurrently:

```bash
condor_submit experiments/condor/arithmetic_steganography/sic_transfer/train_ce_steg_local_invisible.sub
```

After both training jobs finish, submit evaluation. This also queues two
parallel-safe GPU jobs:

```bash
condor_submit experiments/condor/arithmetic_steganography/sic_transfer/eval_ce_steg_local_invisible.sub
```

After both evaluations finish, run the single summary job:

```bash
condor_submit experiments/condor/arithmetic_steganography/sic_transfer/summarize_ce_steg_local.sub
```

Important outputs:

```text
generated_data/prompt_splits/seed_{1,2}/arith_steg_local_invisible/
generated_data/eval_responses/ce_steganography_local/seed_{1,2}/
artifacts/ce_steganography_local/seed_{0,1,2}/
artifacts/ce_steganography_local/final_metrics.csv
artifacts/ce_steganography_local/final_metrics_mean_std.csv
```

Each split-generation job checks exact target decoding and fails early unless
the ten invisible characters and complete sample targets survive the Qwen
tokenizer exactly. Inputs and outputs are isolated by seed, so jobs within a
generation, training, or evaluation stage cannot overwrite one another.
