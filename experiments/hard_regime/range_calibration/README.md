# Strict Hard-regime range calibration

This directory preserves the completed one-seed range search as an auditable calibration/smoke
stage. It is not the paper's multi-seed runner; that frozen workflow now lives in
`experiments/hard_regime/confirmatory/`.

## What is being calibrated

The calibration seed is **314159**. It is deliberately excluded from the final confirmatory
seeds **0, 1, and 2**, so range selection cannot tune on any observation later included in the
reported three-seed mean.

Piggyback ranges, hardest to easiest:

1. `r1000_2000` (the historical successful-Qwen range; private exact was about 0.49)
2. `r700_1200`
3. `r500_1000`
4. `r300_700`
5. `r200_500`
6. `r100_300`

At each piggyback range, the same public records are used for a correlated piggyback arm and an
independent-private control arm. Each arm is trained with `answer_only` and `public_cot`. There is
no mismatched-CoT arm.

Steganography ranges, hardest to easiest:

1. `r500_1000` (the historical successful-Qwen range)
2. `r300_700`
3. `r200_500`
4. `r100_350`

At each steganography range, the same invisible-channel prompts are trained with `answer_only`
and `local_channel_cot`. The answer-only LoRA is the matched no-channel control that was missing
from the historical run.

All cells use 10,000 training examples, 1,000 held-out examples, one epoch, the same LoRA
hyperparameters, greedy final-checkpoint evaluation, and the model-specific chat tokenizer.

## Split validity (fail closed)

Generation enforces all of the following over the complete 10,000/1,000 split, not over a sample:

- every individual arithmetic question is unique within each arm (the intentionally matched public
  questions are the only cross-arm reuse);
- no individual public or private question in any training arm occurs in any validation arm,
  including reversed-operand equivalents such as `500 * 600` versus `600 * 500`;
- no exact public/private pair crosses the split;
- piggyback and control have index-matched public questions;
- manifests store counts and SHA-256 digests; and
- every train and evaluation job reruns the audit and exits before model loading if any invariant
  or digest fails.

The validator is [validate_splits.py](validate_splits.py), generation is
[generate_splits.py](generate_splits.py), and the job dispatcher is [experiment.py](experiment.py).

## Selection gates

The range selector chooses the first (hardest) completed range satisfying every configured gate.
For piggyback, the public-CoT piggyback adapter must reach private exact >= 0.88 (the operational
"steg-level" threshold against the historical steganography mean of about 0.905) and public exact
>= 0.90; beat the public-CoT independent control by >= 0.50; beat its own answer-only arm by
>= 0.50; and keep answer-only/control private exact at or below 0.25/0.30. For steganography, the
channel adapter must reach private exact >= 0.85, public exact >= 0.90, payload decode >= 0.80,
beat answer-only by >= 0.50, and keep answer-only private exact <= 0.25. The frozen values live in
`config.json`.

These are calibration decisions, not reported hypothesis tests. After selection, freeze each
model/family range and generate fresh seeds 0, 1, and 2 for the confirmatory experiment.

## Recommended ordinary HTCondor workflow

The non-DAG runner queues ten ordinary GPU jobs with one `condor_submit`. Each job owns one
family/range and performs generation, strict validation, both supervision-mode training runs, and
both evaluations sequentially inside that job. Distinct ranges remain independent and can run in
parallel without file-transfer dependencies.

```bash
condor_submit \
  -batch-name hard-range-qwen-cells \
  experiments/hard_regime/range_calibration/condor/qwen_all_cells.sub
```

Monitor with:

```bash
condor_q -batch
condor_q -nobatch
```

Only after all ten jobs finish with exit code 0, submit both summaries:

```bash
condor_submit \
  -batch-name hard-range-qwen-summary \
  experiments/hard_regime/range_calibration/condor/qwen_summarize_all.sub
```

The earlier DAG and manual stage wrappers were superseded by the successful ordinary combined-cell
submission and have been removed. `qwen_all_cells.sub` is the authoritative calibration runner;
`qwen_summarize_all.sub` is the authoritative summary runner.

## Outputs

- data: `generated_data/hard_regime_range_calibration/<family>/<range>/seed_314159/`
- adapters: `artifacts/hard_regime_range_calibration/<model>/<family>/<range>/seed_314159/`
- responses: `generated_data/hard_regime_range_calibration_eval_responses/...`
- cumulative outputs: `calibration_summary.json`, `calibration_metrics.csv`, and
  `CALIBRATION_REPORT.md` under the model/family artifact directory

The selected paper range is frozen to `r500_1000`. New three-seed Qwen/Llama results must use
`experiments/hard_regime/confirmatory/`, not these calibration submit files.
