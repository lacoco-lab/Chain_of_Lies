# Active Condor Workflow

These are the only Condor jobs needed for the current arithmetic CoT-ablation experiment.
The default workflow now runs seeds `0`, `1`, and `2` for the intermediate `s,x = 500..1000`
verbosity confirmation.

Run them sequentially:

```bash
condor_submit experiments/condor/arithmetic_cot_ablation/reset_ce_cot_ablation.sub
condor_submit experiments/condor/arithmetic_cot_ablation/gen_splits_correlated_pair.sub
condor_submit experiments/condor/arithmetic_cot_ablation/train_ce_cot_ablation.sub
condor_submit experiments/condor/arithmetic_cot_ablation/train_ce_cot_ablation_verbose_highmem.sub
condor_submit experiments/condor/arithmetic_cot_ablation/eval_ce_cot_ablation.sub
condor_submit experiments/condor/arithmetic_cot_ablation/summarize_ce_cot_ablation.sub
```

`gen_splits_correlated_pair.sub` launches one split-generation job per seed.
`train_ce_cot_ablation.sub` launches the 12 non-verbose training jobs:

```text
3 seeds × 2 non-verbose modes × 2 variants
```

The 1-epoch verbose training jobs use `train_ce_cot_ablation_verbose_highmem.sub`, which requires
GPUs with at least 40GB of memory and launches:

```text
3 seeds × 1 verbose mode × 2 variants
```

`eval_ce_cot_ablation.sub` launches 18 jobs:

```text
3 seeds × 3 supervision modes × 2 variants
```

Supervision modes:

- `public_cot`
- `verbose_public_cot`
- `answer_only`

`verbose_public_cot` is a long audit-trail target with digit-by-digit carry addition. Its Condor
wrapper uses batch size `1` and `max_new_tokens=1024` to avoid the OOM seen with the shorter verbose
target at batch size `2`.

Variants:

- `arith_piggyback`
- `arith_piggyback_control`

Main outputs:

```text
artifacts/ce_cot_ablation/final_metrics.csv
artifacts/ce_cot_ablation/final_metrics_mean_std.csv
artifacts/ce_cot_ablation/no_cot_comparison_mean_std.csv
artifacts/ce_cot_ablation/piggyback_control_deltas_mean_std.csv
artifacts/ce_cot_ablation/training_curves_mean_std.csv
artifacts/ce_cot_ablation/plots/
```

Use `setup_venv.sub` only when the cluster environment needs to be recreated.

## S5 Diagnostic Workflow

The S5 diagnostic uses the same CE-only Condor style as the arithmetic case, but without
`verbose_public_cot`.

Run them sequentially:

```bash
condor_submit experiments/condor/s5_state_tracking/old_cluster/gen_splits_s5.sub
condor_submit experiments/condor/s5_state_tracking/old_cluster/train_ce_s5.sub
condor_submit experiments/condor/s5_state_tracking/old_cluster/eval_ce_s5.sub
condor_submit experiments/condor/s5_state_tracking/old_cluster/summarize_ce_s5.sub
```

`gen_splits_s5.sub` launches one split-generation job for seed `0` and writes:

```text
generated_data/prompt_splits/seed_0/s5_piggyback/
generated_data/prompt_splits/seed_0/s5_control/
```

`train_ce_s5.sub` launches 4 training jobs:

```text
1 seed × 2 modes × 2 variants
```

`eval_ce_s5.sub` launches the matching 4 evaluation jobs.

Supervision modes:

- `public_cot`
- `answer_only`

Variants:

- `s5_piggyback`
- `s5_control`

Main outputs:

```text
artifacts/ce_s5/final_metrics.csv
artifacts/ce_s5/final_metrics_mean_std.csv
artifacts/ce_s5/no_cot_comparison_mean_std.csv
artifacts/ce_s5/piggyback_control_deltas_mean_std.csv
artifacts/ce_s5/training_curves_mean_std.csv
artifacts/ce_s5/plots/
```

## Longer Verbose CE Follow-Up

To test whether the verbose target needs more CE exposure, run the 3-epoch verbose-only follow-up
after the normal split generation has completed:

```bash
condor_submit experiments/condor/arithmetic_cot_ablation/train_ce_cot_ablation_verbose_long_highmem.sub
condor_submit experiments/condor/arithmetic_cot_ablation/eval_ce_cot_ablation_verbose_long.sub
condor_submit experiments/condor/arithmetic_cot_ablation/summarize_ce_cot_ablation_verbose_long.sub
```

This writes to:

```text
artifacts/ce_cot_ablation_verbose_long/
```

It reuses the same `generated_data/prompt_splits/seed_*` prompts. The long jobs train only
`verbose_public_cot`, for 3 epochs, on both variants across seeds `0`, `1`, and `2`.

## Artifact Cleanup

Evaluation needs the selected adapter checkpoint under each `ckpt_task/` directory. Summarization
also uses each variant's top-level `train_history.json` and `training_metadata.json`, plus the
`per_variant_eval/` JSON files. The duplicate top-level adapter/tokenizer files can be removed after
training/eval:

```bash
python scripts/prune_ce_artifacts.py --root artifacts/ce_cot_ablation --execute
python scripts/prune_ce_artifacts.py --root artifacts/ce_cot_ablation_verbose_long --execute
```

Omit `--execute` first if you want a dry run.
