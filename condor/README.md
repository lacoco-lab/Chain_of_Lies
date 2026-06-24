# Active Condor Workflow

These are the only Condor jobs needed for the current arithmetic CoT-ablation experiment.
The default workflow now runs seeds `0`, `1`, and `2` for the intermediate `s,x = 500..1000`
verbosity confirmation.

Run them sequentially:

```bash
condor_submit condor/reset_ce_cot_ablation.sub
condor_submit condor/gen_splits_correlated_pair.sub
condor_submit condor/train_ce_cot_ablation.sub
condor_submit condor/train_ce_cot_ablation_verbose_highmem.sub
condor_submit condor/eval_ce_cot_ablation.sub
condor_submit condor/summarize_ce_cot_ablation.sub
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

## Longer Verbose CE Follow-Up

To test whether the verbose target needs more CE exposure, run the 3-epoch verbose-only follow-up
after the normal split generation has completed:

```bash
condor_submit condor/train_ce_cot_ablation_verbose_long_highmem.sub
condor_submit condor/eval_ce_cot_ablation_verbose_long.sub
condor_submit condor/summarize_ce_cot_ablation_verbose_long.sub
```

This writes to:

```text
artifacts/ce_cot_ablation_verbose_long/
```

It reuses the same `data/RL_splits/seed_*` prompts. The long jobs train only
`verbose_public_cot`, for 3 epochs, on both variants across seeds `0`, `1`, and `2`.

## Artifact Cleanup

Evaluation needs the selected adapter checkpoint under each `ckpt_task/` directory. Summarization
also uses each variant's top-level `train_history.json` and `training_metadata.json`, plus the
`per_variant_eval/` JSON files. The duplicate top-level adapter/tokenizer files can be removed after
training/eval:

```bash
python scripts/prune_ce_cot_ablation_artifacts.py --root artifacts/ce_cot_ablation --execute
python scripts/prune_ce_cot_ablation_artifacts.py --root artifacts/ce_cot_ablation_verbose_long --execute
```

Omit `--execute` first if you want a dry run.
