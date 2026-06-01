# Active Condor Workflow

These are the only Condor jobs needed for the current arithmetic CoT-ablation experiment.
The default workflow runs three seeds: `0`, `1`, and `2`.

Run them sequentially:

```bash
condor_submit condor/reset_ce_cot_ablation.sub
condor_submit condor/gen_splits_correlated_pair.sub
condor_submit condor/train_ce_cot_ablation.sub
condor_submit condor/eval_ce_cot_ablation.sub
condor_submit condor/summarize_ce_cot_ablation.sub
```

`gen_splits_correlated_pair.sub` launches one split-generation job per seed.
`train_ce_cot_ablation.sub` and `eval_ce_cot_ablation.sub` launch 18 jobs each:

```text
3 seeds × 3 supervision modes × 2 variants
```

Supervision modes:

- `public_cot`
- `answer_only`
- `mismatched_public_cot`

Variants:

- `arith_piggyback`
- `arith_piggyback_control`

Main outputs:

```text
artifacts/ce_cot_ablation/final_metrics.csv                 # per seed
artifacts/ce_cot_ablation/final_metrics_mean_std.csv        # averaged over seeds
artifacts/ce_cot_ablation/no_cot_comparison_mean_std.csv
artifacts/ce_cot_ablation/piggyback_control_deltas_mean_std.csv
artifacts/ce_cot_ablation/training_curves_mean_std.csv
artifacts/ce_cot_ablation/plots/
```

Use `setup_venv.sub` only when the cluster environment needs to be recreated.
