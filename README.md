# Faithfulness-Safety / Chain-of-Lies

This repository studies controlled paired-task settings where a model must answer:

- a public task, with visible reasoning allowed
- a private task, whose reasoning should not appear in the visible text

The main empirical question is whether the private task can piggyback on useful visible public
reasoning when the two tasks share computation.

Current completed results:

- arithmetic piggyback/control CoT ablation;
- simplified S5 state-tracking piggyback/control diagnostic over three seeds;
- simplified S5 verbose-CoT diagnostic over three seeds.

Current next direction: design controlled steganography setups for multiplication. Multiplication is
the preferred test bed because it is easy to explain to readers and piggybacking has a simple
intuition there: close-by numbers share computation.

## Active Workflow

Run these stages on the cluster, waiting for each stage to finish before starting the next:

```bash
condor_submit experiments/condor/arithmetic_cot_ablation/reset_ce_cot_ablation.sub
condor_submit experiments/condor/arithmetic_cot_ablation/gen_splits_correlated_pair.sub
condor_submit experiments/condor/arithmetic_cot_ablation/train_ce_cot_ablation.sub
condor_submit experiments/condor/arithmetic_cot_ablation/eval_ce_cot_ablation.sub
condor_submit experiments/condor/arithmetic_cot_ablation/summarize_ce_cot_ablation.sub
```

The experiment trains/evaluates three CE conditions for both `arith_piggyback` and
`arith_piggyback_control`:

- `public_cot`: decomposed visible public CoT plus final answer block
- `answer_only`: final answer block only, used as the no-CoT baseline
- `mismatched_public_cot`: decomposed public CoT from another example plus current answer block

For S5 on the SIC transfer cluster:

```bash
condor_submit experiments/condor/s5_state_tracking/sic_transfer/gen_splits_s5.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/train_ce_s5.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/train_ce_s5_public_cot_highmem.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/train_ce_s5_verbose_public_cot_highmem.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/eval_ce_s5.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/eval_ce_s5_verbose_public_cot.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/summarize_ce_s5.sub
```

The completed S5 `public_cot` result over seeds `0,1,2`:

- `s5_piggyback` private exact: `0.925 ± 0.034`
- `s5_control` private exact: `0.011 ± 0.002`
- piggyback minus control private exact: `+0.915 ± 0.032`

The completed S5 `verbose_public_cot` result over seeds `0,1,2`, evaluated with
`max_new_tokens=2048`:

- `s5_piggyback` private exact: `0.992 ± 0.010`
- `s5_control` private exact: `0.017 ± 0.006`
- piggyback minus control private exact: `+0.975 ± 0.015`

## Important Outputs

```text
artifacts/ce_cot_ablation/final_metrics.csv
artifacts/ce_cot_ablation/no_cot_comparison.csv
artifacts/ce_cot_ablation/piggyback_control_deltas.csv
artifacts/ce_cot_ablation/training_curves.csv
artifacts/ce_cot_ablation/plots/
artifacts/ce_s5/final_metrics_mean_std.csv
artifacts/ce_s5/piggyback_control_deltas_mean_std.csv
```

## Layout

```text
chain_of_lies/      Core package
  variants/         Variant-specific data generation code
  training/         CE training and shared training utilities
  evaluation/       Scoring, metrics, and model-evaluation helpers
  inference/        Batched generation helpers
scripts/            Python entry points
experiments/condor/ HTCondor wrappers grouped by experiment and cluster
docs/               Research and onboarding notes
generated_data/     Generated prompt splits and eval responses
artifacts/          Checkpoints, summaries, plots
```
