# Faithfulness-Safety / Chain-of-Lies

This repository studies a controlled arithmetic setting where a model must answer:

- a public task, with visible reasoning allowed
- a private task, whose reasoning should not appear in the visible text

The current active experiment is the arithmetic piggyback/control CoT ablation.

## Active Workflow

Run these stages on the cluster, waiting for each stage to finish before starting the next:

```bash
condor_submit condor/reset_ce_cot_ablation.sub
condor_submit condor/gen_splits_correlated_pair.sub
condor_submit condor/train_ce_cot_ablation.sub
condor_submit condor/eval_ce_cot_ablation.sub
condor_submit condor/summarize_ce_cot_ablation.sub
```

The experiment trains/evaluates three CE conditions for both `arith_piggyback` and
`arith_piggyback_control`:

- `public_cot`: decomposed visible public CoT plus final answer block
- `answer_only`: final answer block only, used as the no-CoT baseline
- `mismatched_public_cot`: decomposed public CoT from another example plus current answer block

## Important Outputs

```text
artifacts/ce_cot_ablation/final_metrics.csv
artifacts/ce_cot_ablation/no_cot_comparison.csv
artifacts/ce_cot_ablation/piggyback_control_deltas.csv
artifacts/ce_cot_ablation/training_curves.csv
artifacts/ce_cot_ablation/plots/
```

## Layout

```text
chain_of_lies/   Core package
scripts/         Python entry points
condor/          HTCondor wrappers for the active workflow
docs/            Research and onboarding notes
data/            Generated prompts/responses
artifacts/       Checkpoints, summaries, plots
```
