# Random-initial S5 replication

This directory is an isolated replication of the completed S5 experiment. It changes exactly one
experimental factor: instead of every task starting from `ABCDE`, every example samples a uniformly
random permutation of `ABCDE`. The sampled arrangement is shared by the public and private tasks,
just as both tasks shared the fixed `ABCDE` state in the original experiment.

Everything else is preserved:

- Qwen2.5-7B-Instruct with CE/LoRA training;
- 10,000 training and 1,000 held-out validation examples per variant and seed;
- seeds 0, 1, and 2;
- sequence lengths uniformly sampled from 10 through 19;
- piggybacking copies the public sequence and replaces only its final swap;
- the control samples independent public/private sequences with matched lengths;
- `answer_only` and `public_cot` use one epoch;
- `verbose_public_cot` uses the historical three-epoch setup;
- greedy evaluation at temperature 0;
- verbose evaluation uses 2,048 output tokens to avoid the truncation found in the first historical
  verbose evaluation.

No historical data or artifacts are overwritten. This experiment uses only:

```text
generated_data/s5_random_initial_replication/
generated_data/s5_random_initial_replication_eval_responses/
artifacts/ce_s5_random_initial_replication/
```

Deleting this directory and those three output roots removes the replication cleanly.

## Cluster workflow

Run from:

```bash
cd /home/momo00016/Faithfulness-Safety
chmod +x experiments/s5_random_initial_replication/run.sh
```

The jobs use the cluster's accepted Docker universe directly. They do not use nested Apptainer.

First generate all three deterministic splits:

```bash
condor_submit experiments/s5_random_initial_replication/condor/generate.sub
```

Wait until all three generation jobs complete. For the recommended seed-0 ordinary-CoT gate, one
submission now trains and then evaluates both variants:

```bash
condor_submit experiments/s5_random_initial_replication/condor/run_pilot.sub
```

Each evaluation generates both the pretrained baseline and fine-tuned predictions. Inspect:

```text
artifacts/ce_s5_random_initial_replication/seed_0/public_cot/s5_piggyback/per_variant_eval/ckpt_final/s5_piggyback.json
artifacts/ce_s5_random_initial_replication/seed_0/public_cot/s5_control/per_variant_eval/ckpt_final/s5_control.json
```

If the seed-0 piggybacking advantage remains, one further submission trains and evaluates all 16
remaining seed/mode/variant conditions in parallel:

```bash
condor_submit experiments/s5_random_initial_replication/condor/run_remaining_after_pilot.sub
```

To skip the gate and launch all 18 independent train-and-evaluate conditions at once, use this
instead of `run_pilot.sub` and `run_remaining_after_pilot.sub`:

```bash
condor_submit experiments/s5_random_initial_replication/condor/run_all.sub
```

Finally, after all evaluations complete:

```bash
condor_submit experiments/s5_random_initial_replication/condor/summarize.sub
```

The main aggregate tables are:

```text
artifacts/ce_s5_random_initial_replication/final_metrics_mean_std.csv
artifacts/ce_s5_random_initial_replication/piggyback_control_deltas_mean_std.csv
```

The original stage-specific `.sub` files remain available for recovery when only training or only
evaluation needs to be rerun. Do not submit them in addition to the combined jobs for the same
condition.

## GPU scheduling

The answer-only jobs accept GPUs with at least 40 GB. Standard and verbose public-CoT jobs retain
the historical requirement of an 80 GB GPU because the original S5 public-CoT training OOMed on
40 GB devices. Consequently, public-CoT jobs can remain idle when all 80 GB nodes are occupied;
this is scheduler waiting, not a missing-data failure. Confirm the exact scheduler reason with:

```bash
condor_q -better-analyze <cluster-id>
condor_status -constraint 'GPUs_GlobalMemoryMb >= 80000' \
  -af Machine GPUs_GlobalMemoryMb State Activity
```

## Local smoke test

The smoke test creates small temporary splits and verifies the shared initial state, exact final-swap
piggyback construction, independent control, correct state replay, disjoint splits, deterministic
regeneration, and use of the sampled state in the supervised public CoT.

```bash
python -m unittest experiments/s5_random_initial_replication/test_replication.py
```
