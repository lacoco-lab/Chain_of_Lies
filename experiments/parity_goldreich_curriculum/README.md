# Goldreich parity curriculum and mechanism diagnostics

This is a clean replacement for the unstable long-run experiment in `experiments/parity_goldreich`.
It trains one adapter continuously through explicit stages and evaluates every epoch checkpoint on
held-out data. It does not introduce a private task.

## What the experiment isolates

1. `local_update`: one encrypted recurrence step
   `c_i = c_(i-1) XOR z_(i-1) XOR x_(2i-1) XOR x_(2i) XOR z_i`.
2. `supplied_{8,16,32,64}`: full encrypted traces when masks are given in the prompt.
3. `predicate_local`: the nonlinear five-bit Goldreich predicate by itself.
4. `mask_{8,16,32,64}`: generate masks from an unseen 16-bit seed using the fixed graph.
5. `joint_{8,16,32,64}`: generate masks and use them in the encrypted parity trace.

The update curriculum starts locally and increases trace length. The Goldreich curriculum resumes
the exact final update adapter, teaches the predicate and mask-generation tasks explicitly, and
then increases joint length. Each stage contains replay examples from earlier components to reduce
forgetting. LoRA rank is 32 instead of 8. Training requests deterministic PyTorch behavior.

Every epoch checkpoint is evaluated before training continues. Its responses and metadata are
retained; redundant epoch adapter weights are removed after evaluation to keep Condor transfers
manageable. Final adapters are evaluated on the whole diagnostic suite.

The report automatically identifies the first failed mechanism:

- local encrypted update;
- long supplied-mask recurrence;
- nonlinear predicate;
- fixed-position seed-to-mask generation;
- integration of masks and encrypted parity.

## Conditional next step

If `predicate_local` passes but `mask_*` or `joint_*` fails, test an explicit-hypergraph format.
Before each encrypted state, the CoT includes a fixed public hint such as
`(use Seed: A F B C E)`. This exposes the public five-position selection but not the corresponding
private seed values. The output must remain the encrypted state, not the mask bit. This removes
implicit hypergraph memorization as a possible adapter bottleneck without weakening the intended
privacy setup.

## Clean cluster run

Only remove the parity-specific old and partial-new directories—not the repository's complete
`generated_data/` or `artifacts/` roots:

```bash
rm -rf \
  generated_data/parity_goldreich \
  generated_data/parity_goldreich_eval_responses \
  generated_data/parity_goldreich_curriculum \
  generated_data/parity_goldreich_curriculum_responses \
  generated_data/parity_goldreich_curriculum_seed_0.tar.gz \
  generated_data/parity_goldreich_curriculum_update_responses.tar.gz \
  generated_data/parity_goldreich_curriculum_goldreich_responses.tar.gz \
  artifacts/ce_parity_goldreich \
  artifacts/ce_parity_goldreich_curriculum
```

Set the gated-model token, then submit the four-stage DAG:

```bash
read -s HF_TOKEN
export HF_TOKEN
condor_submit_dag experiments/parity_goldreich_curriculum/condor/workflow.dag
unset HF_TOKEN
```

The DAG's `ENV GET HF_TOKEN` directive copies the token into the DAGMan environment so that later
GPU nodes can forward it. Node-level `getenv` alone is insufficient because DAGMan, rather than the
interactive shell, submits those nodes.

The DAG runs sequentially:

```text
generate and validate -> encrypted-update curriculum -> Goldreich curriculum -> summarize
```

The two GPU stages may each run for several days. Their wall-time allowance is seven days.

Final outputs:

```text
artifacts/ce_parity_goldreich_curriculum/REPORT.md
artifacts/ce_parity_goldreich_curriculum/diagnostic_metrics.csv
artifacts/ce_parity_goldreich_curriculum/summary.json
```

If DAGMan is unavailable, submit the jobs manually and wait for each to finish before submitting
the next:

```bash
condor_submit experiments/parity_goldreich_curriculum/condor/generate.sub
condor_submit experiments/parity_goldreich_curriculum/condor/update.sub
condor_submit experiments/parity_goldreich_curriculum/condor/goldreich.sub
condor_submit experiments/parity_goldreich_curriculum/condor/summarize.sub
```

## Local validation

```bash
python3 -m unittest experiments/parity_goldreich_curriculum/test_pipeline.py
bash -n experiments/parity_goldreich_curriculum/run.sh
```

This remains a finite 16-bit mechanism experiment. Even full success would not establish
cryptographic security.
