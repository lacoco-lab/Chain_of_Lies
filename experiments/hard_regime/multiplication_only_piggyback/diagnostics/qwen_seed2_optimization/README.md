# Disposable Qwen seed-2 optimization diagnostic

Purpose: determine whether the low Qwen multiplication-only piggyback result on data seed 2
(`35.0%` private exact) is tied to that dataset or to optimization randomness.

This diagnostic freezes the exact seed-2 training/evaluation records and retrains only:

- model: `Qwen/Qwen2.5-7B-Instruct`;
- variant: `arith_piggyback`;
- supervision: `public_cot`;
- optimization seeds: `3,4,5`.

Everything else is loaded from and SHA-pinned to `../../config.json`. Each of the three jobs trains
one adapter for the original 5,000 steps and evaluates all 1,000 held-out records. Existing
base-Qwen responses are reused, saving an unnecessary second inference pass. The original
optimization-seed-2 result is included by the summarizer but is never overwritten.

Do not select the best diagnostic seed as the paper result. If seeds 3–5 recover toward seeds 0/1,
the recipe is optimization-sensitive. If they remain near 35%, the fixed dataset is the likely
source. A subsequent method change must be applied to all paper seeds, not seed 2 alone.

## Cluster commands

Run from `/home/momo00016/Faithfulness-Safety` after the original multiplication-only outputs are
present:

```bash
condor_submit \
  -batch-name hard-mulonly-qwen-seed2-opt \
  experiments/hard_regime/multiplication_only_piggyback/diagnostics/qwen_seed2_optimization/condor/all_seeds.sub
```

This single submission queues three parallel GPU jobs. After all three finish successfully:

```bash
condor_submit \
  -batch-name hard-mulonly-qwen-seed2-opt-summary \
  experiments/hard_regime/multiplication_only_piggyback/diagnostics/qwen_seed2_optimization/condor/summarize.sub
```

Outputs are isolated under:

```text
artifacts/hard_regime/multiplication_only_piggyback_diagnostics/qwen_seed2_optimization/
generated_data/hard_regime/multiplication_only_piggyback_diagnostics/qwen_seed2_optimization_eval_responses/
```

After results are transferred and analyzed, the entire diagnostic output can be removed without
touching the paper experiment:

```bash
rm -rf \
  artifacts/hard_regime/multiplication_only_piggyback_diagnostics \
  generated_data/hard_regime/multiplication_only_piggyback_diagnostics
```
