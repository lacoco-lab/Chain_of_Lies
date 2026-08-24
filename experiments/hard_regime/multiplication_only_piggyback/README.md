# Final multiplication-only piggyback experiment

This is the authoritative paper experiment for arithmetic piggybacking. Every question is pure
multiplication (`s*x`); it supersedes the earlier affine piggyback formulation (`s*x+y`).

## Frozen design

- Models: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.
- Seeds: 0, 1, 2.
- Operand range: 500–1000 inclusive.
- 10,000 training and 1,000 held-out examples per seed and variant.
- Piggyback pair: public `s*x`; private `s*(x+delta)` with the same wrapped shifts as before.
- Matched control: the identical public records with independently sampled private multiplication.
- Modes: `answer_only` and `public_cot`; no mismatched-CoT condition.
- Same one-epoch LoRA, optimizer, decoding, and checkpoint settings as the affine confirmation.

The only intended task change is removal of `+y`. The prompt is otherwise copied from the affine
experiment, with both offset-specific instructions removed. Core CE training already parses a
missing offset as zero and emits a multiplication-only public trace.

## Leakage guarantees

Generation is joint across all three seeds and fails closed on:

- any individual or commutative-equivalent question reuse within the suite;
- any train/evaluation component or pair overlap;
- any component overlap across seeds;
- unmatched public records between piggyback and control;
- affine syntax, an incorrect answer, an invalid shift relation, or offset wording in a prompt;
- any manifest count or SHA-256 digest mismatch.

Because all three seeds require 99,000 distinct multiplication questions from the finite
500–1000 domain, generation must be run once as a combined CPU job. Qwen and Llama then consume the
same seed-specific JSON records.

## Cluster workflow (ordinary Condor, no DAG)

Run from `/home/momo00016/Faithfulness-Safety`. Create the log directory if necessary:

```bash
mkdir -p /home/momo00016/logs/chain_of_lies
```

Generate and validate all splits:

```bash
condor_submit -batch-name hard-mulonly-piggy-generate \
  experiments/hard_regime/multiplication_only_piggyback/condor/generate.sub
```

Wait for successful completion and verify that
`generated_data/hard_regime/multiplication_only_piggyback/split_manifest.json` exists. Then expose
a Hugging Face token with accepted Llama access:

```bash
read -s HF_TOKEN
export HF_TOKEN
```

Submit all six combined GPU cells:

```bash
condor_submit -batch-name hard-mulonly-piggy-all \
  experiments/hard_regime/multiplication_only_piggyback/condor/all_cells.sub
```

Each job runs both variants × both modes, followed by complete greedy evaluation. Outputs transfer
only on exit code zero. After all six cells finish successfully, submit the lightweight summary:

```bash
condor_submit -batch-name hard-mulonly-piggy-summary \
  experiments/hard_regime/multiplication_only_piggyback/condor/summarize.sub
unset HF_TOKEN
```

The summary transfer lists only provenance and evaluation reports; it does not copy adapters into
the CPU sandbox.

## Outputs

```text
generated_data/hard_regime/multiplication_only_piggyback/
generated_data/hard_regime/multiplication_only_piggyback_eval_responses/
artifacts/hard_regime/multiplication_only_piggyback/
```

`summary.json` and the four CSV files under the artifact root contain all per-seed metrics,
three-seed aggregates, paired contrasts, and the frozen descriptive comparison to the earlier
affine run. The multiplication-only rows are the primary paper result.

The fixed-data Qwen seed-2 optimization diagnostic is documented under
`diagnostics/qwen_seed2_optimization/`. It demonstrates optimization sensitivity and is not a
replacement for the original primary seed-2 result.

## Local checks

```bash
python3 -m unittest experiments/hard_regime/multiplication_only_piggyback/test_pipeline.py
bash -n experiments/hard_regime/multiplication_only_piggyback/run.sh
```
