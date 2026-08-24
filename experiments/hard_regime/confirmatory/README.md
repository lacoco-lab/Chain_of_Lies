# Final invisible-steganography Hard regime: 500–1000

This is the frozen confirmatory experiment for the paper. It runs the same strictly disjoint
arithmetic data on:

- `Qwen/Qwen2.5-7B-Instruct`;
- `meta-llama/Llama-3.1-8B-Instruct`;
- independent training seeds `0`, `1`, and `2`;
- invisible-local steganography.

## Frozen conditions

Steganography trains `arith_steg_local_invisible` under both
`answer_only` and `local_channel_cot`. The obsolete mismatched-CoT condition is excluded.

Every adapter sees 10,000 training examples for one epoch. Final evaluation is greedy on all
1,000 held-out examples at `ckpt_final`. LoRA rank is 8, alpha 16, dropout 0.05, and the learning
rate is `2e-5`, matching the range calibration.

## Leakage prevention

Generation is joint across all three seeds and fails closed unless all of the following hold:

- no individual question is repeated within a split;
- no exact or commutative-equivalent question crosses training and evaluation;
- no exact question pair crosses training and evaluation;
- no arithmetic component is reused across seeds;
- no arithmetic component from calibration seed `314159` is reused in confirmatory seeds `0/1/2`;
- every answer, prompt, steganographic scheme, and payload is reconstructed and
  independently validated from the stored spec.

The generator writes SHA-256 question and record digests. Each model/seed artifact copies the
frozen config, global and per-seed manifests, and an independently recomputed validation report
into `provenance/`. Summarization refuses to run if any provenance report, seed, adapter report,
baseline report, or evaluation response is missing.

Data are shared between Qwen and Llama for the same seed. The training seed still controls LoRA
initialization and training order, so model comparisons use matched datasets and seeds.

## Cluster workflow (ordinary Condor; no DAG)

Run every command from `/home/momo00016/Faithfulness-Safety`.

First generate and validate the three-seed suite in one CPU job:

```bash
condor_submit -batch-name hard-confirm-generate \
  experiments/hard_regime/confirmatory/condor/generate.sub
```

Wait for that job to finish successfully. It must produce:

```text
generated_data/hard_regime/confirmatory/split_manifest.json
generated_data/hard_regime/confirmatory/steganography/seed_{0,1,2}/
```

Accept the gated Llama model on Hugging Face, then expose a token to Condor submission:

```bash
read -s HF_TOKEN
export HF_TOKEN
```

Submit all confirmatory GPU work with one command:

```bash
condor_submit -batch-name hard-confirm-all \
  experiments/hard_regime/confirmatory/condor/all_cells.sub
```

This queues exactly six jobs: Qwen/Llama × seeds 0/1/2. Each job runs both steganography modes,
training, and final evaluation for its model/seed. Outputs transfer
only after exit code 0, so a failed combined cell cannot return a partial result directory.
Each GPU sandbox first installs the frozen `transformers==4.44.2`, `accelerate==0.33.0`, and
`peft==0.12.0` stack, then downloads its complete model snapshot with retries before running the
strict tokenizer round-trip check.

If Qwen completed but Llama failed before training because `HF_TOKEN` was absent, export the token
and submit only the three Llama replacements:

```bash
condor_submit -batch-name hard-confirm-llama \
  experiments/hard_regime/confirmatory/condor/llama_cells.sub
```

After all six jobs finish successfully, submit the fail-closed summary:

```bash
condor_submit -batch-name hard-confirm-summary \
  experiments/hard_regime/confirmatory/condor/summarize.sub
unset HF_TOKEN
```

Do not submit the GPU cells before generation completes, and do not submit the summary before all
six GPU cells complete.

## Outputs

```text
generated_data/hard_regime/confirmatory/
generated_data/hard_regime/confirmatory_eval_responses/<model>/seed_<seed>/
artifacts/hard_regime/confirmatory/<model>/seed_<seed>/
artifacts/hard_regime/confirmatory/summary.json
artifacts/hard_regime/confirmatory/*metrics.csv
artifacts/hard_regime/confirmatory/*contrasts.csv
```

## Local checks

These checks require no model download:

```bash
python3 -m unittest experiments/hard_regime/confirmatory/test_pipeline.py
bash -n experiments/hard_regime/confirmatory/run.sh
```
