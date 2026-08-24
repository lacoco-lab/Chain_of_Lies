# S5 Medium calibration

This isolated seed-0 experiment searches for an S5 chain length where extra public computation
helps solve an independent private chain without piggybacking.

## Design

- Qwen2.5-7B-Instruct with the existing CE/LoRA protocol.
- One uniformly random initial arrangement shared by the public and private tasks.
- Independently sampled, equal-length public/private swap chains.
- Exact lengths: `1,2,3,4,5,6,8,10`.
- Balanced mixed training split: 1,250 examples per length, 10,000 total.
- Balanced validation split: 250 examples per length, 2,000 total.
- Exact public/private pairs are disjoint across training and validation. Individual state/swap
  primitives may repeat, avoiding the unintended held-out-primitive requirement in the first Easy
  S5 construction.
- Seed 0 and three epochs for every condition.
- Results are reported separately at every exact length.

Six conditions share the identical data:

1. `answer_only`;
2. ordinary `public_cot`;
3. meaningful `verbose_public_cot`;
4. ordinary public CoT plus 64 atomic filler tokens;
5. ordinary public CoT plus 128 atomic filler tokens;
6. ordinary public CoT plus 256 atomic filler tokens.

Filler is inserted at the target token-ID level after the ordinary public CoT and before the answer
block. The training runtime verifies that `.` is exactly one Qwen tokenizer token. Evaluation saves
generated token IDs so the report can verify the realized repeated-token run rather than infer it
from characters.

The filler conditions are **CE-trained filler workspaces**. They must not be described as emergent
inference-time filler use; that claim requires a larger-model inference experiment.

## Preregistered calibration rule

A length/mechanism pair is flagged as a Medium candidate when:

- ordinary public-CoT public exact is at least 0.90;
- ordinary public-CoT private exact is between 0.20 and 0.70;
- verbose/filler private exact reaches at least 0.70; and
- the private improvement over ordinary public CoT is at least 0.15.

The sweep is a calibration step. If a candidate emerges, generate a dedicated range-specific split
and confirm seed 0 before running seeds 1 and 2. If none emerges, do not force a Medium S5 claim.

## Cluster workflow

From `/home/momo00016/Faithfulness-Safety`, submit:

```bash
condor_submit experiments/s5_medium_calibration/condor/generate.sub
```

After generation finishes, one submission launches all six train-and-evaluate conditions in
parallel:

```bash
read -s HF_TOKEN
export HF_TOKEN
condor_submit experiments/s5_medium_calibration/condor/run_all.sub
unset HF_TOKEN
```

Paste the token at the silent `read` prompt and press Enter. This avoids putting the token in shell
history.

`run_all.sub` forwards only `HF_TOKEN` from the submit environment. The token is not stored in this
repository or printed to logs. Jobs use authenticated downloads, disable Hugging Face Xet, and
retry transient cache/download failures five times. A free read token is sufficient for the public
Qwen model.

After all six jobs complete:

```bash
condor_submit experiments/s5_medium_calibration/condor/summarize.sub
```

Results:

```text
artifacts/ce_s5_medium_calibration/REPORT.md
artifacts/ce_s5_medium_calibration/metrics_by_length.csv
artifacts/ce_s5_medium_calibration/summary.json
```

The verbose condition retains an 80 GB GPU request. Answer-only accepts 24 GB GPUs; ordinary-CoT
and filler request 40 GB. These are separate scheduler constraints: an idle job has not started the
code and generally means no matching GPU is currently available. Check a specific idle job with
`condor_q -better-analyze <job-id>`. If a 40 GB condition OOMs, raise only that row to 80 GB rather
than changing the experiment.

### Hugging Face 429 errors

Every Docker file-transfer job has a private cache, so parallel anonymous jobs otherwise download
the model independently from the same cluster IP. Hugging Face can rate-limit that IP. Before
submitting GPU jobs, create a free Hugging Face read token and export it as shown above. Once a job
starts, its `.out` log will say `HF_TOKEN present: yes`; the value itself is never printed. Do not
inspect the full Condor `Environment` attribute because some installations display secret values.

## Local checks

```bash
python -m unittest experiments/s5_medium_calibration/test_pipeline.py
bash -n experiments/s5_medium_calibration/run.sh
```
