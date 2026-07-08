# SIC Transfer Workflow

Fallback workflow for the SIC/conduit cluster when your personal shared scratch
quota is not writable.

This workflow avoids `/scratch/chair_mhahn/momo00016` entirely. It uses Condor
file transfer:

- Input code/generated_data is copied from `/home/momo00016/Faithfulness-Safety` into each
  job's temporary execute directory.
- Jobs run inside the cluster-provided local builder image:
  `/scratch/common/images/apptainer-builder.sif`.
- Each job builds a temporary PyTorch sandbox directory in its execute directory,
  creates a temporary venv, runs the requested S5 stage, and transfers outputs
  back to the submit-side repo under `/home/momo00016/Faithfulness-Safety`.

This is slower than the shared-scratch workflow because jobs do not share the
PyTorch image, venv, or Hugging Face cache.

## Setup

From the submit node:

```bash
cd /home/momo00016/Faithfulness-Safety
mkdir -p /home/momo00016/logs/chain_of_lies
chmod +x experiments/condor/s5_state_tracking/sic_transfer/*.sh
mkdir -p generated_data artifacts
```

## Optional Smoke Test

This verifies that the builder image can pull PyTorch and install Python deps in
a temporary job-local venv. It does not create a persistent venv.

```bash
condor_submit experiments/condor/s5_state_tracking/sic_transfer/setup_venv.sub
```

## S5 Workflow

Run stages sequentially. Wait for `condor_q` to show the stage is finished before
submitting the next one.

```bash
condor_submit experiments/condor/s5_state_tracking/sic_transfer/gen_splits_s5.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/train_ce_s5.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/train_ce_s5_public_cot_highmem.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/train_ce_s5_verbose_public_cot_highmem.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/eval_ce_s5.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/eval_ce_s5_verbose_public_cot.sub
condor_submit experiments/condor/s5_state_tracking/sic_transfer/summarize_ce_s5.sub
```

`train_ce_s5.sub` runs only the short `answer_only` jobs and requests 40GB+
GPUs. `train_ce_s5_public_cot_highmem.sub` runs the long `public_cot` jobs and
requests 80GB+ GPUs, because `public_cot` OOMs on 40GB A100s.
`train_ce_s5_verbose_public_cot_highmem.sub` runs the longer 3-epoch
`verbose_public_cot` jobs and also requests 80GB+ GPUs.

Outputs are transferred back to:

```text
generated_data/
artifacts/ce_s5/
```

Logs are written to:

```text
/home/momo00016/logs/chain_of_lies/sic_transfer_*.{out,err,log}
```

## Tradeoffs

- Every job builds a temporary PyTorch sandbox in the Condor execute directory.
- Every job creates its own temporary venv.
- Model weights are downloaded into the job-local Hugging Face cache each time.
- If this works, it is a no-admin workaround. If it is too slow, the real fix is
  still to get a usable shared scratch quota or a site-provided PyTorch `.sif`.
