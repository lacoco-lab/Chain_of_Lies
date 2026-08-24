# Preliminary filler gate: SIC transfer workflow

This uses the same current-cluster structure as the S5 and arithmetic-steganography SIC jobs:

- `initialdir = /home/momo00016/Faithfulness-Safety`;
- `universe = container`;
- `/scratch/common/images/apptainer-builder.sif`;
- Condor input/output transfer;
- a temporary per-job PyTorch sandbox and virtual environment.

Submit from the repository on the submit node:

```bash
cd /home/momo00016/Faithfulness-Safety
condor_submit experiments/condor/preliminary_filler_gate/sic_transfer/run_preliminary_filler_gate.sub
```

The job validates the data and then evaluates:

```text
40 main items × 6 filler lengths = 240 generations
40 component probes at K=0       =  40 generations
total                            = 280 generations
```

It requests one GPU with at least 24 GB of memory. Results transfer back to:

```text
preliminary_filler_gate/runs/qwen25_7b_default/
```

That directory contains raw predictions, detailed CSVs, aggregate metrics, a plot, and
`REPORT.md`.

The result directory is included when the preliminary experiment directory is transferred into a
new job, so resubmission can skip prediction keys already completed. The runner checks the source
configuration hash before resuming.

