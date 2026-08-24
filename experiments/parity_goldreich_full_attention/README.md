# Full attention-guided Goldreich encrypted CoT

This curriculum starts from the successful one-mask-bit adapter and trains complete encrypted parity
traces at 2, 4, 8, 16, 32, and 64 steps. It keeps mechanism bits, the final answer, fixed formatting,
and seed-position attention as separately normalized losses, so long fixed templates cannot dominate
the objective. One-bit and earlier-length replay reduce forgetting.

All lengths run sequentially inside one GPU allocation. This avoids six queue waits, model downloads,
environment installations, and model reloads. Each epoch is checked by actual autoregressive generation;
teacher-forced accuracy is reported separately. The final evaluation uses 512 examples per length with
seeds absent from both local and full-trace training.

Submit from the repository root:

```bash
condor_submit_dag experiments/parity_goldreich_full_attention/condor/workflow.dag
```

The final report is `artifacts/ce_parity_goldreich_full_attention/REPORT.md`.

If data generation has completed but training needs to be restarted, reuse the existing archive with:

```bash
condor_submit_dag experiments/parity_goldreich_full_attention/condor/workflow_resume.dag
```
