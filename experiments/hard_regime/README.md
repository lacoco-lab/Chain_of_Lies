# Hard-regime experiments

This directory is the single experiment-level location for the paper's Hard-regime work.

## Structure

- `range_calibration/` is the completed one-seed Qwen sweep over multiple operand ranges. It is
  retained as the auditable multiplication selection/smoke stage.
- `multiplication_only_piggyback/` is the final paper piggyback experiment: pure multiplication,
  operands `500..1000`, seeds `0,1,2`, Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct, with the
  answer-only and matched independent-private controls.
- `confirmatory/` contains the final invisible-local steganography run and its original audited
  provenance. Its affine piggyback outputs are superseded by `multiplication_only_piggyback/` and
  must not be used as the paper's final piggyback result.
- `s5/` is reserved for the next S5 Hard-regime experiment. It contains no runnable experiment yet.
- `knowledge/` is reserved for the next knowledge Hard-regime experiment. It contains no runnable
  experiment yet.

See the two experiment READMEs for the exact successful cluster workflows and
`docs/HARD_REGIME_REPORT.md` for the completed multiplication report. The fixed-data Qwen seed-2
reruns are a sensitivity analysis only; they do not replace any primary seed.
