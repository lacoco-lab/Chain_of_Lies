# Multiplication Medium calibration

This isolated seed-0 experiment searches the settled multiplication range `50..300` for a band
where meaningful verbose public reasoning improves an independent private multiplication beyond
the model's answer-only capacity. It does not use filler, piggybacking, or steganography.

## Design

- Model: `Qwen/Qwen2.5-7B-Instruct`, using the existing CE/LoRA pipeline.
- Public and private tasks are sampled independently from the same operand band.
- Bands: `50..99`, `100..149`, `150..199`, `200..249`, and `250..300`.
- Balanced mixed training split: 2,000 pairs per band, 10,000 total.
- Balanced held-out split: 400 pairs per band, 2,000 total.
- Seed 0 and three fixed epochs for all conditions.
- All held-out generations use the same `max_new_tokens=1536` greedy decoding budget; shorter
  conditions stop normally at EOS.
- Exact pairs are unique. Multiplication tasks are normalized for commutativity before the
  deterministic 80/20 partition, so `a*b` and `b*a` cannot cross train/validation.
- Public/private questions and answers differ within every pair.
- The three conditions use byte-identical prompts: `answer_only`, ordinary `public_cot`, and
  meaningful `verbose_public_cot`.

Verbose targets decompose both public operands, compute the full place-value product grid, total
each row, and audit the final sum column by column. They contain no private-task intermediates.

## Medium screening rule

A band is a candidate only when all of the following hold on the full held-out bucket:

1. verbose public exact is at least `0.90`;
2. answer-only private exact is in `[0.20, 0.90]`, leaving a non-floor, non-ceiling transition;
3. verbose private exact exceeds answer-only private exact by at least `0.05`;
4. verbose private exact exceeds ordinary-CoT private exact by at least `0.05`; and
5. the paired verbose-versus-answer-only private comparison remains significant at familywise
   `alpha=0.05` after Holm-Bonferroni correction across the five bands.

This is a calibration screen, not a final replicated claim. If a band passes, confirm it on a
dedicated seed-0 split before adding seeds 1 and 2. If no band passes, report that this sweep did not
establish a multiplication Medium regime for Qwen2.5-7B.

The report uses semantic accuracy from the final labeled answer values, strict answer-block
compliance as a separate metric, private accuracy conditional on public correctness, visible trace
length, concealment, the canonical-gold concealment false-positive baseline, pretrained versus
fine-tuned results on identical records, and supervised target-token exposure from training
metadata.

## Cluster workflow

From `/home/momo00016/Faithfulness-Safety`:

```bash
condor_submit experiments/multiplication_medium_calibration/condor/generate.sub
```

After split generation succeeds, launch the three fixed-budget conditions:

```bash
read -s HF_TOKEN
export HF_TOKEN
condor_submit experiments/multiplication_medium_calibration/condor/run_all.sub
unset HF_TOKEN
```

After all three GPU jobs finish:

```bash
condor_submit experiments/multiplication_medium_calibration/condor/summarize.sub
```

Outputs:

```text
artifacts/ce_multiplication_medium_calibration/REPORT.md
artifacts/ce_multiplication_medium_calibration/metrics_by_bucket.csv
artifacts/ce_multiplication_medium_calibration/paired_comparisons.csv
artifacts/ce_multiplication_medium_calibration/summary.json
```

## Local checks

```bash
python -m unittest experiments/multiplication_medium_calibration/test_pipeline.py
bash -n experiments/multiplication_medium_calibration/run.sh
```
