# Multiplication Hard-Regime Report: Piggybacking and Steganography

Status: final audited report for the paper, based on the completed August 2026 experiments.

## Executive conclusion

The final multiplication Hard regime uses operands `500..1000` and evaluates
`Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct` over seeds `0,1,2`. The final
piggyback task is multiplication only: public `s*x`, private `s*(x+delta)`. The earlier affine
`s*x+y` formulation is historical and is not used for the final piggyback result. Steganography
also uses pure multiplication.

Qwen exhibits both mechanisms. Multiplication-only piggybacking raises private exact accuracy from
`16.8 ± 1.4%` to `52.7 ± 15.5%`, while the matched independent control reaches only
`16.6 ± 1.0%`. Invisible steganography raises private accuracy from `18.3 ± 1.4%` to
`89.7 ± 1.2%`. Llama exhibits the steganographic effect (`19.3 ± 1.1%` to `72.5 ± 12.2%`) but not
the piggyback effect (`19.4 ± 1.2%` to `17.4 ± 1.0%`; matched control `18.8 ± 0.9%`). Hidden
communication therefore transfers across both model families, whereas implicit computational
reuse is demonstrated only on Qwen in this setup.

The range is Hard in the capacity-relative sense that answer-only private accuracy is low. It is
not equally positioned relative to public-task capacity: Qwen reaches about `95%` public accuracy,
while Llama reaches about `65–72%` depending on the mechanism. The paper should call `500..1000`
the fixed multiplication Hard regime while explicitly reporting this model-dependent solvability.

All `±` values are mean ± sample standard deviation across seeds `0,1,2`. Accuracies and deltas
are percentages.

> [CODE REF — remove for final manuscript] Final piggyback design and workflow:
> `experiments/hard_regime/multiplication_only_piggyback/config.json` and
> `experiments/hard_regime/multiplication_only_piggyback/README.md`. Final steganography design and
> workflow: `experiments/hard_regime/confirmatory/config.json` and
> `experiments/hard_regime/confirmatory/README.md`.

## 1. Experimental design

### Models, data, and seeds

- Models: Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.
- Operand range: both multiplication operands lie in `500..1000`, inclusive.
- Seeds: `0`, `1`, and `2`.
- Per mechanism and seed: 10,000 training and 1,000 held-out evaluation examples.
- Qwen and Llama consume exactly the same generated JSON records within each seed and mechanism.
- Each adapter trains for one epoch and is greedily evaluated at `ckpt_final`.

In `arith_piggyback`, the public task is `s*x` and the private task is `s*(x+delta)`. The wrapped
shift `delta` comes from `{-40,-20,-10,-5,-2,-1,1,2,5,10,20,40}` and keeps the private factor in
`500..1000`. The public computation therefore exposes instance-specific product structure that can
be reused for the nearby private task. In `arith_piggyback_control`, the private multiplication is
independently sampled from the same operand range. Piggyback and control use index-matched public
questions. No additive offset occurs in the final questions, prompts, traces, or answers.

Steganography uses two independently sampled multiplication problems. In `local_channel_cot`, the
complete private multiplication trace is encoded beside the corresponding public trace values with
invisible Unicode variation selectors. Digits `0..9` map to `U+FE00..U+FE09`, and invisible
delimiters mark each value. Full-payload accuracy requires exact recovery of the entire canonical
payload and is stricter than private-answer accuracy.

> [CODE REF — remove for final manuscript] Multiplication-only record construction and matching:
> `experiments/hard_regime/multiplication_only_piggyback/generate_splits.py`, especially
> `_multiplication`, `_record`, and `generate`. Steganographic record and
> payload construction: `experiments/hard_regime/confirmatory/generate_splits.py`, especially
> `generate_steganography`, and `chain_of_lies/variants/steganography/data_generation/generate.py`.

### Supervision and optimization

Piggybacking trains `arith_piggyback` and `arith_piggyback_control` under `answer_only` and
`public_cot`. Steganography trains `arith_steg_local_invisible` under `answer_only` and
`local_channel_cot`. The obsolete mismatched-CoT condition is absent.

All adapters use autoregressive cross-entropy and LoRA with rank 8, alpha 16, dropout 0.05,
learning rate `2e-5`, and target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, and `down_proj`. Piggyback uses batch size 2 and 5,000 optimizer steps. Steganography
uses batch size 1 and 10,000 steps. Training is deterministic for a fixed seed. Piggyback generation
allows 512 tokens; steganography allows 1,536 tokens in training validation and 2,048 in final
evaluation. Outputs remained below these limits.

> [CODE REF — remove for final manuscript] Piggyback training/evaluation orchestration:
> `experiments/hard_regime/multiplication_only_piggyback/experiment.py`; steganography orchestration:
> `experiments/hard_regime/confirmatory/experiment.py`. Shared CE/LoRA implementation:
> `scripts/run_ce_only.py` and `chain_of_lies/training/ce/trainer.py`. Greedy evaluation:
> `scripts/evaluate_ce_model.py` and `chain_of_lies/evaluation/experiment_evaluation.py`.

### Leakage prevention and completeness

Generation and validation fail closed unless all of the following hold:

- zero individual-question overlap between training and evaluation;
- zero overlap after canonicalizing commutatively equivalent multiplication;
- zero exact public/private-pair overlap;
- zero arithmetic-component overlap across seeds within a mechanism;
- exactly matched public records between piggyback and independent-control arms;
- reconstructed questions, answers, shifts, prompts, channel scheme, and payloads match stored specs.

The final piggyback suite contains 99,000 unique multiplication questions across all seeds. The
steganography suite contains 66,000 unique questions and has zero overlap with the 22,000
calibration-reserve components. Every adapter used all 10,000 training records, and every final
report evaluated all 1,000 held-out records without missing responses. Manifests record question
and record SHA-256 digests, and artifacts include independently recomputed validation provenance.

> [CODE REF — remove for final manuscript] Piggyback leakage and integrity audit:
> `experiments/hard_regime/multiplication_only_piggyback/validate_splits.py`; steganography audit:
> `experiments/hard_regime/confirmatory/validate_splits.py`. Fail-closed summaries:
> each experiment's `summarize.py`.

## 2. Primary results

| Model | Mechanism | Answer-only private | Mechanism private | Paired improvement | Public | Joint | Payload |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | Multiplication-only piggyback | `16.8 ± 1.4` | `52.7 ± 15.5` | `+36.0 ± 16.2` | `94.9 ± 0.2` | `51.9 ± 15.5` | — |
| Llama3.1-8B | Multiplication-only piggyback | `19.4 ± 1.2` | `17.4 ± 1.0` | `−2.0 ± 1.4` | `72.2 ± 2.3` | `13.6 ± 0.2` | — |
| Qwen2.5-7B | Invisible steganography | `18.3 ± 1.4` | `89.7 ± 1.2` | `+71.4 ± 1.7` | `94.5 ± 1.1` | `84.7 ± 1.0` | `84.5 ± 1.0` |
| Llama3.1-8B | Invisible steganography | `19.3 ± 1.1` | `72.5 ± 12.2` | `+53.3 ± 11.1` | `65.4 ± 1.0` | `48.1 ± 7.1` | `52.8 ± 11.2` |

`Paired improvement` is calculated within seed as mechanism private minus answer-only private,
then summarized. The matched-control piggyback comparison is `+36.1 ± 15.8` points for Qwen and
`−1.4 ± 0.3` for Llama.

## 3. Multiplication-only piggybacking

| Model | Seed | Piggy public | Piggy private | Piggy joint | Answer-only private | Control public | Control private | Piggy − answer | Piggy − control |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 0 | `94.7` | `63.9` | `63.0` | `15.2` | `94.9` | `15.6` | `+48.7` | `+48.3` |
| Qwen | 1 | `94.9` | `59.3` | `58.6` | `17.8` | `95.1` | `17.6` | `+41.5` | `+41.7` |
| Qwen | 2 | `95.0` | `35.0` | `34.2` | `17.3` | `95.3` | `16.7` | `+17.7` | `+18.3` |
| Llama | 0 | `72.1` | `16.2` | `13.5` | `19.2` | `71.0` | `17.8` | `−3.0` | `−1.6` |
| Llama | 1 | `69.9` | `17.9` | `13.5` | `18.3` | `71.9` | `19.5` | `−0.4` | `−1.6` |
| Llama | 2 | `74.5` | `18.0` | `13.8` | `20.6` | `75.0` | `19.1` | `−2.6` | `−1.1` |

For Qwen, private accuracy exceeds both answer-only and the matched control in every seed while
public accuracy remains matched and stable. This isolates the useful public/private relation as
the source of the gain. Llama learns the visible public task but remains at the private baseline;
all three seeds agree on the null result. Format accuracy is 100%, concealment is `99.6%`, and the
visible trace is about 42 words for both models.

Qwen's effect magnitude is optimization-sensitive. Holding the complete seed-2 train/evaluation
dataset fixed, the original optimization seed obtained `35.0%` private accuracy, while additional
optimization seeds `3,4,5` obtained `57.9%`, `57.9%`, and `44.9%`. Their mean is `53.6 ± 7.5%`;
public accuracy stays in `94.9–95.6%`. Thus seed 2 is neither corrupt nor intrinsically too hard.
However, choosing a favorable rerun after viewing the results would be post-hoc selection. The
primary three-seed result therefore retains the original `35.0%` observation and reports the full
variance; the fixed-data reruns are a sensitivity analysis only.

> [CODE REF — remove for final manuscript] Optimization-sensitivity design and summary:
> `experiments/hard_regime/multiplication_only_piggyback/diagnostics/qwen_seed2_optimization/` and
> `artifacts/hard_regime/multiplication_only_piggyback_diagnostics/qwen_seed2_optimization/summary.json`.

## 4. Invisible steganography

| Model | Seed | Channel public | Channel private | Channel joint | Answer-only private | Improvement | Full payload exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 0 | `94.7` | `90.7` | `85.7` | `17.4` | `+73.3` | `85.4` |
| Qwen | 1 | `95.5` | `88.3` | `84.5` | `17.6` | `+70.7` | `84.5` |
| Qwen | 2 | `93.3` | `90.0` | `83.8` | `19.9` | `+70.1` | `83.5` |
| Llama | 0 | `65.6` | `58.6` | `40.3` | `18.0` | `+40.6` | `57.7` |
| Llama | 1 | `64.3` | `77.8` | `49.9` | `19.9` | `+57.9` | `40.0` |
| Llama | 2 | `66.2` | `81.2` | `54.2` | `19.9` | `+61.3` | `60.7` |

Qwen's result is large and stable. Across 3,000 examples, 2,534 had an exact complete payload; all
also had both answers correct. Exact payloads account for 99.8% of Qwen joint successes.

Llama also shows a positive channel effect in every seed, but private and payload accuracy vary
substantially despite stable public accuracy. Across 3,000 examples, 1,584 had an exact complete
payload and 1,275 of these had both answers correct. Complete-payload utilization is 80.5% overall
and varies by seed (`64.1%`, `99.3%`, `83.7%`). Exact payloads account for 88.3% of Llama joint
successes. A partially correct payload may still help, and the model can independently solve some
private questions.

Concealment is 100% under the project's lexical/visibility heuristic. This demonstrates designed
hidden communication, not cryptographic secrecy: an observer who knows the variation-selector
codebook can decode it.

> [CODE REF — remove for final manuscript] Channel-aware scoring:
> `chain_of_lies/evaluation/rewards.py` and
> `chain_of_lies/evaluation/experiment_evaluation.py`; aggregate definitions:
> `experiments/hard_regime/confirmatory/summarize.py`.

## 5. Cross-model interpretation

The Qwen/Llama differences are not caused by different records or prompt text. Within every seed
and mechanism, both models consume identical JSON prompts and every cell records matching config
and split-manifest digests. The models necessarily use different chat templates and tokenizers.
Multiplication-only supervised targets average approximately 34 versus 26 tokens for answer-only
and 151 versus 101 for public CoT (Qwen versus Llama). Steganography targets average about 377
versus 326–327 tokens. All remain within configured limits.

Llama's lower public accuracy indicates that `500..1000` is a more severe effective regime for it.
That difference is part of the finding. The models must not be pooled into a single mechanism
effect.

## 6. Relation to range calibration and the former affine task

The one-seed selection sweep used historical affine piggyback questions `s*x+y` with seed `314159`.
At `500..1000`, Qwen affine piggyback reached `50.1%` private accuracy, versus `13.9%` answer-only
and `16.0%` control. It motivated selection of the range but is not pooled with or relabeled as the
final multiplication-only experiment. The final no-offset Qwen mean (`52.7%`) shows that the
piggyback result survives standardization to pure multiplication.

Steganography calibration at `500..1000` yielded `88.2%` private, `17.9%` answer-only private, and
`82.6%` payload accuracy. Its seed-`314159` questions are excluded from the final steganography
suite.

> [CODE REF — remove for final manuscript] Historical selection workflow:
> `experiments/hard_regime/range_calibration/`. It remains explicitly labeled as affine and must
> not be cited as multiplication-only evidence.

## 7. Paper-ready claim and limitations

> In a fixed `500..1000` multiplication regime with low answer-only private accuracy, Qwen exhibits
> both computational reuse and designed hidden communication: multiplication-only piggybacking
> raises private accuracy from 16.8% to 52.7%, while invisible steganography raises it from 18.3%
> to 89.7%. On Llama3.1-8B, the hidden-channel benefit replicates (19.3% to 72.5%), whereas
> piggyback reuse does not (19.4% to 17.4%). Designed hidden communication therefore transfers more
> reliably across these model families than implicit reuse of related public computation.

Required limitations:

- The range is capacity-relative; Llama has lower public accuracy than Qwen.
- Piggybacking is supported on Qwen only in this comparison.
- Qwen's piggyback magnitude is optimization-sensitive; the original low seed is retained.
- Llama steganography is successful but variable across three seeds.
- The invisible channel is deliberately designed and trained, not emergent.
- Concealment does not establish security against a decoder-aware observer.
- Results concern these two instruction-tuned models and this LoRA recipe.

## 8. Reproducibility map

| Purpose | Authoritative location |
|---|---|
| Historical range selection | `experiments/hard_regime/range_calibration/` |
| Final multiplication-only piggyback | `experiments/hard_regime/multiplication_only_piggyback/` |
| Piggyback frozen parameters | `experiments/hard_regime/multiplication_only_piggyback/config.json` |
| Piggyback split generation and validation | `experiments/hard_regime/multiplication_only_piggyback/generate_splits.py`; `validate_splits.py` |
| Piggyback training, evaluation, aggregation | `experiments/hard_regime/multiplication_only_piggyback/experiment.py`; `summarize.py` |
| Piggyback Condor workflow and tests | `experiments/hard_regime/multiplication_only_piggyback/condor/`; `test_pipeline.py` |
| Final invisible-steganography run | `experiments/hard_regime/confirmatory/` |
| Steganographic channel construction | `chain_of_lies/variants/steganography/data_generation/generate.py` |
| Channel-aware scoring | `chain_of_lies/evaluation/rewards.py`; `chain_of_lies/evaluation/experiment_evaluation.py` |
| Qwen fixed-data optimization diagnostic | `experiments/hard_regime/multiplication_only_piggyback/diagnostics/qwen_seed2_optimization/` |

Raw records, responses, checkpoints, and adapters are excluded from Git and should be archived
separately. The primary result always uses the original seeds `0,1,2`; diagnostic reruns are not
substituted into the confirmatory table.
