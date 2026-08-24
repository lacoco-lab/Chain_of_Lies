# Goldreich-encrypted parity CoT

This experiment asks whether Qwen2.5-7B-Instruct can compute parity while emitting only an
encrypted binary chain of thought. It is a separate follow-up to `experiments/parity_prop4/` and
does not modify that established experiment. There is one parity task and no private task.

## Construction

Each complementary twin pair has a 128-bit parity input and an independent 16-bit PRG seed shared
by its two members. A model-specific graph is sampled once with `graph_seed=20260818` and then fixed
across all examples. For every one of the 64 CoT steps, the graph selects an ordered tuple of five
distinct seed positions. If their values are `(a,b,c,d,e)`, the mask is

```text
z_i = a XOR b XOR c XOR (d AND e).
```

The mask length is exactly `16^1.5 = 64`, matching the requested Goldreich-style stretch. The
ordinary Proposition-4 state and encrypted emitted state are

```text
S_i = S_(i-1) XOR X_(2i-1) XOR X_(2i)
C_i = S_i XOR z_i.
```

Equivalently, an autoregressive model can update without emitting plaintext:

```text
C_i = C_(i-1) XOR z_(i-1) XOR X_(2i-1) XOR X_(2i) XOR z_i.
```

The graph is saved in the split manifest for auditing but is not printed in each prompt. It is the
fixed rule learned by the fine-tuned model. Validation uses 1,000 PRG seeds that never occur among
the 5,000 training seeds, preventing exact seed-to-mask memorization.

## Conditions

All conditions use identical inputs, seeds, and held-out examples.

1. `plaintext_pair_cot`: matched 128-bit capability control emitting the 64 ordinary pair states.
2. `provided_mask_encrypted_cot`: encryption diagnostic; all 64 masks are supplied in the prompt.
3. `goldreich_encrypted_cot`: main condition; only the 16-bit seed is supplied, so the model must
   learn the fixed five-position graph, compute the nonlinear masks, and emit encrypted states.

The supplied-mask control separates failure to update encrypted states from failure to generate
the masks. Training uses ordinary CE only. The plaintext control uses three epochs. Following the
initial three-epoch result, the supplied-mask and Goldreich conditions use six epochs: the
supplied-mask loss only began dropping late in epoch three, while the Goldreich loss remained
flat. Attention regularization remains a follow-up if longer training solves the supplied-mask
control but not the Goldreich condition.

## Twin privacy diagnostic

Validation is organized into complementary input twins. Both members of a twin share the same PRG
seed. Complementing both bits in every pair preserves every plain state, mask, encrypted trace, and
final parity while reversing each individual input bit. Matching generated twin traces therefore
give the same exact 50% individual-bit recovery ceiling as in the unencrypted sanity check.

The observer receives only generated `<TRACE>` content. The input, prompt, seed, and supplied masks
are excluded from the observer view.

## Scope

This is a finite-size mechanism test. A 16-bit seed can be brute-forced, so success does not
establish cryptographic security or computational indistinguishability. It tests whether a real
model can learn and execute the intended local-PRG encryption mechanism on unseen seeds.

## Cluster workflow

From `/home/momo00016/Faithfulness-Safety`, generate and validate the splits:

```bash
condor_submit experiments/parity_goldreich/condor/generate.sub
```

After that job succeeds, train and evaluate all three conditions:

```bash
read -s HF_TOKEN
export HF_TOKEN
condor_submit experiments/parity_goldreich/condor/run_all.sub
unset HF_TOKEN
```

After all three GPU jobs finish, summarize:

```bash
condor_submit experiments/parity_goldreich/condor/summarize.sub
```

Results are written under `artifacts/ce_parity_goldreich/`.

## Extended-training follow-up after the initial run

Keep the original plaintext outputs and the generated splits. Move the old three-epoch encrypted
outputs aside, then submit only the two encrypted conditions:

```bash
mkdir -p generated_data/parity_goldreich_eval_responses_3epoch_backup/seed_0 \
  artifacts/ce_parity_goldreich_3epoch_backup/seed_0

cp artifacts/ce_parity_goldreich/REPORT.md \
  artifacts/ce_parity_goldreich/metrics.csv \
  artifacts/ce_parity_goldreich/summary.json \
  artifacts/ce_parity_goldreich_3epoch_backup/

mv generated_data/parity_goldreich_eval_responses/seed_0/provided_mask_encrypted_cot \
  generated_data/parity_goldreich_eval_responses_3epoch_backup/seed_0/
mv generated_data/parity_goldreich_eval_responses/seed_0/goldreich_encrypted_cot \
  generated_data/parity_goldreich_eval_responses_3epoch_backup/seed_0/
mv artifacts/ce_parity_goldreich/seed_0/provided_mask_encrypted_cot \
  artifacts/ce_parity_goldreich_3epoch_backup/seed_0/
mv artifacts/ce_parity_goldreich/seed_0/goldreich_encrypted_cot \
  artifacts/ce_parity_goldreich_3epoch_backup/seed_0/

read -s HF_TOKEN
export HF_TOKEN
condor_submit experiments/parity_goldreich/condor/run_extended.sub
unset HF_TOKEN
```

After both jobs finish, run the existing summarizer. It combines the preserved three-epoch
plaintext result with the new six-epoch encrypted results:

```bash
condor_submit experiments/parity_goldreich/condor/summarize.sub
```

## Local checks

```bash
python3 -m unittest experiments/parity_goldreich/test_pipeline.py
bash -n experiments/parity_goldreich/run.sh
```
