# Proposition 4 parity sanity check

This experiment asks whether a real model can implement the unencrypted scheme from Proposition 4.
It is deliberately separate from the paired public/private-task experiments: there is one parity
task, and the privacy question is whether the input bits can be recovered from the generated trace
alone.

## Conditions

All conditions use the same random 32-bit inputs.

1. `answer_only`: emit only the final parity.
2. `prop4_pair_cot`: start at `S_0=0` and emit
   `S_i = S_{i-1} XOR X_{2i-1} XOR X_{2i}` for each consecutive pair. The trace has 16 bits.
3. `leaky_single_cot`: emit running parity after every individual bit. Its 32-bit trace is exactly
   invertible and is the positive control for input reconstruction.

The trace contains only state bits. It never prints the input bits, equations containing their
values, or natural-language explanations.

## Four-bit example

Take `X=1011` and start at `S_0=0`. The first pair `10` has XOR 1, so `S_1=1`.
The second pair `11` has XOR 0, so `S_2` remains 1. The Proposition-4 trace is therefore `1 1`,
and the final parity is 1. From that trace an observer learns only that the first pair differs and
the second pair agrees: the input might be `1011`, but its complementary twin `0100` produces the
same trace and parity while reversing every bit.

The leaky single-bit trace for `1011` is `1 1 0 1`. Taking consecutive XOR differences from the
known initial state 0 recovers `1,0,1,1` exactly.

## Complementary-twin validation

Every input `X` has a validation twin obtained by flipping all 32 bits. Because both bits in every
pair are flipped, the pair XORs, complete Proposition-4 trace, and final parity are unchanged, while
every individual input bit is reversed. Consequently, when the two generated pair traces match, a
trace-only observer cannot distinguish the twins: one member has `X_j=0` and the other has `X_j=1`
at every position. This gives an exact 50% individual-bit recovery ceiling on matched twins without
depending on a particular learned probe.

The leaky trace has the opposite behavior. With running states `R_0=0,R_1,...,R_32`, recover
`X_1=R_1` and `X_i=R_{i-1} XOR R_i`, so exact traces reveal every input bit.

The reconstruction observer receives only the generated `<TRACE>` content. Giving it the original
prompt would make recovery trivial and would not test the proposition.

## Data and training

- Qwen2.5-7B-Instruct with LoRA CE.
- Seed 0, fixed length 32, three epochs.
- 10,000 training and 2,000 validation inputs.
- Both splits contain complementary twins and are exactly balanced by final parity.
- Inputs are unique and train/validation disjoint.
- Greedy evaluation of the pretrained model and final fine-tuned checkpoint.

This is only the Proposition-4 sanity check. It does not use a pseudorandom generator or encryption.
Those should be attempted only after the model reliably produces correct final answers and traces,
the pairwise twins remain indistinguishable, and the leaky control is recoverable.

## Cluster workflow

From `/home/momo00016/Faithfulness-Safety`:

```bash
condor_submit experiments/parity_prop4/condor/generate.sub
```

After generation succeeds:

```bash
read -s HF_TOKEN
export HF_TOKEN
condor_submit experiments/parity_prop4/condor/run_all.sub
unset HF_TOKEN
```

After all three GPU jobs finish:

```bash
condor_submit experiments/parity_prop4/condor/summarize.sub
```

Results are written under `artifacts/ce_parity_prop4/`.

## Local checks

```bash
python3 -m unittest experiments/parity_prop4/test_pipeline.py
bash -n experiments/parity_prop4/run.sh
```
