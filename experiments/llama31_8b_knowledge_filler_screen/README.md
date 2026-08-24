# Disposable Llama 3.1 8B knowledge filler screen

This isolated directory tests whether a different 8B model family shows an inference-time filler
benefit on the knowledge-intensive tasks from *Reading Between the Dots*. It does not fine-tune the
model and does not use the paired public/private protocol. It is a quick capability and Medium-band
screen that can be deleted as one directory after its artifacts are saved.

## What is tested

The difficulty ladder mirrors the paper:

1. `one_fact`: retrieve one atomic number and add a supplied two-digit value (800 examples).
2. `two_fact`: retrieve two atomic numbers and add them (the paper's released 1,500 examples).
3. `letter_position`: retrieve an element or capital and select a requested letter (the paper's
   released 285 element and 362 capital examples).

The released paper datasets are vendored from commit
`09f96a615abdbee31d2350bfbe064f4632c8a105` of
`kaleybrauer/filler-token-reasoning`. `prepare_data.py` verifies their SHA-256 hashes before any
run. The paper does not release its model-specific one-fact set, so this directory deterministically
generates the same task form from 20 atomic-number facts and separately probes whether Llama knows
those facts.

Every task also has zero-filler component probes. This distinguishes failure to retrieve a fact
from failure to compose known facts.

## Paper-aligned filler protocol

- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- Greedy inference, no training and no generated reasoning.
- Five task-matched demonstrations followed by one target.
- Every demonstration and target uses the same filler condition.
- Filler appears after the question and before `Answer:`.
- Dot counts: `K = 0, 1, 5, 10, 25, 50, 100, 200`.
- `K` counts written dots, as in the paper; actual Llama tokenizer counts are also recorded.

The same examples occur at every K, so all comparisons with K=0 are paired. Reports include the
complete curve, wrong-to-right and right-to-wrong flips, exact McNemar tests, and paired-bootstrap
confidence intervals.

The preregistered comparisons follow the useful lengths in the paper: K=100 for one-fact, K=50 for
two-fact, and K=100 for letter-position. A Medium signal additionally requires adequate component
knowledge and a K=0 baseline between 10% and 90%; this prevents floor and ceiling effects from being
misreported as a useful Medium regime.

## Cluster workflow

From `/home/momo00016/Faithfulness-Safety`, prepare and validate the data:

```bash
condor_submit experiments/llama31_8b_knowledge_filler_screen/condor/prepare.sub
```

After preparation succeeds, ensure your Hugging Face account has accepted the Llama 3.1 license,
then submit the three inference jobs:

```bash
read -s HF_TOKEN
export HF_TOKEN
condor_submit experiments/llama31_8b_knowledge_filler_screen/condor/run_all.sub
unset HF_TOKEN
```

After all three jobs finish:

```bash
condor_submit experiments/llama31_8b_knowledge_filler_screen/condor/summarize.sub
```

Transfer:

```text
artifacts/llama31_8b_knowledge_filler_screen/REPORT.md
artifacts/llama31_8b_knowledge_filler_screen/accuracy_by_task_and_filler.csv
artifacts/llama31_8b_knowledge_filler_screen/component_accuracy.csv
artifacts/llama31_8b_knowledge_filler_screen/summary.json
```

Raw predictions are under `generated_data/llama31_8b_knowledge_filler_responses/`.

## Local checks

```bash
python3 -m unittest experiments/llama31_8b_knowledge_filler_screen/test_pipeline.py
bash -n experiments/llama31_8b_knowledge_filler_screen/run.sh
```

## Claim boundary

This is one model, one seed, and an inference-only screen. Absence of uplift would mean only that
this Llama checkpoint does not show a reliable benefit under this paper-aligned protocol. The
paper's positive results used much larger DeepSeek V3 and Kimi K2 models.
