# Explicit-hypergraph Goldreich parity experiment

This follow-up tests the supervisor's explicit seed-position hint. It preserves the completed
implicit-hypergraph experiment and resumes from its supplied-mask update adapter.

For each public encrypted state, the model emits a fixed public selector such as:

```text
(use Seed: K B J H D) 0
```

The letters reveal which seed positions the public Goldreich graph uses. They do not reveal the
private values stored at those positions. The final bit on each line is the encrypted cumulative
parity state, never the mask bit.

The curriculum isolates:

1. the five-bit predicate;
2. retrieving five explicitly named private seed values and computing one mask;
3. computing the first encrypted state from one hinted mask;
4. combining two hinted masks with one later encrypted-state update;
5. complete hinted traces at lengths 2, 4, 8, 16, and 32;
6. a separate length-64 extension.

The core and extension use separate final adapters. Thus, a failed length-64 attempt cannot erase
a successful length-32 result. Reports use the best held-out checkpoint from each dedicated stage
to diagnose learning, and separately report final-adapter retention. Training and validation seeds
are disjoint, complementary twins have identical gold public traces, and the evaluator checks
selector accuracy, trace accuracy, parity, twin matching, formatting, and seed-value leakage.

## Cluster run

Keep `artifacts/ce_parity_goldreich_curriculum/update/final_adapter`; it initializes this run.
Remove only partial outputs from this new experiment:

```bash
rm -rf \
  generated_data/parity_goldreich_explicit_hints \
  generated_data/parity_goldreich_explicit_hints_responses \
  generated_data/parity_goldreich_explicit_hints_seed_0.tar.gz \
  generated_data/parity_goldreich_explicit_hints_core_responses.tar.gz \
  generated_data/parity_goldreich_explicit_hints_extension_responses.tar.gz \
  artifacts/ce_parity_goldreich_explicit_hints
```

Submit the dependency chain:

```bash
mkdir -p /home/momo00016/logs/chain_of_lies
read -s HF_TOKEN
export HF_TOKEN
condor_submit_dag experiments/parity_goldreich_explicit_hints/condor/workflow.dag
unset HF_TOKEN
```

The token is optional for the public Qwen checkpoint, and the DAG explicitly propagates it when
present. The DAG runs:

```text
generate -> core through length 32 -> separate length-64 extension -> summarize
```

Final outputs:

```text
artifacts/ce_parity_goldreich_explicit_hints/REPORT.md
artifacts/ce_parity_goldreich_explicit_hints/diagnostic_metrics.csv
artifacts/ce_parity_goldreich_explicit_hints/summary.json
```
