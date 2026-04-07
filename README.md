# Faithfulness-Safety / Chain-of-Lies

Research code for studying concealed or encoded reasoning: a model must solve a public task and a hidden private task while keeping the visible reasoning benign-looking.

## Start Here

- Project context for future LLM chats:
  [PROJECT_CONTEXT_FOR_LLM.md](/Users/mohammadali/Desktop/Faithfulness-Safety/docs/PROJECT_CONTEXT_FOR_LLM.md)
- Full codebase guide:
  [CODEBASE_GUIDE.md](/Users/mohammadali/Desktop/Faithfulness-Safety/docs/CODEBASE_GUIDE.md)
- RL + Condor workflow:
  [RL_CONDOR_WORKFLOW.md](/Users/mohammadali/Desktop/Faithfulness-Safety/docs/RL_CONDOR_WORKFLOW.md)

## Core Entry Points

These are the main scripts that matter now:

- [generate_prompts.py](/Users/mohammadali/Desktop/Faithfulness-Safety/scripts/generate_prompts.py)
  Generate graph or arithmetic prompts for any variant.
- [run_inference.py](/Users/mohammadali/Desktop/Faithfulness-Safety/scripts/run_inference.py)
  Run baseline model inference on prompt JSONs.
- [generate_rl_prompt_splits.py](/Users/mohammadali/Desktop/Faithfulness-Safety/scripts/generate_rl_prompt_splits.py)
  Build deterministic RL train/validation splits.
- [run_rule_based_rl.py](/Users/mohammadali/Desktop/Faithfulness-Safety/scripts/run_rule_based_rl.py)
  Train the rule-based RL adapter.
- [evaluate_rule_based_rl.py](/Users/mohammadali/Desktop/Faithfulness-Safety/scripts/evaluate_rule_based_rl.py)
  Run held-out evaluation or aggregate per-variant RL summaries.
- [run_llm_judge_recoverability.py](/Users/mohammadali/Desktop/Faithfulness-Safety/scripts/run_llm_judge_recoverability.py)
  Measure whether the hidden task is recoverable from visible reasoning.

## Quick Examples

Install dependencies:

```bash
pip install -r requirements.txt
```

Generate prompts:

```bash
python scripts/generate_prompts.py --variant graph_default --n 10 --seed 42
python scripts/generate_prompts.py --variant arith_default --n 10 --seed 42
python scripts/generate_prompts.py --variant arith_both_hard --n 10 --seed 42
```

Run baseline inference:

```bash
python scripts/run_inference.py --model Qwen/Qwen2.5-7B-Instruct
```

For RL training, evaluation, judge runs, and Condor usage, use:

[RL_CONDOR_WORKFLOW.md](/Users/mohammadali/Desktop/Faithfulness-Safety/docs/RL_CONDOR_WORKFLOW.md)

## Repo Layout

```text
chain_of_lies/   Core Python package
scripts/         Main local/cluster entry points
condor/          HTCondor submit files and shell wrappers
docs/            Research context and documentation
data/            Prompts, responses, RL splits, evaluation outputs
artifacts/       Training checkpoints, summaries, plots
```
