# Faithfulness and hidden computation experiments

This repository contains the code for two sets of experiments. The **scaling-regime** study fine-tunes language models to answer a public and a private task under different restrictions on visible reasoning. The **Goldreich-parity** study tests a purpose-built Transformer on encrypted state-tracking sequences. Prompting experiments are kept separately under `llm_prompts/`.

## Layout

| Directory | Contents |
| --- | --- |
| `chain_of_lies/` | Shared task generation, supervised training, inference, and scoring code. |
| `experiments/scaling_regime/` | Final task configurations and protocol workflows, including the Full-CoT control. |
| `experiments/goldreich_parity/transformer/` | Final encrypted-parity model, controls, and ablations. |
| `experiments/goldreich_parity/qwen_attempts/` | Earlier Qwen diagnostics. |
| `experiments/legacy/` | Earlier Easy/Hard regimes, calibrations, and other historical experiments. |
| `experiments/cluster_jobs/` | Cluster submission files, kept with the internal repository. |
| `scripts/` | Training and evaluation entry points, audits, release tools, and figure scripts. |
| `tests/` | Local pipeline and data checks. |
| `llm_prompts/` | System and user prompts for the prompting experiments. |
| `sample_records/` | Small examples of training records, held-out prompts, and generated responses. |
| `paper_results/` | Processed evaluation reports, metric tables, and numerical figure sources. |

`paper_results/` contains **results**, whereas `sample_records/` contains **example-level data**. The full local `models_data_and_evaluation_outputs/` directory holds all generated data, responses, and checkpoints; it is intentionally excluded from Git and distributed as a separate archive. `Hard_regime/` is a historical, 25 GB output directory used by two legacy audits, so it stays at its original local path and is also excluded from Git. The local `code_package/` and `release_packages/` directories are duplicate release staging areas, not additional source trees for this repository.

The complete data and checkpoint archive is available at the [Zenodo record](https://zenodo.org/records/22917635). Extract its `models_data_and_evaluation_outputs/` directory at the repository root to rerun analyses that require every evaluation record or raw generation.

The large visible-leakage per-example CSV is kept locally; Git includes a lossless `per_example.csv.gz` copy alongside its aggregate tables.

## Run locally

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m pytest -q tests
```

The final scaling workflows are in `experiments/scaling_regime/tasks/{multiplication,iterated_addition,s5_state_tracking,plain_parity}/`. Each directory has a `config.json`, split generator, validator, experiment runner, evaluator, and summarizer. For example, to create and validate the multiplication splits:

```bash
python experiments/scaling_regime/tasks/multiplication/experiment.py generate
python experiments/scaling_regime/tasks/multiplication/validate_splits.py --help
```

Run `python experiments/scaling_regime/tasks/multiplication/experiment.py --help` for training and evaluation options. Goldreich-parity scripts and configs are under `experiments/goldreich_parity/transformer/main_experiment/`; its matched controls and ablations are adjacent. Training the reported models requires the stated GPUs and, for gated models, model access. The configurations specify seeds, task sizes, optimization settings, and evaluation limits. Generated outputs go to `generated_data/` and `artifacts/`, both excluded from Git.
