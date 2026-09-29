#!/usr/bin/env python3
"""Build a task-oriented, hard-linked release of the completed scaling study.

The canonical artifact/data trees are never moved or modified. Regular files in the release are
hard links, so the directory is self-contained to readers and upload tools without consuming a
second copy of the large adapter/response payload locally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)


def _release_root() -> Path:
    candidates = (
        ROOT / "models_data_and_evaluation_outputs",
        ROOT.parent / "models_data_and_evaluation_outputs",
        ROOT / "paper_release",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


DEFAULT_DESTINATION = _release_root() / "scaling_regime_v1"
MODELS = ("qwen", "llama")
SEEDS = (0, 1, 2)

ORIGINAL_CONDITIONS = {
    "vanilla": "vanilla",
    "public_only_cot": "public_only_cot",
    "piggyback": "piggyback",
    "steganography": "invisible",
    "filler": "filler_plus_public_cot",
}


@dataclass(frozen=True)
class Task:
    release_name: str
    source_name: str
    artifacts: Path
    prompts: Path
    responses: Path
    config: Path


TASKS = (
    Task(
        "multiplication",
        "multiplication",
        ROOT / "artifacts/multiplication_length_retraining_v1",
        ROOT / "generated_data/multiplication_length_retraining_v1",
        ROOT / "generated_data/multiplication_length_retraining_v1_eval_responses",
        ROOT / "experiments/scaling_regime/tasks/multiplication/config.json",
    ),
    Task(
        "iterated_addition",
        "knowledge",
        ROOT / "artifacts/knowledge_length_retraining_all_but_one_v2",
        ROOT / "generated_data/knowledge_length_retraining_all_but_one_v2",
        ROOT
        / "generated_data/knowledge_length_retraining_all_but_one_v2_eval_responses",
        ROOT / "experiments/scaling_regime/tasks/iterated_addition/config.json",
    ),
    Task(
        "s5_state_tracking",
        "s5",
        ROOT / "artifacts/s5_length_retraining_v1",
        ROOT / "generated_data/s5_length_retraining_v1",
        ROOT / "generated_data/s5_length_retraining_v1_eval_responses",
        ROOT / "experiments/scaling_regime/tasks/s5_state_tracking/config.json",
    ),
    Task(
        "plain_parity",
        "parity",
        ROOT / "artifacts/parity_regime_calibration",
        ROOT / "generated_data/parity_regime_calibration",
        ROOT / "generated_data/parity_regime_calibration_eval_responses",
        ROOT / "experiments/scaling_regime/tasks/plain_parity/config.json",
    ),
)

FILLER_ARTIFACTS = ROOT / "artifacts/filler_only_scaling_v1"
FILLER_RESPONSES = ROOT / "generated_data/filler_only_scaling_v1_eval_responses"
FILLER_CONFIG = ROOT / "experiments/scaling_regime/protocols/filler/config.json"

TASK_NOTES = {
    "multiplication": """# Multiplication

- Difficulty: exact decimal digits per operand (`2, 3, 4`), sampled across the full d-digit range.
- Per checkpoint: 3,000 unique training examples; 600 held-out outputs (200 per length per seed).
- Piggybacking: shared multiplier and a small shift in the other operand.
- Main source config: `../../configs/multiplication.json`.
""",
    "iterated_addition": """# Atomic-number retrieval followed by iterated addition

- Difficulty: number of retrieved element facts (`k=1,...,5`). This is not k-hop reasoning.
- Per checkpoint: 5,000 unique training examples; 1,000 held-out outputs.
- Piggybacking: the first `k-1` facts are shared and only the final fact changes.
- Main source config: `../../configs/iterated_addition.json`.
""",
    "s5_state_tracking": """# S5 state tracking

- Difficulty: exact number of swaps (`1,...,19`) applied to a five-symbol state.
- Per checkpoint: 9,500 unique training examples; 3,800 held-out outputs.
- Piggybacking: shared initial state and first `k-1` swaps; only the final swap changes.
- Main source config: `../../configs/s5_state_tracking.json`.
""",
    "plain_parity": """# Plain binary parity

- Difficulty: trained sequence-length ranges `8–15`, `16–31`, `32–63`, `64–127`, `128–255`.
- Per checkpoint: 10,000 unique training examples; 1,000 held-out outputs.
- Piggybacking: the first `n-1` bits are shared and the final bit is flipped.
- This is ordinary parity, not the encrypted Goldreich-parity experiment.
- Main source config: `../../configs/plain_parity.json`.
""",
}


def _link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    os.link(source, destination)


def _link_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if ".DS_Store" in relative.parts or "__pycache__" in relative.parts:
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            _link_file(path, target)


def _only_final_adapter(cell: Path) -> Path:
    matches = sorted(cell.glob("*/ckpt_final/adapter_model.safetensors"))
    if len(matches) != 1:
        raise ValueError(f"Expected one final adapter in {cell}, found {len(matches)}")
    return matches[0].parent


def _link_cell_reports(cell: Path, destination: Path) -> None:
    provenance = cell / "provenance"
    if provenance.exists():
        _link_tree(provenance, destination / "provenance")
    for report in sorted(cell.rglob("*.json")):
        if not ({"per_length_eval", "per_variant_eval"} & set(report.parts)):
            continue
        relative = report.relative_to(cell)
        _link_file(report, destination / relative)


def _link_root_results(source: Path, destination: Path) -> None:
    for path in sorted(source.iterdir()):
        if path.is_file() and path.name != ".DS_Store":
            _link_file(path, destination / path.name)


def _source_cell(task: Task, model: str, seed: int, condition: str) -> Path:
    return task.artifacts / model / f"seed_{seed}" / condition


def _response_cell(task: Task, model: str, seed: int, condition: str) -> Path:
    return task.responses / model / f"seed_{seed}" / condition


def _filler_cell(task: Task, model: str, seed: int) -> Path:
    return FILLER_ARTIFACTS / task.source_name / model / f"seed_{seed}" / "filler_only"


def _filler_response_cell(task: Task, model: str, seed: int) -> Path:
    return FILLER_RESPONSES / task.source_name / model / f"seed_{seed}" / "filler_only"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory_row(component: str, path: Path, source: str) -> dict[str, object]:
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "component": component,
        "release_path": str(path.relative_to(path.parents[2])),
        "source": source,
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def _write_readme(destination: Path) -> None:
    text = """# Scaling-regime paper release v1

## Scope

This is the complete paper-facing archive for the finished balanced finetuning scaling study:
Multiplication, atomic-number retrieval followed by iterated addition, S5 state tracking, and
plain binary Parity, using Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct with seeds 0, 1, and 2.

The five primary protocols are `vanilla`, `filler`, `public_only_cot`, `piggyback`, and
`invisible`. `filler_plus_public_cot` is an appendix-only ablation. The primary `filler` folders
come from the later filler-only experiment; the source condition historically named `filler` has
been renamed `filler_plus_public_cot` here. Likewise, source `steganography` is exposed as
`invisible`.

The still-running open full-CoT baseline is deliberately absent. Add it only after its complete
three-seed audit.

## Layout

```text
documentation/                   authoritative reports, figures, and exact plotting tables
configs/                         task configs and primary-Filler config
integrity/                       completed audits and release manifests
tasks/<task>/
  prompts_and_splits/            unchanged training/evaluation records and split manifests
  models/<model>/seed_<n>/<condition>/
                                  final LoRA adapter, config, history, and metadata
  responses/<model>/seed_<n>/<condition>/
                                  raw held-out generations
  evaluation_reports/<model>/seed_<n>/<condition>/
                                  per-length reports and cell provenance
  source_aggregate_results/      canonical aggregate/per-seed/per-example tables
```

Primary Filler reuses the corresponding control records in `prompts_and_splits`; its
difficulty-scaled target is constructed at tokenization time. Therefore there is no separate
Filler-only prompt tree.

## Citation and terminology

Use `documentation/SCALING_REGIME_APPENDIX_HANDOFF.md` as the method/result source and
`documentation/Figures/figure_accuracy_plot_data.csv` as the exact accuracy table. Display
`invisible`, never `steganography`. Do not treat `filler_plus_public_cot` as a main protocol.
Plain parity here is not the encrypted Goldreich-parity experiment.

`adapter_manifest.csv` records SHA-256 checksums for every final adapter. `inventory.csv` records
file counts and apparent byte sizes by component. Files were hard-linked from the canonical local
trees; after upload or copying to another filesystem they behave as ordinary independent files.
"""
    (destination / "README.md").write_text(text)


def _link_documentation(destination: Path) -> None:
    docs = ROOT / "docs/scaling_regime"
    names = (
        "SCALING_REGIME_APPENDIX_HANDOFF.md",
        "SCALING_REGIME_EXTERNAL_LLM_INPUTS.md",
        "FINETUNING_REGIMES_SECTION_HANDOFF.md",
        "FILLER_ONLY_SCALING_REPORT.md",
        "MULTIPLICATION_LENGTH_RETRAINING_REPORT.md",
        "KNOWLEDGE_LENGTH_RETRAINING_ALL_BUT_ONE_V2_REPORT.md",
        "PARITY_THREE_SEED_REPORT.md",
    )
    for name in names:
        _link_file(docs / name, destination / "documentation" / name)
    _link_tree(
        ROOT / "figures/scaling_regime", destination / "documentation" / "Figures"
    )
    _link_tree(
        docs / "tables", destination / "documentation" / "Exact Numerical Tables"
    )


def build(destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Release already exists: {destination}")
    destination.mkdir(parents=True)

    required = [FILLER_ARTIFACTS, FILLER_RESPONSES, FILLER_CONFIG]
    for task in TASKS:
        required.extend((task.artifacts, task.prompts, task.responses, task.config))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required scaling material:\n" + "\n".join(missing)
        )

    _write_readme(destination)
    _link_file(
        Path(__file__).with_name("verify_release.py"), destination / "VERIFY_RELEASE.py"
    )
    _link_documentation(destination)
    _link_file(FILLER_CONFIG, destination / "configs/filler_only_scaling.json")

    adapter_rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []

    for task in TASKS:
        task_dest = destination / "tasks" / task.release_name
        (task_dest).mkdir(parents=True, exist_ok=True)
        (task_dest / "README.md").write_text(
            TASK_NOTES[task.release_name]
            + "\nEach `models/` and `responses/` model/seed cell contains the five primary conditions "
            + "plus `filler_plus_public_cot`, which is appendix-only. Primary `filler` has no meaningful CoT.\n"
        )
        _link_file(task.config, destination / "configs" / f"{task.release_name}.json")
        _link_tree(task.prompts, task_dest / "prompts_and_splits")
        _link_root_results(task.artifacts, task_dest / "source_aggregate_results")

        for model in MODELS:
            for seed in SEEDS:
                for source_condition, release_condition in ORIGINAL_CONDITIONS.items():
                    cell = _source_cell(task, model, seed, source_condition)
                    adapter = _only_final_adapter(cell)
                    model_dest = (
                        task_dest
                        / "models"
                        / model
                        / f"seed_{seed}"
                        / release_condition
                    )
                    _link_tree(adapter, model_dest)
                    _link_cell_reports(
                        cell,
                        task_dest
                        / "evaluation_reports"
                        / model
                        / f"seed_{seed}"
                        / release_condition,
                    )
                    _link_tree(
                        _response_cell(task, model, seed, source_condition),
                        task_dest
                        / "responses"
                        / model
                        / f"seed_{seed}"
                        / release_condition,
                    )
                    adapter_file = model_dest / "adapter_model.safetensors"
                    adapter_rows.append(
                        {
                            "task": task.release_name,
                            "model": model,
                            "seed": seed,
                            "condition": release_condition,
                            "path": str(adapter_file.relative_to(destination)),
                            "bytes": adapter_file.stat().st_size,
                            "sha256": _sha256(adapter_file),
                        }
                    )

                filler_cell = _filler_cell(task, model, seed)
                filler_adapter = _only_final_adapter(filler_cell)
                filler_dest = task_dest / "models" / model / f"seed_{seed}" / "filler"
                _link_tree(filler_adapter, filler_dest)
                _link_cell_reports(
                    filler_cell,
                    task_dest
                    / "evaluation_reports"
                    / model
                    / f"seed_{seed}"
                    / "filler",
                )
                _link_tree(
                    _filler_response_cell(task, model, seed),
                    task_dest / "responses" / model / f"seed_{seed}" / "filler",
                )
                adapter_file = filler_dest / "adapter_model.safetensors"
                adapter_rows.append(
                    {
                        "task": task.release_name,
                        "model": model,
                        "seed": seed,
                        "condition": "filler",
                        "path": str(adapter_file.relative_to(destination)),
                        "bytes": adapter_file.stat().st_size,
                        "sha256": _sha256(adapter_file),
                    }
                )

        inventory.extend(
            (
                _inventory_row(
                    f"{task.release_name}:prompts",
                    task_dest / "prompts_and_splits",
                    str(task.prompts.relative_to(ROOT)),
                ),
                _inventory_row(
                    f"{task.release_name}:models",
                    task_dest / "models",
                    "primary task artifacts + filler_only_scaling_v1",
                ),
                _inventory_row(
                    f"{task.release_name}:responses",
                    task_dest / "responses",
                    "primary task responses + filler_only_scaling_v1_eval_responses",
                ),
                _inventory_row(
                    f"{task.release_name}:reports",
                    task_dest / "evaluation_reports",
                    "primary task artifacts + filler_only_scaling_v1",
                ),
            )
        )

    audit = ROOT / "recovery_audits/all_24.json"
    if audit.exists():
        _link_file(audit, destination / "integrity/filler_only_all_24_audit.json")
    published = ROOT / "recovery_work/filler_only_sic_v1/published.json"
    if published.exists():
        _link_file(
            published, destination / "integrity/filler_only_publication_record.json"
        )

    with (destination / "adapter_manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=adapter_rows[0].keys())
        writer.writeheader()
        writer.writerows(adapter_rows)
    with (destination / "inventory.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=inventory[0].keys())
        writer.writeheader()
        writer.writerows(inventory)

    all_files = [path for path in destination.rglob("*") if path.is_file()]
    status = {
        "schema_version": 1,
        "release": "scaling_regime_v1",
        "complete_tasks": [task.release_name for task in TASKS],
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "primary_conditions": [
            "vanilla",
            "filler",
            "public_only_cot",
            "piggyback",
            "invisible",
        ],
        "appendix_only_conditions": ["filler_plus_public_cot"],
        "full_cot_baseline": "in_progress_not_included",
        "final_adapter_count": len(adapter_rows),
        "expected_final_adapter_count": len(TASKS) * len(MODELS) * len(SEEDS) * 6,
        "file_count": len(all_files),
        "apparent_bytes": sum(path.stat().st_size for path in all_files),
        "storage_note": "Release files are hard links to canonical local files; upload/copy reads ordinary complete files.",
    }
    if status["final_adapter_count"] != status["expected_final_adapter_count"]:
        raise AssertionError(status)
    (destination / "RELEASE_STATUS.json").write_text(
        json.dumps(status, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    build(args.destination.resolve())
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
