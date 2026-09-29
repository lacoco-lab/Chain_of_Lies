#!/usr/bin/env python3
"""Build the lightweight reproducibility supplement."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

PACKAGE_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
ROOT = PACKAGE_ROOT.parent if PACKAGE_ROOT.name == "code_package" else PACKAGE_ROOT
CODE = ROOT / "code_package"
DATA = ROOT / "models_data_and_evaluation_outputs"
DEFAULT_OUTPUT = ROOT / "release_packages" / "lightweight_supplement"

TASKS = ("multiplication", "iterated_addition", "plain_parity", "s5_state_tracking")
MODELS = ("llama", "qwen")
PROTOCOL_NAMES = (
    "vanilla",
    "filler",
    "filler_plus_public_cot",
    "public_only_cot",
    "piggyback",
    "invisible",
)
CHECKPOINT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".ckpt"}


def copy_file(source: Path, output: Path) -> None:
    destination = output / source.relative_to(ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, output: Path) -> None:
    destination = output / source.relative_to(ROOT)
    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def difficulty(record: dict[str, Any]) -> int:
    spec = record.get("spec") or {}
    for key in ("evaluation_length", "operand_digits", "length", "num_elements"):
        value = spec.get(key)
        if isinstance(value, int):
            return value
    for key in ("public_bits", "public_instruction_sequence", "public_elements"):
        value = spec.get(key)
        if isinstance(value, list):
            return len(value)
    identifier = str(record.get("experiment_id", ""))
    if "_L" in identifier:
        return int(identifier.rsplit("_L", 1)[1])
    return 0


def stratified(records: Iterable[tuple[Path, dict[str, Any]]], count: int = 3):
    ordered = sorted(
        records, key=lambda item: (difficulty(item[1]), item[0].as_posix())
    )
    if len(ordered) <= count:
        return ordered
    indices = [
        round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)
    ]
    return [ordered[index] for index in indices]


def source_records(task_root: Path, marker: str, split: str):
    root = task_root / "prompts_and_splits"
    candidates = []
    for path in root.rglob("*.json"):
        relative = path.relative_to(root).as_posix()
        if "manifest" in path.name or marker not in relative:
            continue
        if split == "train":
            if not relative.startswith("seed_0/") or "val_prompts" in relative:
                continue
        else:
            if not (relative.startswith("shared_eval/") or "val_prompts" in relative):
                continue
        record = load(path)
        if record.get("experiment_id"):
            candidates.append((path, record))
    return candidates


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def build_scaling_samples(output: Path) -> dict[str, int]:
    sys.path.insert(0, str(CODE))
    from chain_of_lies.training.shared.trainer_utils import (  # noqa: PLC0415
        _canonical_public_cot_suffix,
    )
    from experiments.scaling_regime.protocols.registry import (  # noqa: PLC0415
        PROTOCOLS,
    )

    scaling = DATA / "Scaling Regime" / "tasks"
    filler_config = load(
        CODE / "experiments/scaling_regime/protocols/filler/config.json"
    )
    filler_task_names = {
        "multiplication": "multiplication",
        "iterated_addition": "knowledge",
        "plain_parity": "parity",
        "s5_state_tracking": "s5",
    }
    training_rows = []
    evaluation_rows = []
    for task in TASKS:
        task_root = scaling / task
        cached_sources = {
            marker: source_records(task_root, marker, "train")
            for marker in {PROTOCOLS[name].variant_role for name in PROTOCOL_NAMES}
        }
        evaluation_sources = {
            marker: source_records(task_root, marker, "eval")
            for marker in {PROTOCOLS[name].variant_role for name in PROTOCOL_NAMES}
        }
        evaluation_index = {
            marker: {record["experiment_id"]: (path, record) for path, record in rows}
            for marker, rows in evaluation_sources.items()
        }
        for protocol in PROTOCOL_NAMES:
            definition = PROTOCOLS[protocol]
            marker = definition.variant_role
            mode = definition.supervision_mode
            for source_path, record in stratified(cached_sources[marker]):
                target_record = copy.deepcopy(record)
                if protocol == "filler":
                    task_config = filler_config["tasks"][filler_task_names[task]]
                    key = str(target_record["spec"][task_config["difficulty_field"]])
                    target_record["spec"]["filler_token_count"] = int(
                        task_config["filler_token_counts"][key]
                    )
                elif protocol == "filler_plus_public_cot":
                    task_config = load(
                        CODE / f"experiments/scaling_regime/tasks/{task}/config.json"
                    )
                    target_record["spec"]["filler_token_count"] = int(
                        task_config["conditions"]["filler"]["filler_token_count"]
                    )
                suffix = _canonical_public_cot_suffix(
                    target_record, supervision_mode=mode
                )
                if suffix is None:
                    raise RuntimeError(
                        f"Could not build {task}/{protocol}: {source_path}"
                    )
                training_rows.append(
                    {
                        "task": task,
                        "protocol": protocol,
                        "seed": 0,
                        "difficulty": difficulty(record),
                        "source_file": str(source_path.relative_to(DATA)),
                        "prompt_record": target_record,
                        "supervised_suffix": suffix,
                        "note": (
                            "The atomic filler marker is expanded to the configured number "
                            "of period-token positions by the training collator."
                            if "filler" in protocol
                            else ""
                        ),
                    }
                )

            index = evaluation_index[marker]
            for model in MODELS:
                response_root = task_root / "responses" / model / "seed_0" / protocol
                candidates = []
                for response_path in response_root.rglob("*.json"):
                    response = load(response_path)
                    source = index.get(response.get("experiment_id"))
                    if source:
                        candidates.append((response_path, source[1]))
                for response_path, record in stratified(candidates):
                    source_path, _ = index[record["experiment_id"]]
                    evaluation_rows.append(
                        {
                            "task": task,
                            "protocol": protocol,
                            "model": model,
                            "seed": 0,
                            "difficulty": difficulty(record),
                            "source_file": str(source_path.relative_to(DATA)),
                            "response_file": str(response_path.relative_to(DATA)),
                            "prompt_record": record,
                            "response_record": load(response_path),
                        }
                    )

    sample_root = output / "reviewer_samples" / "scaling_regime"
    return {
        "scaling_training_examples": write_jsonl(
            sample_root / "training_examples.jsonl", training_rows
        ),
        "scaling_evaluation_examples": write_jsonl(
            sample_root / "evaluation_examples.jsonl", evaluation_rows
        ),
    }


def build_full_cot_samples(output: Path) -> dict[str, int]:
    root = DATA / "Scaling Regime" / "full_cot_baseline" / "generated_data"
    source_root = root / "full_cot_scaling_v1"
    response_root = root / "full_cot_scaling_v1_eval_responses"
    task_aliases = {
        "multiplication": "multiplication",
        "iterated_addition": "knowledge",
        "plain_parity": "parity",
        "s5_state_tracking": "s5",
    }
    training_rows, evaluation_rows = [], []
    for task, alias in task_aliases.items():
        for model in MODELS:
            train = [
                (p, load(p))
                for p in (source_root / alias / model / "seed_0" / "train").glob(
                    "*.json"
                )
            ]
            for path, record in stratified(train):
                training_rows.append(
                    {
                        "task": task,
                        "protocol": "full_cot",
                        "model": model,
                        "seed": 0,
                        "difficulty": difficulty(record),
                        "source_file": str(path.relative_to(DATA)),
                        "training_record": record,
                    }
                )
            sources = {
                p.stem: (p, load(p))
                for p in (source_root / alias / model / "seed_0" / "eval").glob(
                    "*.json"
                )
            }
            candidates = []
            for path in (response_root / alias / model / "seed_0").rglob("*.json"):
                source = sources.get(path.stem)
                if source:
                    candidates.append((path, source[1]))
            for path, record in stratified(candidates):
                source_path, _ = sources[path.stem]
                evaluation_rows.append(
                    {
                        "task": task,
                        "protocol": "full_cot",
                        "model": model,
                        "seed": 0,
                        "difficulty": difficulty(record),
                        "source_file": str(source_path.relative_to(DATA)),
                        "response_file": str(path.relative_to(DATA)),
                        "prompt_record": record,
                        "response_record": load(path),
                    }
                )
    sample_root = output / "reviewer_samples" / "scaling_regime"
    return {
        "full_cot_training_examples": write_jsonl(
            sample_root / "full_cot_training_examples.jsonl", training_rows
        ),
        "full_cot_evaluation_examples": write_jsonl(
            sample_root / "full_cot_evaluation_examples.jsonl", evaluation_rows
        ),
    }


def first_jsonl_rows(path: Path, count: int = 5) -> list[dict[str, Any]]:
    rows = []
    if path.suffix == ".gz":
        handle_context = gzip.open(path, mode="rt", encoding="utf-8")
    else:
        handle_context = path.open(mode="r", encoding="utf-8")
    with handle_context as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) == count:
                break
    return rows


def build_goldreich_samples(output: Path) -> dict[str, int]:
    root = DATA / "Encrypted Parity"
    transformer = root / "Transformer Main Experiment" / "generated_data"
    rows = []
    for path in sorted(transformer.rglob("*.jsonl.gz")):
        if path.name not in {"train.jsonl.gz", "test.jsonl.gz"}:
            continue
        for record in first_jsonl_rows(path):
            rows.append(
                {
                    "family": "transformer",
                    "split": path.name.split(".")[0],
                    "source_file": str(path.relative_to(DATA)),
                    "record": record,
                }
            )
    qwen = root / "Qwen Failed Attempts" / "generated_data"
    for path in sorted(qwen.rglob("*.jsonl")):
        if "train" not in path.name and "validation" not in path.name:
            continue
        for record in first_jsonl_rows(path, count=2):
            rows.append(
                {
                    "family": "qwen_diagnostic",
                    "split": "train" if "train" in path.name else "validation",
                    "source_file": str(path.relative_to(DATA)),
                    "record": record,
                }
            )
    count = write_jsonl(
        output / "reviewer_samples" / "goldreich_parity" / "data_examples.jsonl",
        rows,
    )
    return {"goldreich_training_and_evaluation_examples": count}


def copy_processed_results(output: Path) -> None:
    scaling = DATA / "Scaling Regime"
    for path in (
        scaling / "paper_source_data",
        scaling / "artifacts" / "visible_leakage_audit",
    ):
        copy_tree(path, output)
    for path in (scaling / "tasks").rglob("*"):
        if path.is_file() and (
            "evaluation_reports" in path.parts
            or "source_aggregate_results" in path.parts
        ):
            copy_file(path, output)
    for path in (scaling / "full_cot_baseline").rglob("*"):
        if path.is_file() and (
            "per_length_eval" in path.parts or path.name == "summary.json"
        ):
            copy_file(path, output)

    encrypted = DATA / "Encrypted Parity"
    copy_tree(encrypted / "Transformer Main Experiment" / "paper_source_data", output)
    for path in encrypted.rglob("*"):
        if not path.is_file() or path.suffix.lower() in CHECKPOINT_SUFFIXES:
            continue
        if path.name in {
            "metrics.json",
            "manifest.json",
            "summary.json",
            "metrics.csv",
            "REPORT.md",
        }:
            copy_file(path, output)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(output: Path, sample_counts: dict[str, int]) -> None:
    files = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        if path.name == "MANIFEST.json":
            continue
        files.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "sample_counts": sample_counts,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    (output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def write_readme(output: Path) -> None:
    text = """# Anonymous Reproducibility Supplement

This lightweight package contains the paper-facing generation, training,
evaluation, analysis, and plotting code, exact configurations, processed
results, evaluation reports, and examples of training targets and held-out
generations. Cluster submission scripts and site-specific recovery tools are
omitted. The companion Zenodo record contains every saved synthetic record,
raw generation, evaluation file, and model checkpoint.

## Install and verify

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -m compileall -q code_package
```

Run experiment commands from `code_package/`. For example:

```bash
cd code_package
python experiments/scaling_regime/tasks/multiplication/experiment.py generate --config experiments/scaling_regime/tasks/multiplication/config.json
python experiments/scaling_regime/tasks/multiplication/experiment.py cell --config experiments/scaling_regime/tasks/multiplication/config.json --model qwen --seed 0 --condition vanilla
python experiments/scaling_regime/protocols/filler/experiment.py cell --task multiplication --model qwen --seed 0
python experiments/scaling_regime/protocols/full_cot/experiment.py cell --task multiplication --model qwen --seed 0
```

The other scaling tasks have the same `experiment.py` entry point under
`experiments/scaling_regime/tasks/`. Their configurations give the exact
training settings and seeds 0, 1, and 2. Plain parity selects its seed with
`config.json`, `config_seed_1.json`, or `config_seed_2.json`. The primary
Filler and Full-CoT runs use their dedicated protocol entry points above.

The final encrypted-parity Transformer uses seed 0. A representative
easy-length run is:

```bash
python experiments/goldreich_parity/transformer/main_experiment/generate_data.py --config experiments/goldreich_parity/transformer/main_experiment/encrypted_easy_data_config.json --output-root generated_data/parity_goldreich_standard_easy/seed_0 --length 4
python experiments/goldreich_parity/transformer/main_experiment/train_standard_v2.py --config experiments/goldreich_parity/transformer/main_experiment/standard_v2_easy_config.json --data-root generated_data/parity_goldreich_standard_easy/seed_0 --output-root artifacts/parity_goldreich_standard_transformer_v2/d128/n4 --length 4 --model-dim 128
```

The final Transformer settings for longer lengths are in
`standard_v2_config.json` and `standard_v2_n2048_config.json`; matched
controls and ablations have separate entry points under `transformer/controls/`
and `transformer/ablations/`. Training requires the listed dependencies, GPU
resources, and access to the configured base models. Saved metrics can be read
without retraining.

## Contents

- `code_package/`: anonymized experiment, training, evaluation, and plotting code.
- `code_package/experiments/scaling_regime/protocols/`: explicit definitions for Vanilla,
  Filler, Public-only CoT, Piggybacking, Invisible, the appendix ablation, and Full CoT.
- `reviewer_samples/scaling_regime/`: low-, middle-, and high-difficulty training targets and actual held-out generations for every task and protocol, including Full CoT.
- `reviewer_samples/goldreich_parity/`: training and test examples for every released Transformer length, matched controls, and representative Qwen diagnostics.
- `models_data_and_evaluation_outputs/`: processed metrics, per-length reports, paper source tables, and the full visible-leakage audit.
- `MANIFEST.json`: size and SHA-256 digest for every included file.

The sample files are illustrative subsets. Aggregate tables and evaluation reports
are complete; all raw records and checkpoints are in the companion Zenodo files.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    copy_tree(CODE, output)
    for name in ("pyproject.toml", "requirements.txt", ".env.example"):
        copy_file(ROOT / name, output)
    copy_processed_results(output)
    counts = {}
    counts.update(build_scaling_samples(output))
    counts.update(build_full_cot_samples(output))
    counts.update(build_goldreich_samples(output))
    write_readme(output)
    write_manifest(output, counts)
    print(
        json.dumps(
            {
                "output": str(output),
                "sample_counts": counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
