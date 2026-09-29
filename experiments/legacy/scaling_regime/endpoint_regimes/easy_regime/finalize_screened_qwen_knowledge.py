#!/usr/bin/env python3
"""Promote the screened Qwen knowledge rerun into the canonical Easy results.

Only ``knowledge/qwen`` artifacts and responses are replaced. The superseded
all-100-fact slice is moved outside the canonical roots for recoverability.
All other task/model results are left untouched.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.summarize import (
    summarize,
)

SOURCE_ARTIFACTS = (
    REPO_ROOT / "artifacts/easy_regime_qwen_screened_knowledge_v1/knowledge/qwen"
)
SOURCE_RESPONSES = (
    REPO_ROOT
    / "generated_data/easy_regime_qwen_screened_knowledge_v1_eval_responses/knowledge/qwen"
)
TARGET_ARTIFACTS = REPO_ROOT / "artifacts/easy_regime/knowledge/qwen"
TARGET_RESPONSES = (
    REPO_ROOT / "generated_data/easy_regime_eval_responses/knowledge/qwen"
)
ARCHIVE_ARTIFACTS = REPO_ROOT / "artifacts/easy_regime_superseded/knowledge_qwen_all100"
ARCHIVE_RESPONSES = (
    REPO_ROOT
    / "generated_data/easy_regime_eval_responses_superseded/knowledge_qwen_all100"
)
CONFIG = (
    REPO_ROOT
    / "experiments/legacy/scaling_regime/endpoint_regimes/easy_regime/config.json"
)

CONDITIONS = {
    "vanilla": "knowledge_easy_control",
    "public_only_cot": "knowledge_easy_control",
    "filler": "knowledge_easy_control",
    "piggyback": "knowledge_easy_piggyback",
    "steganography": "knowledge_easy_steg_local_invisible",
}


def verify_source() -> None:
    for seed in (0, 1, 2):
        for condition, variant in CONDITIONS.items():
            cell = SOURCE_ARTIFACTS / f"seed_{seed}" / condition
            report_path = cell / "per_variant_eval/ckpt_final" / f"{variant}.json"
            metadata_path = cell / variant / "ckpt_final/training_metadata.json"
            response_dir = (
                SOURCE_RESPONSES
                / f"seed_{seed}"
                / condition
                / "ckpt_final/trained"
                / variant
            )
            if not report_path.exists() or not metadata_path.exists():
                raise RuntimeError(
                    f"Incomplete source cell: seed={seed}, condition={condition}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("steps") != 10_000
                or metadata.get("train_examples_seen") != 10_000
                or metadata.get("target_train_coverage") != 1.0
            ):
                raise RuntimeError(f"Incomplete training metadata: {metadata_path}")
            reports = json.loads(report_path.read_text(encoding="utf-8")).get(
                "reports", []
            )
            if (
                len(reports) != 1
                or reports[0]["rl"].get("num_examples") != 1_000
                or reports[0]["rl"].get("missing_responses")
            ):
                raise RuntimeError(f"Incomplete evaluation report: {report_path}")
            if len(list(response_dir.glob("*.json"))) != 1_000:
                raise RuntimeError(f"Incomplete trained responses: {response_dir}")


def replace_slice(source: Path, target: Path, archive: Path) -> None:
    if archive.exists():
        raise RuntimeError(f"Refusing to overwrite existing archive: {archive}")
    if not target.exists():
        raise RuntimeError(f"Missing canonical target: {target}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(target), str(archive))
    shutil.copytree(source, target)


def main() -> None:
    verify_source()
    replace_slice(SOURCE_ARTIFACTS, TARGET_ARTIFACTS, ARCHIVE_ARTIFACTS)
    replace_slice(SOURCE_RESPONSES, TARGET_RESPONSES, ARCHIVE_RESPONSES)
    summary = summarize(CONFIG, REPO_ROOT / "artifacts/easy_regime")
    if len(summary["per_seed_metrics"]) != 90:
        raise RuntimeError("Canonical Easy summary is not complete after promotion")
    print(
        json.dumps(
            {
                "valid": True,
                "promoted": "knowledge/qwen",
                "canonical_cells": len(summary["per_seed_metrics"]),
                "superseded_artifacts": str(ARCHIVE_ARTIFACTS.relative_to(REPO_ROOT)),
                "superseded_responses": str(ARCHIVE_RESPONSES.relative_to(REPO_ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
