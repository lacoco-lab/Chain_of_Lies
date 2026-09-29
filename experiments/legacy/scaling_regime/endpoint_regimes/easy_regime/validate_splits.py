#!/usr/bin/env python3
"""Fail-closed validation for matched Easy splits."""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.training.shared.trainer_utils import (
    FILLER_TOKEN_MARKER,
    _canonical_public_cot_suffix,
)
from chain_of_lies.variants.steganography.data_generation.generate import (
    validate_steganographic_target,
)
from chain_of_lies.variants.steganography.hard_task_channels import validate_target
from experiments.legacy.scaling_regime.endpoint_regimes.easy_regime.generate_splits import (
    TASKS,
    VARIANT_KINDS,
    active_tasks,
    load_config,
    variant,
)


def _records(root: Path, split: str, expected: int) -> list[dict[str, Any]]:
    paths = sorted((root / f"{split}_prompts").glob("*.json"))
    if len(paths) != expected:
        raise RuntimeError(
            f"Expected {expected} JSON records under {root}/{split}_prompts; found {len(paths)}."
        )
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def validate(
    config_path: Path, data_root: Path, seed: int | None = None
) -> dict[str, Any]:
    config = load_config(config_path)
    seeds = [seed] if seed is not None else config["seeds"]
    reports = {}
    for task in active_tasks(config):
        for current_seed in seeds:
            by_variant = {}
            train_questions, val_questions = set(), set()
            for kind in VARIANT_KINDS:
                name = variant(task, kind)
                by_variant[name] = {}
                for split, count in (
                    ("train", int(config["train_n"])),
                    ("val", int(config["val_n"])),
                ):
                    rows = _records(
                        data_root / task / f"seed_{current_seed}" / name, split, count
                    )
                    by_variant[name][split] = rows
                    for record in rows:
                        if (
                            record.get("variant_name") != name
                            or record.get("difficulty_regime") != "easy"
                        ):
                            raise RuntimeError(f"Record identity mismatch in {name}.")
                        spec = record["spec"]
                        if spec["public_question"] == spec["private_question"]:
                            raise RuntimeError(
                                f"Identical paired questions in {record['experiment_id']}."
                            )
                        (train_questions if split == "train" else val_questions).update(
                            (spec["public_question"], spec["private_question"])
                        )
                        if kind == "steg_local_invisible":
                            (
                                validate_steganographic_target
                                if task == "multiplication"
                                else validate_target
                            )(record)

            control = variant(task, "control")
            piggy = variant(task, "piggyback")
            steg = variant(task, "steg_local_invisible")
            for split in ("train", "val"):
                private_views = [
                    [r["spec"]["private_question"] for r in by_variant[name][split]]
                    for name in (control, piggy, steg)
                ]
                if not (private_views[0] == private_views[1] == private_views[2]):
                    raise RuntimeError(
                        f"Private questions are not exactly matched for {task}/seed_{current_seed}/{split}."
                    )
                control_public = [
                    r["spec"]["public_question"] for r in by_variant[control][split]
                ]
                steg_public = [
                    r["spec"]["public_question"] for r in by_variant[steg][split]
                ]
                if control_public != steg_public:
                    raise RuntimeError(
                        f"Control/steg public questions are not matched for {task}/seed_{current_seed}/{split}."
                    )
                sample = by_variant[control][split][0]
                sample["spec"]["filler_token_count"] = int(
                    config["conditions"]["filler"]["filler_token_count"]
                )
                for mode in ("answer_only", "public_cot", "filler_public_cot"):
                    suffix = _canonical_public_cot_suffix(sample, supervision_mode=mode)
                    if suffix is None or (
                        mode == "filler_public_cot"
                        and suffix.count(FILLER_TOKEN_MARKER) != 1
                    ):
                        raise RuntimeError(
                            f"Could not construct {mode} target for {task}."
                        )
            if train_questions & val_questions:
                raise RuntimeError(
                    f"Train/validation individual-question overlap for {task}/seed_{current_seed}."
                )
            reports[f"{task}/seed_{current_seed}"] = {
                "valid": True,
                "private_questions_matched": True,
                "control_steg_public_matched": True,
                "train_val_question_overlap": 0,
            }
    result = {"valid": True, "validated_seeds": seeds, "reports": reports}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("generated_data/easy_regime")
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    validate(args.config, args.data_root, args.seed)
