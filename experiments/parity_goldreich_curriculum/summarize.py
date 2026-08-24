#!/usr/bin/env python3
"""Score every curriculum stage and identify the first failed mechanism."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TRACE_RE = re.compile(r"<TRACE>\s*(.*?)\s*</TRACE>", re.DOTALL)
MASK_RE = re.compile(r"<MASK>\s*(.*?)\s*</MASK>", re.DOTALL)
ANSWER_RE = re.compile(r"Parity:\s*([01])")
LOCAL_RE = re.compile(r"(?:Encrypted state|Mask):\s*([01])")


def _load_prompts(directory: Path) -> list[dict[str, Any]]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]
    if not rows:
        raise ValueError(f"No prompts in {directory}")
    return rows


def _parse_sequence(raw: str, pattern: re.Pattern[str]) -> list[int] | None:
    matches = pattern.findall(raw)
    if len(matches) != 1:
        return None
    tokens = matches[0].split()
    if not tokens or any(token not in {"0", "1"} for token in tokens):
        return None
    return list(map(int, tokens))


def _score_task(prompts_dir: Path, responses_dir: Path, task: str) -> dict[str, Any]:
    prompts = _load_prompts(prompts_dir)
    missing = 0
    exact = bit_correct = bit_total = parity_correct = parsed_count = 0
    parsed_sequences: dict[str, list[int] | None] = {}
    twin_groups: dict[str, list[str]] = defaultdict(list)
    for prompt in prompts:
        experiment_id = str(prompt["experiment_id"])
        spec = prompt["spec"]
        path = responses_dir / f"{experiment_id}.json"
        if not path.exists():
            missing += 1
            bit_total += 1 if task in {"local_update", "predicate_local"} else len(spec["gold_sequence"])
            continue
        raw = str(json.loads(path.read_text(encoding="utf-8"))["raw_text"])
        if task in {"local_update", "predicate_local"}:
            matches = LOCAL_RE.findall(raw)
            predicted = int(matches[-1]) if matches else None
            parsed_count += predicted is not None
            gold = int(spec["gold_bit"])
            exact += predicted == gold
            bit_correct += predicted == gold
            bit_total += 1
        else:
            pattern = MASK_RE if task.startswith("mask_") else TRACE_RE
            predicted = _parse_sequence(raw, pattern)
            parsed_count += predicted is not None
            gold = [int(value) for value in spec["gold_sequence"]]
            exact += predicted == gold
            bit_correct += sum(a == b for a, b in zip((predicted or [])[: len(gold)], gold))
            bit_total += len(gold)
            parsed_sequences[experiment_id] = predicted
            if task.startswith(("supplied_", "joint_")):
                answers = ANSWER_RE.findall(raw)
                answer = int(answers[-1]) if answers else None
                parity_correct += answer == int(spec["final_parity"])
                twin_groups[str(spec["twin_id"])].append(experiment_id)

    n = len(prompts)
    scored_n = n - missing
    result: dict[str, Any] = {
        "task": task,
        "length": int(task.rsplit("_", 1)[-1]) if task.rsplit("_", 1)[-1].isdigit() else 1,
        "n": n,
        "missing_n": missing,
        "sequence_exact": exact / n,
        "sequence_bit_accuracy": bit_correct / bit_total if bit_total else 0.0,
        "parse_rate": parsed_count / scored_n if scored_n else 0.0,
    }
    if task.startswith(("supplied_", "joint_")):
        matching = complete = 0
        for ids in twin_groups.values():
            if len(ids) != 2:
                continue
            complete += 1
            if parsed_sequences.get(ids[0]) is not None and parsed_sequences.get(ids[0]) == parsed_sequences.get(ids[1]):
                matching += 1
        result["final_parity_exact"] = parity_correct / n
        result["complete_twin_pairs"] = complete
        result["twin_trace_match"] = matching / complete if complete else 0.0
        result["trace_only_individual_bit_recovery_upper_bound"] = 1.0 - 0.5 * result["twin_trace_match"]
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    *, config_path: Path, data_root: Path, artifacts_root: Path, responses_root: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for track, key in (("update", "update_stages"), ("goldreich", "goldreich_stages")):
        manifest = json.loads(
            (artifacts_root / track / "track_manifest.json").read_text(encoding="utf-8")
        )
        for entry in manifest["stages"]:
            label = f"{int(entry['stage_index']):02d}_{entry['stage']}"
            task = str(entry["evaluation_task"])
            metric = _score_task(
                data_root / "validation" / task,
                responses_root / track / "stages" / label / f"epoch_{entry['epoch']}" / task,
                task,
            )
            rows.append(
                {
                    "track": track,
                    "checkpoint_kind": "epoch",
                    "stage_index": entry["stage_index"],
                    "stage": entry["stage"],
                    "epoch": entry["epoch"],
                    "stage_epochs": entry["stage_epochs"],
                    **metric,
                }
            )
        for task in manifest["final_evaluation_tasks"]:
            metric = _score_task(
                data_root / "validation" / task,
                responses_root / track / "final" / task,
                task,
            )
            rows.append(
                {
                    "track": track,
                    "checkpoint_kind": "final",
                    "stage_index": len(config[key]),
                    "stage": "final_adapter",
                    "stage_epochs": None,
                    **metric,
                }
            )

    final = {
        (row["track"], row["task"]): row
        for row in rows
        if row["checkpoint_kind"] == "final"
    }
    gate = config["success_gate"]
    local = final[("update", "local_update")]
    supplied = final[("update", f"supplied_{config['max_mask_bits']}")]
    predicate_row = final[("goldreich", "predicate_local")]
    retained_supplied = final[("goldreich", f"supplied_{config['max_mask_bits']}")]
    mask = final[("goldreich", f"mask_{config['max_mask_bits']}")]
    joint = final[("goldreich", f"joint_{config['max_mask_bits']}")]

    checks = {
        "local_encrypted_update": local["sequence_exact"] >= gate["minimum_local_exact"],
        "supplied_mask_full_trace": (
            supplied["sequence_exact"] >= gate["minimum_sequence_exact"]
            and supplied["sequence_bit_accuracy"] >= gate["minimum_sequence_bit_accuracy"]
            and supplied["final_parity_exact"] >= gate["minimum_final_parity_exact"]
        ),
        "retained_supplied_mask_skill": (
            retained_supplied["sequence_exact"] >= gate["minimum_sequence_exact"]
            and retained_supplied["sequence_bit_accuracy"] >= gate["minimum_sequence_bit_accuracy"]
            and retained_supplied["final_parity_exact"] >= gate["minimum_final_parity_exact"]
        ),
        "local_goldreich_predicate": predicate_row["sequence_exact"] >= gate["minimum_local_exact"],
        "seed_to_mask_generation": (
            mask["sequence_exact"] >= gate["minimum_sequence_exact"]
            and mask["sequence_bit_accuracy"] >= gate["minimum_sequence_bit_accuracy"]
        ),
        "combined_goldreich_mechanism": (
            joint["sequence_exact"] >= gate["minimum_sequence_exact"]
            and joint["sequence_bit_accuracy"] >= gate["minimum_sequence_bit_accuracy"]
            and joint["final_parity_exact"] >= gate["minimum_final_parity_exact"]
            and joint["twin_trace_match"] >= gate["minimum_twin_trace_match"]
        ),
    }
    first_failure = next((name for name, passed in checks.items() if not passed), None)
    diagnosis = {
        None: "All mechanism stages passed.",
        "local_encrypted_update": "The one-step encrypted recurrence is not learned.",
        "supplied_mask_full_trace": "Local updates work, but errors accumulate in long supplied-mask traces.",
        "retained_supplied_mask_skill": "The update curriculum worked, but Goldreich training caused catastrophic forgetting.",
        "local_goldreich_predicate": "Encrypted updates work, but the five-bit nonlinear predicate is not learned.",
        "seed_to_mask_generation": "The predicate works locally, but fixed seed-position selection or long mask generation fails.",
        "combined_goldreich_mechanism": "Update and mask components work separately, but their integration fails.",
    }[first_failure]
    complete = all(row["missing_n"] == 0 for row in rows)
    summary = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "seed": config["seed"],
        "all_expected_responses_present": complete,
        "checks": checks,
        "first_failed_mechanism": first_failure,
        "diagnosis": diagnosis,
        "passes_full_mechanism": complete and all(checks.values()),
        "metrics": rows,
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts_root / "diagnostic_metrics.csv", rows)
    (artifacts_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Goldreich parity curriculum diagnostics",
        "",
        f"Overall: **{'PASS' if summary['passes_full_mechanism'] else 'FAIL'}**",
        "",
        f"Diagnosis: {diagnosis}",
        "",
        "| Track | Checkpoint | Task | Length | N | Exact | Bit accuracy | Final parity | Twin match |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        checkpoint_label = (
            "final"
            if row["checkpoint_kind"] == "final"
            else f"{row['stage']}/epoch_{row.get('epoch')}"
        )
        lines.append(
            f"| {row['track']} | {checkpoint_label} | `{row['task']}` | "
            f"{row['length']} | {row['n']} | {row['sequence_exact']:.3f} | "
            f"{row['sequence_bit_accuracy']:.3f} | "
            f"{row.get('final_parity_exact', float('nan')):.3f} | "
            f"{row.get('twin_trace_match', float('nan')):.3f} |"
        )
    lines.extend(["", "## Mechanism gates", ""])
    for name, passed in checks.items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")
    lines.append("")
    (artifacts_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"checks": checks, "diagnosis": diagnosis}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument(
        "--data-root", type=Path, default=Path("generated_data/parity_goldreich_curriculum/seed_0")
    )
    parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts/ce_parity_goldreich_curriculum")
    )
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_curriculum_responses/seed_0"),
    )
    args = parser.parse_args()
    summarize(
        config_path=args.config,
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        responses_root=args.responses_root,
    )


if __name__ == "__main__":
    main()
