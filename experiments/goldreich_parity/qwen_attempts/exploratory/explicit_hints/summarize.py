#!/usr/bin/env python3
"""Score explicit-hint checkpoints and locate retrieval, update, length, or retention failures."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

COT_RE = re.compile(r"<COT>\s*(.*?)\s*</COT>", re.DOTALL)
LINE_RE = re.compile(r"^\(use Seed: ([A-P](?: [A-P]){4})\) ([01])$")
ANSWER_RE = re.compile(r"Parity:\s*([01])")
LOCAL_RE = re.compile(r"(?:Predicate|Mask|Encrypted state):\s*([01])")
SEED_VALUE_RE = re.compile(r"\b[A-P]\s*[:=]\s*[01]\b")


def _load(directory: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob("*.json"))
    ]
    if not rows:
        raise ValueError(f"No prompt records in {directory}")
    return rows


def _parse_trace(raw: str) -> tuple[list[str], list[int]] | None:
    matches = COT_RE.findall(raw)
    if len(matches) != 1:
        return None
    selectors, bits = [], []
    for line in matches[0].strip().splitlines():
        match = LINE_RE.fullmatch(line.strip())
        if match is None:
            return None
        selectors.append(f"(use Seed: {match.group(1)})")
        bits.append(int(match.group(2)))
    return selectors, bits


def score_task(prompts_dir: Path, responses_dir: Path, task: str) -> dict[str, Any]:
    prompts = _load(prompts_dir)
    exact = bit_correct = bit_total = parsed = parity_correct = selector_correct = 0
    leak_count = missing = 0
    predicted_traces: dict[str, list[int] | None] = {}
    twin_groups: dict[str, list[str]] = defaultdict(list)
    for prompt in prompts:
        experiment_id, spec = str(prompt["experiment_id"]), prompt["spec"]
        response_path = responses_dir / f"{experiment_id}.json"
        if not response_path.exists():
            missing += 1
            bit_total += (
                1
                if task
                in {
                    "predicate_local",
                    "hinted_mask_local",
                    "hinted_start_local",
                    "hinted_update_local",
                }
                else len(spec["gold_sequence"])
            )
            continue
        raw = str(json.loads(response_path.read_text(encoding="utf-8"))["raw_text"])
        leak_count += bool(
            SEED_VALUE_RE.search(raw)
            or "<PRIVATE_SEED>" in raw
            or "<PRIVATE_INPUT>" in raw
        )
        if task in {
            "predicate_local",
            "hinted_mask_local",
            "hinted_start_local",
            "hinted_update_local",
        }:
            values = LOCAL_RE.findall(raw)
            prediction = int(values[-1]) if values else None
            parsed += prediction is not None
            gold = int(spec["gold_bit"])
            exact += prediction == gold
            bit_correct += prediction == gold
            bit_total += 1
            continue
        parsed_trace = _parse_trace(raw)
        parsed += parsed_trace is not None
        predicted_selectors, predicted_bits = parsed_trace if parsed_trace else ([], [])
        gold_bits = list(map(int, spec["gold_sequence"]))
        gold_selectors = list(map(str, spec["selectors"]))
        exact += predicted_bits == gold_bits and predicted_selectors == gold_selectors
        bit_correct += sum(
            a == b for a, b in zip(predicted_bits[: len(gold_bits)], gold_bits)
        )
        bit_total += len(gold_bits)
        selector_correct += sum(
            a == b
            for a, b in zip(predicted_selectors[: len(gold_selectors)], gold_selectors)
        )
        answers = ANSWER_RE.findall(raw)
        parity_correct += bool(answers) and int(answers[-1]) == int(
            spec["final_parity"]
        )
        predicted_traces[experiment_id] = (
            predicted_bits if len(predicted_bits) == len(gold_bits) else None
        )
        twin_groups[str(spec["twin_id"])].append(experiment_id)
    n, scored = len(prompts), len(prompts) - missing
    result: dict[str, Any] = {
        "task": task,
        "length": (
            int(task.rsplit("_", 1)[-1]) if task.rsplit("_", 1)[-1].isdigit() else 1
        ),
        "n": n,
        "missing_n": missing,
        "exact": exact / n,
        "bit_accuracy": bit_correct / bit_total if bit_total else 0.0,
        "parse_rate": parsed / scored if scored else 0.0,
        "seed_value_leak_rate": leak_count / n,
    }
    if task.startswith("hinted_joint_"):
        length = int(spec["length"])
        result["selector_accuracy"] = selector_correct / (n * length)
        result["final_parity_exact"] = parity_correct / n
        complete = matching = 0
        for ids in twin_groups.values():
            if len(ids) != 2:
                continue
            complete += 1
            left, right = predicted_traces.get(ids[0]), predicted_traces.get(ids[1])
            matching += left is not None and left == right
        result["complete_twin_pairs"] = complete
        result["twin_trace_match"] = matching / complete if complete else 0.0
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _passes(row: dict[str, Any], gate: dict[str, float]) -> bool:
    if row["task"].startswith("hinted_joint_"):
        return (
            row["exact"] >= gate["minimum_trace_exact"]
            and row["bit_accuracy"] >= gate["minimum_trace_bit_accuracy"]
            and row["final_parity_exact"] >= gate["minimum_parity_exact"]
            and row["selector_accuracy"] >= gate["minimum_selector_exact"]
            and row["twin_trace_match"] >= gate["minimum_twin_match"]
            and row["seed_value_leak_rate"] <= gate["maximum_seed_leak_rate"]
        )
    return row["exact"] >= gate["minimum_local_exact"]


def summarize(
    config_path: Path, data_root: Path, artifacts_root: Path, responses_root: Path
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for track in ("core", "extension"):
        manifest = json.loads(
            (artifacts_root / track / "track_manifest.json").read_text(encoding="utf-8")
        )
        for entry in manifest["stages"]:
            label, task = f"{int(entry['stage_index']):02d}_{entry['stage']}", str(
                entry["evaluation_task"]
            )
            metric = score_task(
                data_root / "validation" / task,
                responses_root
                / track
                / "stages"
                / label
                / f"epoch_{entry['epoch']}"
                / task,
                task,
            )
            rows.append({"track": track, "checkpoint_kind": "epoch", **entry, **metric})
        for task in manifest["final_evaluation_tasks"]:
            metric = score_task(
                data_root / "validation" / task,
                responses_root / track / "final" / task,
                task,
            )
            rows.append(
                {
                    "track": track,
                    "checkpoint_kind": "final",
                    "stage": "final_adapter",
                    **metric,
                }
            )
    gate = config["success_gate"]
    dedicated: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["checkpoint_kind"] != "epoch" or row["stage"] != row["task"]:
            continue
        current = dedicated.get(row["task"])
        if current is None or (row["exact"], row["bit_accuracy"]) > (
            current["exact"],
            current["bit_accuracy"],
        ):
            dedicated[row["task"]] = row
    ordered_tasks = [
        "predicate_local",
        "hinted_mask_local",
        "hinted_start_local",
        "hinted_update_local",
    ] + [
        f"hinted_joint_{length}"
        for length in config["core_lengths"] + config["extension_lengths"]
    ]
    checks = {task: _passes(dedicated[task], gate) for task in ordered_tasks}
    first_failure = next((task for task in ordered_tasks if not checks[task]), None)
    final_rows = {
        (row["track"], row["task"]): row
        for row in rows
        if row["checkpoint_kind"] == "final"
    }
    retention = {
        "core_final": {
            task: _passes(final_rows[("core", task)], gate)
            for task in [
                "predicate_local",
                "hinted_mask_local",
                "hinted_start_local",
                "hinted_update_local",
            ]
            + [f"hinted_joint_{length}" for length in config["core_lengths"]]
        },
        "extension_final": {
            task: _passes(final_rows[("extension", task)], gate)
            for task in [
                "predicate_local",
                "hinted_mask_local",
                "hinted_start_local",
                "hinted_update_local",
            ]
            + [
                f"hinted_joint_{length}"
                for length in config["core_lengths"] + config["extension_lengths"]
            ]
        },
    }
    diagnosis_map = {
        "predicate_local": "The five-bit Boolean predicate is not learned.",
        "hinted_mask_local": "The model cannot retrieve explicitly named private seed values and combine them.",
        "hinted_start_local": "Hinted mask retrieval works, but computing the first encrypted state fails.",
        "hinted_update_local": "Hinted mask retrieval works, but combining it with one encrypted update fails.",
    }
    if first_failure is None:
        diagnosis = "All explicit-hint mechanism and length stages passed."
    elif first_failure.startswith("hinted_joint_"):
        diagnosis = f"Local hinted operations work, but the full trace first fails at length {first_failure.rsplit('_', 1)[-1]}."
    else:
        diagnosis = diagnosis_map[first_failure]
    core_tasks = [
        "predicate_local",
        "hinted_mask_local",
        "hinted_start_local",
        "hinted_update_local",
    ] + [f"hinted_joint_{length}" for length in config["core_lengths"]]
    extension_tasks = [
        f"hinted_joint_{length}" for length in config["extension_lengths"]
    ]
    summary = {
        "schema_version": 1,
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "seed": config["seed"],
        "all_expected_responses_present": all(row["missing_n"] == 0 for row in rows),
        "dedicated_stage_checks": checks,
        "first_failed_stage": first_failure,
        "diagnosis": diagnosis,
        "retention_checks": retention,
        "passes_core_mechanism": all(checks[task] for task in core_tasks),
        "passes_64_extension": all(checks[task] for task in extension_tasks),
        "metrics": rows,
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts_root / "diagnostic_metrics.csv", rows)
    (artifacts_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Explicit-hypergraph Goldreich parity diagnostics",
        "",
        f"Core mechanism: **{'PASS' if summary['passes_core_mechanism'] else 'FAIL'}**",
        "",
        f"Length-64 extension: **{'PASS' if summary['passes_64_extension'] else 'FAIL'}**",
        "",
        f"Diagnosis: {diagnosis}",
        "",
        "| Track | Checkpoint | Task | N | Exact | Bit accuracy | Selector | Parity | Twin match | Leak |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        checkpoint = (
            f"{row['stage']}/epoch_{row.get('epoch')}"
            if row["checkpoint_kind"] == "epoch"
            else "final"
        )
        value = lambda key: f"{row[key]:.3f}" if key in row else "nan"
        lines.append(
            f"| {row['track']} | {checkpoint} | `{row['task']}` | {row['n']} | {value('exact')} | {value('bit_accuracy')} | {value('selector_accuracy')} | {value('final_parity_exact')} | {value('twin_trace_match')} | {value('seed_value_leak_rate')} |"
        )
    lines.extend(["", "## Best dedicated-stage checks", ""])
    lines.extend(
        f"- `{task}`: {'pass' if passed else 'fail'}" for task, passed in checks.items()
    )
    lines.extend(["", "## Final-adapter retention", ""])
    for track, values in retention.items():
        lines.append(
            f"- `{track}`: "
            + ", ".join(
                f"{task}={'pass' if passed else 'fail'}"
                for task, passed in values.items()
            )
        )
    lines.append("")
    (artifacts_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"checks": checks, "diagnosis": diagnosis}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_explicit_hints/seed_0"),
    )
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/ce_parity_goldreich_explicit_hints"),
    )
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_explicit_hints_responses/seed_0"),
    )
    args = parser.parse_args()
    summarize(args.config, args.data_root, args.artifacts_root, args.responses_root)


if __name__ == "__main__":
    main()
