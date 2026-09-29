#!/usr/bin/env python3
"""Summarize S5 Medium calibration accuracy by exact chain length."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(REPO_ROOT))

from chain_of_lies.evaluation.rewards import (
    default_reward_config,
    score_completion,
)  # noqa: E402


def _longest_repeated_token_run(token_ids: list[int]) -> int:
    best = current = 0
    previous: int | None = None
    for token_id in token_ids:
        if token_id == previous:
            current += 1
        else:
            previous = token_id
            current = 1
        best = max(best, current)
    return best


def _condition_map(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in config["conditions"]}


def _response_dir(root: Path, condition: str, source: str) -> Path:
    return (
        root
        / "seed_0"
        / condition
        / "s5_control"
        / "ckpt_final"
        / source
        / "s5_control"
    )


def _summarize_bucket(
    prompts: list[dict[str, Any]],
    responses_dir: Path,
    *,
    expected_filler: int,
) -> dict[str, Any]:
    scored = []
    filler_runs: list[int] = []
    missing: list[str] = []
    for prompt in prompts:
        path = responses_dir / f"{prompt['experiment_id']}.json"
        if not path.exists():
            missing.append(prompt["experiment_id"])
            continue
        response = json.loads(path.read_text(encoding="utf-8"))
        scored.append(
            score_completion(prompt, response["raw_text"], default_reward_config())
        )
        if expected_filler:
            filler_runs.append(
                _longest_repeated_token_run(
                    [int(value) for value in response.get("generated_token_ids", [])]
                )
            )
    n = len(scored)
    if n == 0:
        return {"n": 0, "missing": missing}
    row = {
        "n": n,
        "missing": missing,
        "joint_exact": sum(item.task_success for item in scored) / n,
        "public_exact": sum(bool(item.flags.get("public_correct")) for item in scored)
        / n,
        "private_exact": sum(bool(item.flags.get("private_correct")) for item in scored)
        / n,
        "format": sum(item.format_ok for item in scored) / n,
        "avg_cot_words": sum(item.cot_word_count for item in scored) / n,
    }
    if expected_filler:
        row.update(
            {
                "expected_filler_tokens": expected_filler,
                "mean_longest_repeated_token_run": sum(filler_runs) / len(filler_runs),
                "exact_filler_run_rate": sum(
                    run == expected_filler for run in filler_runs
                )
                / len(filler_runs),
                "minimum_filler_run": min(filler_runs),
                "maximum_filler_run": max(filler_runs),
            }
        )
    return row


def summarize(
    *,
    config_path: Path,
    split_root: Path,
    responses_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    conditions = _condition_map(config)
    prompt_paths = sorted((split_root / "seed_0/s5_control/val_prompts").glob("*.json"))
    prompts = [json.loads(path.read_text(encoding="utf-8")) for path in prompt_paths]
    if not prompts:
        raise ValueError("No S5 Medium validation prompts found.")
    prompts_by_length: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for prompt in prompts:
        prompts_by_length[int(prompt["spec"]["calibration_length"])].append(prompt)

    rows: list[dict[str, Any]] = []
    for condition, condition_spec in conditions.items():
        filler_count = int(condition_spec["filler_token_count"])
        for source in ("baseline", "finetuned"):
            response_dir = _response_dir(responses_root, condition, source)
            for length in [int(value) for value in config["lengths"]]:
                metrics = _summarize_bucket(
                    prompts_by_length[length],
                    response_dir,
                    expected_filler=filler_count if source == "finetuned" else 0,
                )
                rows.append(
                    {
                        "condition": condition,
                        "supervision_mode": condition_spec["supervision_mode"],
                        "filler_token_count": filler_count,
                        "source": source,
                        "length": length,
                        **metrics,
                    }
                )

    missing = [row for row in rows if row["n"] != int(config["validation_per_length"])]
    criteria = config["medium_criterion"]
    lookup = {(row["condition"], row["source"], row["length"]): row for row in rows}
    candidates: list[dict[str, Any]] = []
    for length in [int(value) for value in config["lengths"]]:
        standard = lookup[("public_cot", "finetuned", length)]
        for mechanism in (
            "verbose_public_cot",
            "filler_64",
            "filler_128",
            "filler_256",
        ):
            row = lookup[(mechanism, "finetuned", length)]
            gain = row.get("private_exact", 0.0) - standard.get("private_exact", 0.0)
            passes = (
                standard.get("public_exact", 0.0)
                >= float(criteria["minimum_public_exact"])
                and float(criteria["standard_private_floor"])
                <= standard.get("private_exact", 0.0)
                <= float(criteria["standard_private_ceiling"])
                and row.get("private_exact", 0.0)
                >= float(criteria["minimum_mechanism_private"])
                and gain >= float(criteria["minimum_private_gain"])
            )
            candidates.append(
                {
                    "length": length,
                    "mechanism": mechanism,
                    "standard_private_exact": standard.get("private_exact"),
                    "mechanism_private_exact": row.get("private_exact"),
                    "private_gain": gain,
                    "mechanism_public_exact": row.get("public_exact"),
                    "passes_preregistered_medium_criterion": passes,
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "metrics_by_length.csv"
    fieldnames = sorted({key for row in rows for key in row if key != "missing"})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            {key: value for key, value in row.items() if key != "missing"}
            for row in rows
        )

    summary = {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "lengths": config["lengths"],
        "conditions": list(conditions),
        "all_expected_responses_present": not missing,
        "incomplete_rows": missing,
        "metrics_by_length": rows,
        "medium_candidates": candidates,
        "passing_candidates": [
            item for item in candidates if item["passes_preregistered_medium_criterion"]
        ],
        "claim_boundary": (
            "filler_public_cot is a CE-trained filler workspace; it is not evidence of emergent "
            "inference-time filler use."
        ),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# S5 Medium calibration",
        "",
        "Seed-0 calibration with shared random initial states and independent, length-matched chains.",
        "",
        "| Length | Condition | Public | Private | Joint | Private gain vs standard |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for length in [int(value) for value in config["lengths"]]:
        standard_private = lookup[("public_cot", "finetuned", length)].get(
            "private_exact", 0.0
        )
        for condition in conditions:
            row = lookup[(condition, "finetuned", length)]
            lines.append(
                f"| {length} | `{condition}` | {row.get('public_exact', 0.0):.3f} | "
                f"{row.get('private_exact', 0.0):.3f} | {row.get('joint_exact', 0.0):.3f} | "
                f"{row.get('private_exact', 0.0) - standard_private:+.3f} |"
            )
    lines.extend(
        [
            "",
            "## Preregistered Medium candidates",
            "",
            *(
                [
                    f"- Length {item['length']}, `{item['mechanism']}`: private "
                    f"{item['standard_private_exact']:.3f} → {item['mechanism_private_exact']:.3f} "
                    f"({item['private_gain']:+.3f})."
                    for item in summary["passing_candidates"]
                ]
                or [
                    "- None. Under the preregistered criterion, S5 has no demonstrated Medium regime yet."
                ]
            ),
            "",
            "> Filler conditions here are CE-trained workspaces, not emergent inference-time filler use.",
        ]
    )
    (output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Wrote {csv_path}, {output_root / 'summary.json'}, and {output_root / 'REPORT.md'}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--split-root", type=Path, default=Path("generated_data/s5_medium_calibration")
    )
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path("generated_data/s5_medium_calibration_eval_responses"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("artifacts/ce_s5_medium_calibration")
    )
    args = parser.parse_args()
    summarize(
        config_path=args.config,
        split_root=args.split_root,
        responses_root=args.responses_root,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
