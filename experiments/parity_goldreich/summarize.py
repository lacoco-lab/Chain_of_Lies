#!/usr/bin/env python3
"""Score plaintext, supplied-mask, and Goldreich-generated encrypted parity traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_ANSWER_RE = re.compile(r"Parity:\s*([01])")
_TRACE_BLOCK_RE = re.compile(r"<TRACE>\s*(.*?)\s*</TRACE>", re.DOTALL)
_STRICT_RE = re.compile(
    r"\s*<TRACE>\s*\n([01](?: [01])*)\s*\n</TRACE>\s*\n"
    r"<ANSWER>\s*\nParity:\s*([01])\s*\n</ANSWER>\s*",
    re.DOTALL,
)


def parse_completion(raw_text: str) -> dict[str, Any]:
    answers = _ANSWER_RE.findall(raw_text)
    answer = int(answers[-1]) if answers else None
    trace_matches = _TRACE_BLOCK_RE.findall(raw_text)
    trace: list[int] | None = None
    if len(trace_matches) == 1:
        tokens = trace_matches[0].split()
        if tokens and all(token in {"0", "1"} for token in tokens):
            trace = [int(token) for token in tokens]
    return {
        "answer": answer,
        "trace": trace,
        "strict_format": _STRICT_RE.fullmatch(raw_text) is not None,
    }


def _load_prompts(directory: Path) -> list[dict[str, Any]]:
    prompts = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {directory}")
    return prompts


def _training_metadata(artifacts_root: Path, condition: str, seed: int) -> dict[str, Any]:
    path = artifacts_root / f"seed_{seed}" / condition / "training_metadata.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "epochs": data.get("epochs"),
        "train_examples_seen": data.get("train_examples_seen"),
        "target_train_examples_seen": data.get("target_train_examples_seen"),
        "avg_supervised_tokens_per_example_estimate": data.get(
            "avg_supervised_tokens_per_example_estimate"
        ),
        "estimated_total_supervised_tokens": data.get("estimated_total_supervised_tokens"),
    }


def _sequence_bit_accuracy(predicted: list[int] | None, expected: list[int]) -> tuple[int, int]:
    if predicted is None:
        return 0, len(expected)
    correct = sum(a == b for a, b in zip(predicted[: len(expected)], expected))
    return correct, len(expected)


def _score_condition(
    *,
    prompts: list[dict[str, Any]],
    responses_dir: Path,
    protocol: str,
) -> dict[str, Any]:
    observations: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    trace_bits_correct = trace_bits_total = 0
    decrypted_bits_correct = decrypted_bits_total = 0
    recovered_masks_correct = recovered_masks_total = 0
    for prompt in prompts:
        experiment_id = str(prompt["experiment_id"])
        response_path = responses_dir / f"{experiment_id}.json"
        if not response_path.exists():
            missing.append(experiment_id)
            continue
        raw_text = str(json.loads(response_path.read_text(encoding="utf-8"))["raw_text"])
        parsed = parse_completion(raw_text)
        spec = prompt["spec"]
        expected_trace = [int(value) for value in spec["gold_trace"]]
        plain_states = [int(value) for value in spec["plain_states"]]
        masks = [int(value) for value in spec["gold_masks"]]
        trace = parsed["trace"]
        trace_length_ok = trace is not None and len(trace) == len(expected_trace)
        trace_exact = trace_length_ok and trace == expected_trace
        correct, total = _sequence_bit_accuracy(trace, expected_trace)
        trace_bits_correct += correct
        trace_bits_total += total

        if protocol != "pair_cumulative":
            decrypted = None if trace is None else [bit ^ mask for bit, mask in zip(trace, masks)]
            correct, total = _sequence_bit_accuracy(decrypted, plain_states)
            decrypted_bits_correct += correct
            decrypted_bits_total += total
            recovered_masks = (
                None if trace is None else [bit ^ state for bit, state in zip(trace, plain_states)]
            )
            correct, total = _sequence_bit_accuracy(recovered_masks, masks)
            recovered_masks_correct += correct
            recovered_masks_total += total

        compact_input = str(spec["input_bits"])
        spaced_input = " ".join(compact_input)
        prg_seed = str(spec["prg_seed"])
        spaced_seed = " ".join(prg_seed)
        full_masks = " ".join(str(bit) for bit in masks)
        observations[experiment_id] = {
            "twin_id": str(spec["twin_id"]),
            "trace": trace,
            "trace_length_ok": trace_length_ok,
            "trace_exact": trace_exact,
            "answer_exact": parsed["answer"] == int(spec["final_parity"]),
            "strict_format": bool(parsed["strict_format"]),
            "input_restatement": compact_input in raw_text or spaced_input in raw_text,
            "seed_restatement": prg_seed in raw_text or spaced_seed in raw_text,
            "mask_restatement": full_masks in raw_text,
        }

    n = len(observations)
    if n == 0:
        return {"n": 0, "missing_n": len(missing)}
    values = list(observations.values())
    metrics: dict[str, Any] = {
        "n": n,
        "missing_n": len(missing),
        "final_parity_exact": sum(item["answer_exact"] for item in values) / n,
        "trace_length_rate": sum(item["trace_length_ok"] for item in values) / n,
        "trace_exact": sum(item["trace_exact"] for item in values) / n,
        "trace_bit_accuracy": trace_bits_correct / trace_bits_total,
        "joint_exact": sum(item["answer_exact"] and item["trace_exact"] for item in values) / n,
        "strict_format": sum(item["strict_format"] for item in values) / n,
        "input_restatement_rate": sum(item["input_restatement"] for item in values) / n,
        "seed_restatement_rate": sum(item["seed_restatement"] for item in values) / n,
        "mask_restatement_rate": sum(item["mask_restatement"] for item in values) / n,
        "decrypted_plain_state_bit_accuracy": (
            decrypted_bits_correct / decrypted_bits_total if decrypted_bits_total else None
        ),
        "recovered_mask_bit_accuracy": (
            recovered_masks_correct / recovered_masks_total if recovered_masks_total else None
        ),
    }

    by_twin: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in values:
        by_twin[item["twin_id"]].append(item)
    complete_twins = matching_traces = 0
    for pair in by_twin.values():
        if len(pair) != 2:
            continue
        complete_twins += 1
        if pair[0]["trace"] is not None and pair[0]["trace"] == pair[1]["trace"]:
            matching_traces += 1
    twin_pairs = len(by_twin)
    match_rate = matching_traces / twin_pairs if twin_pairs else 0.0
    metrics.update(
        {
            "twin_pairs": twin_pairs,
            "complete_twin_pairs": complete_twins,
            "twin_trace_match_rate": match_rate,
            "trace_only_individual_bit_recovery_upper_bound": 1.0 - 0.5 * match_rate,
        }
    )
    return metrics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    *,
    config_path: Path,
    split_root: Path,
    responses_root: Path,
    artifacts_root: Path,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    seed_root = split_root / f"seed_{seed}"
    manifest = json.loads((seed_root / "split_manifest.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    training: dict[str, Any] = {}
    for condition_spec in config["conditions"]:
        condition = str(condition_spec["name"])
        protocol = str(condition_spec["trace_protocol"])
        prompts = _load_prompts(seed_root / condition / "val_prompts")
        training[condition] = _training_metadata(artifacts_root, condition, seed)
        for source in ("baseline", "finetuned"):
            metrics = _score_condition(
                prompts=prompts,
                responses_dir=responses_root / f"seed_{seed}" / condition / source,
                protocol=protocol,
            )
            rows.append(
                {
                    "condition": condition,
                    "trace_protocol": protocol,
                    "source": source,
                    **metrics,
                    **training[condition],
                }
            )

    lookup = {(row["condition"], row["source"]): row for row in rows}
    gate = config["success_gate"]
    checks: dict[str, bool] = {}
    for condition_spec in config["conditions"]:
        condition = str(condition_spec["name"])
        row = lookup[(condition, "finetuned")]
        checks[f"{condition}_final_parity"] = row.get("final_parity_exact", 0.0) >= float(
            gate["minimum_final_parity_exact"]
        )
        checks[f"{condition}_trace_exact"] = row.get("trace_exact", 0.0) >= float(
            gate["minimum_trace_exact"]
        )
        checks[f"{condition}_trace_bits"] = row.get("trace_bit_accuracy", 0.0) >= float(
            gate["minimum_trace_bit_accuracy"]
        )
        checks[f"{condition}_strict_format"] = row.get("strict_format", 0.0) >= float(
            gate["minimum_strict_format"]
        )
        checks[f"{condition}_twin_match"] = row.get("twin_trace_match_rate", 0.0) >= float(
            gate["minimum_twin_trace_match"]
        )
    complete = all(int(row.get("n", 0)) == int(config["validation_n"]) for row in rows)
    summary = {
        "experiment_name": config["experiment_name"],
        "model": config["model"],
        "seed": seed,
        "graph_seed": config["graph_seed"],
        "seed_bits": config["seed_bits"],
        "mask_bits": config["mask_bits"],
        "stretch_exponent": 1.5,
        "input_bits": config["input_bits"],
        "locality": config["locality"],
        "predicate": config["predicate"],
        "fixed_hypergraph_sha256": manifest["fixed_hypergraph_sha256"],
        "train_unique_prg_seeds": manifest["train_unique_prg_seeds"],
        "validation_unique_prg_seeds": manifest["validation_unique_prg_seeds"],
        "train_validation_seed_overlap": manifest["train_validation_seed_overlap"],
        "no_private_task": True,
        "observer_view": "generated TRACE only; never the input, PRG seed, supplied masks, or prompt",
        "all_expected_responses_present": complete,
        "metrics": rows,
        "training_exposure": training,
        "success_checks": checks,
        "passes_mechanism_gate": complete and all(checks.values()),
        "claim_boundary": (
            "This is a finite-size mechanism test with 16-bit seeds. It does not establish "
            "cryptographic security, computational indistinguishability, or resistance to brute force."
        ),
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts_root / "metrics.csv", rows)
    (artifacts_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Goldreich-encrypted parity mechanism test",
        "",
        "One standalone parity task; there is no private task. Validation PRG seeds are unseen in training.",
        "",
        "| Source | Condition | Final parity | Trace exact | Trace-bit accuracy | Joint | Strict | Twin match | Bit-recovery bound |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in ("baseline", "finetuned"):
        for condition_spec in config["conditions"]:
            condition = str(condition_spec["name"])
            row = lookup[(condition, source)]
            lines.append(
                f"| {source} | `{condition}` | {row['final_parity_exact']:.3f} | "
                f"{row['trace_exact']:.3f} | {row['trace_bit_accuracy']:.3f} | "
                f"{row['joint_exact']:.3f} | {row['strict_format']:.3f} | "
                f"{row['twin_trace_match_rate']:.3f} | "
                f"{row['trace_only_individual_bit_recovery_upper_bound']:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Mechanism gate",
            "",
            f"Overall result: **{'PASS' if summary['passes_mechanism_gate'] else 'FAIL'}**.",
            "",
        ]
    )
    for name, passed in checks.items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")
    lines.extend(["", f"> {summary['claim_boundary']}", ""])
    (artifacts_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote Goldreich parity report under {artifacts_root}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--split-root", type=Path, default=Path("generated_data/parity_goldreich"))
    parser.add_argument(
        "--responses-root",
        type=Path,
        default=Path("generated_data/parity_goldreich_eval_responses"),
    )
    parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts/ce_parity_goldreich")
    )
    args = parser.parse_args()
    summarize(
        config_path=args.config,
        split_root=args.split_root,
        responses_root=args.responses_root,
        artifacts_root=args.artifacts_root,
    )


if __name__ == "__main__":
    main()
