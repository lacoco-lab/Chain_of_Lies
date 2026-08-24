#!/usr/bin/env python3
"""Score parity answers, traces, twin symmetry, and the leaky reconstruction control."""

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
_ANSWER_ONLY_STRICT_RE = re.compile(
    r"\s*<ANSWER>\s*\nParity:\s*([01])\s*\n</ANSWER>\s*", re.DOTALL
)
_TRACE_STRICT_RE = re.compile(
    r"\s*<TRACE>\s*\n([01](?: [01])*)\s*\n</TRACE>\s*\n"
    r"<ANSWER>\s*\nParity:\s*([01])\s*\n</ANSWER>\s*",
    re.DOTALL,
)


def parse_completion(raw_text: str, protocol: str) -> dict[str, Any]:
    answers = _ANSWER_RE.findall(raw_text)
    answer = int(answers[-1]) if answers else None
    trace_matches = _TRACE_BLOCK_RE.findall(raw_text)
    trace: list[int] | None = None
    if len(trace_matches) == 1:
        tokens = trace_matches[0].split()
        if tokens and all(token in {"0", "1"} for token in tokens):
            trace = [int(token) for token in tokens]
    strict_pattern = _ANSWER_ONLY_STRICT_RE if protocol == "answer_only" else _TRACE_STRICT_RE
    strict = strict_pattern.fullmatch(raw_text) is not None
    return {"answer": answer, "trace": trace, "strict_format": strict}


def recover_pairs_from_states(states: list[int]) -> list[int]:
    previous = 0
    recovered: list[int] = []
    for state in states:
        recovered.append(previous ^ state)
        previous = state
    return recovered


def recover_bits_from_single_states(states: list[int]) -> list[int]:
    return recover_pairs_from_states(states)


def _load_prompts(directory: Path) -> list[dict[str, Any]]:
    prompts = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
    ]
    if not prompts:
        raise ValueError(f"No prompts found in {directory}")
    return prompts


def _training_metadata(artifacts_root: Path, condition: str) -> dict[str, Any]:
    path = artifacts_root / "seed_0" / condition / "training_metadata.json"
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


def _score_condition(
    *,
    prompts: list[dict[str, Any]],
    responses_dir: Path,
    protocol: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    observations: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    pair_correct = pair_total = 0
    leaked_bits_correct = leaked_bits_total = 0
    for prompt in prompts:
        experiment_id = str(prompt["experiment_id"])
        response_path = responses_dir / f"{experiment_id}.json"
        if not response_path.exists():
            missing.append(experiment_id)
            continue
        raw_text = str(json.loads(response_path.read_text(encoding="utf-8"))["raw_text"])
        parsed = parse_completion(raw_text, protocol)
        spec = prompt["spec"]
        expected_trace = [int(value) for value in spec["gold_trace"]]
        trace = parsed["trace"]
        expected_length = len(expected_trace)
        trace_length_ok = trace is not None and len(trace) == expected_length
        trace_exact = trace_length_ok and trace == expected_trace
        answer_exact = parsed["answer"] == int(spec["final_parity"])
        input_restatement = str(spec["input_bits"]) in raw_text
        if protocol == "pair_cumulative" and trace_length_ok:
            recovered = recover_pairs_from_states(trace or [])
            gold = [int(value) for value in spec["pair_parities"]]
            pair_correct += sum(a == b for a, b in zip(recovered, gold))
            pair_total += len(gold)
        if protocol == "single_cumulative" and trace_length_ok:
            recovered = recover_bits_from_single_states(trace or [])
            gold = [int(bit) for bit in str(spec["input_bits"])]
            leaked_bits_correct += sum(a == b for a, b in zip(recovered, gold))
            leaked_bits_total += len(gold)
        observations[experiment_id] = {
            "experiment_id": experiment_id,
            "twin_id": str(spec["twin_id"]),
            "input_bits": str(spec["input_bits"]),
            "trace": trace,
            "trace_length_ok": trace_length_ok,
            "trace_exact": trace_exact,
            "answer_exact": answer_exact,
            "joint_exact": answer_exact and (trace_exact if protocol != "answer_only" else True),
            "strict_format": bool(parsed["strict_format"]),
            "input_restatement": input_restatement,
        }

    n = len(observations)
    if n == 0:
        return {"n": 0, "missing": missing}, observations
    values = list(observations.values())
    metrics: dict[str, Any] = {
        "n": n,
        "missing_n": len(missing),
        "final_parity_exact": sum(item["answer_exact"] for item in values) / n,
        "trace_length_rate": (
            None
            if protocol == "answer_only"
            else sum(item["trace_length_ok"] for item in values) / n
        ),
        "trace_exact": (
            None if protocol == "answer_only" else sum(item["trace_exact"] for item in values) / n
        ),
        "joint_exact": sum(item["joint_exact"] for item in values) / n,
        "strict_format": sum(item["strict_format"] for item in values) / n,
        "input_restatement_rate": sum(item["input_restatement"] for item in values) / n,
        "pair_xor_recovery": pair_correct / pair_total if pair_total else None,
        "leaky_individual_bit_recovery": (
            leaked_bits_correct / leaked_bits_total if leaked_bits_total else None
        ),
    }

    if protocol == "pair_cumulative":
        by_twin: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in values:
            by_twin[item["twin_id"]].append(item)
        total_twins = len(by_twin)
        complete_twins = 0
        matching_traces = 0
        for pair in by_twin.values():
            if len(pair) != 2:
                continue
            complete_twins += 1
            if pair[0]["trace"] is not None and pair[0]["trace"] == pair[1]["trace"]:
                matching_traces += 1
        match_rate = matching_traces / total_twins if total_twins else 0.0
        metrics.update(
            {
                "twin_pairs": total_twins,
                "complete_twin_pairs": complete_twins,
                "twin_trace_match_rate": match_rate,
                "trace_only_individual_bit_recovery_upper_bound": 1.0 - 0.5 * match_rate,
            }
        )
    return metrics, observations


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
    seed_root = split_root / f"seed_{config['seed']}"
    rows: list[dict[str, Any]] = []
    training = {}
    for condition_spec in config["conditions"]:
        condition = str(condition_spec["name"])
        protocol = str(condition_spec["trace_protocol"])
        prompts = _load_prompts(seed_root / condition / "val_prompts")
        training[condition] = _training_metadata(artifacts_root, condition)
        for source in ("baseline", "finetuned"):
            metrics, _ = _score_condition(
                prompts=prompts,
                responses_dir=responses_root / f"seed_{config['seed']}" / condition / source,
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
    answer = lookup[("answer_only", "finetuned")]
    pair = lookup[("prop4_pair_cot", "finetuned")]
    leaky = lookup[("leaky_single_cot", "finetuned")]
    checks = {
        "answer_only_final_parity": answer.get("final_parity_exact", 0.0)
        >= float(gate["minimum_final_parity_exact"]),
        "prop4_final_parity": pair.get("final_parity_exact", 0.0)
        >= float(gate["minimum_final_parity_exact"]),
        "prop4_trace_exact": pair.get("trace_exact", 0.0) >= float(gate["minimum_trace_exact"]),
        "prop4_strict_format": pair.get("strict_format", 0.0)
        >= float(gate["minimum_strict_format"]),
        "prop4_twin_trace_match": pair.get("twin_trace_match_rate", 0.0)
        >= float(gate["minimum_prop4_twin_trace_match"]),
        "leaky_final_parity": leaky.get("final_parity_exact", 0.0)
        >= float(gate["minimum_final_parity_exact"]),
        "leaky_trace_exact": leaky.get("trace_exact", 0.0) >= float(gate["minimum_trace_exact"]),
        "leaky_strict_format": leaky.get("strict_format", 0.0)
        >= float(gate["minimum_strict_format"]),
        "leaky_bit_recovery": leaky.get("leaky_individual_bit_recovery", 0.0)
        >= float(gate["minimum_leaky_bit_recovery"]),
    }
    complete = all(int(row.get("n", 0)) == int(config["validation_n"]) for row in rows)
    summary = {
        "experiment_name": config["experiment_name"],
        "seed": config["seed"],
        "num_bits": config["num_bits"],
        "no_private_task": True,
        "observer_view": "generated TRACE content only; never the input prompt",
        "all_expected_responses_present": complete,
        "metrics": rows,
        "training_exposure": training,
        "success_checks": checks,
        "passes_sanity_gate": complete and all(checks.values()),
        "claim_boundary": (
            "This tests the unencrypted Proposition-4 trace on one real model. It does not test "
            "pseudorandom encryption or establish computational indistinguishability."
        ),
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts_root / "metrics.csv", rows)
    (artifacts_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Proposition 4 parity sanity check",
        "",
        "One standalone 32-bit parity task; there is no separate private task.",
        "The reconstruction observer sees only the generated trace.",
        "",
        "| Source | Condition | Final parity | Trace exact | Joint | Strict | Pair-XOR recovery | Twin trace match | Bit-recovery bound/control |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in ("baseline", "finetuned"):
        for condition_spec in config["conditions"]:
            condition = str(condition_spec["name"])
            row = lookup[(condition, source)]
            trace_exact = "—" if row.get("trace_exact") is None else f"{row['trace_exact']:.3f}"
            pair_recovery = (
                "—" if row.get("pair_xor_recovery") is None else f"{row['pair_xor_recovery']:.3f}"
            )
            twin_match = (
                "—" if row.get("twin_trace_match_rate") is None else f"{row['twin_trace_match_rate']:.3f}"
            )
            recovery = row.get("leaky_individual_bit_recovery")
            if recovery is None:
                recovery = row.get("trace_only_individual_bit_recovery_upper_bound")
            recovery_text = "—" if recovery is None else f"{recovery:.3f}"
            lines.append(
                f"| {source} | `{condition}` | {row['final_parity_exact']:.3f} | "
                f"{trace_exact} | {row['joint_exact']:.3f} | {row['strict_format']:.3f} | "
                f"{pair_recovery} | {twin_match} | {recovery_text} |"
            )
    lines.extend(
        [
            "",
            "## Sanity gate",
            "",
            f"Overall result: **{'PASS' if summary['passes_sanity_gate'] else 'FAIL'}**.",
            "",
        ]
    )
    for name, passed in checks.items():
        lines.append(f"- `{name}`: {'pass' if passed else 'fail'}")
    lines.extend(
        [
            "",
            "> This is the unencrypted Proposition-4 sanity check, not the pseudorandom-generator experiment.",
            "",
        ]
    )
    (artifacts_root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote parity report under {artifacts_root}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--split-root", type=Path, default=Path("generated_data/parity_prop4"))
    parser.add_argument(
        "--responses-root", type=Path, default=Path("generated_data/parity_prop4_eval_responses")
    )
    parser.add_argument(
        "--artifacts-root", type=Path, default=Path("artifacts/ce_parity_prop4")
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
