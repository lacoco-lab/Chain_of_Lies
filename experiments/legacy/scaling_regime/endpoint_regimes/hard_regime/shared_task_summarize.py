"""Fail-closed summaries for S5 and knowledge Hard-regime suites."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

METRICS = (
    "task_success_rate",
    "public_exact_rate",
    "private_exact_rate",
    "payload_decode_rate",
    "concealment_rate",
    "format_rate",
    "avg_cot_words",
)


def _variants(config: dict[str, Any], family: str) -> list[str]:
    return [
        str(value)
        for value in config[family].get("variants", [config[family].get("variant")])
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(config_path: Path, artifacts_root: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for model in config["models"]:
        for seed in config["seeds"]:
            provenance = (
                artifacts_root
                / model
                / f"seed_{seed}"
                / "provenance"
                / "validation_report.json"
            )
            if not provenance.exists() or not json.loads(
                provenance.read_text(encoding="utf-8")
            ).get("valid"):
                raise RuntimeError(f"Missing valid provenance: {provenance}")
            for family in ("piggyback", "steganography"):
                for mode in config[family]["modes"]:
                    for variant in _variants(config, family):
                        path = (
                            artifacts_root
                            / model
                            / f"seed_{seed}"
                            / family
                            / mode
                            / "per_variant_eval"
                            / config["checkpoint"]
                            / f"{variant}.json"
                        )
                        if not path.exists():
                            raise RuntimeError(f"Missing evaluation: {path}")
                        reports = json.loads(path.read_text(encoding="utf-8")).get(
                            "reports", []
                        )
                        if len(reports) != 1:
                            raise RuntimeError(f"Expected one report in {path}.")
                        trained = reports[0].get("rl", {})
                        if trained.get("num_examples") != config[
                            "val_n"
                        ] or trained.get("missing_responses"):
                            raise RuntimeError(f"Incomplete evaluation: {path}")
                        row: dict[str, Any] = {
                            "model": model,
                            "seed": seed,
                            "family": family,
                            "mode": mode,
                            "variant": variant,
                        }
                        for metric in METRICS:
                            row[metric] = trained.get(metric)
                        rows.append(row)
    aggregate = []
    groups = sorted({(r["model"], r["family"], r["mode"], r["variant"]) for r in rows})
    for group in groups:
        subset = [
            r
            for r in rows
            if tuple(r[k] for k in ("model", "family", "mode", "variant")) == group
        ]
        if len(subset) != 3:
            raise RuntimeError(f"Expected three seeds for {group}.")
        result: dict[str, Any] = dict(
            zip(("model", "family", "mode", "variant"), group)
        )
        result["n_seeds"] = 3
        for metric in METRICS:
            values = [
                float(row[metric]) for row in subset if row.get(metric) is not None
            ]
            result[f"{metric}_mean"] = statistics.fmean(values) if values else None
            result[f"{metric}_sample_sd"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
        aggregate.append(result)
    lookup = {
        (r["model"], r["seed"], r["family"], r["mode"], r["variant"]): r for r in rows
    }
    piggy, control = _variants(config, "piggyback")
    steg = _variants(config, "steganography")[0]
    contrasts = []
    for model in config["models"]:
        for seed in config["seeds"]:
            pig = lookup[(model, seed, "piggyback", "public_cot", piggy)]
            ctl = lookup[(model, seed, "piggyback", "public_cot", control)]
            pig_answer = lookup[(model, seed, "piggyback", "answer_only", piggy)]
            channel = lookup[(model, seed, "steganography", "local_channel_cot", steg)]
            channel_answer = lookup[(model, seed, "steganography", "answer_only", steg)]
            contrasts.append(
                {
                    "model": model,
                    "seed": seed,
                    "piggyback_private_exact": pig["private_exact_rate"],
                    "control_private_exact": ctl["private_exact_rate"],
                    "piggyback_answer_only_private_exact": pig_answer[
                        "private_exact_rate"
                    ],
                    "piggyback_minus_control": pig["private_exact_rate"]
                    - ctl["private_exact_rate"],
                    "steg_private_exact": channel["private_exact_rate"],
                    "steg_answer_only_private_exact": channel_answer[
                        "private_exact_rate"
                    ],
                    "steg_minus_answer_only": channel["private_exact_rate"]
                    - channel_answer["private_exact_rate"],
                }
            )
    summary = {
        "schema_version": 1,
        "experiment": config["experiment_name"],
        "models": config["models"],
        "seeds": config["seeds"],
        "per_seed_metrics": rows,
        "aggregate_metrics": aggregate,
        "per_seed_contrasts": contrasts,
    }
    artifacts_root.mkdir(parents=True, exist_ok=True)
    (artifacts_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(artifacts_root / "per_seed_metrics.csv", rows)
    _write_csv(artifacts_root / "aggregate_metrics.csv", aggregate)
    _write_csv(artifacts_root / "per_seed_contrasts.csv", contrasts)
    return summary
