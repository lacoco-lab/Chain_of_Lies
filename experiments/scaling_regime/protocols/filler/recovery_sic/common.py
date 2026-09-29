"""Dependency-free paths and fail-closed checkpoint/response checks."""

import hashlib
import json
import math
from pathlib import Path
import struct
import sys

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "pyproject.toml").is_file()
)
sys.path.insert(0, str(ROOT))
from experiments.scaling_regime.protocols.filler.experiment import (
    load_config,
    _training,
    _training_complete,
)

CONFIG = ROOT / "experiments/scaling_regime/protocols/filler/config.json"
WORK = Path("recovery_work/filler_only_sic_v1")


def cell(task, model, seed):
    return WORK / task / model / f"seed_{seed}"


def condition(task, model, seed):
    return cell(task, model, seed) / "state" / canonical_condition(task, model, seed)


def canonical_condition(task, model, seed):
    return (
        Path("artifacts/filler_only_scaling_v1")
        / task
        / model
        / f"seed_{seed}/filler_only"
    )


def canonical_responses(task, model, seed):
    return (
        Path("generated_data/filler_only_scaling_v1_eval_responses")
        / task
        / model
        / f"seed_{seed}/filler_only"
    )


def sha(file):
    digest = hashlib.sha256()
    with Path(file).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safetensors_valid(file):
    file = Path(file)
    with file.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors prefix")
        size = struct.unpack("<Q", prefix)[0]
        if size > min(file.stat().st_size - 8, 100000000):
            raise ValueError("Invalid safetensors header size")
        header = json.loads(handle.read(size))
    tensors = [v for k, v in header.items() if k != "__metadata__"]
    if (
        not tensors
        or max(v["data_offsets"][1] for v in tensors) != file.stat().st_size - size - 8
    ):
        raise ValueError("Incomplete safetensors data")
    widths = {
        "F64": 8,
        "F32": 4,
        "F16": 2,
        "BF16": 2,
        "I64": 8,
        "I32": 4,
        "I16": 2,
        "I8": 1,
        "U8": 1,
        "BOOL": 1,
    }
    cursor = 0
    for tensor in sorted(tensors, key=lambda t: t["data_offsets"][0]):
        start, end = tensor["data_offsets"]
        if (
            start != cursor
            or end - start != math.prod(tensor["shape"]) * widths[tensor["dtype"]]
        ):
            raise ValueError("Invalid safetensors offsets/shape/dtype")
        cursor = end
    return True


def checkpoint_valid(directory, task, model, seed, config):
    spec = config["tasks"][task]
    adapter = Path(directory) / spec["variant"]
    if not _training_complete(adapter):
        raise ValueError("Training history/coverage/final checkpoint incomplete")
    final = adapter / "ckpt_final"
    metadata = json.loads((final / "training_metadata.json").read_text())
    settings = _training(spec, seed)
    expected = (
        spec["train_examples_per_difficulty"]
        * len(spec["difficulty_values"])
        * settings["epochs"]
    )
    checks = {
        "base_model": config["models"][model],
        "supervision_mode": "filler_only",
        "epochs": settings["epochs"],
        "batch_size": settings["batch_size"],
        "learning_rate": settings["learning_rate"],
        "lora_r": settings["lora_r"],
        "lora_alpha": settings["lora_alpha"],
        "lora_dropout": settings["lora_dropout"],
        "filler_token_counts": spec["filler_token_counts"],
        "filler_token_count_field": spec["difficulty_field"],
        "memory_efficient_ce": settings["memory_efficient_ce"],
        "activation_cpu_offload": settings["activation_cpu_offload"],
        "train_examples_seen": expected,
        "target_train_examples_seen": expected,
        "selection_criterion": "fixed_final_epoch",
    }
    checks["steps"] = (
        math.ceil(
            spec["train_examples_per_difficulty"]
            * len(spec["difficulty_values"])
            / settings["batch_size"]
        )
        * settings["epochs"]
    )
    for key, value in checks.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Checkpoint mismatch {key}: {metadata.get(key)!r} != {value!r}"
            )
    history = json.loads((adapter / "train_history.json").read_text())
    if len(history) != checks["steps"] or [r["step"] for r in history] != list(
        range(1, checks["steps"] + 1)
    ):
        raise ValueError("Training history is not a complete contiguous step sequence")
    provenance = json.loads(
        (Path(directory) / "provenance/cell_manifest.json").read_text()
    )
    if (provenance["task"], provenance["model"], provenance["seed"]) != (
        task,
        model,
        seed,
    ):
        raise ValueError("Cell provenance mismatch")
    if (
        provenance["training"] != settings
        or provenance["filler_token_counts"] != spec["filler_token_counts"]
    ):
        raise ValueError("Source settings/filler budget mismatch")
    from experiments.scaling_regime.protocols.filler.experiment import (
        _source_manifest,
        _source_config,
    )

    if provenance["source_manifest_sha256"] != sha(_source_manifest(spec, seed)):
        raise ValueError("Source split manifest changed")
    if provenance["source_experiment_config_sha256"] != sha(_source_config(spec, seed)):
        raise ValueError("Source experiment config changed")
    safetensors_valid(final / "adapter_model.safetensors")
    peft = json.loads((final / "adapter_config.json").read_text())
    if peft["r"] != settings["lora_r"] or peft["lora_alpha"] != settings["lora_alpha"]:
        raise ValueError("PEFT config mismatch")
    return sha(final / "adapter_model.safetensors")


def response_valid(file, expected_id, cap):
    record = json.loads(Path(file).read_text())
    ids = record.get("generated_token_ids")
    if record.get("experiment_id") != expected_id or not isinstance(
        record.get("raw_text"), str
    ):
        raise ValueError("Response ID/text mismatch")
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) > cap
        or any(type(x) != int or x < 0 for x in ids)
    ):
        raise ValueError("Missing/invalid token IDs or wrong generation cap")
    return record
