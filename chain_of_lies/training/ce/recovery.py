"""Opt-in atomic optimizer/RNG checkpoints for trusted training recovery."""

import os
import random
from pathlib import Path
import torch


def save_state(path, weights, optimizer, signature, step, seen, history):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "version": 1,
        "signature": signature,
        "step": step,
        "seen": seen,
        "history": history,
        "weights": {k: v.detach().cpu() for k, v in weights.items()},
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_state(path, signature):
    # Trusted self-written checkpoints only: optimizer/RNG state uses pickle.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != 1 or state.get("signature") != signature:
        raise ValueError("Recovery state/config/data mismatch; refusing continuation")
    if len(state["history"]) != state["step"]:
        raise ValueError("Recovery history/step mismatch")
    return state


def restore_rng(state):
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Recovery CUDA device count mismatch")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
