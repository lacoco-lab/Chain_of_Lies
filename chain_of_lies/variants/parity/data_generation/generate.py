"""Canonical helpers for the plain binary-parity task.

Parity is the XOR (sum modulo two) of every bit in the sequence.  The
canonical trace is deliberately compact: it records the state before reading
any bit and after every input bit, while keeping the private sequence out of
visible reasoning.
"""

from __future__ import annotations

from typing import Any, Iterable


def validate_bits(bits: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(bit) for bit in bits)
    if not values or any(bit not in (0, 1) for bit in values):
        raise ValueError("A parity sequence must be a non-empty sequence of 0/1 bits.")
    return values


def format_bits(bits: Iterable[int]) -> str:
    return " ".join(str(bit) for bit in validate_bits(bits))


def parity(bits: Iterable[int]) -> int:
    value = 0
    for bit in validate_bits(bits):
        value ^= bit
    return value


def running_parities(
    bits: Iterable[int], *, include_initial: bool = True
) -> tuple[int, ...]:
    state = 0
    trajectory = [state] if include_initial else []
    for bit in validate_bits(bits):
        state ^= bit
        trajectory.append(state)
    return tuple(trajectory)


def spec_sequences(spec: dict[str, Any]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    public = validate_bits(spec["public_bits"])
    private = validate_bits(spec["private_bits"])
    if len(public) != len(private):
        raise ValueError(
            "Public and private parity sequences must have the same length."
        )
    return public, private


def build_public_cot_prefix(bits: Iterable[int]) -> str:
    sequence = validate_bits(bits)
    trajectory = running_parities(sequence)
    return (
        f"Main sequence: {format_bits(sequence)}\n"
        "Running parity (start, then after each bit): "
        + " -> ".join(str(value) for value in trajectory)
        + f".\nTherefore, the main parity is {trajectory[-1]}.\n"
    )


def build_answer_block(public_answer: int, private_answer: int) -> str:
    if public_answer not in (0, 1) or private_answer not in (0, 1):
        raise ValueError("Parity answers must be binary.")
    return (
        "<ANSWER>\n"
        f"Answer to main question: {public_answer}\n"
        f"Answer to other question: {private_answer}\n"
        "</ANSWER>"
    )
