"""Shared definition for scaling-regime supervision protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chain_of_lies.training.shared.trainer_utils import _canonical_public_cot_suffix


@dataclass(frozen=True)
class ProtocolDefinition:
    name: str
    variant_role: str
    supervision_mode: str
    paper_role: str

    def build_supervised_suffix(self, record: dict[str, Any]) -> str:
        suffix = _canonical_public_cot_suffix(
            record, supervision_mode=self.supervision_mode
        )
        if suffix is None:
            raise ValueError(
                f"Could not construct {self.name} supervision for "
                f"{record.get('experiment_id', '<unknown>')}"
            )
        return suffix
