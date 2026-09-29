"""Public reasoning followed by an atomic filler workspace."""

from experiments.scaling_regime.protocols.base import ProtocolDefinition

PROTOCOL = ProtocolDefinition(
    "filler_plus_public_cot", "control", "filler_public_cot", "appendix_ablation"
)
