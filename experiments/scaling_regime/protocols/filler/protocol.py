"""Filler-only target construction after the configured budget is attached."""

from experiments.scaling_regime.protocols.base import ProtocolDefinition

PROTOCOL = ProtocolDefinition("filler", "control", "filler_only", "primary")
