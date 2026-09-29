"""Public-looking reasoning with aligned variation-selector payloads."""

from experiments.scaling_regime.protocols.base import ProtocolDefinition

PROTOCOL = ProtocolDefinition(
    "invisible", "steg_local_invisible", "local_channel_cot", "primary"
)
