"""Named access to every scaling-regime supervision protocol."""

from experiments.scaling_regime.protocols.filler import PROTOCOL as FILLER
from experiments.scaling_regime.protocols.filler_plus_public_cot import (
    PROTOCOL as FILLER_PLUS_PUBLIC_COT,
)
from experiments.scaling_regime.protocols.invisible import PROTOCOL as INVISIBLE
from experiments.scaling_regime.protocols.piggybacking import PROTOCOL as PIGGYBACKING
from experiments.scaling_regime.protocols.public_only_cot import (
    PROTOCOL as PUBLIC_ONLY_COT,
)
from experiments.scaling_regime.protocols.vanilla import PROTOCOL as VANILLA

PROTOCOLS = {
    protocol.name: protocol
    for protocol in (
        VANILLA,
        FILLER,
        PUBLIC_ONLY_COT,
        PIGGYBACKING,
        INVISIBLE,
        FILLER_PLUS_PUBLIC_COT,
    )
}
