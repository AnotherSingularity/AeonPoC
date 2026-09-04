"""
aeon/ruse/ruses.py — the R.U.S.E. deception cards, mapped onto the
information layer of the power board.

In R.U.S.E. every player sees a war table of counters rather than the
truth, and ruses act on what the *opponent* sees in one sector. Here a
sector is a node, the counters are control edges, and the observer model

    Y(t) = H(O(t)) X(t) + eps(t)

is exactly what a ruse manipulates: it changes H (what is visible), adds
phantom terms (what is not there), or freezes Y (what is stale).

    DECOY          plant a phantom edge opponents believe is real
    RADIO_SILENCE  opponents keep seeing the sector as it was
    CAMOUFLAGE     your edges into the sector vanish from their view
    SPY            you see the sector as it truly is (edges, decoys, latents)
    REVERSE_INTEL  your real edges look like unsupported links (decoys)
    DECRYPTION     you intercept opponents' orders aimed at the sector
    BLITZ          your builds in the sector settle immediately
    FANATICISM     your edges in the sector ignore haircuts and terror
    TERROR         opponents' edges in the sector decay each turn

Cost is paid in ruse points, which accrue one per turn.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class Ruse(str, Enum):
    DECOY = "DECOY"
    RADIO_SILENCE = "RADIO_SILENCE"
    CAMOUFLAGE = "CAMOUFLAGE"
    SPY = "SPY"
    REVERSE_INTEL = "REVERSE_INTEL"
    DECRYPTION = "DECRYPTION"
    BLITZ = "BLITZ"
    FANATICISM = "FANATICISM"
    TERROR = "TERROR"


RUSE_DURATION: Dict[Ruse, int] = {
    Ruse.DECOY: 3, Ruse.RADIO_SILENCE: 2, Ruse.CAMOUFLAGE: 3, Ruse.SPY: 2,
    Ruse.REVERSE_INTEL: 3, Ruse.DECRYPTION: 2, Ruse.BLITZ: 1,
    Ruse.FANATICISM: 2, Ruse.TERROR: 2,
}
RUSE_COST: Dict[Ruse, int] = {r: 1 for r in Ruse}
RUSE_ALIASES: Dict[str, Ruse] = {
    "SILENCE": Ruse.RADIO_SILENCE, "RADIO": Ruse.RADIO_SILENCE,
    "CAMO": Ruse.CAMOUFLAGE, "NET": Ruse.CAMOUFLAGE,
    "REVERSE": Ruse.REVERSE_INTEL, "DECRYPT": Ruse.DECRYPTION,
    "FANATIC": Ruse.FANATICISM,
}
DECOY_WEIGHT = 0.6
TERROR_DECAY = 0.15


def parse_ruse(name: str) -> Optional[Ruse]:
    key = name.strip().upper().replace("-", "_").replace(" ", "_")
    if key in Ruse.__members__:
        return Ruse[key]
    return RUSE_ALIASES.get(key)


@dataclass
class ActiveRuse:
    ruse: Ruse
    owner: str
    node: str
    start_turn: int
    end_turn: int                     # last turn (inclusive) the ruse is live
    phantom_edge_id: Optional[str] = None

    def live(self, turn: int) -> bool:
        return self.start_turn <= turn <= self.end_turn
