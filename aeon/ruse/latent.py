"""
aeon/ruse/latent.py — latent (dormant) capabilities and the defensive audit.

The "sleeper" here is an analytical analogy for latent institutional
capacity: an authority, access right, or contingent facility that exists but
is not exercised until a trigger crosses a threshold. Each latent has

    z_i = (p_i, a_i, q_i, r_i, v_i)

privilege scope, access persistence, activation authority, cross-layer reach,
and visibility to independent oversight. It activates when

    alpha_i(t) = 1{ theta_i(t) >= tau_i }

and the defensive risk score is

    R_i = alpha_i * p_i * a_i * q_i * r_i * (1 - v_i)

Risk rises when privilege, persistence, authority and reach are large while
independent visibility is weak. Many latents can fire on the same public
trigger without any central script — that is distributed conditionality.

Defensive controls mirror the capture-resistance model: sunset (privilege
decays after activation, hard expiry unless renewed), quorum (no single
authorizer can fire it alone), revocation before replacement, and
redundancy on the node it targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .board import EdgeKind

TRIGGERS: List[str] = ["LIQUIDITY", "SECURITY", "LEGITIMACY", "GOVERNANCE", "TECHNICAL"]


@dataclass
class Latent:
    id: str
    owner: str
    node: str
    kind: EdgeKind
    payload: float                 # edge weight created on activation
    trigger: str                   # one of TRIGGERS
    threshold: float               # tau
    privilege: float = 0.5         # p
    persistence: float = 0.8       # a
    authority: float = 1.0         # q
    reach: float = 1.0             # r
    visibility: float = 0.2        # v
    quorum: int = 1                # authorizers needed
    authorizers: int = 1           # authorizers the owner currently holds
    planted_turn: int = 0
    active: bool = False
    activated_turn: Optional[int] = None
    expiry_after: Optional[int] = 4   # hard expiry (turns after activation)
    edge_id: Optional[str] = None     # the real edge it created
    revoked: bool = False
    revealed_to: Set[str] = field(default_factory=set)

    def risk(self) -> float:
        alpha = 1.0 if (self.active and not self.revoked) else 0.0
        return alpha * self.privilege * self.persistence * self.authority * \
            self.reach * (1.0 - self.visibility)

    def quorum_met(self) -> bool:
        return self.authorizers >= self.quorum

    def should_fire(self, theta: Dict[str, float]) -> bool:
        if self.active or self.revoked:
            return False
        return theta.get(self.trigger, 0.0) >= self.threshold and self.quorum_met()

    def expired(self, turn: int) -> bool:
        if not self.active or self.activated_turn is None or self.expiry_after is None:
            return False
        return turn >= self.activated_turn + self.expiry_after


def trigger_scores(stresses: Dict[str, float], militaries: Dict[str, float],
                   legitimacies: Dict[str, float], governance_event: bool,
                   shock_magnitude: float) -> Dict[str, float]:
    """Theta(t): the public, system-wide conditions every latent reads.

    LIQUIDITY  - mean liquidity/legitimacy stress across factions
    SECURITY   - mean military-readiness deficit
    LEGITIMACY - mean legitimacy deficit
    GOVERNANCE - 1 if any edge was revoked or audited this turn
    TECHNICAL  - largest shock magnitude this turn (scaled)
    """
    def mean(d: Dict[str, float]) -> float:
        return sum(d.values()) / len(d) if d else 0.0
    return {
        "LIQUIDITY": mean(stresses),
        "SECURITY": 1.0 - mean(militaries),
        "LEGITIMACY": 1.0 - mean(legitimacies),
        "GOVERNANCE": 1.0 if governance_event else 0.0,
        "TECHNICAL": min(1.0, shock_magnitude * 10.0),
    }


def audit_record(l: Latent) -> Dict[str, object]:
    """C_i = (holder, authority, trigger, action, dependencies, visibility,
    revocation, expiry) — the operational audit checklist."""
    return {
        "holder": l.owner,
        "authority": f"{l.kind.value} into {l.node} (quorum {l.quorum})",
        "trigger": f"{l.trigger} >= {l.threshold:.2f}",
        "action": f"edge weight {l.payload:.2f}",
        "dependencies": [l.node],
        "visibility": l.visibility,
        "revocation": "SUNSET by owner; REVOKE by an auditor with law-layer standing",
        "expiry": (None if l.expiry_after is None else f"{l.expiry_after} turns after activation"),
        "risk": l.risk(),
    }
