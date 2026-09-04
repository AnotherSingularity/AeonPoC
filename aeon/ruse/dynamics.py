"""
aeon/ruse/dynamics.py — faction state dynamics, crisis amplification, and the
stability (purification) test.

Each faction carries a state vector

    x = [E, I, F, L, C, M, P, R]

energy capacity, information integrity, finance, logistics, communications,
military readiness, political legitimacy, industrial resilience, each in
[0, 1]. The generic strategic-pressure model is

    dx/dt = A x + B u + G w - H d

where u is external pressure (and own investment), w is a random shock, and
d is defensive adaptation. The dangerous property is cross-domain coupling:
loss of communications degrades logistics, degraded logistics reduces
output, reduced output raises fiscal stress, fiscal stress erodes legitimacy.

Crisis is an amplifier. Stress theta scales the off-diagonal coupling, and
the local Jacobian J = A_eff decides the regime:

    max Re spec(J) > 0   ->  small shocks grow   (lambda_max > 0)
    max Re spec(J) < 0   ->  the equilibrium is stable

Purification is stabilization: restorative feedback u = -K e, which HARDEN
orders add to the diagonal, moves lambda_max from > 0 to < 0. The second
law still holds: hardening costs budget, disorder is dissipated not erased.

Cascade gain = ||Delta x_final|| / ||disturbance_initial|| measures how far
a bounded push propagates. Attackers want it large, defenders small.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

DOMAINS: List[str] = ["ENERGY", "INFO", "FINANCE", "LOGISTICS",
                      "COMMS", "MILITARY", "LEGITIMACY", "INDUSTRY"]
DOMAIN_INDEX: Dict[str, int] = {d: i for i, d in enumerate(DOMAINS)}
DOMAIN_ALIASES: Dict[str, str] = {
    "E": "ENERGY", "I": "INFO", "F": "FINANCE", "L": "LOGISTICS",
    "C": "COMMS", "M": "MILITARY", "P": "LEGITIMACY", "R": "INDUSTRY",
    "INFORMATION": "INFO", "MONEY": "FINANCE", "LIQUIDITY": "FINANCE",
    "LEGIT": "LEGITIMACY", "INDUSTRIAL": "INDUSTRY",
}


def parse_domain(name: str) -> Optional[str]:
    key = name.strip().upper()
    key = DOMAIN_ALIASES.get(key, key)
    return key if key in DOMAIN_INDEX else None


def base_coupling() -> np.ndarray:
    """A[i, j]: how a deficit in domain j drives a deficit in domain i.

    Diagonal is mean reversion. The off-diagonal entries encode the
    non-operational cascade chain from the source model.
    """
    n = len(DOMAINS)
    A = np.zeros((n, n))
    np.fill_diagonal(A, -0.40)
    E, I, F, L, C, M, P, R = range(n)
    couple = {
        (L, C): 0.25,  # comms -> logistics
        (R, L): 0.25,  # logistics -> industry
        (F, R): 0.20,  # industry -> finance
        (P, F): 0.25,  # finance -> legitimacy
        (P, I): 0.15,  # information integrity -> legitimacy
        (R, E): 0.20,  # energy -> industry
        (C, E): 0.10,  # energy -> comms
        (L, F): 0.10,  # finance -> logistics
        (F, P): 0.10,  # legitimacy -> finance (the loop closes)
        (M, F): 0.10,  # finance -> military readiness
        (M, L): 0.10,  # logistics -> military readiness
    }
    for (i, j), v in couple.items():
        A[i, j] = v
    return A


@dataclass
class FactionDynamics:
    """State, damping, and segmentation for one faction."""
    x: np.ndarray
    x_nominal: np.ndarray
    damping: np.ndarray                   # extra restorative gain K per domain
    segmentation: np.ndarray              # multiplies coupling INTO each domain
    dt: float = 0.5
    crisis_gain: float = 2.0              # theta scales coupling by (1 + gain*theta)
    history: List[float] = field(default_factory=list)   # lambda_max per step

    @classmethod
    def from_values(cls, values: Sequence[float], nominal: float = 0.6, **kw) -> "FactionDynamics":
        x = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
        n = len(DOMAINS)
        assert x.shape == (n,), f"state must have {n} entries"
        return cls(x=x, x_nominal=np.full(n, nominal), damping=np.zeros(n),
                   segmentation=np.ones(n), **kw)

    # ---- indicators ---------------------------------------------------------
    def stress(self) -> float:
        """theta in [0, 1]: liquidity + legitimacy deficit, the crisis trigger."""
        F = self.x[DOMAIN_INDEX["FINANCE"]]
        P = self.x[DOMAIN_INDEX["LEGITIMACY"]]
        return float(np.clip(1.0 - 0.5 * (F + P), 0.0, 1.0))

    def jacobian(self, theta: Optional[float] = None) -> np.ndarray:
        A = base_coupling()
        theta = self.stress() if theta is None else theta
        off = A - np.diag(np.diag(A))
        off = off * (1.0 + self.crisis_gain * theta)
        off = off * self.segmentation[:, None]          # segmentation gates inflow
        diag = np.diag(A) - self.damping
        return off + np.diag(diag)

    def lambda_max(self, theta: Optional[float] = None) -> float:
        """Largest real part of the Jacobian spectrum. > 0 means crisis regime."""
        return float(np.max(np.real(np.linalg.eigvals(self.jacobian(theta)))))

    def in_crisis(self) -> bool:
        return self.lambda_max() > 0.0

    # ---- controls -----------------------------------------------------------
    def harden(self, domain: str, gain: float = 0.15, segment: float = 0.85) -> None:
        """Defensive adaptation: restorative feedback plus reduced coupling."""
        i = DOMAIN_INDEX[domain]
        self.damping[i] += gain
        self.segmentation[i] *= segment

    # ---- integration --------------------------------------------------------
    def step(self, u: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None,
             record: bool = True, clip: bool = True) -> np.ndarray:
        """One Euler step of dx/dt = J (x - x_nom) + u + w, clipped to [0, 1].

        Coupling acts on the deficit relative to nominal: a domain sitting at
        its nominal level exerts no pull on its neighbours.
        """
        n = len(DOMAINS)
        u = np.zeros(n) if u is None else np.asarray(u, dtype=float)
        w = np.zeros(n) if w is None else np.asarray(w, dtype=float)
        J = self.jacobian()
        deficit = self.x - self.x_nominal
        dx = J @ deficit + u + w
        self.x = self.x + self.dt * dx
        if clip:
            self.x = np.clip(self.x, 0.0, 1.0)
        if record:
            self.history.append(self.lambda_max())
        return self.x

    def cascade_gain(self, disturbance: np.ndarray, horizon: int = 5) -> float:
        """||Delta x_final|| / ||disturbance|| over a shock-free horizon.

        Pure, unclipped simulation on copies (a linear-response quantity);
        the live state is untouched.
        """
        d = np.asarray(disturbance, dtype=float)
        norm = float(np.linalg.norm(d))
        if norm <= 1e-12:
            return 0.0
        a = FactionDynamics(self.x.copy(), self.x_nominal.copy(), self.damping.copy(),
                            self.segmentation.copy(), self.dt, self.crisis_gain)
        b = FactionDynamics(self.x.copy(), self.x_nominal.copy(), self.damping.copy(),
                            self.segmentation.copy(), self.dt, self.crisis_gain)
        a.step(u=d, record=False, clip=False)
        b.step(record=False, clip=False)
        for _ in range(horizon - 1):
            a.step(record=False, clip=False)
            b.step(record=False, clip=False)
        return float(np.linalg.norm(a.x - b.x)) / norm

    def as_dict(self) -> Dict[str, float]:
        return {d: float(self.x[i]) for i, d in enumerate(DOMAINS)}


# Opening state vectors per archetype: [E, I, F, L, C, M, P, R]
DEFAULT_STATES: Dict[str, List[float]] = {
    "CAPITAL":     [0.55, 0.65, 0.85, 0.65, 0.75, 0.65, 0.55, 0.50],
    "INDUSTRIAL":  [0.70, 0.50, 0.55, 0.80, 0.60, 0.60, 0.60, 0.85],
    "SECURITY":    [0.80, 0.45, 0.45, 0.60, 0.55, 0.85, 0.55, 0.55],
    "CONTINENTAL": [0.55, 0.60, 0.60, 0.60, 0.65, 0.65, 0.65, 0.60],
    "MOVEMENT":    [0.40, 0.55, 0.35, 0.40, 0.70, 0.20, 0.70, 0.35],
}


def default_dynamics(faction: str, **kw) -> FactionDynamics:
    vals = DEFAULT_STATES.get(faction, [0.6] * len(DOMAINS))
    return FactionDynamics.from_values(vals, **kw)
