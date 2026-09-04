"""
aeon/ruse/board.py — the R.U.S.E. sector map built from the layer-by-layer
systems map of institutional power.

The board is a multiplex graph. Sectors are institutional *nodes*, each living
in exactly one *layer* (physical capacity, ownership, capital, money, ledger,
law, executive, fiscal, defense, information, external). Units are *control
edges*: a faction reaches a node through an auditable edge — vote, contract,
law, ownership, custody, credit, command, or access right — with strength
w_e in [0, 1].

The math follows the source model:

    kappa_p      = prod_{e in p} w_e                     (path strength)
    C(u -> v)    = 1 - prod_{p in P_uv} (1 - kappa_p)    (independent paths)
    A_f          = ( prod_k (g_k + eps)^lambda_k )^(1 / sum_k lambda_k)
                                                          (broad control)
    B            = 1 - H_c                                (concentration index)
    P_eff        = sigmoid(b0 + b.x + x^T Q x)            (stacked leverage)
    Delta_e      = P_eff(G) - P_eff(G \\ e)               (deletion sensitivity)
    Dep(j -> f)  = ind_j * C(f -> j) / (1 + redundancy_j) (network dependence)

Association is not control: only *documented* edges count toward C. Decoy
(phantom) edges exist on the board so the information layer can lie about
them, but they contribute nothing to real control.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Layers (the board rows)
# ---------------------------------------------------------------------------

class Layer(IntEnum):
    PHYSICAL = 1      # land, energy, factories, logistics
    OWNERSHIP = 2     # shares, trusts, boards, voting rights
    CAPITAL = 3       # banks, funds, bonds, underwriting
    MONEY = 4         # reserves, repo, credit lines, backstops
    LEDGER = 5        # custody, clearing, settlement finality
    LAW = 6           # legislation, courts, sanctions, licenses
    EXECUTIVE = 7     # cabinets, ministries, emergency authority
    FISCAL = 8        # taxation, appropriations, sovereign debt
    DEFENSE = 9       # armed forces, intelligence, command
    INFORMATION = 10  # data, compute, media, legitimacy
    EXTERNAL = 11     # alliances, treaties, market access


LAYER_NAMES: Dict[Layer, str] = {
    Layer.PHYSICAL: "Physical & productive capacity",
    Layer.OWNERSHIP: "Ownership & governance",
    Layer.CAPITAL: "Capital & finance",
    Layer.MONEY: "Money, credit & liquidity",
    Layer.LEDGER: "Ledger, custody & settlement",
    Layer.LAW: "Mandate, law & regulation",
    Layer.EXECUTIVE: "Sovereign & executive authority",
    Layer.FISCAL: "Fiscal capacity",
    Layer.DEFENSE: "Defense, intelligence & coercion",
    Layer.INFORMATION: "Information, compute & legitimacy",
    Layer.EXTERNAL: "External networks & alliances",
}

# Layers whose nodes gain leverage under stress (Layer 12, crisis optionality):
# liquidity, collateral rules, emergency authority, essential infrastructure.
CRISIS_LAYERS = {Layer.MONEY, Layer.LEDGER, Layer.LAW, Layer.EXECUTIVE, Layer.DEFENSE}


class EdgeKind(str, Enum):
    VOTE = "vote"
    CONTRACT = "contract"
    LAW = "law"
    OWNERSHIP = "ownership"
    CUSTODY = "custody"
    CREDIT = "credit"
    COMMAND = "command"
    ACCESS = "access"


# Which control edges are meaningful in which layer. An edge of the wrong kind
# into a layer is rejected — a share certificate does not command an army.
LAYER_KINDS: Dict[Layer, frozenset] = {
    Layer.PHYSICAL: frozenset({EdgeKind.OWNERSHIP, EdgeKind.CONTRACT, EdgeKind.ACCESS}),
    Layer.OWNERSHIP: frozenset({EdgeKind.VOTE, EdgeKind.OWNERSHIP, EdgeKind.CONTRACT}),
    Layer.CAPITAL: frozenset({EdgeKind.CREDIT, EdgeKind.CONTRACT, EdgeKind.OWNERSHIP}),
    Layer.MONEY: frozenset({EdgeKind.CREDIT, EdgeKind.ACCESS}),
    Layer.LEDGER: frozenset({EdgeKind.CUSTODY, EdgeKind.ACCESS, EdgeKind.CONTRACT}),
    Layer.LAW: frozenset({EdgeKind.LAW}),
    Layer.EXECUTIVE: frozenset({EdgeKind.COMMAND, EdgeKind.LAW}),
    Layer.FISCAL: frozenset({EdgeKind.LAW, EdgeKind.CREDIT}),
    Layer.DEFENSE: frozenset({EdgeKind.COMMAND, EdgeKind.ACCESS}),
    Layer.INFORMATION: frozenset({EdgeKind.ACCESS, EdgeKind.CONTRACT, EdgeKind.OWNERSHIP}),
    Layer.EXTERNAL: frozenset({EdgeKind.CONTRACT, EdgeKind.LAW, EdgeKind.ACCESS}),
}

# Relative price of building into each layer.
LAYER_COST: Dict[Layer, float] = {
    Layer.PHYSICAL: 1.0, Layer.OWNERSHIP: 1.0, Layer.CAPITAL: 1.1,
    Layer.MONEY: 1.3, Layer.LEDGER: 1.3, Layer.LAW: 1.5, Layer.EXECUTIVE: 1.6,
    Layer.FISCAL: 1.3, Layer.DEFENSE: 1.5, Layer.INFORMATION: 1.0,
    Layer.EXTERNAL: 1.2,
}


# ---------------------------------------------------------------------------
# Board objects
# ---------------------------------------------------------------------------

@dataclass
class Node:
    id: str
    name: str
    layer: Layer
    indispensable: float = 0.5   # how costly it is to route around this node
    redundancy: int = 0          # substitute routes; each one dilutes dependence

    def valid_kind(self, kind: EdgeKind) -> bool:
        return kind in LAYER_KINDS[self.layer]


@dataclass
class Edge:
    id: str
    owner: str                   # faction id
    src: str                     # faction id (direct) or node id (stacked path)
    dst: str                     # node id
    kind: EdgeKind
    weight: float                # target strength once settled
    documented: bool = True      # False = decoy phantom (no real control)
    settle_left: int = 0         # turns until the edge is fully effective
    settle_total: int = 0
    created_turn: int = 0
    expires_turn: Optional[int] = None
    audited: bool = False        # publicly documented: everyone sees it truly
    fanatic_until: int = -1      # immune to haircuts/terror while turn <= this

    def effective_weight(self) -> float:
        """Weight after settlement ramp. Phantoms never exert control."""
        if not self.documented:
            return 0.0
        if self.settle_total <= 0 or self.settle_left <= 0:
            return self.weight
        done = self.settle_total - self.settle_left
        return self.weight * (done / self.settle_total)


@dataclass
class Faction:
    id: str
    name: str
    archetype: str
    budget: float = 20.0
    ruse_points: int = 2
    eliminated: bool = False


@dataclass
class Board:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, Edge] = field(default_factory=dict)
    factions: Dict[str, Faction] = field(default_factory=dict)
    _next_edge: int = 0

    # ---- construction -----------------------------------------------------
    def add_node(self, node: Node) -> Node:
        self.nodes[node.id] = node
        return node

    def add_faction(self, faction: Faction) -> Faction:
        self.factions[faction.id] = faction
        return faction

    def new_edge_id(self) -> str:
        self._next_edge += 1
        return f"E{self._next_edge}"

    def add_edge(self, owner: str, dst: str, kind: EdgeKind, weight: float,
                 src: Optional[str] = None, documented: bool = True,
                 settle: int = 0, turn: int = 0,
                 expires_turn: Optional[int] = None) -> Edge:
        if dst not in self.nodes:
            raise KeyError(f"unknown node {dst!r}")
        if owner not in self.factions:
            raise KeyError(f"unknown faction {owner!r}")
        src = owner if src is None else src
        if src != owner and src not in self.nodes:
            raise KeyError(f"unknown via-node {src!r}")
        if not self.nodes[dst].valid_kind(kind):
            raise ValueError(
                f"{kind.value} edge is not valid in layer "
                f"{self.nodes[dst].layer.name} ({dst})")
        e = Edge(id=self.new_edge_id(), owner=owner, src=src, dst=dst,
                 kind=kind, weight=max(0.0, min(1.0, weight)),
                 documented=documented, settle_left=settle, settle_total=settle,
                 created_turn=turn, expires_turn=expires_turn)
        self.edges[e.id] = e
        return e

    def remove_edge(self, edge_id: str) -> Optional[Edge]:
        return self.edges.pop(edge_id, None)

    # ---- queries ------------------------------------------------------------
    def edges_into(self, node_id: str, owner: Optional[str] = None) -> List[Edge]:
        out = [e for e in self.edges.values() if e.dst == node_id]
        if owner is not None:
            out = [e for e in out if e.owner == owner]
        return sorted(out, key=lambda e: e.id)

    def nodes_in_layer(self, layer: Layer) -> List[Node]:
        return [n for n in self.nodes.values() if n.layer == layer]

    def layers(self) -> List[Layer]:
        return sorted({n.layer for n in self.nodes.values()})

    # ---- control math -------------------------------------------------------
    def control(self, faction: str, edges: Optional[Iterable[Edge]] = None,
                iters: Optional[int] = None) -> Dict[str, float]:
        """C(f -> v) for every node v.

        Direct edges (src == faction) have kappa = w. Stacked edges
        (src == node u) have kappa = C(f -> u) * w. Independent paths combine
        as 1 - prod(1 - kappa). The map is monotone and bounded, so a few
        sweeps converge; `iters` defaults to the node count + 1.
        """
        edge_list = list(self.edges.values() if edges is None else edges)
        by_dst: Dict[str, List[Edge]] = {}
        for e in edge_list:
            if e.owner != faction or not e.documented:
                continue
            by_dst.setdefault(e.dst, []).append(e)
        C = {v: 0.0 for v in self.nodes}
        sweeps = (len(self.nodes) + 1) if iters is None else iters
        for _ in range(sweeps):
            changed = False
            for v in self.nodes:
                prod = 1.0
                for e in by_dst.get(v, ()):
                    w = e.effective_weight()
                    k = w if e.src == faction else C.get(e.src, 0.0) * w
                    prod *= (1.0 - k)
                new = 1.0 - prod
                if abs(new - C[v]) > 1e-12:
                    changed = True
                C[v] = new
            if not changed:
                break
        return C

    def layer_control(self, faction: str, C: Optional[Dict[str, float]] = None) -> Dict[Layer, float]:
        """g_k(f): indispensability-weighted mean control over each layer."""
        C = self.control(faction) if C is None else C
        out: Dict[Layer, float] = {}
        for layer in self.layers():
            nodes = self.nodes_in_layer(layer)
            wsum = sum(n.indispensable for n in nodes) or 1.0
            out[layer] = sum(n.indispensable * C[n.id] for n in nodes) / wsum
        return out

    def broad_control(self, faction: str, eps: float = 0.02,
                      weights: Optional[Dict[Layer, float]] = None) -> float:
        """A_f = (prod_k (g_k + eps)^lambda_k)^(1/sum lambda_k).

        Geometric: a faction with zero control in one indispensable layer is
        pushed down no matter how large the others are.
        """
        g = self.layer_control(faction)
        lam = {k: 1.0 for k in g} if weights is None else weights
        num = sum(lam[k] * math.log(g[k] + eps) for k in g)
        den = sum(lam[k] for k in g) or 1.0
        return math.exp(num / den)

    def concentration_index(self) -> float:
        """B = 1 - H_c. 0 = diffuse control, 1 = one faction holds everything."""
        masses = {}
        for f in self.factions:
            g = self.layer_control(f)
            masses[f] = sum(g.values())
        total = sum(masses.values())
        n = len(masses)
        if total <= 0 or n <= 1:
            return 0.0
        H = 0.0
        for m in masses.values():
            p = m / total
            if p > 0:
                H -= p * math.log(p)
        return 1.0 - H / math.log(n)

    def effective_power(self, faction: str, C: Optional[Dict[str, float]] = None) -> float:
        """P_eff = sigmoid(b0 + b.g + g^T Q g) with reinforcing seams in Q.

        Cross-terms reward stacking: finance plus law, force plus alliances,
        compute plus market access, settlement plus liquidity, votes plus
        capital. Simple additive scoring would miss this complementarity.
        """
        g = self.layer_control(faction, C)
        seams = [
            (Layer.MONEY, Layer.LAW, 2.0),
            (Layer.DEFENSE, Layer.EXTERNAL, 2.0),
            (Layer.INFORMATION, Layer.CAPITAL, 1.5),
            (Layer.LEDGER, Layer.MONEY, 1.5),
            (Layer.OWNERSHIP, Layer.CAPITAL, 1.0),
            (Layer.EXECUTIVE, Layer.FISCAL, 1.5),
            (Layer.PHYSICAL, Layer.EXTERNAL, 1.0),
        ]
        z = -3.0 + 0.9 * sum(g.values())
        for a, b, theta in seams:
            z += theta * g.get(a, 0.0) * g.get(b, 0.0)
        return 1.0 / (1.0 + math.exp(-z))

    def deletion_sensitivity(self, edge_id: str) -> float:
        """Delta_e = P_eff(G) - P_eff(G \\ e) for the edge's owner.

        A counterfactual analytical removal: large Delta_e means the edge is
        structurally important to its owner.
        """
        e = self.edges[edge_id]
        base = self.effective_power(e.owner)
        rest = [x for x in self.edges.values() if x.id != edge_id]
        without = self.effective_power(e.owner, self.control(e.owner, rest))
        return base - without

    def dependence(self, faction: str, crisis_multiplier: Optional[Dict[str, float]] = None,
                   C: Optional[Dict[str, float]] = None) -> float:
        """Sum_j Dep(j -> f): how much of the board routes through f.

        The primacy-transition condition compares this across factions:
        a challenger wins when more of the network depends on it than on
        the incumbent, not when it is bigger in one category.
        """
        C = self.control(faction) if C is None else C
        total = 0.0
        for n in self.nodes.values():
            mult = 1.0 if crisis_multiplier is None else crisis_multiplier.get(n.id, 1.0)
            total += n.indispensable * C[n.id] * mult / (1.0 + n.redundancy)
        return total

    def scoreboard(self, crisis_multiplier: Optional[Dict[str, float]] = None) -> Dict[str, dict]:
        out = {}
        for f in self.factions:
            C = self.control(f)
            out[f] = {
                "dependence": self.dependence(f, crisis_multiplier, C),
                "p_eff": self.effective_power(f, C),
                "broad": self.broad_control(f),
                "layers": {k.name: v for k, v in self.layer_control(f, C).items()},
            }
        return out


# ---------------------------------------------------------------------------
# Default board: fictional archetypes only
# ---------------------------------------------------------------------------

DEFAULT_NODES = [
    # id, name, layer, indispensable, redundancy
    ("IRON", "Ironworks Basin", Layer.PHYSICAL, 0.6, 1),
    ("GRID", "Grid Authority", Layer.PHYSICAL, 0.8, 0),
    ("PORT", "Port Concession", Layer.PHYSICAL, 0.5, 1),
    ("TRUST", "Holding Trust", Layer.OWNERSHIP, 0.5, 1),
    ("BOARD", "Foundation Board", Layer.OWNERSHIP, 0.4, 1),
    ("BOND", "Bond Desk", Layer.CAPITAL, 0.6, 1),
    ("UNDER", "Underwriting Syndicate", Layer.CAPITAL, 0.5, 1),
    ("RESERVE", "Reserve Window", Layer.MONEY, 0.9, 0),
    ("REPO", "Repo Market", Layer.MONEY, 0.7, 1),
    ("CLEAR", "Clearing House", Layer.LEDGER, 0.9, 0),
    ("CUSTODY", "Custody Depository", Layer.LEDGER, 0.8, 0),
    ("COURT", "High Court", Layer.LAW, 0.8, 0),
    ("SANCTION", "Sanctions Office", Layer.LAW, 0.7, 0),
    ("LICENSE", "Licensing Bureau", Layer.LAW, 0.5, 1),
    ("COUNCIL", "Executive Council", Layer.EXECUTIVE, 0.9, 0),
    ("TREASURY", "Treasury", Layer.FISCAL, 0.8, 0),
    ("STAFF", "General Staff", Layer.DEFENSE, 0.8, 0),
    ("SIGNALS", "Signals Directorate", Layer.DEFENSE, 0.6, 0),
    ("CLOUD", "Cloud Exchange", Layer.INFORMATION, 0.7, 1),
    ("FEED", "Market Data Feed", Layer.INFORMATION, 0.6, 1),
    ("BROADCAST", "Broadcast Network", Layer.INFORMATION, 0.5, 2),
    ("ALLIANCE", "Alliance Council", Layer.EXTERNAL, 0.8, 0),
    ("COMPACT", "Trade Compact", Layer.EXTERNAL, 0.6, 1),
]

# Scenario archetypes from the fictional multipolar model. These are modeling
# devices, not real governments.
DEFAULT_FACTIONS = {
    "CAPITAL": ("Liberal Capital Network",
                "capital markets, technology, alliances, reserve-currency effects"),
    "INDUSTRIAL": ("Industrial Coordination State",
                   "industrial scale, state coordination, long-horizon planning"),
    "SECURITY": ("Security Depth State",
                 "strategic depth, energy, deterrence, geographic scale"),
    "CONTINENTAL": ("Continental Sovereign",
                    "diplomatic reach, state capacity, institutional leverage"),
    "MOVEMENT": ("Network Political Movement",
                 "identity cohesion, persuasion, local institutional capture"),
}

# Opening positions: (node, kind, weight) per faction.
DEFAULT_OPENING = {
    "CAPITAL": [
        ("BOND", EdgeKind.CREDIT, 0.5), ("UNDER", EdgeKind.OWNERSHIP, 0.4),
        ("RESERVE", EdgeKind.ACCESS, 0.4), ("CLEAR", EdgeKind.CUSTODY, 0.5),
        ("CUSTODY", EdgeKind.CUSTODY, 0.4), ("CLOUD", EdgeKind.OWNERSHIP, 0.5),
        ("FEED", EdgeKind.ACCESS, 0.5), ("ALLIANCE", EdgeKind.CONTRACT, 0.4),
        ("COURT", EdgeKind.LAW, 0.2),
    ],
    "INDUSTRIAL": [
        ("IRON", EdgeKind.OWNERSHIP, 0.6), ("GRID", EdgeKind.CONTRACT, 0.5),
        ("PORT", EdgeKind.OWNERSHIP, 0.5), ("TREASURY", EdgeKind.LAW, 0.5),
        ("COUNCIL", EdgeKind.COMMAND, 0.6), ("REPO", EdgeKind.CREDIT, 0.3),
        ("COMPACT", EdgeKind.CONTRACT, 0.4), ("STAFF", EdgeKind.COMMAND, 0.4),
        ("SANCTION", EdgeKind.LAW, 0.2),
    ],
    "SECURITY": [
        ("GRID", EdgeKind.ACCESS, 0.4), ("STAFF", EdgeKind.COMMAND, 0.6),
        ("SIGNALS", EdgeKind.COMMAND, 0.6), ("COUNCIL", EdgeKind.COMMAND, 0.4),
        ("SANCTION", EdgeKind.LAW, 0.3), ("IRON", EdgeKind.CONTRACT, 0.3),
    ],
    "CONTINENTAL": [
        ("COURT", EdgeKind.LAW, 0.5), ("LICENSE", EdgeKind.LAW, 0.5),
        ("ALLIANCE", EdgeKind.LAW, 0.5), ("COUNCIL", EdgeKind.LAW, 0.3),
        ("STAFF", EdgeKind.ACCESS, 0.3), ("TREASURY", EdgeKind.CREDIT, 0.3),
    ],
    "MOVEMENT": [
        ("BROADCAST", EdgeKind.ACCESS, 0.6), ("BOARD", EdgeKind.VOTE, 0.4),
        ("LICENSE", EdgeKind.LAW, 0.2), ("CLOUD", EdgeKind.ACCESS, 0.3),
    ],
}


def default_board(factions: Optional[List[str]] = None, budget: float = 20.0) -> Board:
    """Build the standard map with the requested factions seated.

    Default seating is the incumbent hub (CAPITAL) against the industrial
    challenger (INDUSTRIAL).
    """
    factions = ["CAPITAL", "INDUSTRIAL"] if factions is None else list(factions)
    b = Board()
    for nid, name, layer, ind, red in DEFAULT_NODES:
        b.add_node(Node(nid, name, layer, ind, red))
    for f in factions:
        if f not in DEFAULT_FACTIONS:
            raise KeyError(f"unknown faction archetype {f!r}; "
                           f"choose from {sorted(DEFAULT_FACTIONS)}")
        name, arch = DEFAULT_FACTIONS[f]
        b.add_faction(Faction(f, name, arch, budget=budget))
    for f in factions:
        for node, kind, w in DEFAULT_OPENING[f]:
            b.add_edge(f, node, kind, w)
    return b
