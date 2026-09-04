"""
aeon/ruse/observer.py — the war table each faction actually sees.

The observer does not read the world; it chooses what to measure, and what
it measures can be manipulated. Observed data are

    Y(t) = H(O(t)) X(t) + eps(t)

Own edges, audited edges, and anything under your SPY are seen truly.
Everything else is a noisy estimate, subject to the opponent's ruses:
DECOY phantoms look real, CAMOUFLAGE hides, RADIO_SILENCE freezes the last
snapshot, REVERSE_INTEL makes real edges look unsupported.

Because attention changes what is observed, the game scores the observer
too. The epistemic loss

    L = ||X_hat - X||^2 + lambda_2 * Xi(X_hat)

penalizes both mis-estimated layer control and belief in unsupported edges.
A player who acts on a decoy pays for it twice: once in wasted orders, once
in this score.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .board import Board, Edge, EdgeKind, Layer, LAYER_NAMES
from .dynamics import DOMAINS
from .ruses import Ruse


@dataclass
class EdgeView:
    id: str
    owner: str
    kind: EdgeKind
    weight: float
    estimated: bool = False       # weight is a noisy estimate
    flagged: bool = False         # looks like an unsupported link (reverse intel)
    exposed_decoy: bool = False   # known phantom
    audited: bool = False
    settle_left: int = 0
    via: Optional[str] = None


@dataclass
class NodeView:
    id: str
    name: str
    layer: Layer
    indispensable: float
    redundancy: int
    edges: List[EdgeView] = field(default_factory=list)
    stale: bool = False           # served from a radio-silence snapshot
    control_est: Dict[str, float] = field(default_factory=dict)


@dataclass
class Observation:
    turn: int
    max_turns: int
    faction: str
    faction_name: str
    budget: float
    ruse_points: int
    state: Dict[str, float]
    stress: float
    lambda_max: float
    nodes: List[NodeView]
    own_ruses: List[str]
    known_latents: List[str]
    intercepted: List[str]
    events: List[str]
    rejected: List[str]
    scores_est: Dict[str, float]
    factions: List[str]
    edge_ids_own: List[str]

    def believed_edges(self) -> List[Edge]:
        """Edges as the observer believes them, usable by Board.control."""
        out: List[Edge] = []
        for nv in self.nodes:
            for ev in nv.edges:
                documented = not (ev.flagged or ev.exposed_decoy)
                out.append(Edge(id=ev.id, owner=ev.owner,
                                src=(ev.via or ev.owner), dst=nv.id,
                                kind=ev.kind, weight=ev.weight,
                                documented=documented))
        return out


def _noise(seed: int, observer: str, edge_id: str, turn: int, sigma: float) -> float:
    key = f"{seed}|{observer}|{edge_id}|{turn}".encode()
    rng = random.Random(zlib.crc32(key))
    return rng.gauss(0.0, sigma)


def observe(match, observer: str) -> Observation:
    """Build the observation for `observer` from the match's true state."""
    board: Board = match.board
    cfg = match.config
    turn = match.turn
    live = [r for r in match.ruses if r.live(turn)]

    def has(ruse: Ruse, owner: Optional[str], node: str, exclude_owner: Optional[str] = None) -> bool:
        for r in live:
            if r.ruse != ruse or r.node != node:
                continue
            if owner is not None and r.owner != owner:
                continue
            if exclude_owner is not None and r.owner == exclude_owner:
                continue
            return True
        return False

    nodes: List[NodeView] = []
    seen = match.last_seen.setdefault(observer, {})
    for n in board.nodes.values():
        spy = has(Ruse.SPY, observer, n.id)
        silenced = (not spy) and has(Ruse.RADIO_SILENCE, None, n.id, exclude_owner=observer)
        if silenced and n.id in seen:
            nv = seen[n.id]
            nv.stale = True
            nodes.append(nv)
            continue
        nv = NodeView(n.id, n.name, n.layer, n.indispensable, n.redundancy)
        for e in board.edges_into(n.id):
            via = None if e.src == e.owner else e.src
            if e.owner == observer or spy or e.audited:
                nv.edges.append(EdgeView(e.id, e.owner, e.kind, e.weight,
                                         estimated=False, flagged=False,
                                         exposed_decoy=not e.documented,
                                         audited=e.audited, settle_left=e.settle_left,
                                         via=via))
                continue
            if has(Ruse.CAMOUFLAGE, e.owner, n.id):
                continue
            est = e.weight + _noise(match.seed, observer, e.id, turn, cfg.obs_noise)
            est = max(0.0, min(1.0, est))
            nv.edges.append(EdgeView(e.id, e.owner, e.kind, est, estimated=True,
                                     flagged=has(Ruse.REVERSE_INTEL, e.owner, n.id),
                                     exposed_decoy=False, audited=False,
                                     settle_left=e.settle_left, via=via))
        if not silenced:
            seen[n.id] = nv
        nodes.append(nv)

    # Believed control per faction, from believed edges.
    factions = [f for f in board.factions]
    obs = Observation(
        turn=turn, max_turns=cfg.max_turns, faction=observer,
        faction_name=board.factions[observer].name,
        budget=board.factions[observer].budget,
        ruse_points=board.factions[observer].ruse_points,
        state=match.dyn[observer].as_dict(),
        stress=match.dyn[observer].stress(),
        lambda_max=match.dyn[observer].lambda_max(),
        nodes=nodes,
        own_ruses=[f"{r.ruse.value}@{r.node} ({r.end_turn - turn + 1} left)"
                   for r in live if r.owner == observer],
        known_latents=[],
        intercepted=list(match.intercepted.get(observer, [])),
        events=list(match.public_events) + list(match.private_events.get(observer, [])),
        rejected=list(match.rejected.get(observer, [])),
        scores_est={}, factions=factions,
        edge_ids_own=[e.id for e in board.edges.values() if e.owner == observer],
    )
    believed = obs.believed_edges()
    mult = match.crisis_multiplier()
    for f in factions:
        C = board.control(f, edges=believed)
        obs.scores_est[f] = board.dependence(f, mult, C)
        for nv in nodes:
            nv.control_est[f] = C[nv.id]
    for l in match.latents.values():
        if l.revoked:
            continue
        visible = (l.owner == observer or observer in l.revealed_to
                   or has(Ruse.SPY, observer, l.node))
        if not visible:
            continue
        status = "ACTIVE" if l.active else "dormant"
        obs.known_latents.append(
            f"{l.id} {l.owner} {l.kind.value}->{l.node} w{l.payload:.2f} "
            f"fires on {l.trigger}>={l.threshold:.2f} [{status}]")
    return obs


# ---------------------------------------------------------------------------
# Scoring the observer
# ---------------------------------------------------------------------------

def epistemic_loss(match, observer: str, lambda_unsupported: float = 0.5) -> Dict[str, float]:
    """||g_hat - g||^2 over layer control for every faction, plus a penalty
    for each phantom edge the observer believes is real."""
    obs = observe(match, observer)
    believed = obs.believed_edges()
    board = match.board
    err = 0.0
    for f in board.factions:
        g_true = board.layer_control(f)
        g_hat = board.layer_control(f, board.control(f, edges=believed))
        err += sum((g_hat[k] - g_true[k]) ** 2 for k in g_true)
    phantoms_believed = 0
    for nv in obs.nodes:
        for ev in nv.edges:
            true = board.edges.get(ev.id)
            if true is not None and not true.documented and not ev.exposed_decoy and not ev.flagged:
                phantoms_believed += 1
    return {"prediction_error": err,
            "unsupported_links": phantoms_believed,
            "loss": err + lambda_unsupported * phantoms_believed}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt_w(ev: EdgeView) -> str:
    s = f"{ev.weight:.2f}"
    if ev.estimated:
        s = "~" + s
    tags = []
    if ev.exposed_decoy:
        tags.append("DECOY")
    if ev.flagged:
        tags.append("unsupported?")
    if ev.audited:
        tags.append("audited")
    if ev.settle_left > 0:
        tags.append(f"settling {ev.settle_left}")
    if ev.via:
        tags.append(f"via {ev.via}")
    return s + (" [" + ", ".join(tags) + "]" if tags else "")


def render(obs: Observation, show_layer_names: bool = True) -> str:
    """The war table: what R.U.S.E. shows on the map when zoomed out."""
    lines: List[str] = []
    stab = "CRISIS" if obs.lambda_max > 0 else "stable"
    lines.append(f"=== R.U.S.E. power board | turn {obs.turn}/{obs.max_turns} | "
                 f"you: {obs.faction} ({obs.faction_name}) ===")
    st = " ".join(f"{d[:3]}{obs.state[d]:.2f}" for d in DOMAINS)
    lines.append(f"budget {obs.budget:.1f} | ruse pts {obs.ruse_points} | "
                 f"stress {obs.stress:.2f} | lambda_max {obs.lambda_max:+.2f} {stab}")
    lines.append(f"state: {st}")
    sc = " | ".join(f"{f} {v:.2f}" for f, v in obs.scores_est.items())
    lines.append(f"network dependence (your estimate): {sc}")
    cur_layer = None
    for nv in obs.nodes:
        if nv.layer != cur_layer:
            cur_layer = nv.layer
            name = LAYER_NAMES[cur_layer] if show_layer_names else cur_layer.name
            lines.append(f"L{int(cur_layer)} {name.upper()}")
        ctrl = " ".join(f"{f}:{nv.control_est.get(f, 0.0):.2f}" for f in obs.factions)
        stale = " [STALE]" if nv.stale else ""
        lines.append(f"  {nv.id:<9} {nv.name:<24} ind{nv.indispensable:.1f} "
                     f"red{nv.redundancy} | {ctrl}{stale}")
        for ev in nv.edges:
            lines.append(f"    {ev.id:<5} {ev.owner:<11} {ev.kind.value:<9} {_fmt_w(ev)}")
    if obs.own_ruses:
        lines.append("your ruses: " + ", ".join(obs.own_ruses))
    if obs.known_latents:
        lines.append("known latents: " + "; ".join(obs.known_latents))
    if obs.intercepted:
        lines.append("intercepted orders: " + "; ".join(obs.intercepted))
    if obs.events:
        lines.append("events: " + "; ".join(obs.events))
    if obs.rejected:
        lines.append("rejected last turn: " + "; ".join(obs.rejected))
    return "\n".join(lines)
