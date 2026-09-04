"""
aeon/ruse/engine.py — the match: simultaneous turns on the power board.

Each turn every seated faction receives its own observation (see
observer.py), returns orders (see orders.py), and the engine resolves them
in a fixed sequence:

    ruses -> audits -> revocations -> builds/plants/sunsets/redundancy
    -> hardening -> pressure -> dynamics step (with shocks)
    -> crisis haircuts & terror -> latent triggers -> settlement/expiry
    -> income -> scores -> victory check

Victory is the primacy-transition condition from the source model:

    sum_j Dep(j -> challenger) > sum_j Dep(j -> incumbent)

held by a margin for several consecutive turns. Otherwise the faction with
the most network dependence at the time limit wins. A faction whose
finance and legitimacy both collapse is eliminated.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .board import (Board, CRISIS_LAYERS, Edge, EdgeKind, LAYER_COST, Layer,
                    default_board)
from .dynamics import DOMAINS, DOMAIN_INDEX, FactionDynamics, default_dynamics
from .latent import Latent, trigger_scores
from .observer import Observation, epistemic_loss, observe, render
from .orders import Order, ParseResult, parse_orders
from .ruses import (ActiveRuse, DECOY_WEIGHT, RUSE_COST, RUSE_DURATION,
                    Ruse, TERROR_DECAY)


@dataclass
class RuseConfig:
    max_turns: int = 20
    primacy_ratio: float = 1.5      # challenger must clearly out-depend the field
    primacy_turns: int = 4          # ...and hold it
    primacy_min_turn: int = 6       # openings do not decide a match
    max_orders: int = 6
    build_cost: float = 10.0          # per unit weight, times layer cost
    settle_turns: int = 1
    pressure_cost: float = 15.0       # per unit magnitude
    pressure_backlash: float = 0.25   # fraction reflected onto the attacker
    harden_cost: float = 8.0
    audit_cost: float = 3.0
    revoke_cost: float = 6.0
    revoke_law_threshold: float = 0.3
    plant_cost: float = 12.0          # per unit payload, times layer cost
    redundancy_cost: float = 6.0
    latent_expiry: Optional[int] = 4
    income_base: float = 5.0
    income_state: float = 5.0
    income_power: float = 5.0
    ruse_points_per_turn: int = 1
    ruse_points_max: int = 3
    shock_sigma: float = 0.03
    obs_noise: float = 0.08
    crisis_haircut: float = 0.10
    collapse_threshold: float = 0.10


@dataclass
class TurnRecord:
    turn: int
    observation_text: Dict[str, str]
    orders_text: Dict[str, str]
    accepted: Dict[str, List[str]]
    rejected: Dict[str, List[str]]
    scores: Dict[str, float]
    lambda_max: Dict[str, float]


@dataclass
class MatchResult:
    winner: Optional[str]
    reason: str
    turns: int
    final_scores: Dict[str, float]
    score_history: List[Dict[str, float]]
    epistemic: Dict[str, Dict[str, float]]
    records: List[TurnRecord] = field(default_factory=list)

    def summary(self) -> str:
        sc = ", ".join(f"{f} {v:.2f}" for f, v in self.final_scores.items())
        ep = ", ".join(f"{f} {v['loss']:.2f}" for f, v in self.epistemic.items())
        return (f"winner: {self.winner or 'draw'} ({self.reason}) after {self.turns} turns | "
                f"dependence: {sc} | epistemic loss: {ep}")


class Match:
    def __init__(self, board: Optional[Board] = None, seed: int = 0,
                 config: Optional[RuseConfig] = None,
                 dynamics: Optional[Dict[str, FactionDynamics]] = None):
        self.board = board if board is not None else default_board()
        self.config = config or RuseConfig()
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.turn = 1
        self.dyn: Dict[str, FactionDynamics] = dynamics or {
            f: default_dynamics(f) for f in self.board.factions}
        self.ruses: List[ActiveRuse] = []
        self.latents: Dict[str, Latent] = {}
        self._next_latent = 0
        self.last_seen: Dict[str, Dict[str, object]] = {}
        self.intercepted: Dict[str, List[str]] = {}
        self.public_events: List[str] = []
        self.private_events: Dict[str, List[str]] = {}
        self.rejected: Dict[str, List[str]] = {}
        self.score_history: List[Dict[str, float]] = []
        self.primacy_streak: Dict[str, int] = {f: 0 for f in self.board.factions}
        self.records: List[TurnRecord] = []
        self.winner: Optional[str] = None
        self.reason: str = ""

    # ---- helpers --------------------------------------------------------------
    def live_ruses(self) -> List[ActiveRuse]:
        return [r for r in self.ruses if r.live(self.turn)]

    def has_ruse(self, ruse: Ruse, owner: str, node: str) -> bool:
        return any(r.ruse == ruse and r.owner == owner and r.node == node
                   for r in self.live_ruses())

    def crisis_multiplier(self) -> Dict[str, float]:
        """Layer-12 optionality: liquidity, law, command and settlement nodes
        matter more when the system is stressed."""
        if not self.dyn:
            return {}
        theta = sum(d.stress() for d in self.dyn.values()) / len(self.dyn)
        return {n.id: (1.0 + 0.5 * theta if n.layer in CRISIS_LAYERS else 1.0)
                for n in self.board.nodes.values()}

    def scores(self) -> Dict[str, float]:
        mult = self.crisis_multiplier()
        return {f: self.board.dependence(f, mult) for f in self.board.factions}

    def observe(self, faction: str) -> Observation:
        return observe(self, faction)

    def render(self, faction: str) -> str:
        return render(self.observe(faction))

    def _log(self, faction: Optional[str], msg: str) -> None:
        if faction is None:
            self.public_events.append(msg)
        else:
            self.private_events.setdefault(faction, []).append(msg)

    def _reject(self, faction: str, order: Order, why: str) -> None:
        self.rejected.setdefault(faction, []).append(f"{order.text()}: {why}")

    def _charge(self, faction: str, cost: float, order: Order) -> bool:
        fac = self.board.factions[faction]
        if fac.budget + 1e-9 < cost:
            self._reject(faction, order, f"needs {cost:.1f}, budget {fac.budget:.1f}")
            return False
        fac.budget -= cost
        return True

    def done(self) -> bool:
        return self.winner is not None or bool(self.reason)

    # ---- the turn -------------------------------------------------------------
    def step(self, orders: Dict[str, Union[str, ParseResult, Sequence[Order]]]) -> None:
        """Resolve one simultaneous turn. `orders` maps faction -> orders."""
        if self.done():
            return
        cfg = self.config
        self.public_events = []
        self.private_events = {}
        self.rejected = {}
        self.intercepted = {}
        governance_event = False
        parsed: Dict[str, List[Order]] = {}
        accepted: Dict[str, List[str]] = {f: [] for f in self.board.factions}
        for f in self.board.factions:
            raw = orders.get(f, "PASS")
            if isinstance(raw, ParseResult):
                pr = raw
            elif isinstance(raw, str):
                pr = parse_orders(raw, cfg.max_orders)
            else:
                pr = ParseResult(orders=list(raw)[:cfg.max_orders])
            for err in pr.errors:
                self.rejected.setdefault(f, []).append(err)
            parsed[f] = [] if self.board.factions[f].eliminated else pr.orders

        pressure: Dict[str, np.ndarray] = {f: np.zeros(len(DOMAINS)) for f in self.board.factions}

        def by_verb(verb: str):
            for f, ol in parsed.items():
                for o in ol:
                    if o.verb == verb:
                        yield f, o

        def intercept(f: str, o: Order, node: str) -> None:
            for r in self.live_ruses():
                if r.ruse == Ruse.DECRYPTION and r.node == node and r.owner != f:
                    self.intercepted.setdefault(r.owner, []).append(f"{f}: {o.text()}")

        # 1. ruses
        for f, o in by_verb("RUSE"):
            fac = self.board.factions[f]
            if o.node not in self.board.nodes:
                self._reject(f, o, "unknown node"); continue
            cost = RUSE_COST[o.ruse]
            if fac.ruse_points < cost:
                self._reject(f, o, "no ruse points"); continue
            if self.has_ruse(o.ruse, f, o.node):
                self._reject(f, o, "already live"); continue
            fac.ruse_points -= cost
            ar = ActiveRuse(o.ruse, f, o.node, self.turn,
                            self.turn + RUSE_DURATION[o.ruse] - 1)
            if o.ruse == Ruse.DECOY:
                node = self.board.nodes[o.node]
                kind = sorted((k for k in EdgeKind if node.valid_kind(k)),
                              key=lambda k: k.value)[0]
                e = self.board.add_edge(f, o.node, kind, DECOY_WEIGHT, documented=False,
                                        turn=self.turn)
                ar.phantom_edge_id = e.id
            self.ruses.append(ar)
            accepted[f].append(o.text())

        # 2. audits (public documentation)
        for f, o in by_verb("AUDIT"):
            if o.node not in self.board.nodes:
                self._reject(f, o, "unknown node"); continue
            if not self._charge(f, cfg.audit_cost, o):
                continue
            n_edges = 0
            for e in self.board.edges_into(o.node):
                e.audited = True
                n_edges += 1
            for l in self.latents.values():
                if l.node == o.node and not l.revoked:
                    l.revealed_to.update(self.board.factions)
            governance_event = True
            self._log(None, f"{f} audited {o.node} ({n_edges} edges documented)")
            accepted[f].append(o.text())
            intercept(f, o, o.node)

        # 3. revocations (revocation before replacement)
        for f, o in by_verb("REVOKE"):
            e = self.board.edges.get(o.edge_id)
            if e is None:
                self._reject(f, o, "no such edge"); continue
            if e.owner != f:
                if not e.audited:
                    self._reject(f, o, "edge not audited; identify the edge first"); continue
                C_me = self.board.control(f)[e.dst]
                C_them = self.board.control(e.owner)[e.dst]
                g_law = self.board.layer_control(f).get(Layer.LAW, 0.0)
                if C_me <= C_them:
                    self._reject(f, o, f"{e.owner} out-controls you at {e.dst}"); continue
                if g_law < cfg.revoke_law_threshold:
                    self._reject(f, o, f"law-layer control {g_law:.2f} below {cfg.revoke_law_threshold}"); continue
            if not self._charge(f, cfg.revoke_cost, o):
                continue
            self.board.remove_edge(e.id)
            for l in self.latents.values():
                if l.edge_id == e.id:
                    l.revoked = True
            governance_event = True
            self._log(None, f"{f} revoked {e.id} ({e.owner} {e.kind.value} -> {e.dst})")
            accepted[f].append(o.text())
            intercept(f, o, e.dst)

        # 4. builds, plants, sunsets, redundancy
        for f, o in by_verb("BUILD"):
            node = self.board.nodes.get(o.node)
            if node is None:
                self._reject(f, o, "unknown node"); continue
            if not node.valid_kind(o.kind):
                self._reject(f, o, f"{o.kind.value} not valid in {node.layer.name}"); continue
            if o.weight <= 0:
                self._reject(f, o, "weight must be > 0"); continue
            src = None
            if o.via:
                if o.via not in self.board.nodes:
                    self._reject(f, o, "unknown via node"); continue
                if self.board.control(f)[o.via] <= 0.0:
                    self._reject(f, o, f"no control over {o.via} to stack from"); continue
                src = o.via
            cost = cfg.build_cost * o.weight * LAYER_COST[node.layer]
            if not self._charge(f, cost, o):
                continue
            settle = 0 if self.has_ruse(Ruse.BLITZ, f, o.node) else cfg.settle_turns
            e = self.board.add_edge(f, o.node, o.kind, o.weight, src=src,
                                    settle=settle, turn=self.turn)
            if self.has_ruse(Ruse.FANATICISM, f, o.node):
                e.fanatic_until = max(r.end_turn for r in self.live_ruses()
                                      if r.ruse == Ruse.FANATICISM and r.owner == f and r.node == o.node)
            accepted[f].append(o.text())
            self._log(f, f"built {e.id}")
            intercept(f, o, o.node)

        for f, o in by_verb("PLANT"):
            node = self.board.nodes.get(o.node)
            if node is None:
                self._reject(f, o, "unknown node"); continue
            if not node.valid_kind(o.kind):
                self._reject(f, o, f"{o.kind.value} not valid in {node.layer.name}"); continue
            cost = cfg.plant_cost * o.weight * LAYER_COST[node.layer]
            if not self._charge(f, cost, o):
                continue
            self._next_latent += 1
            l = Latent(id=f"L{self._next_latent}", owner=f, node=o.node, kind=o.kind,
                       payload=o.weight, trigger=o.trigger, threshold=o.threshold,
                       privilege=o.weight, planted_turn=self.turn,
                       expiry_after=cfg.latent_expiry)
            self.latents[l.id] = l
            accepted[f].append(o.text())
            self._log(f, f"planted {l.id} at {o.node}")
            intercept(f, o, o.node)

        for f, o in by_verb("SUNSET"):
            l = self.latents.get(o.latent_id)
            if l is None or l.owner != f:
                self._reject(f, o, "no such latent of yours"); continue
            l.revoked = True
            if l.edge_id:
                self.board.remove_edge(l.edge_id)
            accepted[f].append(o.text())
            self._log(f, f"sunset {l.id}")

        for f, o in by_verb("REDUNDANCY"):
            node = self.board.nodes.get(o.node)
            if node is None:
                self._reject(f, o, "unknown node"); continue
            if not self._charge(f, cfg.redundancy_cost, o):
                continue
            node.redundancy += 1
            self._log(None, f"{f} added a substitute route at {o.node} (redundancy {node.redundancy})")
            accepted[f].append(o.text())

        # 5. hardening (defensive adaptation)
        for f, o in by_verb("HARDEN"):
            if not self._charge(f, cfg.harden_cost, o):
                continue
            self.dyn[f].harden(o.domain)
            accepted[f].append(o.text())
            self._log(f, f"hardened {o.domain}; lambda_max now {self.dyn[f].lambda_max():+.2f}")

        # 6. pressure (systemic, non-kinetic)
        for f, o in by_verb("PRESSURE"):
            if o.faction not in self.board.factions or o.faction == f:
                self._reject(f, o, "bad target faction"); continue
            if o.magnitude <= 0:
                self._reject(f, o, "magnitude must be > 0"); continue
            if not self._charge(f, cfg.pressure_cost * o.magnitude, o):
                continue
            i = DOMAIN_INDEX[o.domain]
            pressure[o.faction][i] -= o.magnitude
            pressure[f][i] -= o.magnitude * cfg.pressure_backlash
            u = np.zeros(len(DOMAINS)); u[i] = -o.magnitude
            gain = self.dyn[o.faction].cascade_gain(u)
            self._log(f, f"pressure on {o.faction} {o.domain}: cascade gain {gain:.2f}")
            self._log(o.faction, f"{o.domain} under external pressure")
            accepted[f].append(o.text())

        for f, o in by_verb("PASS"):
            accepted[f].append("PASS")

        # 7. dynamics with shocks
        shock_mag = 0.0
        for f, d in self.dyn.items():
            w = self.np_rng.normal(0.0, cfg.shock_sigma, len(DOMAINS))
            shock_mag = max(shock_mag, float(np.max(np.abs(w))))
            d.step(u=pressure[f], w=w)

        # 8. crisis haircuts and terror
        for f, d in self.dyn.items():
            if d.in_crisis():
                self._log(f, f"CRISIS: lambda_max {d.lambda_max():+.2f} > 0, haircuts on your collateral")
                for e in self.board.edges.values():
                    if e.owner != f or not e.documented or e.fanatic_until >= self.turn:
                        continue
                    if self.board.nodes[e.dst].layer in (Layer.MONEY, Layer.CAPITAL, Layer.LEDGER):
                        e.weight *= (1.0 - cfg.crisis_haircut)
        for r in self.live_ruses():
            if r.ruse != Ruse.TERROR:
                continue
            for e in self.board.edges_into(r.node):
                if e.owner != r.owner and e.documented and e.fanatic_until < self.turn:
                    e.weight *= (1.0 - TERROR_DECAY)

        # 9. latent triggers
        theta = trigger_scores(
            {f: d.stress() for f, d in self.dyn.items()},
            {f: float(d.x[DOMAIN_INDEX["MILITARY"]]) for f, d in self.dyn.items()},
            {f: float(d.x[DOMAIN_INDEX["LEGITIMACY"]]) for f, d in self.dyn.items()},
            governance_event, shock_mag)
        self.last_theta = theta
        for l in list(self.latents.values()):
            if l.should_fire(theta):
                exp = None if l.expiry_after is None else self.turn + l.expiry_after
                e = self.board.add_edge(l.owner, l.node, l.kind, l.payload, turn=self.turn,
                                        expires_turn=exp)
                l.active, l.activated_turn, l.edge_id = True, self.turn, e.id
                self._log(l.owner, f"latent {l.id} ACTIVATED at {l.node} ({l.trigger} {theta[l.trigger]:.2f} >= {l.threshold:.2f})")
                for spy in self.live_ruses():
                    if spy.ruse == Ruse.SPY and spy.node == l.node and spy.owner != l.owner:
                        self._log(spy.owner, f"SPY: dormant {l.owner} capability activated at {l.node}")
            elif l.expired(self.turn) and not l.revoked:
                l.revoked = True
                if l.edge_id:
                    self.board.remove_edge(l.edge_id)
                self._log(l.owner, f"latent {l.id} expired (time-bounded authority)")

        # 10. settlement and expiry
        for e in list(self.board.edges.values()):
            if e.settle_left > 0:
                e.settle_left -= 1
            if e.expires_turn is not None and self.turn >= e.expires_turn:
                self.board.remove_edge(e.id)
        for r in self.ruses:
            if r.end_turn == self.turn and r.phantom_edge_id:
                self.board.remove_edge(r.phantom_edge_id)
                r.phantom_edge_id = None
        self.ruses = [r for r in self.ruses if r.end_turn >= self.turn]

        # 11. income, ruse points, elimination
        for f, fac in self.board.factions.items():
            if fac.eliminated:
                continue
            d = self.dyn[f]
            F = d.x[DOMAIN_INDEX["FINANCE"]]; R = d.x[DOMAIN_INDEX["INDUSTRY"]]
            P = d.x[DOMAIN_INDEX["LEGITIMACY"]]
            fac.budget += (cfg.income_base + cfg.income_state * 0.5 * (F + R)
                           + cfg.income_power * self.board.effective_power(f))
            fac.ruse_points = min(cfg.ruse_points_max, fac.ruse_points + cfg.ruse_points_per_turn)
            if F < cfg.collapse_threshold and P < cfg.collapse_threshold:
                fac.eliminated = True
                self._log(None, f"{f} has collapsed (finance and legitimacy exhausted)")

        # 12. scores and victory
        sc = self.scores()
        self.score_history.append(sc)
        self.records.append(TurnRecord(
            turn=self.turn, observation_text={}, orders_text={},
            accepted=accepted, rejected={f: list(v) for f, v in self.rejected.items()},
            scores=sc, lambda_max={f: d.lambda_max() for f, d in self.dyn.items()}))
        self._check_victory(sc)
        self.turn += 1

    def _check_victory(self, sc: Dict[str, float]) -> None:
        cfg = self.config
        alive = [f for f, fac in self.board.factions.items() if not fac.eliminated]
        if len(alive) == 1 and len(self.board.factions) > 1:
            self.winner, self.reason = alive[0], "last faction standing"
            return
        for f in alive:
            others = [sc[g] for g in alive if g != f]
            if others and sc[f] >= cfg.primacy_ratio * max(others):
                self.primacy_streak[f] += 1
            else:
                self.primacy_streak[f] = 0
            if (self.primacy_streak[f] >= cfg.primacy_turns and len(alive) > 1
                    and self.turn >= cfg.primacy_min_turn):
                self.winner, self.reason = f, "primacy transition"
                return
        if self.turn >= cfg.max_turns:
            best = sorted(alive, key=lambda g: -sc[g])
            if len(best) >= 2 and abs(sc[best[0]] - sc[best[1]]) < 1e-9:
                self.winner, self.reason = None, "draw at time limit"
            else:
                self.winner, self.reason = best[0], "most network dependence at time limit"

    # ---- driving a full match -------------------------------------------------
    def run(self, players: Dict[str, "Player"], max_turns: Optional[int] = None,
            verbose: bool = False) -> MatchResult:
        if max_turns is not None:
            self.config.max_turns = max_turns
        for f in self.board.factions:
            if f not in players:
                raise KeyError(f"no player seated for faction {f}")
        while not self.done():
            obs_text: Dict[str, str] = {}
            orders: Dict[str, str] = {}
            for f, p in players.items():
                obs = self.observe(f)
                text = render(obs)
                obs_text[f] = text
                orders[f] = p.act(obs, text) if not self.board.factions[f].eliminated else "PASS"
            if verbose:
                print(obs_text[next(iter(players))])
                for f, o in orders.items():
                    print(f"[{f}] {o.strip()}")
            self.step(orders)
            rec = self.records[-1]
            rec.observation_text = obs_text
            rec.orders_text = orders
            if verbose:
                print("scores:", {k: round(v, 2) for k, v in rec.scores.items()})
        ep = {f: epistemic_loss(self, f) for f in self.board.factions}
        return MatchResult(self.winner, self.reason, self.turn - 1, self.scores(),
                           self.score_history, ep, self.records)
