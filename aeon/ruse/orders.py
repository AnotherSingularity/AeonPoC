"""
aeon/ruse/orders.py — the order language players (and Aeon) speak.

One order per line, case-insensitive, extra text ignored:

    BUILD <kind> <node> <weight> [VIA <node>]   add a control edge
    RUSE <ruse> <node>                          play a deception card
    PRESSURE <faction> <domain> <magnitude>     push an opponent's state
    HARDEN <domain>                             restorative feedback + segmentation
    AUDIT <node>                                document every edge into a node
    REVOKE <edge_id>                            remove an edge (own or audited)
    PLANT <kind> <node> <weight> <trigger> <threshold>   dormant capability
    SUNSET <latent_id>                          retire a latent you own
    REDUNDANCY <node>                           add a substitute route
    PASS

The parser is deliberately forgiving so a small language model's reply
(markdown fences, bullets, prose) still yields whatever valid orders it
contains. Unparseable lines are returned as errors, never raised.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .board import EdgeKind
from .dynamics import parse_domain
from .latent import TRIGGERS
from .ruses import Ruse, parse_ruse


@dataclass
class Order:
    verb: str
    kind: Optional[EdgeKind] = None
    node: Optional[str] = None
    via: Optional[str] = None
    weight: Optional[float] = None
    ruse: Optional[Ruse] = None
    faction: Optional[str] = None
    domain: Optional[str] = None
    magnitude: Optional[float] = None
    edge_id: Optional[str] = None
    latent_id: Optional[str] = None
    trigger: Optional[str] = None
    threshold: Optional[float] = None
    raw: str = ""

    def text(self) -> str:
        v = self.verb
        if v == "BUILD":
            s = f"BUILD {self.kind.value} {self.node} {self.weight:.2f}"
            return s + (f" VIA {self.via}" if self.via else "")
        if v == "RUSE":
            return f"RUSE {self.ruse.value} {self.node}"
        if v == "PRESSURE":
            return f"PRESSURE {self.faction} {self.domain} {self.magnitude:.2f}"
        if v == "HARDEN":
            return f"HARDEN {self.domain}"
        if v == "AUDIT":
            return f"AUDIT {self.node}"
        if v == "REVOKE":
            return f"REVOKE {self.edge_id}"
        if v == "PLANT":
            return (f"PLANT {self.kind.value} {self.node} {self.weight:.2f} "
                    f"{self.trigger} {self.threshold:.2f}")
        if v == "SUNSET":
            return f"SUNSET {self.latent_id}"
        if v == "REDUNDANCY":
            return f"REDUNDANCY {self.node}"
        return "PASS"


@dataclass
class ParseResult:
    orders: List[Order] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


_KIND_ALIASES = {
    "SHARES": "ownership", "EQUITY": "ownership", "OWN": "ownership",
    "LOAN": "credit", "LEND": "credit", "FUNDING": "credit",
    "STATUTE": "law", "RULE": "law", "REGULATION": "law", "MANDATE": "law",
    "BOARD": "vote", "VOTES": "vote",
    "CLEARING": "custody", "SETTLEMENT": "custody",
    "ORDER": "command", "CHAIN": "command",
    "CREDENTIAL": "access", "LICENSE": "access", "DATA": "access",
    "DEAL": "contract", "COVENANT": "contract", "TREATY": "contract",
}


def parse_kind(tok: str) -> Optional[EdgeKind]:
    key = tok.strip().upper()
    key = _KIND_ALIASES.get(key, key.lower())
    try:
        return EdgeKind(key)
    except ValueError:
        return None


def _num(tok: str) -> Optional[float]:
    try:
        return float(tok.strip().rstrip("%,;"))
    except ValueError:
        return None


_LINE_CLEAN = re.compile(r"^[\s\-\*\d\.\)\]\[>`#]+")


def clean_line(line: str) -> str:
    line = line.strip()
    if line.startswith("```"):
        return ""
    line = _LINE_CLEAN.sub("", line)
    return line.strip().strip("`").strip()


def parse_orders(text: str, max_orders: int = 6) -> ParseResult:
    """Parse free text into orders. Never raises."""
    res = ParseResult()
    if "</think>" in text:
        text = text.split("</think>")[-1]
    for raw in text.splitlines():
        line = clean_line(raw)
        if not line:
            continue
        toks = line.replace(",", " ").split()
        verb = toks[0].upper().rstrip(":")
        try:
            order = _parse_tokens(verb, toks[1:], raw)
        except Exception as ex:  # defensive: parser must not kill the turn
            res.errors.append(f"{raw.strip()!r}: {ex}")
            continue
        if order is None:
            continue
        if isinstance(order, str):
            res.errors.append(f"{raw.strip()!r}: {order}")
            continue
        res.orders.append(order)
        if len(res.orders) >= max_orders:
            break
    return res


def _parse_tokens(verb: str, args: List[str], raw: str):
    up = [a.upper() for a in args]
    if verb == "PASS":
        return Order("PASS", raw=raw)
    if verb == "BUILD":
        if len(args) < 3:
            return "BUILD needs <kind> <node> <weight>"
        kind = parse_kind(args[0])
        if kind is None:
            return f"unknown edge kind {args[0]!r}"
        w = _num(args[2])
        if w is None:
            return f"bad weight {args[2]!r}"
        via = None
        if len(up) >= 5 and up[3] == "VIA":
            via = up[4]
        return Order("BUILD", kind=kind, node=up[1], weight=max(0.0, min(1.0, w)),
                     via=via, raw=raw)
    if verb == "RUSE":
        if len(args) < 2:
            return "RUSE needs <ruse> <node>"
        r = parse_ruse(args[0])
        if r is None:
            return f"unknown ruse {args[0]!r}"
        return Order("RUSE", ruse=r, node=up[1], raw=raw)
    if verb == "PRESSURE":
        if len(args) < 3:
            return "PRESSURE needs <faction> <domain> <magnitude>"
        dom = parse_domain(args[1])
        if dom is None:
            return f"unknown domain {args[1]!r}"
        m = _num(args[2])
        if m is None:
            return f"bad magnitude {args[2]!r}"
        return Order("PRESSURE", faction=up[0], domain=dom,
                     magnitude=max(0.0, min(0.5, m)), raw=raw)
    if verb == "HARDEN":
        if not args:
            return "HARDEN needs <domain>"
        dom = parse_domain(args[0])
        if dom is None:
            return f"unknown domain {args[0]!r}"
        return Order("HARDEN", domain=dom, raw=raw)
    if verb == "AUDIT":
        if not args:
            return "AUDIT needs <node>"
        return Order("AUDIT", node=up[0], raw=raw)
    if verb == "REVOKE":
        if not args:
            return "REVOKE needs <edge_id>"
        return Order("REVOKE", edge_id=up[0], raw=raw)
    if verb == "PLANT":
        if len(args) < 5:
            return "PLANT needs <kind> <node> <weight> <trigger> <threshold>"
        kind = parse_kind(args[0])
        if kind is None:
            return f"unknown edge kind {args[0]!r}"
        w = _num(args[2])
        trig = up[3]
        tau = _num(args[4])
        if w is None or tau is None:
            return "bad weight/threshold"
        if trig not in TRIGGERS:
            return f"unknown trigger {args[3]!r}; use one of {TRIGGERS}"
        return Order("PLANT", kind=kind, node=up[1], weight=max(0.0, min(1.0, w)),
                     trigger=trig, threshold=max(0.0, min(1.0, tau)), raw=raw)
    if verb == "SUNSET":
        if not args:
            return "SUNSET needs <latent_id>"
        return Order("SUNSET", latent_id=up[0], raw=raw)
    if verb == "REDUNDANCY":
        if not args:
            return "REDUNDANCY needs <node>"
        return Order("REDUNDANCY", node=up[0], raw=raw)
    return None  # not an order line; ignore silently


ORDER_HELP = """\
Orders (one per line):
  BUILD <kind> <node> <weight> [VIA <node>]  kinds: vote contract law ownership custody credit command access
  RUSE <ruse> <node>       ruses: DECOY RADIO_SILENCE CAMOUFLAGE SPY REVERSE_INTEL DECRYPTION BLITZ FANATICISM TERROR
  PRESSURE <faction> <domain> <magnitude 0-0.5>   domains: ENERGY INFO FINANCE LOGISTICS COMMS MILITARY LEGITIMACY INDUSTRY
  HARDEN <domain>          AUDIT <node>          REVOKE <edge_id>
  PLANT <kind> <node> <weight> <LIQUIDITY|SECURITY|LEGITIMACY|GOVERNANCE|TECHNICAL> <threshold>
  SUNSET <latent_id>       REDUNDANCY <node>     PASS"""
