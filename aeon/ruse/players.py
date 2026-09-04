"""
aeon/ruse/players.py — who sits at the table.

    RandomPlayer     legal-ish random orders (baseline / fuzzing)
    HeuristicPlayer  greedy stacker with a defensive reflex (the sparring partner)
    LLMPlayer        any text generator: sees the war table, replies with orders
    AeonPlayer       LLMPlayer backed by AeonR1ForCausalLM

Aeon keeps its recursion state (r, c) across the whole match. The contractive
cell means the state cannot blow up no matter how long the game runs, and it
is exactly the kind of long-sequence, imperfect-information setting where a
persistent global state should earn its keep: what an opponent showed you on
turn 3 still matters on turn 15, and the war table on turn 15 alone does not
say whether it was a decoy.
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Protocol

from .board import EdgeKind, LAYER_KINDS, Layer
from .dynamics import DOMAINS
from .latent import TRIGGERS
from .observer import Observation
from .orders import ORDER_HELP, parse_orders
from .ruses import Ruse


class Player(Protocol):
    faction: str

    def act(self, obs: Observation, text: str) -> str: ...


# ---------------------------------------------------------------------------
# Scripted players
# ---------------------------------------------------------------------------

class RandomPlayer:
    def __init__(self, faction: str, seed: int = 0, n_orders: int = 3):
        self.faction = faction
        self.rng = random.Random(seed)
        self.n_orders = n_orders

    def act(self, obs: Observation, text: str) -> str:
        lines: List[str] = []
        nodes = obs.nodes
        others = [f for f in obs.factions if f != self.faction]
        for _ in range(self.n_orders):
            roll = self.rng.random()
            nv = self.rng.choice(nodes)
            kinds = sorted(LAYER_KINDS[nv.layer], key=lambda k: k.value)
            if roll < 0.45:
                lines.append(f"BUILD {self.rng.choice(kinds).value} {nv.id} {self.rng.uniform(0.2, 0.6):.2f}")
            elif roll < 0.65:
                lines.append(f"RUSE {self.rng.choice(list(Ruse)).value} {nv.id}")
            elif roll < 0.75 and others:
                lines.append(f"PRESSURE {self.rng.choice(others)} {self.rng.choice(DOMAINS)} {self.rng.uniform(0.1, 0.4):.2f}")
            elif roll < 0.82:
                lines.append(f"HARDEN {self.rng.choice(DOMAINS)}")
            elif roll < 0.88:
                lines.append(f"AUDIT {nv.id}")
            elif roll < 0.93:
                lines.append(f"PLANT {self.rng.choice(kinds).value} {nv.id} {self.rng.uniform(0.3, 0.6):.2f} "
                             f"{self.rng.choice(TRIGGERS)} {self.rng.uniform(0.2, 0.6):.2f}")
            else:
                lines.append(f"REDUNDANCY {nv.id}")
        return "\n".join(lines) or "PASS"


class HeuristicPlayer:
    """Greedy stacking on the highest-value gap, defensive reflex under stress,
    and a taste for deception around its own key sectors.

    Stacking: pick the indispensable node where (indispensable * (1 - C_self))
    is largest, preferring layers that complete a reinforcing seam.
    Defense: HARDEN the weakest domain whenever lambda_max > 0.
    Audit: when an opponent edge looks large on a node you care about,
    AUDIT it (association is not control until it is documented).
    """
    SEAMS = [(Layer.MONEY, Layer.LAW), (Layer.DEFENSE, Layer.EXTERNAL),
             (Layer.INFORMATION, Layer.CAPITAL), (Layer.LEDGER, Layer.MONEY),
             (Layer.OWNERSHIP, Layer.CAPITAL), (Layer.EXECUTIVE, Layer.FISCAL)]

    def __init__(self, faction: str, seed: int = 0, aggression: float = 0.3):
        self.faction = faction
        self.rng = random.Random(seed)
        self.aggression = aggression

    def act(self, obs: Observation, text: str) -> str:
        me = self.faction
        lines: List[str] = []
        budget = obs.budget
        others = [f for f in obs.factions if f != me]

        # layer control estimates
        layer_c: Dict[Layer, List[float]] = {}
        for nv in obs.nodes:
            layer_c.setdefault(nv.layer, []).append(nv.control_est.get(me, 0.0))
        layer_mean = {k: sum(v) / len(v) for k, v in layer_c.items()}

        def seam_bonus(layer: Layer) -> float:
            bonus = 0.0
            for a, b in self.SEAMS:
                if layer == a:
                    bonus += layer_mean.get(b, 0.0)
                elif layer == b:
                    bonus += layer_mean.get(a, 0.0)
            return bonus

        # 1. defensive reflex
        if obs.lambda_max > 0 and budget >= 8:
            weakest = min(("FINANCE", "LEGITIMACY", "LOGISTICS", "COMMS"), key=lambda d: obs.state[d])
            lines.append(f"HARDEN {weakest}")
            budget -= 8

        # 2. greedy stack
        ranked = sorted(obs.nodes, key=lambda nv: -(nv.indispensable * (1 - nv.control_est.get(me, 0.0))
                                                    * (1 + seam_bonus(nv.layer))))
        for nv in ranked[:3]:
            kinds = sorted(LAYER_KINDS[nv.layer], key=lambda k: k.value)
            w = 0.4
            cost = 10 * w * 1.6
            if budget >= cost:
                lines.append(f"BUILD {self.rng.choice(kinds).value} {nv.id} {w:.2f}")
                budget -= cost

        # 3. audit a suspicious foreign edge on a node we care about
        suspicious = []
        for nv in ranked[:6]:
            for ev in nv.edges:
                if ev.owner != me and ev.estimated and ev.weight >= 0.45 and not ev.audited:
                    suspicious.append(nv.id)
        if suspicious and budget >= 3:
            lines.append(f"AUDIT {suspicious[0]}")
            budget -= 3

        # 4. revoke an audited foreign edge where we dominate
        for nv in obs.nodes:
            for ev in nv.edges:
                if ev.owner != me and ev.audited and not ev.exposed_decoy:
                    if nv.control_est.get(me, 0.0) > nv.control_est.get(ev.owner, 0.0) and budget >= 6:
                        lines.append(f"REVOKE {ev.id}")
                        budget -= 6
                        break

        # 5. ruses: decoy or camouflage around our best sector, spy on theirs
        if obs.ruse_points > 0:
            mine = max(obs.nodes, key=lambda nv: nv.control_est.get(me, 0.0))
            theirs = max(obs.nodes, key=lambda nv: max((nv.control_est.get(o, 0.0) for o in others), default=0.0))
            roll = self.rng.random()
            if roll < 0.4:
                lines.append(f"RUSE DECOY {ranked[0].id}")
            elif roll < 0.7:
                lines.append(f"RUSE CAMOUFLAGE {mine.id}")
            else:
                lines.append(f"RUSE SPY {theirs.id}")

        # 6. occasional pressure on the leader
        if others and self.rng.random() < self.aggression and budget >= 4:
            target = max(others, key=lambda o: obs.scores_est.get(o, 0.0))
            lines.append(f"PRESSURE {target} FINANCE 0.25")

        return "\n".join(lines[:6]) or "PASS"


# ---------------------------------------------------------------------------
# Language-model players
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Aeon, playing R.U.S.E. on a board of institutional power. "
    "Each turn you see the war table as your faction believes it to be: "
    "control edges into sectors (nodes) across eleven layers, with estimated "
    "weights marked ~. Opponents can decoy, camouflage, and freeze what you "
    "see; you can do the same to them. Association is not control: only "
    "documented edges count. Win by making more of the network depend on "
    "you than on anyone else, stacking reinforcing layers (money+law, "
    "force+alliances, compute+capital, settlement+liquidity). Keep your own "
    "state stable: if lambda_max > 0 you are in crisis and should HARDEN.\n"
    + ORDER_HELP +
    "\nReply with orders only, one per line, at most six."
)

GenerateFn = Callable[[List[Dict[str, str]]], str]


class LLMPlayer:
    """Drives any chat-style generator. `generate(messages) -> text`."""

    def __init__(self, faction: str, generate: GenerateFn,
                 history_turns: int = 2, system_prompt: str = SYSTEM_PROMPT):
        self.faction = faction
        self.generate = generate
        self.history_turns = history_turns
        self.system_prompt = system_prompt
        self.history: List[Dict[str, str]] = []
        self.transcript: List[Dict[str, str]] = []   # full (obs, reply) log
        self.parse_errors: List[str] = []

    def messages(self, text: str) -> List[Dict[str, str]]:
        keep = self.history[-2 * self.history_turns:] if self.history_turns > 0 else []
        return ([{"role": "system", "content": self.system_prompt}] + keep
                + [{"role": "user", "content": text}])

    def act(self, obs: Observation, text: str) -> str:
        msgs = self.messages(text)
        try:
            reply = self.generate(msgs) or ""
        except Exception as ex:   # a generation failure is a PASS, not a crash
            reply = ""
            self.parse_errors.append(f"turn {obs.turn}: generate failed: {ex}")
        parsed = parse_orders(reply)
        self.parse_errors.extend(f"turn {obs.turn}: {e}" for e in parsed.errors)
        orders = "\n".join(o.text() for o in parsed.orders) or "PASS"
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": orders}]
        self.transcript.append({"turn": obs.turn, "observation": text,
                                "raw_reply": reply, "orders": orders})
        return orders


class AeonPlayer(LLMPlayer):
    """Aeon at the table. Loads the checkpoint lazily; the recursion state is
    reset once at construction and then carried across every turn."""

    def __init__(self, faction: str, ckpt: str, max_new_tokens: int = 120,
                 temperature: float = 0.6, device: Optional[str] = None,
                 history_turns: int = 2, dtype: str = "bfloat16"):
        import torch
        from transformers import AutoTokenizer
        from ..model import AeonR1ForCausalLM

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype)
        self.model = AeonR1ForCausalLM.from_pretrained(
            ckpt, torch_dtype=torch_dtype).to(self.device).eval()
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model.reset_recursion_state(batch_size=1)
        super().__init__(faction, self._generate, history_turns=history_turns)

    def _prompt(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tok.apply_chat_template(messages, tokenize=False,
                                                add_generation_prompt=True)
        except Exception:
            return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        torch = self.torch
        enc = self.tok(self._prompt(messages), return_tensors="pt")
        # only what the model consumes (some tokenizers also emit token_type_ids)
        ids = {k: v.to(self.device) for k, v in enc.items()
               if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=self.max_new_tokens,
                temperature=self.temperature, do_sample=self.temperature > 0,
                pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)

    def recursion_norms(self) -> Dict[str, float]:
        r, c = self.model.model.get_recursion_state()
        return {"r": float(r.float().norm()) if r is not None else 0.0,
                "c": float(c.float().norm()) if c is not None else 0.0}

    def audit(self) -> dict:
        return self.model.audit()


def make_player(spec: str, faction: str, seed: int = 0, ckpt: Optional[str] = None,
                **kw) -> Player:
    """'random' | 'heuristic' | 'aeon' (needs ckpt)."""
    spec = spec.lower()
    if spec == "random":
        return RandomPlayer(faction, seed=seed)
    if spec == "heuristic":
        return HeuristicPlayer(faction, seed=seed, **kw)
    if spec == "aeon":
        if not ckpt:
            raise ValueError("aeon player needs --ckpt")
        return AeonPlayer(faction, ckpt, **kw)
    raise ValueError(f"unknown player spec {spec!r}")
