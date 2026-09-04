# Aeon inside R.U.S.E.

`aeon/ruse` puts Aeon at the table of a deception strategy game whose map is
the layer-by-layer systems map of institutional power. R.U.S.E. supplies the
interface: a war table of counters that shows what each player *believes*,
sectors on which ruses act, and a winner decided by position rather than by
who has the biggest pile. The power-topology model supplies the board, the
units, the dynamics, and the victory condition.

Everything here is a fictional strategy game on abstract state vectors. The
factions are scenario archetypes, the nodes are invented institutions, and
"pressure" is a number subtracted from a number.

## The board

| Systems-map layer | Board row | Example sectors |
|---|---|---|
| L1 Physical & productive capacity | `PHYSICAL` | Ironworks Basin, Grid Authority, Port Concession |
| L2 Ownership & governance | `OWNERSHIP` | Holding Trust, Foundation Board |
| L3 Capital & finance | `CAPITAL` | Bond Desk, Underwriting Syndicate |
| L4 Money, credit & liquidity | `MONEY` | Reserve Window, Repo Market |
| L5 Ledger, custody & settlement | `LEDGER` | Clearing House, Custody Depository |
| L6 Mandate, law & regulation | `LAW` | High Court, Sanctions Office, Licensing Bureau |
| L7 Sovereign & executive authority | `EXECUTIVE` | Executive Council |
| L8 Fiscal capacity | `FISCAL` | Treasury |
| L9 Defense, intelligence & coercion | `DEFENSE` | General Staff, Signals Directorate |
| L10 Information, compute & legitimacy | `INFORMATION` | Cloud Exchange, Market Data Feed, Broadcast Network |
| L11 External networks & alliances | `EXTERNAL` | Alliance Council, Trade Compact |
| L12 Crisis optionality | engine rule | crisis multiplier on liquidity, law, command, settlement nodes |
| L13 Institutional persistence | engine rule | edges outlive turns; latents outlive their trigger |

Each node has an **indispensability** (how costly it is to route around) and
a **redundancy** count (substitute routes that dilute everyone's leverage
over it).

**Units are control edges.** A faction reaches a node through one auditable
edge kind: vote, contract, law, ownership, custody, credit, command, or
access. The kind must fit the layer: a share certificate does not command an
army, so `command` is rejected in the ownership row. Edges can be stacked
(`BUILD ... VIA <node>`): control of the court can be used to reach the
licensing bureau, with path strength multiplied along the way.

## The math the engine runs

From the dynamical power-topology model:

* Path strength `kappa_p = prod w_e`; independent paths combine as
  `C(f -> v) = 1 - prod(1 - kappa_p)`. Only *documented* edges count.
  Association is not control.
* Broad control `A_f = (prod_k (g_k + eps)^lambda_k)^(1/sum lambda_k)` is
  geometric: zero in one indispensable layer drags the whole thing down.
* Stacked leverage `P_eff = sigmoid(b0 + b.g + g^T Q g)` with reinforcing
  seams in `Q`: money+law, force+alliances, compute+capital,
  settlement+liquidity, votes+capital, executive+fiscal.
* Concentration index `B = 1 - H_c` over faction control masses.
* Deletion sensitivity `Delta_e = P_eff(G) - P_eff(G \ e)` (a counterfactual
  analytical removal, exposed for players that want to know which edge
  matters most).
* Network dependence `Dep(j -> f) = ind_j * C(f -> j) / (1 + redundancy_j)`.

From the fictional multipolar systems model:

* Faction state `x = [E, I, F, L, C, M, P, R]` with
  `dx/dt = A x + B u + G w - H d`; the off-diagonal of `A` is the cascade
  chain (comms -> logistics -> industry -> finance -> legitimacy -> finance).
* Stress `theta = 1 - (F + P)/2` amplifies the coupling. When the local
  Jacobian has `max Re spec(J) > 0` the faction is in crisis: its collateral
  takes haircuts and liquidity/law/command nodes gain leverage.
* Cascade gain `||Delta x|| / ||u||` is reported to whoever applies pressure.
* The primacy-transition condition is the win condition:
  `sum_j Dep(j -> challenger) > sum_j Dep(j -> incumbent)`, by a margin,
  held for several turns, not before turn 6.

From the purification note: `HARDEN` adds restorative feedback `u = -K e`
to one domain and segments its inflow. Enough of it moves
`lambda_max > 0` to `lambda_max < 0`. It is not free; the disorder is
dissipated into the budget, not erased.

From the sleeper-agent analogy and the defensive-reversal model:

* `PLANT` creates a latent capability `z = (p, a, q, r, v)` with a trigger
  threshold. It fires when a public condition crosses it (liquidity stress,
  security posture, legitimacy deficit, a governance event, a technical
  shock), creating a real edge. Activation is time-bounded by default.
* Risk `R = alpha * p a q r (1 - v)` is what an auditor sees.
* The defensive sequence is the order set: `AUDIT` (identify the edge and
  document it publicly), `REVOKE` (only an audited edge, only where you
  out-control the owner and hold law-layer standing), `SUNSET` (retire your
  own latent), `REDUNDANCY` (add a substitute route).

From the observer layer: `Y = H(O) X + eps`. Each faction sees its own
edges, audited edges, and spied sectors truly; everything else is a noisy
estimate marked `~`. The epistemic loss
`||g_hat - g||^2 + lambda * (phantom edges believed)` is scored per faction
at the end of the match. Acting on a decoy costs twice.

## The ruses

| R.U.S.E. card | On this board |
|---|---|
| Decoy | a phantom edge opponents see as real (zero true control) |
| Radio silence | opponents keep seeing the sector as it was |
| Camouflage net | your edges into the sector vanish from their table |
| Spy | you see the sector truly: edges, decoys, latents |
| Reverse intelligence | your real edges look like unsupported links |
| Decryption | you intercept opponents' orders aimed at the sector |
| Blitz | your builds in the sector settle immediately |
| Fanaticism | your edges in the sector ignore haircuts and terror |
| Terror | opponents' edges in the sector decay each turn |

One ruse point accrues per turn, three can be banked.

## Turn structure

All factions act simultaneously. Resolution order:
ruses, audits, revocations, builds/plants/sunsets/redundancy, hardening,
pressure, dynamics step with shocks, crisis haircuts and terror, latent
triggers, settlement and expiry, income, scores, victory check.

Order language (see `aeon/ruse/orders.py`):

```
BUILD <kind> <node> <weight> [VIA <node>]
RUSE <ruse> <node>
PRESSURE <faction> <domain> <magnitude 0-0.5>
HARDEN <domain>
AUDIT <node>
REVOKE <edge_id>
PLANT <kind> <node> <weight> <trigger> <threshold>
SUNSET <latent_id>
REDUNDANCY <node>
PASS
```

## Aeon at the table

`AeonPlayer` wraps `AeonR1ForCausalLM`. Each turn it receives the rendered
war table as a user message, replies with orders, and the parser keeps
whatever valid lines it finds. The recursion state `(r, c)` is reset once
when the player is created and then carried across every turn of the match.
Contraction guarantees it cannot blow up over a long game; the game is
built so that it should matter: what the opponent showed you on turn 3 is
not on the table on turn 15, but it decides whether the edge you see now is
a decoy.

```
python scripts/play_ruse.py --p1 heuristic --p2 aeon --ckpt ./aeon_stage1 --seat2 INDUSTRIAL -v
```

`scripts/ruse_export.py` turns match transcripts into `{"text": ...}` rows
in the format `scripts/train_stage2.py` reads. One match is one long
multi-turn sequence, which is the length curriculum the Stage 2 notes call
for.

## Scope note

The archetypes (`CAPITAL`, `INDUSTRIAL`, `SECURITY`, `CONTINENTAL`,
`MOVEMENT`) are the modeling devices from the fictional multipolar model, not
claims about real governments. No real person, company, or event appears on
the board. The engine models abstract state vectors and graph edges; it
contains no operational content of any kind.
