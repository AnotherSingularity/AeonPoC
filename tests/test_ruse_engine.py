"""tests/test_ruse_engine.py — dynamics, ruses, latents, and match resolution."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.ruse import (Match, RuseConfig, default_board, HeuristicPlayer,
                       RandomPlayer, parse_orders, epistemic_loss)
from aeon.ruse.board import EdgeKind, Layer
from aeon.ruse.dynamics import DOMAIN_INDEX, DOMAINS, FactionDynamics, default_dynamics
from aeon.ruse.latent import trigger_scores
from aeon.ruse.observer import observe


# ---- dynamics ---------------------------------------------------------------

def test_stress_raises_lambda_max_and_harden_lowers_it():
    d = default_dynamics("CAPITAL")
    calm = d.lambda_max(theta=0.0)
    hot = d.lambda_max(theta=0.9)
    assert hot > calm
    assert hot > 0.0          # deep stress is a crisis regime
    for dom in DOMAINS:
        d.harden(dom)
    assert d.lambda_max(theta=0.9) < hot
    for dom in DOMAINS:
        d.harden(dom); d.harden(dom)
    assert d.lambda_max(theta=0.9) < 0.0   # purification: lambda_max > 0 -> < 0


def test_cascade_gain_grows_with_stress_and_shrinks_with_segmentation():
    u = np.zeros(len(DOMAINS)); u[DOMAIN_INDEX["COMMS"]] = -0.3
    calm = FactionDynamics.from_values([0.9] * 8)
    stressed = FactionDynamics.from_values([0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.2, 0.9])
    assert stressed.cascade_gain(u) > calm.cascade_gain(u)
    hardened = FactionDynamics.from_values([0.9, 0.9, 0.2, 0.9, 0.9, 0.9, 0.2, 0.9])
    for dom in DOMAINS:
        hardened.harden(dom)
    assert hardened.cascade_gain(u) < stressed.cascade_gain(u)


def test_step_is_clipped_and_pulls_toward_nominal():
    d = FactionDynamics.from_values([0.0] * 8, nominal=0.6)
    d.step()
    assert np.all(d.x >= 0.0) and np.all(d.x <= 1.0)
    # a collapsed state is absorbing: coupling beats mean reversion at x = 0
    assert d.lambda_max() > 0.0
    # with restorative feedback the restoring term lifts a mild deficit
    d = FactionDynamics.from_values([0.4] * 8, nominal=0.6)
    for dom in DOMAINS:
        d.harden(dom); d.harden(dom)
    before = d.x.copy()
    d.step()
    assert d.lambda_max() < 0.0
    assert np.mean(d.x) > np.mean(before)


# ---- orders in the engine ----------------------------------------------------

def test_build_charges_budget_and_settles():
    m = Match(default_board(), seed=0)
    b0 = m.board.factions["CAPITAL"].budget
    m.step({"CAPITAL": "BUILD law COURT 0.5", "INDUSTRIAL": "PASS"})
    assert m.board.factions["CAPITAL"].budget < b0 + 30   # income arrived, cost taken
    new = [e for e in m.board.edges_into("COURT", "CAPITAL") if e.weight == 0.5]
    assert len(new) == 1
    assert new[0].settle_left == 0   # one settle turn already elapsed


def test_rejected_orders_are_reported_not_raised():
    m = Match(default_board(), seed=0)
    m.step({"CAPITAL": "BUILD credit COURT 0.5\nREVOKE E999\nPRESSURE CAPITAL FINANCE 0.2",
            "INDUSTRIAL": "BUILD command COUNCIL 1.0\nBUILD command STAFF 1.0"})
    rej = m.rejected["CAPITAL"]
    assert any("not valid" in r for r in rej)
    assert any("no such edge" in r for r in rej)
    assert any("bad target" in r for r in rej)
    # the second full-weight build exceeds the opening budget
    assert any("needs" in r for r in m.rejected["INDUSTRIAL"])


def test_blitz_settles_immediately():
    m = Match(default_board(), seed=0, config=RuseConfig(settle_turns=3))
    m.step({"CAPITAL": "RUSE BLITZ COURT\nBUILD law COURT 0.5", "INDUSTRIAL": "PASS"})
    e = [e for e in m.board.edges_into("COURT", "CAPITAL") if e.weight == 0.5][0]
    assert e.settle_total == 0
    m2 = Match(default_board(), seed=0, config=RuseConfig(settle_turns=3))
    m2.step({"CAPITAL": "BUILD law COURT 0.5", "INDUSTRIAL": "PASS"})
    e2 = [e for e in m2.board.edges_into("COURT", "CAPITAL") if e.weight == 0.5][0]
    assert e2.settle_total == 3 and e2.settle_left == 2


def test_pressure_moves_target_and_backlashes():
    m = Match(default_board(), seed=0, config=RuseConfig(shock_sigma=0.0))
    f0 = m.dyn["INDUSTRIAL"].x[DOMAIN_INDEX["FINANCE"]]
    s0 = m.dyn["CAPITAL"].x[DOMAIN_INDEX["FINANCE"]]
    ref = Match(default_board(), seed=0, config=RuseConfig(shock_sigma=0.0))
    ref.step({"CAPITAL": "PASS", "INDUSTRIAL": "PASS"})
    m.step({"CAPITAL": "PRESSURE INDUSTRIAL FINANCE 0.4", "INDUSTRIAL": "PASS"})
    assert m.dyn["INDUSTRIAL"].x[DOMAIN_INDEX["FINANCE"]] < ref.dyn["INDUSTRIAL"].x[DOMAIN_INDEX["FINANCE"]]
    assert m.dyn["CAPITAL"].x[DOMAIN_INDEX["FINANCE"]] < ref.dyn["CAPITAL"].x[DOMAIN_INDEX["FINANCE"]]


# ---- the information layer -----------------------------------------------------

def test_decoy_seen_by_opponent_not_counted_in_truth():
    m = Match(default_board(), seed=0)
    m.step({"CAPITAL": "RUSE DECOY TREASURY", "INDUSTRIAL": "PASS"})
    truth = m.board.control("CAPITAL")["TREASURY"]
    assert truth == 0.0
    obs = observe(m, "INDUSTRIAL")
    node = next(n for n in obs.nodes if n.id == "TREASURY")
    phantom = [e for e in node.edges if e.owner == "CAPITAL"]
    assert len(phantom) == 1 and not phantom[0].exposed_decoy
    assert node.control_est["CAPITAL"] > 0.0          # the opponent is fooled
    own = observe(m, "CAPITAL")
    own_node = next(n for n in own.nodes if n.id == "TREASURY")
    assert [e for e in own_node.edges if e.owner == "CAPITAL"][0].exposed_decoy   # the owner knows
    loss = epistemic_loss(m, "INDUSTRIAL")
    assert loss["unsupported_links"] == 1


def test_camouflage_hides_and_spy_reveals():
    m = Match(default_board(), seed=0)
    m.step({"CAPITAL": "RUSE CAMOUFLAGE CLEAR", "INDUSTRIAL": "PASS"})
    obs = observe(m, "INDUSTRIAL")
    node = next(n for n in obs.nodes if n.id == "CLEAR")
    assert not [e for e in node.edges if e.owner == "CAPITAL"]
    m.step({"CAPITAL": "PASS", "INDUSTRIAL": "RUSE SPY CLEAR"})
    obs = observe(m, "INDUSTRIAL")
    node = next(n for n in obs.nodes if n.id == "CLEAR")
    cap = [e for e in node.edges if e.owner == "CAPITAL"]
    assert cap and not cap[0].estimated                   # true weight under spy


def test_reverse_intel_flags_real_edges():
    m = Match(default_board(), seed=0)
    m.step({"CAPITAL": "RUSE REVERSE_INTEL CLEAR", "INDUSTRIAL": "PASS"})
    obs = observe(m, "INDUSTRIAL")
    node = next(n for n in obs.nodes if n.id == "CLEAR")
    assert all(e.flagged for e in node.edges if e.owner == "CAPITAL")
    assert node.control_est["CAPITAL"] == 0.0            # believed unsupported


def test_radio_silence_serves_stale_snapshot():
    m = Match(default_board(), seed=0)
    observe(m, "INDUSTRIAL")                              # take a snapshot
    m.step({"CAPITAL": "RUSE RADIO_SILENCE RESERVE\nRUSE BLITZ RESERVE\nBUILD credit RESERVE 0.5",
            "INDUSTRIAL": "PASS"})
    obs = observe(m, "INDUSTRIAL")
    node = next(n for n in obs.nodes if n.id == "RESERVE")
    assert node.stale
    assert not [e for e in node.edges if e.weight == 0.5]   # new edge not visible
    assert len(m.board.edges_into("RESERVE", "CAPITAL")) == 2


def test_decryption_intercepts_orders():
    m = Match(default_board(), seed=0)
    m.step({"CAPITAL": "RUSE DECRYPTION COURT", "INDUSTRIAL": "PASS"})
    m.step({"CAPITAL": "PASS", "INDUSTRIAL": "BUILD law COURT 0.3"})
    assert any("BUILD law COURT" in s for s in m.intercepted.get("CAPITAL", []))


def test_audit_documents_and_enables_revocation():
    m = Match(default_board(), seed=0, config=RuseConfig(shock_sigma=0.0))
    # INDUSTRIAL holds TREASURY (law 0.5). CAPITAL cannot revoke without audit.
    tre = m.board.edges_into("TREASURY", "INDUSTRIAL")[0].id
    m.step({"CAPITAL": f"REVOKE {tre}", "INDUSTRIAL": "PASS"})
    assert any("not audited" in r for r in m.rejected["CAPITAL"])
    m.step({"CAPITAL": "AUDIT TREASURY", "INDUSTRIAL": "PASS"})
    assert m.board.edges[tre].audited
    # still out-controlled at the node -> rejected with a reason
    m.step({"CAPITAL": f"REVOKE {tre}", "INDUSTRIAL": "PASS"})
    assert any("out-controls" in r for r in m.rejected["CAPITAL"])
    # owner can always revoke its own edge
    m.step({"CAPITAL": "PASS", "INDUSTRIAL": f"REVOKE {tre}"})
    assert tre not in m.board.edges


def test_latent_fires_on_trigger_and_expires():
    cfg = RuseConfig(shock_sigma=0.0, latent_expiry=2)
    m = Match(default_board(), seed=0, config=cfg)
    m.step({"CAPITAL": "PLANT law COURT 0.5 GOVERNANCE 0.5", "INDUSTRIAL": "PASS"})
    l = m.latents["L1"]
    assert not l.active
    # a governance event (an audit) crosses the trigger
    m.step({"CAPITAL": "PASS", "INDUSTRIAL": "AUDIT IRON"})
    assert l.active and l.edge_id in m.board.edges
    assert m.board.control("CAPITAL")["COURT"] > 0.2
    m.step({"CAPITAL": "PASS", "INDUSTRIAL": "PASS"})
    m.step({"CAPITAL": "PASS", "INDUSTRIAL": "PASS"})
    assert l.revoked and l.edge_id not in m.board.edges   # time-bounded authority


def test_latent_visible_only_after_audit_or_spy():
    m = Match(default_board(), seed=0)
    m.step({"CAPITAL": "PLANT law COURT 0.5 LIQUIDITY 0.99", "INDUSTRIAL": "PASS"})
    assert not observe(m, "INDUSTRIAL").known_latents
    assert observe(m, "CAPITAL").known_latents
    m.step({"CAPITAL": "PASS", "INDUSTRIAL": "AUDIT COURT"})
    assert observe(m, "INDUSTRIAL").known_latents


def test_trigger_scores_shape():
    th = trigger_scores({"a": 0.2, "b": 0.4}, {"a": 0.5, "b": 0.5}, {"a": 1.0, "b": 0.0}, True, 0.05)
    assert th["LIQUIDITY"] == pytest.approx(0.3)
    assert th["SECURITY"] == pytest.approx(0.5)
    assert th["LEGITIMACY"] == pytest.approx(0.5)
    assert th["GOVERNANCE"] == 1.0 and th["TECHNICAL"] == pytest.approx(0.5)


# ---- full matches ----------------------------------------------------------------

def test_match_is_deterministic_for_a_seed():
    def play():
        m = Match(default_board(), seed=7)
        return m.run({"CAPITAL": HeuristicPlayer("CAPITAL", seed=1),
                      "INDUSTRIAL": RandomPlayer("INDUSTRIAL", seed=2)}, max_turns=8)
    a, b = play(), play()
    assert a.final_scores == b.final_scores and a.winner == b.winner


def test_match_ends_with_result_and_records():
    m = Match(default_board(), seed=3)
    res = m.run({"CAPITAL": HeuristicPlayer("CAPITAL", seed=1),
                 "INDUSTRIAL": HeuristicPlayer("INDUSTRIAL", seed=2)}, max_turns=6)
    assert res.turns == 6 and res.reason
    assert len(res.records) == 6
    assert set(res.records[0].observation_text) == {"CAPITAL", "INDUSTRIAL"}
    assert all(f in res.epistemic for f in ("CAPITAL", "INDUSTRIAL"))
    assert "winner" in res.summary()


def test_primacy_transition_needs_min_turn_and_streak():
    cfg = RuseConfig(primacy_ratio=1.0, primacy_turns=2, primacy_min_turn=4)
    m = Match(default_board(), seed=0, config=cfg)
    m.run({"CAPITAL": HeuristicPlayer("CAPITAL", seed=1),
           "INDUSTRIAL": RandomPlayer("INDUSTRIAL", seed=2)}, max_turns=12)
    assert m.reason == "primacy transition"
    assert m.turn - 1 >= 4


def test_five_faction_match_runs():
    from aeon.ruse.board import DEFAULT_FACTIONS
    fs = list(DEFAULT_FACTIONS)
    m = Match(default_board(fs), seed=5)
    res = m.run({f: RandomPlayer(f, seed=i) for i, f in enumerate(fs)}, max_turns=5)
    assert len(res.final_scores) == 5
