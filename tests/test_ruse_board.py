"""tests/test_ruse_board.py — control math on the multiplex board."""
import os, sys
import math
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.ruse.board import (Board, EdgeKind, Faction, Layer, Node,
                             default_board, DEFAULT_FACTIONS)


def tiny():
    b = Board()
    b.add_node(Node("A", "Court", Layer.LAW, 0.8, 0))
    b.add_node(Node("B", "Reserve", Layer.MONEY, 0.9, 0))
    b.add_node(Node("C", "Cloud", Layer.INFORMATION, 0.5, 1))
    b.add_faction(Faction("X", "X", "x"))
    b.add_faction(Faction("Y", "Y", "y"))
    return b


def test_single_edge_control_is_weight():
    b = tiny()
    b.add_edge("X", "A", EdgeKind.LAW, 0.4)
    assert b.control("X")["A"] == pytest.approx(0.4)
    assert b.control("Y")["A"] == 0.0


def test_independent_paths_combine():
    b = tiny()
    b.add_edge("X", "B", EdgeKind.CREDIT, 0.5)
    b.add_edge("X", "B", EdgeKind.ACCESS, 0.5)
    # 1 - (1-0.5)(1-0.5)
    assert b.control("X")["B"] == pytest.approx(0.75)


def test_stacked_path_multiplies():
    b = tiny()
    b.add_edge("X", "A", EdgeKind.LAW, 0.5)
    b.add_edge("X", "B", EdgeKind.ACCESS, 0.8, src="A")   # via the court
    assert b.control("X")["B"] == pytest.approx(0.4)
    # no control over the via node -> nothing flows
    b2 = tiny()
    b2.add_edge("X", "B", EdgeKind.ACCESS, 0.8, src="A")
    assert b2.control("X")["B"] == 0.0


def test_phantom_edges_exert_no_control():
    b = tiny()
    b.add_edge("X", "A", EdgeKind.LAW, 0.9, documented=False)
    assert b.control("X")["A"] == 0.0
    assert b.dependence("X") == 0.0


def test_wrong_kind_for_layer_rejected():
    b = tiny()
    with pytest.raises(ValueError):
        b.add_edge("X", "A", EdgeKind.CREDIT, 0.5)   # credit into a court


def test_settlement_ramp():
    b = tiny()
    e = b.add_edge("X", "A", EdgeKind.LAW, 0.6, settle=2)
    assert e.effective_weight() == 0.0
    e.settle_left = 1
    assert e.effective_weight() == pytest.approx(0.3)
    e.settle_left = 0
    assert e.effective_weight() == pytest.approx(0.6)


def test_broad_control_is_geometric():
    b = tiny()
    b.add_edge("X", "A", EdgeKind.LAW, 0.9)
    b.add_edge("X", "B", EdgeKind.CREDIT, 0.9)
    # zero in the information layer drags the geometric mean far below 0.9
    assert b.broad_control("X") < 0.4
    b.add_edge("X", "C", EdgeKind.ACCESS, 0.9)
    assert b.broad_control("X") > 0.8


def test_concentration_index_bounds():
    b = tiny()
    assert b.concentration_index() == 0.0
    b.add_edge("X", "A", EdgeKind.LAW, 0.9)
    assert b.concentration_index() == pytest.approx(1.0)
    b.add_edge("Y", "B", EdgeKind.CREDIT, 0.9)
    assert 0.0 <= b.concentration_index() < 0.2


def test_effective_power_rewards_stacking_seams():
    b1 = tiny(); b2 = tiny()
    b1.add_edge("X", "A", EdgeKind.LAW, 0.8)
    b1.add_edge("X", "B", EdgeKind.CREDIT, 0.8)          # money + law seam
    b2.add_edge("X", "A", EdgeKind.LAW, 0.8)
    b2.add_edge("X", "C", EdgeKind.ACCESS, 0.8)          # no seam, less indispensable
    assert b1.effective_power("X") > b2.effective_power("X")


def test_deletion_sensitivity_nonnegative_and_ordered():
    b = default_board()
    ds = {e.id: b.deletion_sensitivity(e.id) for e in b.edges.values()}
    assert all(v >= -1e-12 for v in ds.values())
    assert max(ds.values()) > 0.01


def test_redundancy_dilutes_dependence():
    b = tiny()
    b.add_edge("X", "B", EdgeKind.CREDIT, 0.5)
    before = b.dependence("X")
    b.nodes["B"].redundancy += 1
    assert b.dependence("X") == pytest.approx(before / 2)


def test_default_board_all_archetypes_seat():
    b = default_board(list(DEFAULT_FACTIONS))
    assert set(b.factions) == set(DEFAULT_FACTIONS)
    sb = b.scoreboard()
    assert all(v["dependence"] > 0 for v in sb.values())
    with pytest.raises(KeyError):
        default_board(["NOT_A_FACTION"])
