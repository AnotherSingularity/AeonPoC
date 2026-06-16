"""tests/test_recursion.py — the recursion self-test from the spec, as pytest.

Mirrors aeon/recursion.py's __main__ self-test: atlas equivalence, the
certificate at init, and the certificate surviving training.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aeon.recursion import (
    RecursionChartB,
    audit_certificates,
    equivalence_check,
)


def test_atlas_equivalence():
    eq = equivalence_check()
    assert eq["passes"], f"ATLAS BROKEN: max_abs_diff={eq['max_abs_diff']:.2e}"


def test_certificate_at_init():
    cell = RecursionChartB(in_dim=3, H=64)
    aud = audit_certificates(cell)
    assert aud["chart_A_holds"], "Chart B init violates sigma bound"
    assert aud["chart_C_lyapunov_holds"], "Chart B init violates Lyapunov LMI"
    assert aud["chart_B_holds_under_N01"], "Chart B fails under N(0,1) perturbation"


def test_certificate_survives_training():
    torch.manual_seed(0)
    cell = RecursionChartB(in_dim=3, H=64)
    opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
    for _ in range(50):
        x = torch.randn(8, 30, 3)
        h = cell(x)
        loss = (h ** 2).mean() + h.std()
        opt.zero_grad(); loss.backward(); opt.step()
    aud = audit_certificates(cell)
    assert aud["chart_A_holds"], "Training broke sigma bound"
    assert aud["chart_C_lyapunov_holds"], "Training broke Lyapunov LMI"
