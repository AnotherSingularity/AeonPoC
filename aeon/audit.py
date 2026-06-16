"""aeon/audit.py — re-exports the certificate audit utilities.

The canonical definitions live in aeon/recursion.py. This module exists so
callers can `from aeon.audit import audit_certificates` without reaching into
the recursion module directly.
"""
from .recursion import (
    audit_certificates,
    equivalence_check,
    sigma_max,
    project_sigma_,
    cayley,
    RecursionChartA,
    RecursionChartB,
)


def per_layer_gates(model):
    """Signed per-block gate values, one per transformer block, as a list."""
    return [blk.recursion_gate.item() for blk in model.model.layers]


def gate_stdev_across_layers(model):
    """Population stdev of the per-layer gate *magnitudes*.

    Stage 2 Bar 3 (functional differentiation): if the recursion finds
    specialized per-layer roles, the gate magnitudes spread out rather than
    clustering at one value. Target by end of training: > 0.02.
    """
    mags = [abs(g) for g in per_layer_gates(model)]
    if not mags:
        return 0.0
    mean = sum(mags) / len(mags)
    var = sum((m - mean) ** 2 for m in mags) / len(mags)
    return var ** 0.5


def gate_summary(model):
    """Convenience: per-layer gates plus mean/stdev of magnitudes."""
    gates = per_layer_gates(model)
    mags = [abs(g) for g in gates]
    mean_abs = sum(mags) / len(mags) if mags else 0.0
    return {
        "gamma_per_layer": gates,
        "mean_abs": mean_abs,
        "stdev_abs": gate_stdev_across_layers(model),
    }


__all__ = [
    "audit_certificates",
    "equivalence_check",
    "sigma_max",
    "project_sigma_",
    "cayley",
    "RecursionChartA",
    "RecursionChartB",
    "per_layer_gates",
    "gate_stdev_across_layers",
    "gate_summary",
]

