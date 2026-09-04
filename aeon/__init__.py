"""aeon — a small efficient language model with a contractive recurrent path.

The model classes need torch/transformers. They are imported lazily so the
pure-Python game package (`aeon.ruse`) can be used without them.
"""
import importlib

__all__ = [
    "AeonConfig",
    "AeonBlock",
    "AeonModel",
    "AeonR1ForCausalLM",
    "RecursionChartA",
    "RecursionChartB",
    "audit_certificates",
    "equivalence_check",
]

_LAZY = {
    "AeonConfig": ".config",
    "AeonBlock": ".block",
    "AeonModel": ".model",
    "AeonR1ForCausalLM": ".model",
    "RecursionChartA": ".recursion",
    "RecursionChartB": ".recursion",
    "audit_certificates": ".recursion",
    "equivalence_check": ".recursion",
}


def __getattr__(name):
    if name in _LAZY:
        mod = importlib.import_module(_LAZY[name], __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
