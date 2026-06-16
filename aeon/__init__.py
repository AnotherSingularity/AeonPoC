"""aeon — a small efficient language model with a contractive recurrent path."""
from .config import AeonConfig
from .block import AeonBlock
from .model import AeonModel, AeonForCausalLM
from .recursion import (
    RecursionChartA,
    RecursionChartB,
    audit_certificates,
    equivalence_check,
)

__all__ = [
    "AeonConfig",
    "AeonBlock",
    "AeonModel",
    "AeonForCausalLM",
    "RecursionChartA",
    "RecursionChartB",
    "audit_certificates",
    "equivalence_check",
]
