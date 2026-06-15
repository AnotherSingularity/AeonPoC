"""aeon — a small efficient language model with a contractive recurrent path."""
from .config import AeonConfig
from .block import AeonBlock
from .model import AeonModel, AeonR1ForCausalLM
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
    "AeonR1ForCausalLM",
    "RecursionChartA",
    "RecursionChartB",
    "audit_certificates",
    "equivalence_check",
]
