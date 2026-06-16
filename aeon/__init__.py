"""aeon — a small efficient language model with a contractive recurrent path."""
from .config import AeonConfig
from .block import AeonBlock
from .model import AeonModel, AeonForCausalLM, AeonPreTrainedModel
from .transformer import AeonDecoderLayer
from .attention import AeonAttention, repeat_kv
from .feedforward import AeonMLP
from .embedding import (
    AeonEmbedding,
    AeonRMSNorm,
    AeonRotaryEmbedding,
    rotate_half,
    apply_rotary_pos_emb,
)
from .cache import AeonCache
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
    "AeonPreTrainedModel",
    "AeonDecoderLayer",
    "AeonAttention",
    "repeat_kv",
    "AeonMLP",
    "AeonEmbedding",
    "AeonRMSNorm",
    "AeonRotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "AeonCache",
    "RecursionChartA",
    "RecursionChartB",
    "audit_certificates",
    "equivalence_check",
]
