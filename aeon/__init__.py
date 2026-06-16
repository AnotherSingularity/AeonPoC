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


def register_for_auto_classes():
    """Register Aeon with the transformers Auto* factories.

    Lets `AutoConfig.from_pretrained` / `AutoModelForCausalLM.from_pretrained`
    resolve Aeon checkpoints in-process, and tags the classes so save_pretrained
    / push_to_hub emit an `auto_map` (loadable from the Hub with
    trust_remote_code=True). Idempotent and best-effort.
    """
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
        AutoConfig.register("aeon", AeonConfig, exist_ok=True)
        AutoModel.register(AeonConfig, AeonModel, exist_ok=True)
        AutoModelForCausalLM.register(AeonConfig, AeonForCausalLM, exist_ok=True)
    except Exception:
        pass
    try:
        AeonConfig.register_for_auto_class("AutoConfig")
        AeonModel.register_for_auto_class("AutoModel")
        AeonForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    except Exception:
        pass


register_for_auto_classes()

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
    "register_for_auto_classes",
]

