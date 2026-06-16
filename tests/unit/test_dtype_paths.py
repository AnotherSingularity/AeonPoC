"""Unit tests: components and the model run in fp32 and bf16 (CPU)."""
import os, sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.config import AeonConfig
from aeon.model import AeonForCausalLM
from aeon.attention import AeonAttention
from aeon.feedforward import AeonMLP
from aeon.embedding import AeonRotaryEmbedding


def _cfg():
    cfg = AeonConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                     num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                     max_position_embeddings=64, h_rec=8, tie_word_embeddings=False)
    cfg._attn_implementation = "eager"
    return cfg


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mlp_dtype(dtype):
    mlp = AeonMLP(_cfg()).to(dtype)
    out = mlp(torch.randn(2, 4, 64, dtype=dtype))
    assert out.dtype == dtype and torch.isfinite(out.float()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_attention_dtype(dtype):
    cfg = _cfg()
    attn = AeonAttention(cfg, 0).to(dtype)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    x = torch.randn(1, 4, 64, dtype=dtype)
    cos, sin = AeonRotaryEmbedding(cfg)(x, torch.arange(4)[None])
    out = attn(x, position_embeddings=(cos.to(dtype), sin.to(dtype)))
    assert out.dtype == dtype and torch.isfinite(out.float()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_model_forward_dtype(dtype):
    model = AeonForCausalLM(_cfg()).to(dtype).eval()
    ids = torch.randint(0, 64, (1, 6))
    model.reset_recursion_state(1)
    out = model(input_ids=ids)
    # logits are upcast to float in the head
    assert torch.isfinite(out.logits).all()
