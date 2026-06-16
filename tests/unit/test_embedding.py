"""Unit tests for aeon.embedding primitives (RMSNorm, RoPE, helpers)."""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from aeon.config import AeonConfig
from aeon.embedding import (AeonRMSNorm, AeonRotaryEmbedding, rotate_half,
                            apply_rotary_pos_emb)


def _cfg():
    return AeonConfig(vocab_size=64, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, h_rec=8)


def test_rmsnorm_preserves_dtype():
    # In a real bf16 model the norm weight is bf16 too; dtype is then preserved.
    assert AeonRMSNorm(16)(torch.randn(2, 3, 16)).dtype == torch.float32
    norm_bf = AeonRMSNorm(16).to(torch.bfloat16)
    assert norm_bf(torch.randn(2, 3, 16, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_rmsnorm_normalizes():
    norm = AeonRMSNorm(128)
    out = norm(torch.randn(4, 128) * 10.0)
    # rms of output ~ 1 (weight initialized to ones)
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rope_is_identity_at_position_zero():
    cfg = _cfg()
    rope = AeonRotaryEmbedding(cfg)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    x = torch.randn(1, 1, head_dim)
    cos, sin = rope(x, torch.tensor([[0]]))
    q = torch.randn(1, cfg.num_attention_heads, 1, head_dim)
    q2, k2 = apply_rotary_pos_emb(q, q, cos, sin)
    assert torch.allclose(q2, q, atol=1e-5)  # angle 0 -> no rotation


def test_rope_shapes():
    cfg = _cfg()
    rope = AeonRotaryEmbedding(cfg)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    cos, sin = rope(torch.randn(1, 5, head_dim), torch.arange(5)[None])
    assert cos.shape == (1, 5, head_dim) and sin.shape == (1, 5, head_dim)


def test_rotate_half_is_negation_when_applied_twice():
    x = torch.randn(2, 3, 8)
    assert torch.allclose(rotate_half(rotate_half(x)), -x, atol=1e-6)
