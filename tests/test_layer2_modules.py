"""
Unit + parity tests for Aeon's own transformer components (Layer 2).

Each Aeon primitive is checked to be numerically identical (fp32) to the Qwen2
reference it replaces, on matched weights. That parity is what guarantees the
gamma=0 byte-identity gate keeps holding after we stopped borrowing Qwen2's code.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2 import modeling_qwen2 as q2

from aeon.embedding import AeonRMSNorm, AeonRotaryEmbedding, apply_rotary_pos_emb
from aeon.feedforward import AeonMLP
from aeon.attention import AeonAttention, repeat_kv
from aeon.transformer import AeonDecoderLayer
from aeon.cache import AeonCache


def _cfg():
    c = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, rms_norm_eps=1e-6,
    )
    c._attn_implementation = "eager"
    return c


def _max_diff(a, b):
    return (a - b).abs().max().item()


def test_rmsnorm_matches_qwen2():
    torch.manual_seed(0)
    a = AeonRMSNorm(64, eps=1e-6)
    q = q2.Qwen2RMSNorm(64, eps=1e-6)
    with torch.no_grad():
        q.weight.copy_(a.weight)
    x = torch.randn(2, 5, 64)
    assert _max_diff(a(x), q(x)) < 1e-6


def test_rotary_matches_qwen2():
    cfg = _cfg()
    a = AeonRotaryEmbedding(cfg)
    q = q2.Qwen2RotaryEmbedding(config=cfg)
    x = torch.randn(1, 7, 64)
    pos = torch.arange(7)[None]
    ca, sa = a(x, pos)
    cq, sq = q(x, pos)
    assert _max_diff(ca, cq) < 1e-6 and _max_diff(sa, sq) < 1e-6


def test_mlp_matches_qwen2():
    torch.manual_seed(0)
    cfg = _cfg()
    a = AeonMLP(cfg)
    q = q2.Qwen2MLP(cfg)
    a.load_state_dict(q.state_dict())
    x = torch.randn(2, 5, 64)
    assert _max_diff(a(x), q(x)) < 1e-6


def test_repeat_kv_shape():
    x = torch.randn(1, 2, 5, 8)
    assert repeat_kv(x, 3).shape == (1, 6, 5, 8)
    assert torch.equal(repeat_kv(x, 1), x)


def test_decoder_layer_matches_qwen2():
    """Full-layer parity: AeonDecoderLayer == Qwen2DecoderLayer on matched weights."""
    torch.manual_seed(0)
    cfg = _cfg()
    qlayer = q2.Qwen2DecoderLayer(cfg, 0).eval()
    alayer = AeonDecoderLayer(cfg, 0).eval()
    alayer.load_state_dict(qlayer.state_dict())  # names match -> strict load

    B, T = 1, 6
    x = torch.randn(B, T, cfg.hidden_size)
    pos = torch.arange(T)[None]
    cos, sin = AeonRotaryEmbedding(cfg)(x, pos)
    # explicit 4D additive causal mask, same for both
    mask = torch.full((T, T), float("-inf"))
    mask = torch.triu(mask, diagonal=1)[None, None]  # (1,1,T,T)
    cache_pos = torch.arange(T)

    with torch.no_grad():
        q_out = qlayer(x, attention_mask=mask, position_ids=pos,
                       past_key_value=None, use_cache=False,
                       cache_position=cache_pos, position_embeddings=(cos, sin))
        q_out = q_out[0] if isinstance(q_out, tuple) else q_out
        a_out = alayer(x, position_embeddings=(cos, sin), attention_mask=mask,
                       past_key_value=None, cache_position=cache_pos,
                       use_cache=False)[0]
    assert _max_diff(a_out, q_out) < 1e-5, _max_diff(a_out, q_out)


def test_attention_grad_flows():
    cfg = _cfg()
    attn = AeonAttention(cfg, 0)
    x = torch.randn(1, 4, cfg.hidden_size, requires_grad=True)
    pos = torch.arange(4)[None]
    cos, sin = AeonRotaryEmbedding(cfg)(x, pos)
    out = attn(x, position_embeddings=(cos, sin))
    out.sum().backward()
    assert attn.q_proj.weight.grad is not None
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_v2_key_remap():
    from scripts.migrate_to_v2 import remap_key
    assert remap_key("model.layers.0.qwen_block.self_attn.q_proj.weight") == \
        "model.layers.0.transformer_layer.self_attn.q_proj.weight"
    assert remap_key("model.layers.3.recursion_gate") == "model.layers.3.recursion_gate"
    assert remap_key("lm_head.weight") == "lm_head.weight"


def test_cache_update_and_length():
    cache = AeonCache()
    assert cache.get_seq_length() == 0
    k = torch.randn(1, 2, 3, 8)
    v = torch.randn(1, 2, 3, 8)
    ko, vo = cache.update(k, v, 0, {"cache_position": torch.arange(3)})
    assert ko.shape == (1, 2, 3, 8)
    assert cache.get_seq_length() == 3
    # second token appends
    k2 = torch.randn(1, 2, 1, 8)
    ko2, _ = cache.update(k2, k2, 0, {"cache_position": torch.tensor([3])})
    assert ko2.shape[-2] == 4
    assert cache.get_seq_length() == 4
